"""C2-Step 10 [LOCAL]: enumerate the full selector x axis x floor matrix + wave plan.

Produces the PLAN (no training): every cell, which cells are decisive/multi-seed, the wave
order, the score-independent kept-count allocation per (axis, floor, budget), and the
prerequisites (which selectors still need a Stage-A scoring pass on the Phase-1 pool, and which
cells already exist from Phase 1 / C2-Step 9 and can be reused).

Matrix:
  selectors (8): perplexity-high/-low/-mid, ifd, semdedup, quality, rdsplus, random
  floors (4)   : none / proportional / absolute / hybrid   (none is axis-independent)
  axes (2)     : language (agg to tier for reporting) | skill (necessity only on movable skills)
  budgets      : 10% (all) + 2% (decisive cells only)
  + full (ceiling, no floor) and base (reference, not trained)

Multi-seed (>=3) = the decisive cells only (biggest eroder x none/prop/absolute per axis):
  language: perplexity-low, quality      skill: semdedup, quality  (+ the C2-9C IF cell)
Everything else single-seed.

Per-cell qualifying-set + kept-count: the ALLOCATION (proportional/absolute/hybrid counts) is
score-independent and emitted here from the metadata + n_abs config. The QUALIFYING set
(needs-floor) is per-selector score-dependent -> emitted at score-time by
audit.stagec.qualifying_report (server, one run per selector's nlls); demoed locally with the
perplexity-low-like proxy in C2-Step 7.

    python -m audit.experiments.plan_matrix
"""
from __future__ import annotations

import argparse
import collections
import json
import os

from audit.stagec.rarity_aware import rarity_aware_select, load_nabs

SELECTORS = ["perplexity-high", "perplexity-low", "perplexity-mid",
             "ifd", "semdedup", "quality", "rdsplus", "random"]
FLOORS = ["none", "proportional", "absolute", "hybrid"]
AXES = ["language", "skill"]
BUDGETS = [0.10, 0.02]
DECISIVE_FLOORS = ["none", "proportional", "absolute"]      # the necessity contrast
DECISIVE_BY_AXIS = {"language": ["perplexity-low", "quality"],
                    "skill": ["semdedup", "quality"]}
MOVABLE_SKILLS = ["math", "instruction_following", "multilingual"]   # C2-Step 8
DECISIVE = ["ceb", "hau", "kir", "mlt", "plt", "som", "zul"]

# Selectors whose per-example scores exist for the Phase-1 pool (all three perplexity
# directions share the one nlls.pkl); random is score-free. The rest need a Stage-A pass.
HAVE_SCORES = {"perplexity-high", "perplexity-low", "perplexity-mid", "random"}
NEED_STAGE_A = [s for s in SELECTORS if s not in HAVE_SCORES]

# Cells already trained (reuse if present): Phase-1 language @10% (perplexity-low + refs) and
# the C2-Step 9 arms.
def already_trained(sel, axis, floor, budget):
    if sel == "perplexity-low" and abs(budget - 0.10) < 1e-9 and axis == "language":
        return "phase1"                                    # none/prop/abs/hybrid all trained
    if sel == "perplexity-low" and abs(budget - 0.02) < 1e-9 and floor in DECISIVE_FLOORS:
        return "c2-9-armA"
    if sel == "perplexity-low" and axis == "skill" and floor in DECISIVE_FLOORS \
            and abs(budget - 0.10) < 1e-9:
        return "c2-9-armC(IF)"                             # IF-protected skill cells
    return None


def is_decisive(sel, axis, floor, budget):
    if floor not in DECISIVE_FLOORS:
        return False
    if sel not in DECISIVE_BY_AXIS.get(axis, []):
        return False
    return True


def enumerate_cells():
    """Unique cells. `none` is axis-independent (emitted once per selector/budget with axis='-').
    2% budget only for decisive cells."""
    cells, seen = [], set()
    for sel in SELECTORS:
        for floor in FLOORS:
            axes = ["-"] if floor == "none" else AXES        # none: no groups -> axis-free
            for axis in axes:
                for budget in BUDGETS:
                    # 2% only where it can be decisive
                    dec = is_decisive(sel, axis if axis != "-" else "language", floor, budget) or \
                          (floor == "none" and sel in set(sum(DECISIVE_BY_AXIS.values(), [])))
                    if abs(budget - 0.02) < 1e-9 and not dec:
                        continue
                    key = (sel, axis, floor, budget)
                    if key in seen:
                        continue
                    seen.add(key)
                    multi = dec and floor in DECISIVE_FLOORS
                    cells.append({
                        "selector": sel, "axis": axis, "floor": floor, "budget": budget,
                        "seeds": [0, 1, 2] if multi else [0], "multi_seed": multi,
                        "decisive": bool(dec), "have_scores": sel in HAVE_SCORES,
                        "needs_stage_a": sel in NEED_STAGE_A,
                        "reuse": already_trained(sel, axis, floor, budget)})
    return cells


def assign_wave(c):
    """Wave 0 = prerequisite Stage-A scoring (not a cell). Wave 1 = decisive multi-seed (the
    necessity map). Wave 2 = rest @10% single-seed. Wave 3 = 2% non-multiseed decisive."""
    if c["reuse"]:
        return "reuse"
    if c["multi_seed"]:
        return "1_decisive_multiseed"
    if abs(c["budget"] - 0.02) < 1e-9:
        return "3_tight_budget"
    return "2_full10pct_singleseed"


def kept_counts(metadata_path):
    """Score-INDEPENDENT floor allocation per (axis, floor, budget): kept-count per group.
    Uses uniform proxy scores (allocation does not depend on the actual scores); protected set =
    the availability-floorable groups (avail >= N_abs) -- the per-selector qualifying rule
    narrows this by natural-retention at score-time."""
    import pandas as pd
    df = pd.read_parquet(metadata_path).sort_values("pool_row_idx").reset_index(drop=True)
    rows = [dict(r) for r in df.to_dict("records")]
    scores = {str(r["id"]): 0.0 for r in rows}
    gate = lambda r: not r.get("is_noised")
    report = {}
    for axis in AXES:
        nabs = load_nabs("skill") if axis == "skill" else 500
        gf = (lambda r: r.get("skill_label")) if axis == "skill" else (lambda r: r.get("language"))
        avail = collections.Counter(gf(r) for r in rows if gate(r))
        from audit.stagec.rarity_aware import _resolve_nabs
        floorable = sorted(g for g in avail if g is not None and avail[g] >= _resolve_nabs(g, nabs))
        # audit protects the AUDITED focus groups (decisive langs / movable skills), not every
        # floorable group -- so absolute floors 7x500 (language) not 8x500. Real per-selector
        # runs narrow this to the qualifying subset (needs-floor) at score-time.
        focus = DECISIVE if axis == "language" else MOVABLE_SKILLS
        from audit.stagec.rarity_aware import _resolve_nabs as _rn
        focus_prot = [g for g in focus if g in avail and avail[g] >= _rn(g, nabs)]
        for budget in BUDGETS:
            k = round(budget * len(rows))
            per_floor, sizes = {}, {}
            for floor in ("proportional", "absolute", "hybrid"):
                r = rarity_aware_select(rows, scores, False, k, floor_mode=floor, n_abs=nabs,
                                        group_key=axis, quality_gate=True, quality_fn=gate,
                                        protected_groups=(focus_prot if floor != "proportional" else None))
                # score-INDEPENDENT FLOOR (requested); leftover refill is per-selector (score-dep)
                per_floor[floor] = {g: r["allocation"].get(g, {}).get("requested", 0) for g in focus}
                sizes[floor] = r["total"]                    # may exceed k (absolute overflow)
            report[f"{axis}@{budget:.0%}"] = {
                "k": k, "floorable_groups": floorable,
                "floor_per_focus_group": per_floor, "subset_size": sizes}
    return report


def main():
    ap = argparse.ArgumentParser(description="Plan the C2-Step 10 full matrix (LOCAL).")
    ap.add_argument("--metadata", default="audit/results/stagec/phase1/pools/stagec_metadata.parquet")
    ap.add_argument("--out_dir", default="audit/results/stagec/phase2/matrix")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cells = enumerate_cells()
    for c in cells:
        c["wave"] = assign_wave(c)

    # ---- counts + rough GPU-hour estimate (@10% ~2.0h, @2% ~0.5h per training) ----
    def cell_hours(c):
        return 2.0 if abs(c["budget"] - 0.10) < 1e-9 else 0.5
    n_trainings = sum(len(c["seeds"]) for c in cells if not c["reuse"])
    gpu_h = sum(cell_hours(c) * len(c["seeds"]) for c in cells if not c["reuse"])
    gpu_h_wave1 = sum(cell_hours(c) * len(c["seeds"]) for c in cells
                      if not c["reuse"] and c["wave"] == "1_decisive_multiseed")
    waves = collections.Counter(c["wave"] for c in cells)
    by_axis = collections.Counter(c["axis"] for c in cells)
    print(f"MATRIX: {len(cells)} unique cells "
          f"({len(SELECTORS)} selectors x floors x axes x budgets; none is axis-free) "
          f"+ full(ceiling) + base(ref)")
    print(f"  cells by axis: {dict(by_axis)}")
    print(f"  cells by wave: {dict(waves)}")
    print(f"  NEW model-trainings (seeds expanded, excl. reuse): {n_trainings} "
          f"(~{gpu_h:.0f} GPU-h; Wave 1 ~{gpu_h_wave1:.0f} GPU-h)")
    print(f"  multi-seed decisive cells: {sum(1 for c in cells if c['multi_seed'])}")
    print(f"  reuse (Phase-1 / C2-9): {sum(1 for c in cells if c['reuse'])}")

    print(f"\nPREREQUISITE (Wave 0) — Stage-A scoring on the Phase-1 pool for: {NEED_STAGE_A}")
    print("  (perplexity-high/-low/-mid share the existing nlls.pkl; random is score-free.)")

    print("\nDECISIVE MULTI-SEED CELLS (Wave 1 — the necessity map):")
    for c in cells:
        if c["multi_seed"]:
            print(f"  {c['selector']:15} axis={c['axis']:9} {c['floor']:12} "
                  f"b={c['budget']:.0%}  seeds={c['seeds']}  "
                  f"{'(needs Stage-A)' if c['needs_stage_a'] else ''}")

    # ---- kept-count allocation (score-independent) ----
    kc = kept_counts(args.metadata) if os.path.exists(args.metadata) else {}
    print("\nFLOOR ALLOCATION (score-independent; focus groups; leftover refill is per-selector) ===")
    for key, d in kc.items():
        print(f"  {key} (k={d['k']}, floorable={len(d['floorable_groups'])} groups)")
        for floor, counts in d["floor_per_focus_group"].items():
            uniform = len(set(counts.values())) == 1
            shown = f"{list(counts.values())[0]}/group" if uniform else counts
            ov = d["subset_size"][floor]
            tag = f"  [subset={ov}{' OVERFLOW' if ov > d['k'] else ''}]"
            print(f"      {floor:13} {shown}{tag}")

    plan = {"selectors": SELECTORS, "floors": FLOORS, "axes": AXES, "budgets": BUDGETS,
            "movable_skills": MOVABLE_SKILLS, "need_stage_a": NEED_STAGE_A,
            "n_unique_cells": len(cells), "n_new_trainings": n_trainings,
            "cells": cells, "kept_counts": kc}
    json.dump(plan, open(os.path.join(args.out_dir, "matrix_plan.json"), "w"), indent=2)
    print(f"\nwrote {os.path.join(args.out_dir, 'matrix_plan.json')}")
    print("\nQualifying-set reports: run per selector at score-time —")
    print("  python -m audit.stagec.qualifying_report --axis {language,skill} --nlls <sel nlls>")
    print("  (local proxy demo in C2-Step 7: language->7 decisive langs; skill->{multilingual,IF,safety}).")


if __name__ == "__main__":
    main()

"""Stage C / C1-Step 2: materialize the perplexity-low variant matrix on the realistic pool.

Conditions (budget b=0.10, group axis = LANGUAGE, protected = the 7 decisive languages):
  perplexity-low  none         gate on   -- the disease (keeps ~0 decisive lang)
  + proportional  proportional gate on   -- shape-restoring; predicted FAIL (110/lang < thr)
  + absolute      absolute     gate on   -- threshold fix; predicted SUCCEED (500/lang >= thr)
  + hybrid        hybrid       gate on   -- cost-aware
  DRoP-analog     proportional gate OFF  -- class-ratio restoration, no quality gate
  random / full                          -- references
random + the floor variants are multi-seeded (>=3).

The floor allocation (kept-count per language) is SCORE-INDEPENDENT, so `--allocation_only`
prints/verifies the acceptance table (proportional < N_abs = absolute) with no model. Full
subset materialization needs the base selector's per-example score -> perplexity-low NLLs
over THIS pool (Qwen2.5-1.5B, a [SERVER] scoring pass); pass --nlls <pkl> to produce subsets.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import pickle
import random

from audit.stagec.rarity_aware import rarity_aware_select, default_quality_gate

DECISIVE = ["ceb", "hau", "kir", "mlt", "plt", "som", "zul"]
BUDGET_B = 0.10
N_ABS = 500
PILOT_THRESHOLD = 250
SEEDS = [0, 1, 2]
# (name, floor_mode, gate, multiseed)
CONDITIONS = [
    ("perplexity-low__none", "none", True, False),
    ("perplexity-low__proportional", "proportional", True, True),
    ("perplexity-low__absolute", "absolute", True, True),
    ("perplexity-low__hybrid", "hybrid", True, True),
    ("perplexity-low__proportional-nogate", "proportional", False, True),   # DRoP analog
]


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def load_pool(pool_path):
    pool = read_jsonl(pool_path)
    n_pool = len(pool)
    k = round(BUDGET_B * n_pool)
    return pool, n_pool, k


def decisive_counts(sel_ids, id2lang):
    c = collections.Counter(id2lang.get(str(i)) for i in sel_ids)
    return {l: c.get(l, 0) for l in DECISIVE}


def allocation_report(pool, id2lang):
    """Score-independent kept-count per decisive language, from the floor allocation only.
    Uses uniform proxy scores (allocation counts don't depend on the actual scores)."""
    scores = {str(r["id"]): 0.0 for r in pool}      # proxy: allocation is score-independent
    k = round(BUDGET_B * len(pool))
    print(f"pool={len(pool)} budget b={BUDGET_B:.0%} -> k={k} | N_abs={N_ABS} thr={PILOT_THRESHOLD}\n")
    print(f"{'condition':38} {'per-decisive-lang kept':>24}  {'total':>6}  {'overflow':>8}")
    rows = {}
    for name, mode, gate, _ in CONDITIONS:
        r = rarity_aware_select(pool, scores, False, k, floor_mode=mode, n_abs=N_ABS,
                                group_key="language", quality_gate=gate,
                                protected_groups=DECISIVE)
        dc = {l: r["allocation"].get(l, {}).get("filled", 0) for l in DECISIVE}
        per = dc[DECISIVE[0]]
        uniform = all(v == per for v in dc.values())
        rows[name] = dc
        print(f"{name:38} {(str(per)+'/lang' if uniform else str(dc)):>24}  "
              f"{r['total']:>6}  {r['overflow']:>8}")
    # acceptance: proportional < N_abs = absolute (per decisive language)
    prop = rows["perplexity-low__proportional"][DECISIVE[0]]
    absl = rows["perplexity-low__absolute"][DECISIVE[0]]
    hybr = rows["perplexity-low__hybrid"][DECISIVE[0]]
    ok = prop < N_ABS == absl == hybr and prop < PILOT_THRESHOLD <= absl
    print(f"\nACCEPTANCE (kept-counts): proportional({prop}) < N_abs({N_ABS}) = absolute({absl}) "
          f"= hybrid({hybr}); proportional < threshold({PILOT_THRESHOLD}) <= absolute : {ok}")
    return ok


def materialize(pool, id2lang, nlls_path, out_dir):
    """Full materialization using perplexity-low NLLs (lower NLL kept: higher_is_better=False)."""
    with open(nlls_path, "rb") as f:
        nll = pickle.load(f)                          # {pool_row_idx or id: nll}
    # map to str(id) scores
    idx2id = {r.get("pool_row_idx"): str(r["id"]) for r in pool}
    scores = {}
    for key, v in nll.items():
        sid = idx2id.get(key, str(key))
        scores[sid] = float(v)
    missing = [str(r["id"]) for r in pool if str(r["id"]) not in scores]
    if missing:
        raise SystemExit(f"{len(missing)} pool rows lack an NLL score (e.g. {missing[:3]}).")
    k = round(BUDGET_B * len(pool))
    os.makedirs(out_dir, exist_ok=True)
    id2row = {str(r["id"]): r for r in pool}
    report = {}
    for name, mode, gate, multi in CONDITIONS:
        for seed in (SEEDS if multi else [0]):
            r = rarity_aware_select(pool, scores, False, k, floor_mode=mode, n_abs=N_ABS,
                                    group_key="language", quality_gate=gate,
                                    protected_groups=DECISIVE)
            tag = f"{name}__s{seed}"
            with open(os.path.join(out_dir, f"{tag}.jsonl"), "w", encoding="utf-8") as f:
                for i in r["selected_ids"]:
                    f.write(json.dumps(id2row[i], ensure_ascii=False) + "\n")
            report[tag] = {"total": r["total"], "gate_drops": sum(r["gate_drops"].values()),
                           "decisive": decisive_counts(r["selected_ids"], id2lang)}
    # random + full references (multi-seed random)
    for seed in SEEDS:
        rng = random.Random(seed)
        ids = [str(r["id"]) for r in pool if default_quality_gate(r)]
        sel = rng.sample(ids, min(k, len(ids)))
        with open(os.path.join(out_dir, f"random__s{seed}.jsonl"), "w", encoding="utf-8") as f:
            for i in sel:
                f.write(json.dumps(id2row[i], ensure_ascii=False) + "\n")
        report[f"random__s{seed}"] = {"total": len(sel),
                                      "decisive": decisive_counts(sel, id2lang)}
    with open(os.path.join(out_dir, "full__s0.jsonl"), "w", encoding="utf-8") as f:
        for r in pool:
            if default_quality_gate(r):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(report, open(os.path.join(out_dir, "variant_report.json"), "w"), indent=2)
    print(f"materialized {len(report)+1} subsets -> {out_dir}")
    for tag, r in report.items():
        print(f"  {tag:44} total={r['total']:6} decisive={r['decisive']}")


def main():
    ap = argparse.ArgumentParser(description="Materialize the Stage-C perplexity-low variants.")
    ap.add_argument("--pool", default="audit/results/stagec/phase1/pools/stagec_pool.jsonl")
    ap.add_argument("--nlls", default=None, help="perplexity-low NLL pickle over THIS pool.")
    ap.add_argument("--out_dir", default="audit/results/stagec/phase1/subsets")
    ap.add_argument("--allocation_only", action="store_true")
    args = ap.parse_args()
    pool = read_jsonl(args.pool)
    id2lang = {str(r["id"]): r.get("language") for r in pool}
    if args.allocation_only or not args.nlls:
        ok = allocation_report(pool, id2lang)
        if not args.allocation_only:
            print("\n(no --nlls given: printed the score-independent allocation only; pass the "
                  "perplexity-low NLLs from the server scoring pass to materialize subsets.)")
        if not ok:
            raise SystemExit(1)
    else:
        materialize(pool, id2lang, args.nlls, args.out_dir)


if __name__ == "__main__":
    main()

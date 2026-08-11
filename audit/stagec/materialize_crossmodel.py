"""A-Step 3: materialize the load-bearing cross-model replication cells (MODEL-INDEPENDENT).

Selection depends only on the selector's scores + the pool, never the target model, so one
materialization serves Aya AND Llama (and reproduces the Gemma design). Cells:

  language necessity  {perplexity-low, quality} x {none, proportional, absolute} @10%,
                      language axis, protect the 7 decisive langs (by the C2 qualifying rule).
                      Tests: does quality's fair-share FAIL where perplexity-low's CURES?
  IFEval necessity    {perplexity-low, perplexity-high, quality} x {none, proportional,
                      absolute}, skill axis, protect instruction_following ONLY (clean IF
                      isolation). Tests: is the absolute floor (500 IF) required vs fair-share (96)?
  math direction-flip {perplexity-high} x {none, absolute}, skill axis, protect math ONLY.
                      Cheap anchor: confirms the new base is in a movable regime where the
                      Stage-A math deletion (perplexity-high zeroes math) propagates to GSM8K.

`none` is axis-independent (one plain top-k per selector, shared across purposes). Floors +
qualifying-by-rule unchanged from Stage C. Uses the EXISTING Gemma-era selector scores
(perplexity nlls, quality scores) -- model-independent.

    python -m audit.stagec.materialize_crossmodel --stage_a audit/results/stagec/phase2/stage_a
"""
from __future__ import annotations

import argparse
import collections
import json
import os

from audit.stagec.rarity_aware import rarity_aware_select, qualifying_groups, load_nabs
from audit.stagec.materialize_stagec_variants import read_jsonl
from audit.stagec.materialize_matrix import load_scores, SPEC

DECISIVE = ["ceb", "hau", "kir", "mlt", "plt", "som", "zul"]
SEEDS = [0, 1, 2]
# purpose -> (group axis, protected focus set, n_abs source)
PURPOSE = {
    "lang":   ("language", DECISIVE,                  "language"),
    "ifeval": ("skill",    ["instruction_following"], "skill"),
    "math":   ("skill",    ["math"],                  "skill"),
}
# axis-free plain-selector baselines (the disease); shared across purposes
NONE = ["perplexity-low", "quality", "perplexity-high"]
# floored cells (selector, purpose, floor)
FLOORED = [
    ("perplexity-low", "lang", "proportional"), ("perplexity-low", "lang", "absolute"),
    ("quality", "lang", "proportional"), ("quality", "lang", "absolute"),
    ("perplexity-low", "ifeval", "proportional"), ("perplexity-low", "ifeval", "absolute"),
    ("perplexity-high", "ifeval", "proportional"), ("perplexity-high", "ifeval", "absolute"),
    ("quality", "ifeval", "proportional"), ("quality", "ifeval", "absolute"),
    ("perplexity-high", "math", "absolute"),
]
BUDGET = 0.10


def counts(sel_ids, id2grp, groups):
    c = collections.Counter(id2grp.get(str(i)) for i in sel_ids)
    return {g: c.get(g, 0) for g in groups}


def materialize(stage_a, pool, ppl_nlls, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    id2row = {str(r["id"]): r for r in pool}
    id2lang = {str(r["id"]): r.get("language") for r in pool}
    id2skill = {str(r["id"]): r.get("skill_label") for r in pool}
    nabs = {"language": 500, "skill": load_nabs("skill")}
    k = round(BUDGET * len(pool))
    report = {}

    def write(name, ids):
        for s in SEEDS:                                  # deterministic -> seeds are training seeds
            with open(os.path.join(out_dir, f"{name}__s{s}.jsonl"), "w", encoding="utf-8") as f:
                for i in ids:
                    f.write(json.dumps(id2row[i], ensure_ascii=False) + "\n")

    print(f"pool={len(pool)} budget={BUDGET:.0%} k={k}\n{'cell':40} {'kept':>5}  focus-counts")
    # (1) axis-free none baselines
    for sel in NONE:
        scores, hib = load_scores(sel, stage_a, pool, 0, ppl_nlls)
        r = rarity_aware_select(pool, scores, hib, k, floor_mode="none",
                                group_key="language", quality_gate=True)
        write(f"{sel}__none", r["selected_ids"])
        fc = {**counts(r["selected_ids"], id2lang, DECISIVE),
              "instruction_following": counts(r["selected_ids"], id2skill, ["instruction_following"])["instruction_following"],
              "math": counts(r["selected_ids"], id2skill, ["math"])["math"]}
        report[f"{sel}__none"] = {"total": r["total"], "focus": fc}
        print(f"{sel+'__none':40} {r['total']:>5}  dec={min(counts(r['selected_ids'],id2lang,DECISIVE).values())}"
              f"-{max(counts(r['selected_ids'],id2lang,DECISIVE).values())}/lang "
              f"IF={fc['instruction_following']} math={fc['math']}")
    # (2) floored cells
    for sel, purpose, floor in FLOORED:
        axis, focus, nk = PURPOSE[purpose]
        scores, hib = load_scores(sel, stage_a, pool, 0, ppl_nlls)
        qual, _ = qualifying_groups(pool, scores, hib, k, n_abs=nabs[nk],
                                    group_key=axis, quality_gate=True)
        protected = [g for g in qual if g in focus]
        r = rarity_aware_select(pool, scores, hib, k, floor_mode=floor, n_abs=nabs[nk],
                                group_key=axis, quality_gate=True, protected_groups=protected)
        name = f"{sel}__{purpose}__{floor}"
        write(name, r["selected_ids"])
        id2grp = id2lang if axis == "language" else id2skill
        fc = counts(r["selected_ids"], id2grp, focus)
        report[name] = {"total": r["total"], "overflow": r["overflow"],
                        "protected": protected, "focus_kept": fc}
        uni = len(set(fc.values())) == 1
        print(f"{name:40} {r['total']:>5}  {(str(list(fc.values())[0])+'/grp') if uni else fc}"
              + (f"  OVERFLOW+{r['overflow']}" if r['overflow'] else ""))

    json.dump(report, open(os.path.join(out_dir, "materialize_report.json"), "w"), indent=2)
    # ---- acceptance: proportional < N_abs = absolute per language; IFEval 96 vs 500 ----
    lp = report.get("perplexity-low__lang__proportional", {}).get("focus_kept", {})
    la = report.get("perplexity-low__lang__absolute", {}).get("focus_kept", {})
    ifp = report.get("perplexity-low__ifeval__proportional", {}).get("focus_kept", {})
    ifa = report.get("perplexity-low__ifeval__absolute", {}).get("focus_kept", {})
    if lp and la:
        pl, al = lp[DECISIVE[0]], la[DECISIVE[0]]
        print(f"\nACCEPTANCE language: proportional({pl}) < N_abs(500) = absolute({al}) : "
              f"{pl < 500 == al}")
    if ifp and ifa:
        pf, af = ifp.get("instruction_following"), ifa.get("instruction_following")
        print(f"ACCEPTANCE IFEval  : proportional({pf}) < absolute({af}=500) : {pf < af == 500}")
    print(f"\nmaterialized {len(report)} cells x {len(SEEDS)} seeds -> {out_dir} "
          f"(model-independent: train on Aya + Llama)")


def main():
    ap = argparse.ArgumentParser(description="Materialize the A-Step 3 cross-model cells.")
    ap.add_argument("--pool", default="audit/results/stagec/phase1/pools/stagec_pool.jsonl")
    ap.add_argument("--stage_a", default="audit/results/stagec/phase2/stage_a")
    ap.add_argument("--ppl_nlls",
                    default="audit/results/stagec/phase1/stage_a_work/ppl_work/nlls.pkl")
    ap.add_argument("--out_dir", default="audit/results/crossmodel/subsets")
    args = ap.parse_args()
    pool = read_jsonl(args.pool)
    materialize(args.stage_a, pool, args.ppl_nlls, args.out_dir)


if __name__ == "__main__":
    main()

"""C2-Step 7: emit the qualifying-group report per (selector, axis, budget).

The rule (rarity_aware.qualifying_groups) is protected-set BY RULE, never an oracle list: a
group is floored iff it is FLOORABLE (available_after_gate >= N_abs) AND NEEDS the floor
(natural_retention under the plain selector < N_abs). N_abs comes from audit/configs/
n_abs_by_axis.json (language: uniform 500; skill: per-skill, derived from Stage B).

The `selector` is whichever per-example scores you pass:
  --nlls <pkl>   perplexity NLLs keyed by pool_row_idx (lower kept -> higher_is_better False)
  --proxy        a perplexity-low-LIKE proxy (low-resource / multilingual rows scored high, so
                 the plain selector deletes them) -- for LOCAL verification with no server nlls.

    # server, real perplexity-low scores:
    python -m audit.stagec.qualifying_report --axis language --nlls .../ppl_work/nlls.pkl
    python -m audit.stagec.qualifying_report --axis skill    --nlls .../ppl_work/nlls.pkl
"""
from __future__ import annotations

import argparse
import json
import os
import pickle

from audit.stagec.rarity_aware import qualifying_groups, load_nabs

DECISIVE = {"ceb", "hau", "kir", "mlt", "plt", "som", "zul"}


def load_meta(metadata_parquet):
    """Rows with the fields the wrapper needs (no message bodies -> gate on is_noised only)."""
    import pandas as pd
    df = pd.read_parquet(metadata_parquet).sort_values("pool_row_idx").reset_index(drop=True)
    cols = ["pool_row_idx", "id", "language", "skill_label", "resource_bucket", "is_noised"]
    return [dict(r) for r in df[cols].to_dict("records")]


def scores_from_nlls(nlls_path, rows):
    with open(nlls_path, "rb") as f:
        nll = pickle.load(f)
    idx2id = {r["pool_row_idx"]: str(r["id"]) for r in rows}
    return {idx2id.get(k, str(k)): float(v) for k, v in nll.items()}


def proxy_scores(rows):
    """perplexity-low-LIKE proxy: low-resource-bucket / multilingual rows are 'high perplexity'
    (score high) so the plain top-k (keep lowest) deletes them; everything else scores low.
    Deterministic; only for LOCAL machinery/expected-group checks, not a research number."""
    sc = {}
    for i, r in enumerate(rows):
        rare = (r.get("language") in DECISIVE or r.get("skill_label") == "multilingual"
                or r.get("resource_bucket") in (0, 1, 2))
        sc[str(r["id"])] = (100.0 + (i % 97) * 0.01) if rare else ((i % 97) * 0.001)
    return sc


def emit(rows, scores, axis, budget, nabs_path, higher_is_better=False, out_dir=None,
         selector="perplexity-low"):
    n_abs = load_nabs(axis, nabs_path)
    gate = lambda r: not r.get("is_noised")            # metadata has no message bodies
    qual, report = qualifying_groups(rows, scores, higher_is_better, budget, n_abs=n_abs,
                                     group_key=axis, quality_gate=True, quality_fn=gate)
    k = round(budget * len(rows))
    print(f"\n=== qualifying report | selector={selector} axis={axis} budget={budget:.0%} "
          f"(k={k}) ===")
    print(f"{'group':24} {'avail':>7} {'natural':>8} {'N_abs':>6} {'floorable':>9} "
          f"{'needs':>6} {'QUALIFIES':>9}")
    for g in sorted(report, key=lambda x: (-report[x]['qualifies'], str(x))):
        d = report[g]
        print(f"{str(g):24} {d['available_after_gate']:>7} {d['natural_retention']:>8} "
              f"{d['n_abs']:>6} {str(d['floorable']):>9} {str(d['needs_floor']):>6} "
              f"{str(d['qualifies']):>9}")
    print(f"QUALIFYING SET ({len(qual)}): {qual}")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        p = os.path.join(out_dir, f"qualifying__{selector}__{axis}__b{budget}.json")
        json.dump({"selector": selector, "axis": axis, "budget": budget, "k": k,
                   "qualifying": qual, "report": report}, open(p, "w"), indent=2)
        print(f"wrote {p}")
    return qual, report


def main():
    ap = argparse.ArgumentParser(description="Emit the qualifying-group report (C2-Step 7).")
    ap.add_argument("--metadata", default="audit/results/stagec/phase1/pools/stagec_metadata.parquet")
    ap.add_argument("--axis", choices=["language", "skill"], required=True)
    ap.add_argument("--budget", type=float, default=0.10)
    ap.add_argument("--nabs_config", default="audit/configs/n_abs_by_axis.json")
    ap.add_argument("--nlls", default=None, help="perplexity NLL pickle (keyed by pool_row_idx).")
    ap.add_argument("--proxy", action="store_true", help="Use the perplexity-low-like proxy score.")
    ap.add_argument("--selector", default="perplexity-low")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    rows = load_meta(args.metadata)
    if args.nlls:
        scores = scores_from_nlls(args.nlls, rows)
    elif args.proxy:
        scores = proxy_scores(rows)
    else:
        raise SystemExit("pass --nlls <pkl> (real selector) or --proxy (local check).")
    emit(rows, scores, args.axis, args.budget, args.nabs_config, out_dir=args.out_dir,
         selector=args.selector)


if __name__ == "__main__":
    main()

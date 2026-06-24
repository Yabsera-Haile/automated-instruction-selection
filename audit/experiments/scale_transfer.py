"""Scale-transfer report: do the pilot distortions hold at scale? (Ivison question)

Compares two audit_results.parquet tables (pilot vs scaled pool) at a fixed budget and,
per selector, measures whether each group's distortion direction is preserved:

  * direction agreement: fraction of shared (resource-bucket + skill) groups that fall
    on the SAME side at both scales — eroded (ratio<0.8), amplified (>1.25), or neutral.
  * low-resource pooled retention ratio (Sum kept / Sum pool over buckets 0-2, / budget)
    at each scale — the corrected aggregate (never a mean of ratios).
  * the groups whose ratio moved the most between scales.

Verdict HOLD / PARTIAL / DIVERGE from the mean direction agreement.

Usage:
    python -m audit.experiments.scale_transfer \
        --pilot audit/results/audit_results.parquet \
        --scale audit/results/scale_100000/audit_results.parquet
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

TARGET_BUDGET = 0.05
LOW_BUCKETS = ["0", "1", "2"]
ERODE, AMPLIFY = 0.8, 1.25


def _class(r: float) -> str:
    return "eroded" if r < ERODE else ("amplified" if r > AMPLIFY else "neutral")


def _ratios(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["budget"] == TARGET_BUDGET]
    return (sub.groupby(["selector", "group_col", "group"], as_index=False)
            .agg(ratio=("representation_ratio", "mean"),
                 kept=("kept_count", "mean"), pool=("pool_count", "mean")))


def _lowres_ratio(rt: pd.DataFrame) -> dict:
    lr = rt[(rt["group_col"] == "resource_bucket") & (rt["group"].isin(LOW_BUCKETS))]
    out = {}
    for sel, g in lr.groupby("selector"):
        pool = g["pool"].sum()
        out[sel] = (g["kept"].sum() / pool / TARGET_BUDGET) if pool else float("nan")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Pilot-vs-scale distortion transfer report.")
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--scale", required=True)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    pr = _ratios(pd.read_parquet(args.pilot))
    sr = _ratios(pd.read_parquet(args.scale))
    pl = _lowres_ratio(pr); sl = _lowres_ratio(sr)

    pkey = pr.set_index(["selector", "group_col", "group"])["ratio"]
    skey = sr.set_index(["selector", "group_col", "group"])["ratio"]
    selectors = sorted(set(pr["selector"]) & set(sr["selector"]))

    rows, agreements = [], []
    movers = []
    for sel in selectors:
        keys = [k for k in pkey.index if k[0] == sel and k in skey.index]
        if not keys:
            continue
        agree = sum(_class(pkey[k]) == _class(skey[k]) for k in keys) / len(keys)
        agreements.append(agree)
        rows.append((sel, agree, pl.get(sel, float("nan")), sl.get(sel, float("nan")), len(keys)))
        for k in keys:
            movers.append((abs(skey[k] - pkey[k]), sel, k[1], k[2], pkey[k], skey[k]))

    mean_agree = sum(agreements) / len(agreements) if agreements else 0.0
    verdict = ("HOLD" if mean_agree >= 0.75 else
               "PARTIAL" if mean_agree >= 0.5 else "DIVERGE")

    lines = ["# Scale-transfer report (do pilot distortions hold at scale?)", "",
             f"Budget = {TARGET_BUDGET}. Direction classes: eroded <{ERODE}, "
             f"amplified >{AMPLIFY}, else neutral.", "",
             f"## Verdict: {verdict}  (mean direction agreement = {mean_agree:.0%})", "",
             "## Per selector",
             "| Selector | direction agreement | low-res ratio (pilot→scale) | groups |",
             "|---|---|---|---|"]
    for sel, agree, lp, ls, n in sorted(rows, key=lambda x: -x[1]):
        lines.append(f"| {sel} | {agree:.0%} | {lp:.2f} → {ls:.2f} | {n} |")

    movers.sort(reverse=True)
    lines += ["", "## Largest per-group shifts (pilot → scale ratio)",
              "| Selector | group | pilot | scale |", "|---|---|---|---|"]
    for _, sel, gc, g, p, s in movers[:15]:
        label = f"bucket {g}" if gc == "resource_bucket" else g
        lines.append(f"| {sel} | {label} | {p:.2f} | {s:.2f} |")

    md = "\n".join(lines) + "\n"
    print(md)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

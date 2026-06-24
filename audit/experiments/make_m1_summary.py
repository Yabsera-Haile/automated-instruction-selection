"""Generate M1_summary.md from audit_results.parquet (Sub-step D).

Builds the Milestone-1 go/no-go summary: representation-ratio tables at the 5% budget
by resource bucket and by skill label, plus an automated GO / NO-GO / UNCLEAR call.

Go/no-go question: do hardness/perplexity selectors show representation ratio < 1 for
low-resource buckets while random stays near 1?

Usage:
    python -m audit.experiments.make_m1_summary --dev
    python -m audit.experiments.make_m1_summary --results audit/results/audit_results.parquet
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from audit.common import results_base

LOW_BUCKETS = ["0", "1", "2"]
TARGET_BUDGET = 0.05
SKILL_ORDER = ["math", "code", "multilingual", "science", "chat", "general",
               "safety", "instruction_following", "other"]


def _pivot(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Mean representation_ratio per (selector, group) at the target budget."""
    sub = df[(df["group_col"] == group_col) & (df["budget"] == TARGET_BUDGET)]
    if sub.empty:
        return pd.DataFrame()
    return sub.pivot_table(index="selector", columns="group",
                           values="representation_ratio", aggfunc="mean")


def _fmt(v) -> str:
    return "-" if pd.isna(v) else f"{v:.2f}"


def _md_table(pivot: pd.DataFrame, col_order: list[str], col_label: str) -> str:
    cols = [c for c in col_order if c in pivot.columns]
    cols += [c for c in pivot.columns if c not in cols]
    header = "| Selector | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    lines = [header, sep]
    for sel in pivot.index:
        row = " | ".join(_fmt(pivot.loc[sel, c]) for c in cols)
        lines.append(f"| {sel} | {row} |")
    return "\n".join(lines)


def _low_avg(pivot: pd.DataFrame, sel: str) -> float:
    if sel not in pivot.index:
        return float("nan")
    cols = [c for c in LOW_BUCKETS if c in pivot.columns]
    return float(pivot.loc[sel, cols].mean()) if cols else float("nan")


def decide(rb_pivot: pd.DataFrame) -> tuple[str, str]:
    """Return (verdict, rationale) from the resource-bucket pivot at 5%."""
    if rb_pivot.empty or "random" not in rb_pivot.index:
        return "UNCLEAR", "Missing random baseline or resource-bucket results."
    rnd = _low_avg(rb_pivot, "random")
    if not (0.7 <= rnd <= 1.3):
        return "UNCLEAR", (f"Random low-resource ratio {rnd:.2f} is not near 1.0, so "
                           "the baseline itself is off — cannot read the contrast.")
    flagged = []
    for sel in ("perplexity-high", "rdsplus"):
        la = _low_avg(rb_pivot, sel)
        if pd.notna(la) and la < 0.75 and la < rnd - 0.15:
            flagged.append(f"{sel} ({la:.2f})")
    if flagged:
        return "GO", (f"Random low-resource ratio ≈ {rnd:.2f} (fair); "
                      f"under-selecting: {', '.join(flagged)} < 1.")
    return "NO-GO", (f"Random low-resource ratio ≈ {rnd:.2f}, but no selector drives "
                     "low-resource representation meaningfully below 1 at 5%.")


def build_markdown(df: pd.DataFrame, dev: bool) -> str:
    rb = _pivot(df, "resource_bucket")
    sk = _pivot(df, "skill_label")
    verdict, rationale = decide(rb)

    parts = ["# Milestone 1 Summary", ""]
    if dev:
        parts += ["> ⚠️ **DEV RUN — small proxy models, NOT research results.**",
                  "> Perplexity = Pythia-160m, RDS+ = all-MiniLM-L6-v2 embeddings on a "
                  "4GB dev GPU. Real M1 numbers come from the GPU server (perplexity "
                  "≥1B, RDS+ 7B). Treat the verdict below as a pipeline demonstration.",
                  ""]
    parts += [
        "## Go/no-go question",
        "Do hardness/perplexity selectors show representation ratio < 1 for low-resource",
        "buckets while random stays near 1?",
        "",
        f"## Result: {verdict}",
        "",
        rationale,
        "",
        "## Perplexity direction (explicit)",
        "The `perplexity-*` selectors differ only in which perplexity band they keep:",
        "- **high** — keep the most-surprising (highest-perplexity) examples. This was the",
        "  original M1 \"perplexity\" selector (repo `ppl_selections.py` default: sort NLL",
        "  descending, take top-k). It is the one the go/no-go question refers to.",
        "- **low** — keep the least-surprising (lowest-perplexity) examples (the common",
        "  practitioner \"remove high-perplexity junk\" filter).",
        "- **mid** — keep the central budget-fraction by perplexity rank (drop both tails;",
        "  the \"When Less is More\" strategy).",
        "",
        "## Representation ratios at 5% budget by resource bucket",
        "(1.0 = fair share; <1 = under-selected. Random averaged over seeds 0,1,2.)",
        "",
        _md_table(rb, [str(i) for i in range(6)], "bucket") if not rb.empty else "_no data_",
        "",
        "## Representation ratios at 5% budget by skill label",
        "",
        _md_table(sk, SKILL_ORDER, "skill") if not sk.empty else "_no data_",
        "",
        "## Interpretation",
        _interpretation(rb, verdict, dev),
        "",
    ]
    return "\n".join(parts)


def _interpretation(rb: pd.DataFrame, verdict: str, dev: bool) -> str:
    rnd = _low_avg(rb, "random")
    ppl = _low_avg(rb, "perplexity-high")
    rds = _low_avg(rb, "rdsplus")
    s = (f"At 5% budget, mean low-resource (buckets 0-2) representation ratios are: "
         f"random {rnd:.2f}, perplexity-high {ppl:.2f}, RDS+ {rds:.2f}. "
         f"A ratio below 1 means a selector keeps low-resource languages at less than "
         f"their pool share. ")
    if dev:
        s += ("Because the dev perplexity model is tiny and English-centric, its "
              "perplexity ranking can behave differently from a real ≥1B model "
              "(small models often assign *high* perplexity to low-resource text, "
              "which top-perplexity selection would then over-select) — so the dev "
              "verdict is illustrative only. The pipeline is confirmed end to end; "
              "rerun on the GPU server for the real go/no-go.")
    else:
        s += ("If perplexity/RDS+ sit well below random here, the project's central "
              "hypothesis holds and we proceed to the noise-disentanglement and "
              "rarity-aware phases.")
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate M1_summary.md.")
    ap.add_argument("--results", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--dev", action="store_true")
    args = ap.parse_args()

    base = results_base(args.dev)
    results = args.results or os.path.join(base, "audit_results.parquet")
    output = args.output or os.path.join(base, "M1_summary.md")

    df = pd.read_parquet(results)
    md = build_markdown(df, args.dev)
    with open(output, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Wrote {output}")
    print("\n" + md)


if __name__ == "__main__":
    main()

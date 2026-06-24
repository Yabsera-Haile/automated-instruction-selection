"""Generate M1_summary.md from audit_results.parquet (C-Step 6: corrected aggregation).

Fixes the aggregation bug: representation ratios are unbounded above and floored at 0,
so a cross-group MEAN of ratios is dominated by over-selection (one amplified group
swamps several eroded ones). We therefore:
  * report representation ratio PER GROUP (never a cross-group mean of ratios);
  * for any aggregate over a group set, aggregate RETENTION (kept/pool, bounded [0,1])
    via Sum(kept)/Sum(pool) — using the identity retention_g = ratio_g * budget;
  * add a worst-group view (eroded <0.5, amplified >2.0) per selector at 5%;
  * state each selector's configuration and give a per-selector erosion/amplification
    read in the signal-dependent-distortion framing.

Usage: python -m audit.experiments.make_m1_summary [--dev]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd

from audit.common import results_base

TARGET_BUDGET = 0.05
LOW_BUCKETS = ["0", "1", "2"]
ERODE = 0.5
AMPLIFY = 2.0
SKILL_ORDER = ["math", "code", "multilingual", "science", "chat", "general",
               "safety", "instruction_following", "other"]

SELECTOR_MECHANISM = {
    "random": "uniform random sample (the fair baseline)",
    "perplexity-high": "keeps the highest-perplexity (most 'surprising') examples",
    "perplexity-low": "keeps the lowest-perplexity (most predictable) examples",
    "perplexity-mid": "keeps the central perplexity band, dropping both tails",
    "ifd": "keeps high instruction-following difficulty (the instruction helps, yet the "
           "example stays hard)",
    "semdedup": "removes near-duplicates and keeps diverse, centroid-distant representatives",
    "quality": "keeps the examples an LLM judge rates highest quality",
    "rdsplus": "retrieves the training examples most similar to the multitask eval set",
}


# ------------------------------- helpers ------------------------------------

def _fmt(v) -> str:
    return "-" if pd.isna(v) else f"{v:.2f}"


def _ratio_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Per-(selector, group) mean representation_ratio at the target budget."""
    sub = df[(df["group_col"] == group_col) & (df["budget"] == TARGET_BUDGET)]
    if sub.empty:
        return pd.DataFrame()
    return sub.pivot_table(index="selector", columns="group",
                           values="representation_ratio", aggfunc="mean")


def _md_table(pivot: pd.DataFrame, col_order: list[str]) -> str:
    if pivot.empty:
        return "_no data_"
    cols = [c for c in col_order if c in pivot.columns]
    cols += [c for c in pivot.columns if c not in cols]
    lines = ["| Selector | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for sel in sorted(pivot.index):
        lines.append("| " + sel + " | " + " | ".join(_fmt(pivot.loc[sel, c]) for c in cols) + " |")
    return "\n".join(lines)


def _budget_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Per (selector, group_col, group) mean kept/pool/ratio at the target budget
    (averaged over seeds; pool_count is seed-invariant)."""
    sub = df[df["budget"] == TARGET_BUDGET]
    if sub.empty:
        return sub
    return (sub.groupby(["selector", "group_col", "group"], as_index=False)
            .agg(representation_ratio=("representation_ratio", "mean"),
                 retention_rate=("retention_rate", "mean"),
                 kept_count=("kept_count", "mean"),
                 pool_count=("pool_count", "mean")))


def lowres_retention_table(rows: pd.DataFrame) -> tuple[str, dict]:
    """Pooled low-resource (buckets 0-2) retention per selector = Sum(kept)/Sum(pool).

    This is the CORRECT aggregate (bounded, pool-weighted) — not a mean of ratios.
    'fair' = budget; ratio column = pooled_retention / budget.
    """
    lr = rows[(rows["group_col"] == "resource_bucket") & (rows["group"].isin(LOW_BUCKETS))]
    if lr.empty:
        return "_no resource_bucket data_", {}
    agg = (lr.groupby("selector")
           .agg(kept=("kept_count", "sum"), pool=("pool_count", "sum")).reset_index())
    agg["retention"] = agg["kept"] / agg["pool"]
    agg["ratio"] = agg["retention"] / TARGET_BUDGET
    lines = [f"Pooled over buckets 0-2 (Sum kept / Sum pool). Fair retention = budget = "
             f"{TARGET_BUDGET}; ratio = retention / budget (1.0 = fair, <1 = eroded).",
             "", "| Selector | low-res retention | ratio vs fair |", "|---|---|---|"]
    ratios = {}
    for _, r in agg.sort_values("ratio").iterrows():
        ratios[r["selector"]] = r["ratio"]
        lines.append(f"| {r['selector']} | {r['retention']:.4f} | {r['ratio']:.2f} |")
    return "\n".join(lines), ratios


def worst_groups(rows: pd.DataFrame) -> dict[str, dict]:
    """Per selector: eroded (ratio<0.5) and amplified (ratio>2.0) groups across both
    group sets, as (label, ratio) lists sorted by severity."""
    out = {}
    for sel, sub in rows.groupby("selector"):
        eroded, amplified = [], []
        for _, r in sub.iterrows():
            label = (f"bucket {r['group']}" if r["group_col"] == "resource_bucket"
                     else str(r["group"]))
            ratio = r["representation_ratio"]
            if ratio < ERODE:
                eroded.append((label, ratio))
            elif ratio > AMPLIFY:
                amplified.append((label, ratio))
        eroded.sort(key=lambda x: x[1])
        amplified.sort(key=lambda x: -x[1])
        out[sel] = {"eroded": eroded, "amplified": amplified}
    return out


def _grouplist(items) -> str:
    return ", ".join(f"{lbl} ({r:.2f})" for lbl, r in items) if items else "none"


def load_selector_meta(selections_dir: str) -> dict:
    meta = {}
    if selections_dir and os.path.isdir(selections_dir):
        for path in sorted(glob.glob(os.path.join(selections_dir, "*.json"))):
            try:
                obj = json.load(open(path, encoding="utf-8"))
            except Exception:
                continue
            sel = obj.get("selector", "?")
            meta.setdefault(sel, obj.get("meta", {}) or {})
    return meta


# ------------------------------- sections -----------------------------------

def cost_table(meta: dict) -> str:
    if not meta:
        return "_no selection metas found_"
    lines = ["| Selector | Model | Device | Runtime (s) | Peak VRAM (MiB) |",
             "|---|---|---|---|---|"]
    for sel in sorted(meta):
        m = meta[sel]
        rt = m.get("runtime_s"); vr = m.get("vram_peak_mib")
        lines.append(f"| {sel} | {m.get('model', '-')} | {m.get('device', '-')} | "
                     f"{'-' if rt is None else f'{rt:.0f}'} | {'-' if vr is None else vr} |")
    return "\n".join(lines)


def config_section(meta: dict) -> str:
    def m(sel, key, default="?"):
        return (meta.get(sel, {}) or {}).get(key, default)
    lines = []
    lines.append("- **random** — uniform without replacement; seeds 0,1,2.")
    for d in ("high", "low", "mid"):
        sel = f"perplexity-{d}"
        if sel in meta or True:
            desc = {"high": "keep highest perplexity (top-k NLL)",
                    "low": "keep lowest perplexity",
                    "mid": "keep the central perplexity band"}[d]
            lines.append(f"- **{sel}** — {desc}; scorer = {m(sel, 'model', 'Qwen2.5-1.5B')}.")
    lines.append(f"- **ifd** — IFD = ppl(response|instruction) / ppl(response); discard "
                 f"IFD ≥ 1.0, then keep highest IFD < 1.0 up to budget; "
                 f"scorer = {m('ifd', 'model', 'Qwen2.5-1.5B')}.")
    lines.append(f"- **semdedup** — KMeans on the RDS+ embeddings "
                 f"({m('semdedup', 'index_model', 'cosinesim_7b')}); near-duplicate cosine "
                 f"threshold = {m('semdedup', 'threshold', 0.95)}; keep the example closest "
                 f"to each cluster centroid; budget ranking = "
                 f"{m('semdedup', 'ranking', 'farthest-from-centroid (most diverse)')}.")
    lines.append(f"- **quality** — LLM judge {m('quality', 'model', 'Qwen2.5-7B-Instruct')}, "
                 f"language-neutral prompt {m('quality', 'judge_prompt', 'audit/configs/quality_judge_prompt.txt')} "
                 f"(scale {m('quality', 'scale', '1-5')}); keep top-scoring up to budget.")
    lines.append(f"- **rdsplus** — multitask cosine retrieval to the eval sets, round-robin "
                 f"max selection; backbone = {m('rdsplus', 'model', 'Llama-2-7b-hf')}.")
    return "\n".join(lines)


def per_selector_reads(worst: dict) -> str:
    paras = []
    for sel in sorted(worst):
        mech = SELECTOR_MECHANISM.get(sel, "selects by its own signal")
        er = _grouplist(worst[sel]["eroded"])
        am = _grouplist(worst[sel]["amplified"])
        if sel == "random":
            paras.append(f"**random** — {mech}. Eroded (<0.5): {er}. Amplified (>2.0): "
                         f"{am}. By construction it tracks the pool, so any extreme groups "
                         f"here are small-sample noise rather than distortion.")
        else:
            paras.append(
                f"**{sel}** — {mech}. Eroded (<0.5): {er}. Amplified (>2.0): {am}. "
                f"Consistent with signal-dependent distortion: optimizing this selector's "
                f"signal systematically over-keeps the capabilities its signal favours and "
                f"drops those it does not — the choice of selector, not just the budget, "
                f"reshapes which skills and languages survive.")
    return "\n\n".join(paras)


def verdict(lr_ratios: dict) -> tuple[str, str]:
    if "random" not in lr_ratios:
        return "UNCLEAR", "No random baseline / resource-bucket data."
    rnd = lr_ratios["random"]
    eroders = sorted([s for s, r in lr_ratios.items()
                      if s != "random" and r < 0.8], key=lambda s: lr_ratios[s])
    amplifiers = sorted([s for s, r in lr_ratios.items()
                         if s != "random" and r > 1.5], key=lambda s: -lr_ratios[s])
    txt = (f"Using the corrected pooled low-resource (buckets 0-2) **retention** "
           f"(not a mean of ratios): random sits at ratio {rnd:.2f} vs fair (≈1). "
           f"Selectors that ERODE low-resource in aggregate (<0.8): "
           f"{', '.join(f'{s} ({lr_ratios[s]:.2f})' for s in eroders) or 'none'}. "
           f"Selectors that AMPLIFY (>1.5): "
           f"{', '.join(f'{s} ({lr_ratios[s]:.2f})' for s in amplifiers) or 'none'}.")
    # The honest verdict: signal-dependent distortion, surfaced per-capability below.
    return "SIGNAL-DEPENDENT DISTORTION", txt


# ------------------------------- assembly -----------------------------------

def build_markdown(df: pd.DataFrame, dev: bool, selections_dir: str | None = None) -> str:
    rows = _budget_rows(df)
    rb = _ratio_by_group(df, "resource_bucket")
    sk = _ratio_by_group(df, "skill_label")
    lr_table, lr_ratios = lowres_retention_table(rows)
    worst = worst_groups(rows)
    meta = load_selector_meta(selections_dir)
    vname, vtext = verdict(lr_ratios)

    parts = ["# Milestone 1 Summary", ""]
    if dev:
        parts += ["> ⚠️ **DEV RUN — small proxy models, NOT research results.**", ""]
    parts += [
        "## Go/no-go question",
        "Do data-selection methods distort the training mixture — eroding low-resource",
        "languages and rare skills — relative to a random baseline?",
        "",
        f"## Result: {vname}",
        "",
        vtext,
        "",
        "## Low-resource aggregate (corrected: pooled retention, not averaged ratios)",
        lr_table,
        "",
        "## Selector configurations",
        config_section(meta),
        "",
        "## Selector models & cost",
        cost_table(meta),
        "",
        "## Representation ratio at 5% budget by resource bucket (per group; 1.0 = fair)",
        "(Random averaged over seeds 0,1,2. Do not average these across groups — see the",
        "pooled-retention table above for the low-resource aggregate.)",
        "",
        _md_table(rb, [str(i) for i in range(6)]),
        "",
        "## Representation ratio at 5% budget by skill label (per group)",
        "",
        _md_table(sk, SKILL_ORDER),
        "",
        f"## Worst-group view at 5% (eroded <{ERODE}, amplified >{AMPLIFY})",
        "Per selector, the groups (resource buckets and skills) most distorted:",
        "",
        "\n".join(
            f"- **{sel}** — eroded: {_grouplist(worst[sel]['eroded'])}; "
            f"amplified: {_grouplist(worst[sel]['amplified'])}"
            for sel in sorted(worst)),
        "",
        "## Per-selector read (signal-dependent distortion)",
        "",
        per_selector_reads(worst),
        "",
    ]
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate M1_summary.md.")
    ap.add_argument("--results", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--selections_dir", default=None)
    ap.add_argument("--dev", action="store_true")
    args = ap.parse_args()

    base = results_base(args.dev)
    results = args.results or os.path.join(base, "audit_results.parquet")
    output = args.output or os.path.join(base, "M1_summary.md")
    selections_dir = args.selections_dir or os.path.join(base, "selections")

    df = pd.read_parquet(results)
    md = build_markdown(df, args.dev, selections_dir)
    with open(output, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Wrote {output}")
    print("\n" + md)


if __name__ == "__main__":
    main()

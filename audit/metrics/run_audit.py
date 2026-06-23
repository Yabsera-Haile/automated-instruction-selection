"""Run the representation audit over a directory of canonical selection JSONs.

For each selection, compute per-group summaries (representation ratio etc.) for each
requested group column, assemble one tidy results table, save it to parquet, and plot
representation ratio vs budget (faceted per group, one line per selector).

Canonical selection JSON (produced in Phase 3; see audit/REPO_MAP.md sec. 9)::

    {"selector": "rds_plus", "budget": 1000, "seed": 42,
     "pool": "audit/results/pool_pilot.jsonl",
     "selected_ids": ["<id>", ...]}          # or "selected_indices": [int, ...]

Usage:
    python -m audit.metrics.run_audit \
        --metadata audit/results/metadata_pilot.parquet \
        --selections_dir audit/results/selections/

No selections yet (Phase 3 makes them) => warn and exit 0, so this is testable now.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os

import pandas as pd

from audit.metrics import audit_metrics as M

logger = logging.getLogger("audit.run_audit")

# Okabe-Ito colorblind-safe palette.
OKABE_ITO = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#E69F00", "#56B4E9", "#F0E442", "#000000",
]

TIDY_COLUMNS = [
    "selector", "budget", "seed", "group_col", "group",
    "pool_count", "pool_share", "kept_count", "kept_share",
    "retention_rate", "representation_ratio",
]


def parse_args():
    p = argparse.ArgumentParser(description="Representation audit over selection JSONs.")
    p.add_argument("--metadata", required=True, help="Path to metadata parquet.")
    p.add_argument("--selections_dir", required=True,
                   help="Directory of canonical selection JSON files.")
    p.add_argument("--output_dir", default="audit/results",
                   help="Where audit_results.parquet and plots/ are written.")
    p.add_argument("--group_cols", nargs="+", default=["resource_bucket", "skill_label"])
    return p.parse_args()


def load_selection(path: str, idx2id: dict) -> dict:
    """Return {selector, budget, seed, selected_ids} for one selection JSON."""
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    stem = os.path.splitext(os.path.basename(path))[0]

    if "selected_ids" in obj and obj["selected_ids"] is not None:
        selected_ids = [str(i) for i in obj["selected_ids"]]
    elif "selected_indices" in obj and obj["selected_indices"] is not None:
        selected_ids = [idx2id[i] for i in obj["selected_indices"] if i in idx2id]
        missing = sum(1 for i in obj["selected_indices"] if i not in idx2id)
        if missing:
            logger.warning("%s: %d selected_indices not in metadata pool.", stem, missing)
    else:
        raise ValueError(f"{path}: no 'selected_ids' or 'selected_indices'.")

    return {
        "selector": obj.get("selector", stem),
        "budget": int(obj.get("budget", len(selected_ids))),
        "seed": obj.get("seed", 0),
        "selected_ids": selected_ids,
    }


def build_tidy(df: pd.DataFrame, selections: list[dict], group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for sel in selections:
        for group_col in group_cols:
            if group_col not in df.columns:
                logger.warning("group_col %r not in metadata; skipping.", group_col)
                continue
            summary = M.summarise(df, sel["selected_ids"], group_col)
            for _, r in summary.iterrows():
                rows.append({
                    "selector": sel["selector"],
                    "budget": sel["budget"],
                    "seed": sel["seed"],
                    "group_col": group_col,
                    # stringify: group ids are heterogeneous across group_cols
                    # (int resource_bucket vs str skill_label) -> keep one dtype.
                    "group": str(r["group"]),
                    "pool_count": r["pool_count"],
                    "pool_share": r["pool_share"],
                    "kept_count": r["kept_count"],
                    "kept_share": r["kept_share"],
                    "retention_rate": r["retention_rate"],
                    "representation_ratio": r["representation_ratio"],
                })
    return pd.DataFrame(rows, columns=TIDY_COLUMNS)


def plot_faceted(tidy: pd.DataFrame, group_col: str, out_path: str) -> None:
    """Faceted plot: one subplot per group, representation ratio vs budget, line/selector."""
    import math

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = tidy[tidy["group_col"] == group_col]
    if sub.empty:
        logger.warning("Nothing to plot for group_col=%s.", group_col)
        return
    groups = sorted(sub["group"].unique(), key=lambda x: (str(type(x)), x))
    selectors = sorted(sub["selector"].unique())

    ncols = min(3, len(groups))
    nrows = math.ceil(len(groups) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows),
                             squeeze=False, sharex=True)
    color = {s: OKABE_ITO[i % len(OKABE_ITO)] for i, s in enumerate(selectors)}

    for ax_idx, g in enumerate(groups):
        ax = axes[ax_idx // ncols][ax_idx % ncols]
        gg = sub[sub["group"] == g]
        for s in selectors:
            d = gg[gg["selector"] == s].sort_values("budget")
            if d.empty:
                continue
            ax.plot(d["budget"], d["representation_ratio"], marker="o",
                    color=color[s], label=s)
        ax.axhline(1.0, ls="--", color="gray", lw=1, label="_fair")
        ax.set_title(f"{group_col} = {g}")
        ax.set_xlabel("budget")
        ax.set_ylabel("representation ratio")
    # hide unused axes
    for ax_idx in range(len(groups), nrows * ncols):
        axes[ax_idx // ncols][ax_idx % ncols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    handles = [h for h, l in zip(handles, labels) if not l.startswith("_")]
    labels = [l for l in labels if not l.startswith("_")]
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 6),
                   bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"Representation ratio vs budget, faceted by {group_col}", y=1.06)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    logger.info("Wrote plot -> %s", out_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    df = pd.read_parquet(args.metadata)
    logger.info("Loaded metadata: %d rows from %s", len(df), args.metadata)

    selection_files = sorted(glob.glob(os.path.join(args.selections_dir, "*.json")))
    if not selection_files:
        logger.warning(
            "No selection JSONs found in %s. Phase 3 creates these; nothing to audit. "
            "Exiting cleanly.", args.selections_dir,
        )
        return

    idx2id = dict(zip(df["pool_row_idx"], df["id"])) if "pool_row_idx" in df.columns else {}
    selections = [load_selection(p, idx2id) for p in selection_files]
    logger.info("Loaded %d selection(s): %s",
                len(selections), [s["selector"] for s in selections])

    tidy = build_tidy(df, selections, args.group_cols)
    os.makedirs(args.output_dir, exist_ok=True)
    out_parquet = os.path.join(args.output_dir, "audit_results.parquet")
    tidy.to_parquet(out_parquet, index=False)
    logger.info("Wrote tidy results: %d rows -> %s", len(tidy), out_parquet)

    plots_dir = os.path.join(args.output_dir, "plots")
    for group_col in args.group_cols:
        plot_faceted(tidy, group_col, os.path.join(plots_dir, f"representation_by_{group_col}.png"))


if __name__ == "__main__":
    main()

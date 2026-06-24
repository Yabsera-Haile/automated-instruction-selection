"""Perplexity selection adapter (Sub-step C / C-Step 1: directional).

Stage 1 SCORE: `minimal_multitask.compute_influence_perplexity` -> nlls.pkl
(per-example NLL = perplexity, keyed by pool_row_idx). Direction-independent, so it
runs once and all directions/budgets are derived from it.

Stage 2 SELECT by --direction:
  * high - keep the MOST-surprising (highest perplexity). This is the original M1
           behavior and matches the repo's scripts/ppl_selections.py default, which
           sorts NLL descending and takes the top-k. (It explains M1: math/code -> 0,
           multilingual -> ~7x, because predictable text is low-perplexity.)
  * low  - keep the LEAST-surprising (lowest perplexity): the common practitioner
           "remove high-perplexity junk" filter.
  * mid  - keep the central budget-fraction by perplexity rank (drop both the lowest
           and highest tails): the "When Less is More" strategy.

We compute selection here from nlls.pkl (not via ppl_selections.py) so we can add the
`low` direction and use a precise central-band `mid` (the repo's --mid_ppl uses a
fixed 33-66 percentile slice, a different definition).

Models (overridable via --model):
  * --dev : EleutherAI/pythia-160m proxy  -> audit/results/dev/. NOT research results.
  * real  : Qwen/Qwen2.5-1.5B (multilingual >=1B) -> audit/results/. Same model for
            all three directions (only the selection rule differs).

Perplexity is deterministic => seed fixed at 0.
filename: perplexity-{direction}__b{budget}__s0.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import subprocess
import sys

import pandas as pd

from audit.common import (PPL_DEV_MODEL, PPL_REAL_MODEL, VramSampler,
                          announce_dev, results_base, run_meta)

logger = logging.getLogger("audit.run_perplexity")

BUDGETS = [0.01, 0.05, 0.10, 0.25, 0.50]
DIRECTIONS = ["high", "low", "mid"]


def score_pool(pool: str, work_dir: str, model: str, dtype: str, batch_size: int,
               force: bool = False) -> tuple[str, VramSampler]:
    """Run compute_influence_perplexity -> work_dir/nlls.pkl. Returns (path, sampler)."""
    os.makedirs(work_dir, exist_ok=True)  # repo script does not mkdir its save_dir
    nlls = os.path.join(work_dir, "nlls.pkl")
    sampler = VramSampler()
    if os.path.exists(nlls) and not force:
        logger.info("Reusing existing perplexity scores: %s", nlls)
        return nlls, sampler
    cmd = [
        sys.executable, "-m", "minimal_multitask.compute_influence_perplexity",
        "--model_name", model,
        "--train_dataset", pool,
        "--save_dir", work_dir,
        "--dtype", dtype,
        "--batch_size", str(batch_size),
        "--seed", "0",
    ]
    logger.info("Scoring perplexity (model=%s, dtype=%s) -> %s", model, dtype, nlls)
    with sampler:
        subprocess.run(cmd, check=True)
    if not os.path.exists(nlls):
        raise RuntimeError(f"Perplexity scoring did not produce {nlls}")
    logger.info("Scoring done in %.1fs, peak VRAM %s MiB", sampler.runtime_s, sampler.peak_mib)
    return nlls, sampler


def select_ids_by_direction(nlls: dict, direction: str, k: int, idx2id: dict) -> list[str]:
    """Select k pool ids by perplexity direction (high|low|mid)."""
    items = sorted(nlls.items(), key=lambda kv: kv[1])  # ascending NLL (perplexity)
    n = len(items)
    k = min(k, n)
    if direction == "high":
        chosen = items[n - k:]            # highest perplexity (most surprising)
    elif direction == "low":
        chosen = items[:k]                # lowest perplexity (least surprising)
    elif direction == "mid":
        start = (n - k) // 2              # central budget-fraction by rank
        chosen = items[start:start + k]
    else:
        raise ValueError(f"Unknown direction: {direction!r}")
    return [idx2id[int(idx)] for idx, _ in chosen if int(idx) in idx2id]


def generate(pool: str, metadata: str, out_dir: str, work_dir: str, model: str,
             dev: bool, directions=DIRECTIONS, dtype: str = "fp32", batch_size: int = 1,
             budgets=BUDGETS, force_score: bool = False) -> list[str]:
    meta_df = pd.read_parquet(metadata)
    n_pool = len(meta_df)
    idx2id = {int(r): str(i) for r, i in zip(meta_df["pool_row_idx"], meta_df["id"])}
    os.makedirs(out_dir, exist_ok=True)

    nlls_path, sampler = score_pool(pool, work_dir, model, dtype, batch_size, force=force_score)
    with open(nlls_path, "rb") as f:
        nlls = pickle.load(f)
    base_meta = run_meta(model, dev, sampler, extra={"selector_kind": "perplexity", "dtype": dtype})

    written = []
    for direction in directions:
        for budget in budgets:
            k = round(budget * n_pool)
            ids = select_ids_by_direction(nlls, direction, k, idx2id)
            obj = {"selector": f"perplexity-{direction}", "budget": budget, "seed": 0,
                   "selected_ids": ids, "meta": {**base_meta, "direction": direction}}
            path = os.path.join(out_dir, f"perplexity-{direction}__b{budget}__s0.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f)
            written.append(path)
            logger.info("perplexity-%s b=%.2f -> %d ids -> %s",
                        direction, budget, len(ids), os.path.basename(path))
    return written


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Directional perplexity selection adapter.")
    ap.add_argument("--pool", default="audit/results/pilot_pool.jsonl")
    ap.add_argument("--metadata", default="audit/results/metadata_pilot.parquet")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--work_dir", default=None)
    ap.add_argument("--model", default=None, help="Override the perplexity model.")
    ap.add_argument("--direction", nargs="+", choices=DIRECTIONS, default=DIRECTIONS,
                    help="Which perplexity direction(s) to emit (default: all three).")
    ap.add_argument("--dtype", default="fp32")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--dev", action="store_true", help="Dev run with proxy model -> dev/.")
    ap.add_argument("--force_score", action="store_true")
    args = ap.parse_args()

    announce_dev(args.dev, logger)
    base = results_base(args.dev)
    out_dir = args.out_dir or os.path.join(base, "selections")
    work_dir = args.work_dir or os.path.join(base, "ppl_work")
    model = args.model or (PPL_DEV_MODEL if args.dev else PPL_REAL_MODEL)

    paths = generate(args.pool, args.metadata, out_dir, work_dir, model, args.dev,
                     directions=args.direction, dtype=args.dtype,
                     batch_size=args.batch_size, force_score=args.force_score)
    logger.info("Wrote %d perplexity selection files to %s", len(paths), out_dir)


if __name__ == "__main__":
    main()

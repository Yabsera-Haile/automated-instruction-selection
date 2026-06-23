"""Perplexity selection adapter (Sub-step C).

Two stages, using the repo's existing scripts:
  1. SCORE: `minimal_multitask.compute_influence_perplexity` -> nlls.pkl (per-example
     NLL, keyed by pool_row_idx). (The brief says "ppl_selections.py scores" — in this
     repo scoring is compute_influence_perplexity; ppl_selections does the selection.)
  2. SELECT: `scripts.ppl_selections` writes a JSONL of the top-NLL examples for a
     given output_size; we read back each `id` and emit canonical selection JSON.

Models (overridable via --model):
  * --dev  : EleutherAI/pythia-160m  (GPT-2-class proxy; 2048 ctx — literal GPT-2's
             1024 ctx crashes the repo's 2048-token tokenization on long examples).
             Output -> audit/results/dev/.  NOT research results.
  * real   : EleutherAI/pythia-1.4b  (>=1B per spec; ungated). Output -> audit/results/.
             The GPU server may override with a larger model via --model.

Perplexity is deterministic => seed fixed at 0. Budgets: fractions of the pool.
filename: perplexity__b{budget}__s0.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys

import pandas as pd

from audit.common import (PPL_DEV_MODEL, PPL_REAL_MODEL, VramSampler,
                          announce_dev, results_base, run_meta)

logger = logging.getLogger("audit.run_perplexity")

BUDGETS = [0.01, 0.05, 0.10, 0.25, 0.50]


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


def select_budget(nlls: str, pool: str, output_size: int, work_dir: str) -> list[str]:
    out_jsonl = os.path.join(work_dir, f"ppl_top_{output_size}.jsonl")
    cmd = [
        sys.executable, "-m", "scripts.ppl_selections",
        "--ppl_scores", nlls,
        "--train_datasets", pool,
        "--output_size", str(output_size),
        "--output_file_path", out_jsonl,
    ]
    subprocess.run(cmd, check=True)
    ids = []
    with open(out_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(str(json.loads(line)["id"]))
    return ids


def generate(pool: str, metadata: str, out_dir: str, work_dir: str, model: str,
             dev: bool, dtype: str = "fp32", batch_size: int = 1,
             budgets=BUDGETS, force_score: bool = False) -> list[str]:
    n_pool = len(pd.read_parquet(metadata))
    os.makedirs(out_dir, exist_ok=True)
    nlls, sampler = score_pool(pool, work_dir, model, dtype, batch_size, force=force_score)
    meta = run_meta(model, dev, sampler, extra={"selector_kind": "perplexity-top",
                                                "dtype": dtype})

    written = []
    for budget in budgets:
        k = round(budget * n_pool)
        ids = select_budget(nlls, pool, k, work_dir)
        obj = {"selector": "perplexity", "budget": budget, "seed": 0,
               "selected_ids": ids, "meta": meta}
        path = os.path.join(out_dir, f"perplexity__b{budget}__s0.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f)
        written.append(path)
        logger.info("perplexity b=%.2f -> %d ids -> %s", budget, len(ids), os.path.basename(path))
    return written


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Perplexity selection adapter.")
    ap.add_argument("--pool", default="audit/results/pilot_pool.jsonl")
    ap.add_argument("--metadata", default="audit/results/metadata_pilot.parquet")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--work_dir", default=None)
    ap.add_argument("--model", default=None, help="Override the perplexity model.")
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
                     dtype=args.dtype, batch_size=args.batch_size,
                     force_score=args.force_score)
    logger.info("Wrote %d perplexity selection files to %s", len(paths), out_dir)


if __name__ == "__main__":
    main()

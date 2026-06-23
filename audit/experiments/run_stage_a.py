"""Stage A (Milestone 1) driver: run the three selectors in sequence (Sub-step D).

Invokes the random, perplexity, and RDS+ adapters as Python functions, writing
canonical selection JSONs to <base>/selections/. Then point run_audit at that dir.

Dev vs real is controlled entirely by --dev (routes output to audit/results/dev/ and
substitutes proxy models in the perplexity/RDS+ adapters). See audit/common.py.

Usage:
    python -m audit.experiments.run_stage_a --dev      # local dev (proxy models)
    python -m audit.experiments.run_stage_a            # GPU server (real models)
"""
from __future__ import annotations

import argparse
import logging
import os

from audit.common import announce_dev, ensure_utf8, results_base

logger = logging.getLogger("audit.run_stage_a")


def main() -> None:
    ensure_utf8("audit.experiments.run_stage_a")  # must precede heavy imports below
    from audit.selectors import run_random, run_perplexity, run_rdsplus
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for n in ("sentence_transformers", "httpx", "urllib3", "datasets", "filelock",
              "huggingface_hub", "fsspec"):
        logging.getLogger(n).setLevel(logging.WARNING)

    ap = argparse.ArgumentParser(description="Stage A / M1 selector driver.")
    ap.add_argument("--pool", default="audit/results/pilot_pool.jsonl")
    ap.add_argument("--metadata", default="audit/results/metadata_pilot.parquet")
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="Selectors to skip: random perplexity rdsplus.")
    args = ap.parse_args()

    announce_dev(args.dev, logger)
    base = results_base(args.dev)
    sel_dir = os.path.join(base, "selections")
    os.makedirs(sel_dir, exist_ok=True)

    if "random" not in args.skip:
        logger.info("=== [1/3] random ===")
        run_random.generate(args.metadata, sel_dir, dev=args.dev)
    if "perplexity" not in args.skip:
        logger.info("=== [2/3] perplexity ===")
        ppl_model = run_perplexity.PPL_DEV_MODEL if args.dev else run_perplexity.PPL_REAL_MODEL
        run_perplexity.generate(
            args.pool, args.metadata, sel_dir, os.path.join(base, "ppl_work"),
            model=ppl_model, dev=args.dev)
    if "rdsplus" not in args.skip:
        logger.info("=== [3/3] rdsplus ===")
        run_rdsplus.generate(
            args.pool, args.metadata, sel_dir, os.path.join(base, "rds_work"),
            model=(run_rdsplus.RDS_DEV_MODEL if args.dev else run_rdsplus.RDS_REAL_MODEL),
            dev=args.dev)

    logger.info("Stage A selections written to %s", sel_dir)
    logger.info("Next: python -m audit.metrics.run_audit %s", "--dev" if args.dev else "")


if __name__ == "__main__":
    main()

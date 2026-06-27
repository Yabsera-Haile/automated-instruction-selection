"""B-Step 3 [SERVER]: Stage-B LoRA training driver — loops over the 9 cells.

For each cell it (re)materializes nothing itself except ensuring the subsets exist (runs
materialize_subsets if needed), then subprocesses audit.stageb.train_one_cell so each
cell trains a FRESH LoRA on the base (process isolation; a crash in one cell doesn't lose
the others). Collects per-cell metrics into audit/results/stageb/train_log.json and prints
a table.

FIXED-EPOCH comparison: every cell trains 3 epochs, so larger subsets get more optimizer
steps. We record steps-per-cell so a fixed-step view can be computed later.

Run on the server (uses GPU0 by default; set CUDA_VISIBLE_DEVICES to pick a card):
    python -m audit.experiments.run_stageb_train
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys

logger = logging.getLogger("audit.run_stageb_train")

# checkpoint/condition name == subset stem; the base (no-SFT) cell is evaluated in B-4.
CELLS = [
    "full__b1.0",
    "random__b0.05", "random__b0.1",
    "perplexity-low__b0.05", "perplexity-low__b0.1",
    "quality__b0.05", "quality__b0.1",
    "perplexity-high__b0.05", "perplexity-high__b0.1",
]


def ensure_subsets(subsets_dir: str) -> None:
    missing = [c for c in CELLS if not os.path.exists(os.path.join(subsets_dir, f"{c}.jsonl"))]
    if missing:
        logger.info("Subsets missing (%d); running materialize_subsets ...", len(missing))
        subprocess.run([sys.executable, "-m", "audit.stageb.materialize_subsets"], check=True)
    if any(not os.path.exists(os.path.join(subsets_dir, f"{c}.jsonl")) for c in CELLS):
        raise SystemExit("Subsets still missing after materialize_subsets — check inputs.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Stage-B LoRA training driver.")
    ap.add_argument("--subsets_dir", default="audit/results/stageb/subsets")
    ap.add_argument("--ckpt_dir", default="audit/results/stageb/checkpoints")
    ap.add_argument("--config", default="audit/configs/stageb_train.yaml")
    ap.add_argument("--only", nargs="*", default=None, help="Subset of cells to run.")
    args = ap.parse_args()

    ensure_subsets(args.subsets_dir)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    cells = [c for c in CELLS if (args.only is None or c in args.only)]

    results = []
    for i, cell in enumerate(cells, 1):
        train = os.path.join(args.subsets_dir, f"{cell}.jsonl")
        out = os.path.join(args.ckpt_dir, cell)
        logger.info("=== [%d/%d] training cell %s ===", i, len(cells), cell)
        cmd = [sys.executable, "-m", "audit.stageb.train_one_cell",
               "--train", train, "--output_dir", out, "--config", args.config]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        sys.stdout.write(proc.stdout[-2000:])
        if proc.returncode != 0:
            logger.error("Cell %s FAILED (exit %d). stderr tail:\n%s",
                         cell, proc.returncode, proc.stderr[-1500:])
            results.append({"condition": cell, "error": proc.stderr[-500:]})
            continue
        metrics = None
        for line in proc.stdout.splitlines():
            if line.startswith("STAGEB_METRICS "):
                metrics = json.loads(line[len("STAGEB_METRICS "):])
        if metrics is None:
            logger.error("Cell %s produced no metrics line.", cell)
            results.append({"condition": cell, "error": "no metrics"})
        else:
            results.append(metrics)
            logger.info("Cell %s: %d examples, %d steps, %.0fs, %d MiB peak",
                        cell, metrics["n_examples"], metrics["steps"],
                        metrics["runtime_s"], metrics["peak_vram_mib"])

    log_path = os.path.join(os.path.dirname(args.ckpt_dir), "train_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # summary table
    logger.info("=" * 78)
    logger.info("%-26s %8s %6s %8s %10s %s", "condition", "examples", "steps",
                "runtime", "VRAM(MiB)", "ok")
    ok_all = True
    for r in results:
        if "error" in r:
            ok_all = False
            logger.info("%-26s %8s %6s %8s %10s ERROR", r["condition"], "-", "-", "-", "-")
        else:
            cpu = r["peak_vram_mib"] < 100   # would indicate a silent CPU fallback
            ok_all = ok_all and not cpu
            logger.info("%-26s %8d %6d %7.0fs %10d %s", r["condition"], r["n_examples"],
                        r["steps"], r["runtime_s"], r["peak_vram_mib"],
                        "OK" if not cpu else "CPU?!")
    logger.info("=" * 78)
    logger.info("Adapters in %s | log -> %s | all-on-GPU=%s", args.ckpt_dir, log_path, ok_all)
    logger.info("Fixed 3 epochs => steps scale with subset size (recorded above for a "
                "fixed-step view later).")


if __name__ == "__main__":
    main()

"""B-Step 3 / R2-Step 2 [SERVER]: Stage-B LoRA training driver.

Trains one FRESH LoRA per cell on a chosen target model (process isolation: a crash in
one cell doesn't lose the others). The target model is a CLI arg (--target_model) and all
outputs are routed to a model-tagged tree so rounds don't collide:

    Round 1: Qwen/Qwen2.5-7B   -> audit/results/stageb/qwen2.5-7b/checkpoints/  (or the
             legacy un-tagged audit/results/stageb/checkpoints/ from the first run)
    Round 2: Qwen/Qwen2.5-1.5B -> audit/results/stageb/qwen2.5-1.5b/checkpoints/

Matrix (13 SFT cells; the no-SFT base is evaluated directly, not trained here):
    full__b1.0                              (whole enriched pilot)
    {random, perplexity-low, quality, perplexity-high} x {0.01, 0.05, 0.10}

FIXED-EPOCH comparison: every cell trains the same epochs (config), so larger subsets get
more optimizer steps; steps-per-cell are recorded for a fixed-step view later.

Cells are dispatched across GPUs with a dynamic pool (one cell per GPU at a time, next
cell to the first free GPU), sorted largest-first so `full` overlaps the tiny 1% cells.

Run on the server (Round 2):
    python -m audit.experiments.run_stageb_train --target_model Qwen/Qwen2.5-1.5B --gpus 0,1,2
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time

from audit.common import model_slug

logger = logging.getLogger("audit.run_stageb_train")

SELECTORS = ["random", "perplexity-low", "quality", "perplexity-high"]
BUDGETS = [0.01, 0.05, 0.10]
# checkpoint/condition name == subset stem. full__b1.0 + 4 selectors x 3 budgets = 13.
CELLS = ["full__b1.0"] + [f"{s}__b{b}" for s in SELECTORS for b in BUDGETS]


def budget_of(cell: str) -> float:
    return float(cell.rpartition("__b")[2])


def ensure_subsets(subsets_dir: str, cells: list[str]) -> None:
    missing = [c for c in cells if not os.path.exists(os.path.join(subsets_dir, f"{c}.jsonl"))]
    if missing:
        logger.info("Subsets missing (%d: %s); running materialize_subsets ...",
                    len(missing), ", ".join(missing))
        subprocess.run([sys.executable, "-m", "audit.stageb.materialize_subsets"], check=True)
    still = [c for c in cells if not os.path.exists(os.path.join(subsets_dir, f"{c}.jsonl"))]
    if still:
        raise SystemExit(f"Subsets still missing after materialize_subsets: {still} — "
                         "check the b0.01 selection JSONs are committed on this machine.")


def parse_metrics(text: str) -> dict | None:
    for line in text.splitlines():
        if line.startswith("STAGEB_METRICS "):
            return json.loads(line[len("STAGEB_METRICS "):])
    return None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Stage-B LoRA training driver (multi-GPU pool).")
    ap.add_argument("--target_model", default="Qwen/Qwen2.5-7B",
                    help="Base model to LoRA-train (BASE, not Instruct).")
    ap.add_argument("--subsets_dir", default="audit/results/stageb/subsets")
    ap.add_argument("--ckpt_dir", default=None,
                    help="Default: audit/results/stageb/<model_slug>/checkpoints.")
    ap.add_argument("--config", default="audit/configs/stageb_train.yaml")
    ap.add_argument("--gpus", default="0,1,2", help="GPUs to distribute cells across.")
    ap.add_argument("--only", nargs="*", default=None, help="Subset of cells to run.")
    args = ap.parse_args()

    slug = model_slug(args.target_model)
    ckpt_dir = args.ckpt_dir or f"audit/results/stageb/{slug}/checkpoints"
    cells = [c for c in CELLS if (args.only is None or c in args.only)]
    ensure_subsets(args.subsets_dir, cells)
    os.makedirs(ckpt_dir, exist_ok=True)
    log_dir = os.path.join(os.path.dirname(ckpt_dir), "train_logs")
    os.makedirs(log_dir, exist_ok=True)

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    pending = sorted(cells, key=budget_of, reverse=True)   # biggest first (balance)
    free = list(gpus)
    running: list[dict] = []
    results: list[dict] = []
    logger.info("Target=%s slug=%s | %d cells across GPUs %s -> %s",
                args.target_model, slug, len(cells), gpus, ckpt_dir)

    while pending or running:
        while free and pending:
            cell = pending.pop(0)
            gpu = free.pop(0)
            train = os.path.join(args.subsets_dir, f"{cell}.jsonl")
            out = os.path.join(ckpt_dir, cell)
            log_path = os.path.join(log_dir, f"{cell}.log")
            lf = open(log_path, "w")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            cmd = [sys.executable, "-m", "audit.stageb.train_one_cell",
                   "--train", train, "--output_dir", out, "--config", args.config,
                   "--model", args.target_model]
            p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
            running.append({"cell": cell, "gpu": gpu, "p": p, "log": log_path, "lf": lf})
            logger.info("launch %-24s on GPU%s (log: %s)", cell, gpu, log_path)

        progressed = False
        still = []
        for r in running:
            if r["p"].poll() is None:
                still.append(r)
                continue
            r["lf"].close()
            free.append(r["gpu"])
            progressed = True
            text = open(r["log"], encoding="utf-8", errors="replace").read()
            rc = r["p"].returncode
            metrics = parse_metrics(text)
            if rc != 0 or metrics is None:
                logger.error("Cell %s FAILED (exit %s). log tail:\n%s",
                             r["cell"], rc, text[-1200:])
                results.append({"condition": r["cell"], "error": text[-500:]})
            else:
                results.append(metrics)
                logger.info("done   %-24s %d ex, %d steps, %.0fs, %d MiB peak (GPU%s)",
                            r["cell"], metrics["n_examples"], metrics["steps"],
                            metrics["runtime_s"], metrics["peak_vram_mib"], r["gpu"])
        running = still
        if running and not progressed:
            time.sleep(5)

    log_path = os.path.join(os.path.dirname(ckpt_dir), "train_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({"target_model": args.target_model, "slug": slug, "cells": results}, f, indent=2)

    logger.info("=" * 82)
    logger.info("%-24s %8s %6s %9s %10s %s", "condition", "examples", "steps",
                "runtime", "VRAM(MiB)", "ok")
    ok_all = True
    for r in results:
        if "error" in r:
            ok_all = False
            logger.info("%-24s %8s %6s %9s %10s ERROR", r["condition"], "-", "-", "-", "-")
        else:
            cpu = r["peak_vram_mib"] < 100   # would indicate a silent CPU fallback
            ok_all = ok_all and not cpu
            logger.info("%-24s %8d %6d %8.0fs %10d %s", r["condition"], r["n_examples"],
                        r["steps"], r["runtime_s"], r["peak_vram_mib"],
                        "OK" if not cpu else "CPU?!")
    logger.info("=" * 82)
    logger.info("%d/%d cells OK | adapters in %s | log -> %s | all-on-GPU=%s",
                sum(1 for r in results if "error" not in r), len(results), ckpt_dir,
                log_path, ok_all)
    logger.info("Base (no-SFT) is evaluated directly in R2-Step 3; not trained here.")


if __name__ == "__main__":
    main()

"""C1-Step 3 [SERVER]: train the Stage-C rarity-aware variant matrix on Gemma-3-4B.

Trains the 17 subsets materialized in C1-Step 2 (results/stagec/phase1/subsets/):
  perplexity-low__none__s0            -- the disease (0 decisive/lang)
  perplexity-low__proportional__s0..2 -- shape-restoring floor (110/lang, < threshold)
  perplexity-low__absolute__s0..2     -- absolute floor (500/lang, >= threshold)
  perplexity-low__hybrid__s0..2       -- max(prop, N_abs) floor (500/lang)
  perplexity-low__proportional-nogate__s0..2 -- DRoP-analog (110/lang, no quality gate)
  random__s0..2                       -- fair-share reference (genuine draws)
  full__s0                            -- ceiling reference

Training path is IDENTICAL to Round 3/4 (train_one_cell: image-text load for the vision-text
Gemma-3-4B, language-tower-only LoRA with the vision=0 assert, add_bos/eager, grad
checkpointing), same LoRA config and fixed epochs -- so Stage C differs from Stage B only in
WHICH rows the wrapper kept, never in the model or scoring.

SEED SEMANTICS (mirrors Round 4):
  * random           -- s0/s1/s2 are 3 genuine selection DRAWS, each trained once.
  * floor conditions -- the floor selection is deterministic, so the 3 seed files are
                        identical; s0/s1/s2 are 3 TRAINING seeds on the same subset -> the
                        run-variance band, exactly as perplexity-low was handled in R4.
  * none / full      -- single run.

The training seed for every cell is parsed from its `__s<N>` suffix.

Checkpoints -> phase1/checkpoints/<cell>/.

    python -m audit.experiments.run_stagec_train --gpus 0,1,2
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import re
import subprocess
import sys
import time

logger = logging.getLogger("audit.run_stagec_train")

SEC_PER_STEP = 42.3          # measured on the 4B pilot (20962s / 495 steps)


def seed_kind(cell: str) -> str:
    """Classify a cell for the report's seed-semantics column."""
    if cell.startswith("random__"):
        return "draw"          # genuine per-seed selection draw
    if cell.startswith("full__") or cell.endswith("__none__s0") or "__none__" in cell:
        return "single"
    return "train"             # deterministic floor: same data, seed varies the training run


def build_cells(subsets_dir: str):
    """[(cell, subset_path, train_seed, seed_kind)] discovered from the materialized subsets.
    The training seed is the `__s<N>` suffix; unseeded files default to seed 0."""
    cells = []
    for path in sorted(glob.glob(os.path.join(subsets_dir, "*.jsonl"))):
        cell = os.path.splitext(os.path.basename(path))[0]
        m = re.search(r"__s(\d+)$", cell)
        seed = int(m.group(1)) if m else 0
        cells.append((cell, path, seed, seed_kind(cell)))
    return cells


def n_rows(path: str) -> int:
    return sum(1 for l in open(path, encoding="utf-8") if l.strip()) if os.path.exists(path) else 0


def parse_metrics(text: str):
    for line in text.splitlines():
        if line.startswith("STAGEB_METRICS "):
            return json.loads(line[len("STAGEB_METRICS "):])
    return None


def parse_towers(text: str):
    m = re.search(r"language=(\d+),\s*vision=(\d+)", text)
    return (int(m.group(1)), int(m.group(2))) if m else (-1, -1)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Train the Stage-C rarity-aware variant matrix.")
    ap.add_argument("--target_model", default="google/gemma-3-4b-pt")
    ap.add_argument("--subsets_dir", default="audit/results/stagec/phase1/subsets")
    ap.add_argument("--ckpt_dir", default="audit/results/stagec/phase1/checkpoints")
    ap.add_argument("--config", default="audit/configs/stageb_train.yaml")
    ap.add_argument("--gpus", default="0,1,2")
    ap.add_argument("--only", nargs="*", default=None, help="Subset of cell names to run.")
    ap.add_argument("--dry_run", action="store_true", help="Print the plan + cost, then exit.")
    args = ap.parse_args()

    cells = build_cells(args.subsets_dir)
    if args.only:
        cells = [c for c in cells if c[0] in set(args.only)]
    if not cells:
        raise SystemExit(f"No *.jsonl subsets found in {args.subsets_dir}")
    # cost estimate: steps = ceil(rows/64)*3 epochs
    total_h = 0.0
    for cell, path, _, _ in cells:
        steps = math.ceil(n_rows(path) / 64) * 3
        total_h += steps * SEC_PER_STEP / 3600
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    logger.info("Matrix: %d cells | est %.0f GPU-hours (~%.0f h wall on %d GPUs)",
                len(cells), total_h, total_h / max(len(gpus), 1), len(gpus))
    if args.dry_run:
        for cell, path, seed, kind in cells:
            logger.info("  %-40s rows=%6d seed=%d (%s)", cell, n_rows(path), seed, kind)
        return

    os.makedirs(args.ckpt_dir, exist_ok=True)
    log_dir = os.path.join(os.path.dirname(args.ckpt_dir), "train_logs")
    os.makedirs(log_dir, exist_ok=True)
    pending = sorted(cells, key=lambda c: n_rows(c[1]), reverse=True)   # biggest first
    free, running, results = list(gpus), [], []

    while pending or running:
        while free and pending:
            cell, path, seed, kind = pending.pop(0)
            gpu = free.pop(0)
            out = os.path.join(args.ckpt_dir, cell)
            if os.path.exists(os.path.join(out, "adapter_config.json")):
                logger.info("skip   %-40s (already trained)", cell)
                free.append(gpu)
                continue
            lf = open(os.path.join(log_dir, f"{cell}.log"), "w")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            cmd = [sys.executable, "-m", "audit.stageb.train_one_cell",
                   "--train", path, "--output_dir", out, "--config", args.config,
                   "--model", args.target_model, "--seed", str(seed)]
            p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
            running.append({"cell": cell, "kind": kind, "seed": seed, "gpu": gpu, "p": p,
                            "log": lf.name, "lf": lf})
            logger.info("launch %-40s on GPU%s (rows=%d, seed=%d/%s)", cell, gpu,
                        n_rows(path), seed, kind)
        progressed, still = False, []
        for r in running:
            if r["p"].poll() is None:
                still.append(r)
                continue
            r["lf"].close()
            free.append(r["gpu"])
            progressed = True
            text = open(r["log"], encoding="utf-8", errors="replace").read()
            m, (lang_n, vis_n) = parse_metrics(text), parse_towers(text)
            if r["p"].returncode != 0 or m is None:
                logger.error("Cell %s FAILED (exit %s):\n%s", r["cell"], r["p"].returncode,
                             text[-1000:])
                results.append({"cell": r["cell"], "error": text[-400:]})
            else:
                results.append({**m, "cell": r["cell"], "seed_kind": r["kind"],
                                "lora_language": lang_n, "lora_vision": vis_n})
                logger.info("done   %-40s %d ex, %d steps, %.0fs, %d MiB | lang=%d vision=%d",
                            r["cell"], m["n_examples"], m["steps"], m["runtime_s"],
                            m["peak_vram_mib"], lang_n, vis_n)
        running = still
        if running and not progressed:
            time.sleep(5)

    log_path = os.path.join(os.path.dirname(args.ckpt_dir), "stagec_train_log.json")
    json.dump({"target_model": args.target_model, "cells": results},
              open(log_path, "w", encoding="utf-8"), indent=2)
    logger.info("=" * 100)
    logger.info("%-40s %8s %6s %9s %10s %s", "cell", "examples", "steps", "runtime",
                "VRAM(MiB)", "lang/vision")
    ok = True
    seeds_seen = set()
    for r in sorted(results, key=lambda x: x["cell"]):
        if "error" in r:
            ok = False
            logger.info("%-40s %8s %6s %9s %10s ERROR", r["cell"], "-", "-", "-", "-")
        else:
            seeds_seen.add(r.get("seed"))
            v_ok = r["lora_vision"] == 0
            ok = ok and v_ok and r["peak_vram_mib"] > 100
            logger.info("%-40s %8d %6d %8.0fs %10d %d/%d %s", r["cell"], r["n_examples"],
                        r["steps"], r["runtime_s"], r["peak_vram_mib"], r["lora_language"],
                        r["lora_vision"], "OK" if v_ok else "VISION!=0!")
    logger.info("=" * 100)
    n_ok = sum(1 for r in results if "error" not in r)
    logger.info("%d/%d cells OK | 0-vision on all=%s | seeds present=%s | ckpts %s | log %s",
                n_ok, len(results), ok, sorted(s for s in seeds_seen if s is not None),
                args.ckpt_dir, log_path)


if __name__ == "__main__":
    main()

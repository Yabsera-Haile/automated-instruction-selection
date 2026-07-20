"""R4-Step 4 [SERVER]: train Gemma-3-4B on the 4 plt density-pilot pools (full-data).

Full-data only -- no selection, no budget sweep. Four LoRA runs (same config / fixed epochs
as Round 3), reusing audit.stageb.train_one_cell unchanged (multimodal load, language-tower-
only LoRA with the vision=0 assert, add_bos/eager, grad checkpointing). Dynamic 3-GPU pool,
biggest pool first. Checkpoints -> round4/pilot/checkpoints/plt_{0,250,1000,4000}/.

If the pools aren't materialized yet, this first runs build_plt_pilot --mode materialize
(deterministic from the committed dose file + pilot pool).

    python -m audit.experiments.run_plt_pilot --target_model google/gemma-3-4b-pt --gpus 0,1,2
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time

logger = logging.getLogger("audit.run_plt_pilot")

DOSES = [0, 250, 1000, 4000]
POOL_SIZE = {0: 10755, 250: 11005, 1000: 11755, 4000: 14755}  # for biggest-first ordering


def ensure_pools(pools_dir: str) -> None:
    missing = [d for d in DOSES
               if not os.path.exists(os.path.join(pools_dir, f"plt_pilot_d{d}.jsonl"))]
    if missing:
        logger.info("Pools missing %s; materializing from the committed dose file ...", missing)
        subprocess.run([sys.executable, "-m", "audit.stageb.build_plt_pilot",
                        "--mode", "materialize", "--out_dir", pools_dir], check=True)
    still = [d for d in DOSES
             if not os.path.exists(os.path.join(pools_dir, f"plt_pilot_d{d}.jsonl"))]
    if still:
        raise SystemExit(f"Pools still missing after materialize: {still}")


def parse_metrics(text: str) -> dict | None:
    for line in text.splitlines():
        if line.startswith("STAGEB_METRICS "):
            return json.loads(line[len("STAGEB_METRICS "):])
    return None


def parse_lora_towers(text: str) -> tuple[int, int] | None:
    """Extract (language, vision) adapted-module counts from train_one_cell's log line."""
    m = re.search(r"language=(\d+),\s*vision=(\d+)", text)
    return (int(m.group(1)), int(m.group(2))) if m else None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Train the plt density pilot (full-data).")
    ap.add_argument("--target_model", default="google/gemma-3-4b-pt")
    ap.add_argument("--pools_dir", default="audit/results/stageb/gemma-3-4b-pt/round4/pools")
    ap.add_argument("--ckpt_dir",
                    default="audit/results/stageb/gemma-3-4b-pt/round4/pilot/checkpoints")
    ap.add_argument("--config", default="audit/configs/stageb_train.yaml")
    ap.add_argument("--gpus", default="0,1,2")
    args = ap.parse_args()

    ensure_pools(args.pools_dir)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    log_dir = os.path.join(os.path.dirname(args.ckpt_dir), "train_logs")
    os.makedirs(log_dir, exist_ok=True)

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    pending = sorted(DOSES, key=lambda d: POOL_SIZE[d], reverse=True)   # biggest first
    free = list(gpus)
    running: list[dict] = []
    results: list[dict] = []
    logger.info("Pilot: %s on GPUs %s -> %s", [f"plt_{d}" for d in DOSES], gpus, args.ckpt_dir)

    while pending or running:
        while free and pending:
            d = pending.pop(0)
            gpu = free.pop(0)
            cell = f"plt_{d}"
            train = os.path.join(args.pools_dir, f"plt_pilot_d{d}.jsonl")
            out = os.path.join(args.ckpt_dir, cell)
            log_path = os.path.join(log_dir, f"{cell}.log")
            lf = open(log_path, "w")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            cmd = [sys.executable, "-m", "audit.stageb.train_one_cell",
                   "--train", train, "--output_dir", out, "--config", args.config,
                   "--model", args.target_model]
            p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
            running.append({"dose": d, "cell": cell, "gpu": gpu, "p": p, "log": log_path, "lf": lf})
            logger.info("launch %-10s on GPU%s (log: %s)", cell, gpu, log_path)

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
            towers = parse_lora_towers(text)
            if rc != 0 or metrics is None:
                logger.error("Cell %s FAILED (exit %s). log tail:\n%s",
                             r["cell"], rc, text[-1200:])
                results.append({"cell": r["cell"], "dose": r["dose"], "error": text[-400:]})
            else:
                lang_n, vis_n = towers if towers else (-1, -1)
                results.append({**metrics, "dose": r["dose"], "lora_language": lang_n,
                                "lora_vision": vis_n})
                logger.info("done   %-10s %d ex, %d steps, %.0fs, %d MiB | LoRA lang=%d vision=%d",
                            r["cell"], metrics["n_examples"], metrics["steps"],
                            metrics["runtime_s"], metrics["peak_vram_mib"], lang_n, vis_n)
        running = still
        if running and not progressed:
            time.sleep(5)

    log_path = os.path.join(os.path.dirname(args.ckpt_dir), "pilot_train_log.json")
    json.dump({"target_model": args.target_model, "cells": results},
              open(log_path, "w", encoding="utf-8"), indent=2)

    logger.info("=" * 84)
    logger.info("%-10s %8s %6s %9s %10s %s", "pool", "examples", "steps", "runtime",
                "VRAM(MiB)", "LoRA(lang/vision)")
    ok_all = True
    for r in sorted(results, key=lambda x: x["dose"]):
        if "error" in r:
            ok_all = False
            logger.info("%-10s %8s %6s %9s %10s ERROR", r["cell"], "-", "-", "-", "-")
        else:
            vis_ok = r["lora_vision"] == 0
            ok_all = ok_all and vis_ok and r["peak_vram_mib"] > 100
            logger.info("%-10s %8d %6d %8.0fs %10d %d/%d %s", r["cell"], r["n_examples"],
                        r["steps"], r["runtime_s"], r["peak_vram_mib"], r["lora_language"],
                        r["lora_vision"], "OK" if vis_ok else "VISION!=0!")
    logger.info("=" * 84)
    logger.info("%d/%d pilot adapters OK | 0-vision on all=%s | ckpts %s | log %s",
                sum(1 for r in results if "error" not in r), len(results), ok_all,
                args.ckpt_dir, log_path)


if __name__ == "__main__":
    main()

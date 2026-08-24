"""Overnight batch: bring the single-seed Round-4 language-axis cells up to 3 seeds.

Round-4 concentrated pool (N=24,477), floor=none, same LoRA/epoch config and same train + eval
path as the existing 3-seed random/perplexity-low cells, so the new seeds are directly comparable.
Deterministic selectors: the extra seeds reuse the s0 subset and vary only the TRAINING seed
(exactly as perplexity-low was multi-seeded).

Cells, in priority order (an early Ctrl-C still leaves the important ones on disk):
    quality b0.1  ->  quality b0.25  ->  perplexity-high b0.1  ->  perplexity-high b0.25
    ->  full b1.0
Each gets seeds 1 and 2 (s0 already exists). Each seed is evaluated on all three table metrics:
held-out FLORES perplexity, FLORES chrF++ (eng->xx), Belebele accuracy -- decisive-language macros.

Robustness: results append to a CSV and fsync after every cell; checkpoints + eval outputs are
reused so a re-run skips finished work; each cell is wrapped so a failure logs and continues; the
orchestrator can be interrupted and everything already written stays.

    python -m audit.experiments.run_seed_completion --gpus 0,1,2
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

from audit.experiments.run_round4_eval import eval_condition, _latest, _num, LANGS, FLORES

CELLS = [("quality", "0.1"), ("quality", "0.25"),
         ("perplexity-high", "0.1"), ("perplexity-high", "0.25"),
         ("full", "1.0")]
NEW_SEEDS = [1, 2]
METRICS = ["heldout_ppl", "chrf_eng_to_xx", "belebele"]
BASE = "google/gemma-3-4b-pt"
ROOT = "audit/results/stageb/gemma-3-4b-pt/round4"
SUBSETS, CKPT, EVALROOT = f"{ROOT}/subsets", f"{ROOT}/checkpoints", f"{ROOT}/eval"
CONFIG, FLORES_TASKS = "audit/configs/stageb_train.yaml", "audit/configs/flores_tasks"
CSVPATH = f"{ROOT}/stageb_seed_results.csv"
COLS = ["selector", "budget", "seed", "metric", "value"]


def ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(m):
    print(f"[{ts()}] {m}", flush=True)


def cell(sel, bud, seed):
    return f"{sel}__b{bud}__s{seed}"


def macros(cellname):
    """Decisive-language macro for each of the three metrics, from the cached eval outputs."""
    out = {}
    p = f"{EVALROOT}/metrics/heldout_ppl/{cellname}.json"
    if os.path.exists(p):
        L = json.load(open(p, encoding="utf-8")).get("languages", {})
        v = [L[l]["ppl"] for l in LANGS if L.get(l, {}).get("ppl") is not None]
        if len(v) == len(LANGS):
            out["heldout_ppl"] = sum(v) / len(v)
    flo = _latest(f"{EVALROOT}/metrics/lm_eval/{cellname}/flores")
    v = [_num(flo.get(f"flores_eng_{l}", {}), "chrf,none", "chrf") for l in LANGS]
    if all(x is not None for x in v):
        out["chrf_eng_to_xx"] = sum(v) / len(v)
    bel = _latest(f"{EVALROOT}/metrics/lm_eval/{cellname}/belebele")
    v = [_num(bel.get(f"belebele_{FLORES[l]}", {}), "acc,none", "acc") for l in LANGS]
    if all(x is not None for x in v):
        out["belebele"] = sum(v) / len(v)
    return out


def append_rows(rows):
    new = not os.path.exists(CSVPATH)
    with open(CSVPATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(COLS)
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def have(sel, bud, seed):
    if not os.path.exists(CSVPATH):
        return set()
    got = set()
    with open(CSVPATH, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["selector"] == sel and r["budget"] == bud and int(r["seed"]) == seed:
                got.add(r["metric"])
    return got


def seed_existing_s0():
    """Record the already-evaluated s0 metrics (from round4_results.parquet) so the CSV carries
    all three seeds and the summary is self-contained."""
    pq = f"{EVALROOT}/round4_results.parquet"
    if not os.path.exists(pq):
        log(f"note: {pq} not found; s0 will be absent from the summary")
        return
    import pandas as pd
    df = pd.read_parquet(pq)
    for sel, bud in CELLS:
        if have(sel, bud, 0) >= set(METRICS):
            continue
        b = float(bud)
        rows = []
        for metric in METRICS:
            s = df[(df.selector == sel) & (abs(df.budget - b) < 1e-9) & (df.seed == 0)
                   & (df.metric == metric) & (df.group.isin(LANGS))]
            if not s.empty:
                rows.append([sel, bud, 0, metric, round(s["value"].mean(), 4)])
        if rows:
            append_rows(rows)
            log(f"seeded s0 for {sel} b{bud}: {[r[3] for r in rows]}")


def run_worker(sel, bud, seed):
    cn = cell(sel, bud, seed)
    out = f"{CKPT}/{cn}"
    subset = f"{SUBSETS}/{sel}__b{bud}__s0.jsonl"
    if not os.path.exists(subset):
        raise SystemExit(f"subset not found: {subset}")
    if not os.path.exists(os.path.join(out, "adapter_config.json")):
        log(f"train {cn} (subset {os.path.basename(subset)}, seed {seed})")
        subprocess.run([sys.executable, "-m", "audit.stageb.train_one_cell", "--train", subset,
                        "--output_dir", out, "--config", CONFIG, "--model", BASE,
                        "--seed", str(seed)], check=True)
    else:
        log(f"adapter present, skip train: {cn}")
    log(f"eval {cn}")
    ea = SimpleNamespace(base_model=BASE, out_root=EVALROOT, batch_size=8,
                         flores_tasks=FLORES_TASKS, flores_limit=200, mmlu_on="__never__")
    eval_condition(cn, out, ea)
    log(f"worker complete {cn}")


def summarize():
    if not os.path.exists(CSVPATH):
        log("no results file to summarize")
        return
    import pandas as pd
    df = pd.read_csv(CSVPATH)
    print(f"\n{'selector':16}{'budget':>7}{'metric':>16}{'mean':>10}{'sd':>9}{'n_seeds':>9}")
    rows = []
    for (sel, bud, metric), g in df.groupby(["selector", "budget", "metric"]):
        vals = list(g["value"])
        mean = sum(vals) / len(vals)
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        rows.append([sel, bud, metric, round(mean, 4), round(sd, 4), len(vals)])
        print(f"{sel:16}{str(bud):>7}{metric:>16}{mean:>10.3f}{sd:>9.3f}{len(vals):>9}")
    import pandas as pd
    pd.DataFrame(rows, columns=["selector", "budget", "metric", "mean", "sd", "n_seeds"]).to_csv(
        f"{ROOT}/stageb_seed_summary.csv", index=False)
    log(f"summary -> {ROOT}/stageb_seed_summary.csv")


def orchestrate(gpus):
    os.makedirs(CKPT, exist_ok=True)
    log(f"seed-completion batch | cells (priority order): "
        f"{[f'{s} b{b}' for s, b in CELLS]} x seeds {NEW_SEEDS} | GPUs {gpus}")
    seed_existing_s0()
    units, skipped = [], 0
    for sel, bud in CELLS:
        for sd in NEW_SEEDS:
            if have(sel, bud, sd) >= set(METRICS):
                skipped += 1
                log(f"skip {cell(sel, bud, sd)} (already complete in CSV)")
            else:
                units.append((sel, bud, sd))
    log(f"{len(units)} units to run, {skipped} already done")
    free, running = list(gpus), []
    try:
        while units or running:
            while free and units:
                sel, bud, sd = units.pop(0)
                g = free.pop(0)
                cn = cell(sel, bud, sd)
                lf = open(f"{ROOT}/seedlog_{cn}.log", "w")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g),
                           PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
                p = subprocess.Popen(
                    [sys.executable, "-m", "audit.experiments.run_seed_completion", "--worker",
                     "--sel", sel, "--bud", bud, "--seed", str(sd)],
                    env=env, stdout=lf, stderr=subprocess.STDOUT)
                running.append({"u": (sel, bud, sd), "g": g, "p": p, "lf": lf})
                log(f"launch {cn} on GPU{g}  (log: seedlog_{cn}.log)")
            prog, still = False, []
            for r in running:
                if r["p"].poll() is None:
                    still.append(r)
                    continue
                prog = True
                free.append(r["g"])
                r["lf"].close()
                sel, bud, sd = r["u"]
                cn = cell(sel, bud, sd)
                if r["p"].returncode == 0:
                    m = macros(cn)
                    rows = [[sel, bud, sd, k, round(m[k], 4)] for k in METRICS if k in m]
                    if len(rows) == len(METRICS):
                        append_rows(rows)
                        log(f"DONE {cn}: " + ", ".join(f"{k}={m[k]:.3f}" for k in METRICS))
                    else:
                        log(f"WARN {cn}: parsed only {[r_[3] for r_ in rows]} "
                            f"(exit 0 but metrics incomplete) — check seedlog_{cn}.log")
                else:
                    log(f"FAILED {cn} (exit {r['p'].returncode}) — logged, continuing to next")
            running = still
            if running and not prog:
                time.sleep(10)
    except KeyboardInterrupt:
        log("INTERRUPT received — terminating running workers; all completed cells are on disk")
        for r in running:
            try:
                r["p"].terminate()
            except Exception:
                pass
    log("batch finished (or interrupted)")
    summarize()


def main():
    ap = argparse.ArgumentParser(description="Round-4 language-axis seed completion (overnight).")
    ap.add_argument("--gpus", default="0,1,2")
    ap.add_argument("--worker", action="store_true", help="internal: train+eval one cell")
    ap.add_argument("--sel")
    ap.add_argument("--bud")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--summary_only", action="store_true")
    args = ap.parse_args()
    os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if args.worker:
        run_worker(args.sel, args.bud, args.seed)
    elif args.summary_only:
        summarize()
    else:
        orchestrate([g.strip() for g in args.gpus.split(",") if g.strip()])


if __name__ == "__main__":
    main()

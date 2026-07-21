"""R4-Step 5 [SERVER]: evaluate the plt density pilot on the sensitive metrics.

For base (no-SFT) + the four pilot models, on plt:
  1. held-out FLORES-200 perplexity  (audit.stageb.heldout_ppl)  -- PRIMARY (continuous, floorless)
  2. FLORES chrF++ both directions   (lm-eval flores_eng_plt / flores_plt_eng)
  3. Belebele plt, full              (lm-eval belebele_plt_Latn) -- secondary, continuity with R3

Assembles round4/pilot/pilot_dose_response.parquet and prints metric-vs-plt-count.
Dynamic GPU pool: one condition per GPU, its three metric jobs run sequentially there.

Protocol: SFT models are scored with their trained Gemma template (tokenizer={adapter},
--apply_chat_template); the base has no template in its tokenizer so lm-eval scores it raw
(as at the R3 gate). heldout_ppl applies the SAME Gemma wrapping to all five conditions
(ensure_chat_template supplies it for the base), so the primary metric is fully comparable.

    python -m audit.experiments.run_pilot_eval --gpus 0,1,2
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import subprocess
import sys
import time

logger = logging.getLogger("audit.run_pilot_eval")

DOSES = [0, 250, 1000, 4000]
LANG = "plt"
FLORES_CODE = "plt_Latn"


def conditions(ckpt_dir: str):
    """[(condition, adapter_or_None, plt_count_or_None)] -- base first, then the doses."""
    out = [("base", None, None)]
    for d in DOSES:
        out.append((f"plt_{d}", os.path.join(ckpt_dir, f"plt_{d}"), d))
    return out


def run_lm_eval(margs, tasks, workdir, batch_size, limit, apply_ct, include_path=None):
    os.makedirs(workdir, exist_ok=True)
    cmd = [sys.executable, "-m", "lm_eval", "--model", "hf", "--model_args", margs,
           "--tasks", ",".join(tasks), "--batch_size", str(batch_size),
           "--output_path", workdir]
    if include_path:
        cmd += ["--include_path", include_path]
    if apply_ct:
        cmd.append("--apply_chat_template")
    if limit and limit > 0:
        cmd += ["--limit", str(limit)]
    print("  $ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def eval_condition(cond, adapter, args) -> None:
    """Run the three metric jobs for one condition on the (single) visible GPU."""
    margs = (f"pretrained={args.base_model},dtype=bfloat16,"
             f"add_bos_token=True,attn_implementation=eager")
    if adapter is not None:
        margs += f",peft={adapter},tokenizer={adapter}"
    apply_ct = adapter is not None

    # 1) held-out perplexity (primary) -- full FLORES devtest, teacher-forced, no generation
    ppl_dir = os.path.join(args.out_root, "metrics", "heldout_ppl")
    if not os.path.exists(os.path.join(ppl_dir, f"{cond}.json")):
        cmd = [sys.executable, "-m", "audit.stageb.heldout_ppl",
               "--base", args.base_model, "--condition", cond, "--langs", LANG,
               "--out_dir", ppl_dir]
        if adapter is not None:
            cmd += ["--adapter", adapter]
        print("  $ " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

    work = os.path.join(args.out_root, "metrics", "lm_eval", cond)
    # 2) Belebele plt, full (loglik)
    if not glob.glob(os.path.join(work, "belebele", "**", "results*.json"), recursive=True):
        run_lm_eval(margs, [f"belebele_{FLORES_CODE}"], os.path.join(work, "belebele"),
                    args.batch_size, 0, apply_ct)
    # 3) FLORES chrF++ both directions (generation)
    if not glob.glob(os.path.join(work, "flores", "**", "results*.json"), recursive=True):
        run_lm_eval(margs, [f"flores_eng_{LANG}", f"flores_{LANG}_eng"],
                    os.path.join(work, "flores"), args.batch_size, args.flores_limit,
                    apply_ct, include_path=args.flores_tasks)
    print(f"[{cond}] all three metrics done", flush=True)


def _latest_results(pattern_dir):
    files = sorted(glob.glob(os.path.join(pattern_dir, "**", "results*.json"), recursive=True),
                   key=os.path.getmtime)
    return json.load(open(files[-1], encoding="utf-8")).get("results", {}) if files else {}


def _pick(d, *keys):
    for k in keys:
        if isinstance(d.get(k), (int, float)):
            return float(d[k])
    for k, v in d.items():
        if isinstance(v, (int, float)) and "stderr" not in k:
            return float(v)
    return None


def assemble(args) -> None:
    import pandas as pd
    rows = []
    for cond, adapter, dose in conditions(args.ckpt_dir):
        ppl_path = os.path.join(args.out_root, "metrics", "heldout_ppl", f"{cond}.json")
        if os.path.exists(ppl_path):
            lang = json.load(open(ppl_path, encoding="utf-8"))["languages"].get(LANG, {})
            if lang.get("ppl") is not None:
                rows.append([cond, dose, "heldout_ppl", lang["ppl"]])
                rows.append([cond, dose, "heldout_nll", lang["nll"]])
        work = os.path.join(args.out_root, "metrics", "lm_eval", cond)
        bel = _latest_results(os.path.join(work, "belebele")).get(f"belebele_{FLORES_CODE}", {})
        v = _pick(bel, "acc,none", "acc")
        if v is not None:
            rows.append([cond, dose, "belebele_plt", v])
        flo = _latest_results(os.path.join(work, "flores"))
        for task, name in ((f"flores_eng_{LANG}", "chrf_eng_to_plt"),
                           (f"flores_{LANG}_eng", "chrf_plt_to_eng")):
            v = _pick(flo.get(task, {}), "chrf,none", "chrf")
            if v is not None:
                rows.append([cond, dose, name, v])
    df = pd.DataFrame(rows, columns=["condition", "plt_count", "metric", "value"])
    out = os.path.join(args.out_root, "pilot_dose_response.parquet")
    os.makedirs(args.out_root, exist_ok=True)
    df.to_parquet(out, index=False)

    order = ["base"] + [f"plt_{d}" for d in DOSES]
    piv = df.pivot_table(index="metric", columns="condition", values="value")
    piv = piv.reindex(columns=[c for c in order if c in piv.columns])
    metric_order = ["heldout_ppl", "heldout_nll", "chrf_eng_to_plt", "chrf_plt_to_eng",
                    "belebele_plt"]
    piv = piv.reindex([m for m in metric_order if m in piv.index])
    print(f"\nAssembled {len(df)} rows -> {out}")
    print("\n=== plt DOSE-RESPONSE (columns = plt examples injected) ===")
    print(piv.round(4).to_string())
    print("\nheldout_ppl is PRIMARY and LOWER IS BETTER; chrF++/Belebele higher is better.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Evaluate the plt density pilot.")
    ap.add_argument("--base_model", default="google/gemma-3-4b-pt")
    ap.add_argument("--out_root", default="audit/results/stageb/gemma-3-4b-pt/round4/pilot")
    ap.add_argument("--ckpt_dir",
                    default="audit/results/stageb/gemma-3-4b-pt/round4/pilot/checkpoints")
    ap.add_argument("--flores_tasks", default="audit/configs/flores_tasks")
    ap.add_argument("--flores_limit", type=int, default=200,
                    help="Sentences/direction for chrF++ (generation is the slow part).")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--gpus", default="0,1,2")
    ap.add_argument("--assemble_only", action="store_true")
    ap.add_argument("--worker_cond", default=None)   # internal
    args = ap.parse_args()

    if args.assemble_only:
        assemble(args)
        return

    conds = conditions(args.ckpt_dir)
    if args.worker_cond is not None:                 # ---- worker: one condition here ----
        for cond, adapter, _ in conds:
            if cond == args.worker_cond:
                eval_condition(cond, adapter, args)
        return

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    pending = [c[0] for c in conds]
    free, running = list(gpus), []
    log_dir = os.path.join(args.out_root, "eval_logs")
    os.makedirs(log_dir, exist_ok=True)
    logger.info("Pilot eval: %s across GPUs %s (flores_limit=%d)", pending, gpus,
                args.flores_limit)

    while pending or running:
        while free and pending:
            cond, gpu = pending.pop(0), free.pop(0)
            lf = open(os.path.join(log_dir, f"{cond}.log"), "w")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            cmd = [sys.executable, "-m", "audit.experiments.run_pilot_eval",
                   "--worker_cond", cond, "--base_model", args.base_model,
                   "--out_root", args.out_root, "--ckpt_dir", args.ckpt_dir,
                   "--flores_tasks", args.flores_tasks,
                   "--flores_limit", str(args.flores_limit),
                   "--batch_size", str(args.batch_size)]
            p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
            running.append({"cond": cond, "gpu": gpu, "p": p, "lf": lf})
            logger.info("launch %-9s on GPU%s (log: %s/%s.log)", cond, gpu, log_dir, cond)
        progressed, still = False, []
        for r in running:
            if r["p"].poll() is None:
                still.append(r)
                continue
            r["lf"].close()
            free.append(r["gpu"])
            progressed = True
            status = "OK" if r["p"].returncode == 0 else f"FAILED({r['p'].returncode})"
            logger.info("done   %-9s %s", r["cond"], status)
        running = still
        if running and not progressed:
            time.sleep(5)
    assemble(args)


if __name__ == "__main__":
    main()

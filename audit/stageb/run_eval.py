"""B-Step 4 [SERVER]: evaluate the 9 fine-tuned models + the no-SFT base.

Uses EleutherAI lm-evaluation-harness (CLI) with the model's chat template (matching
training). Per model, three lm_eval calls to respect per-task few-shot:
  A) Belebele (the held-out per-language metric, 16 langs) + IFEval + MBPP  [task defaults]
  B) GSM8K  --num_fewshot 8
  C) MMLU   --num_fewshot 5
TydiQA is deliberately NOT used (RDS+ retrieves toward it -> contaminated). Belebele is
held out for every selector.

Outputs raw per-model JSON to audit/results/stageb/eval/<condition>__b<budget>.json and a
tidy audit/results/stageb/stageb_results.parquet with columns
[condition, budget, task, group_type(language|skill|average), group, score].

Run on the server (single GPU). First: pip install "lm-eval" (ifeval/mbpp need extras +
code execution). Example:
    pip install "lm-eval[ifeval]"
    HF_ALLOW_CODE_EVAL=1 python -m audit.stageb.run_eval --limit 200
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")   # MBPP executes generated code

BASE_MODEL = "Qwen/Qwen2.5-7B"

# our eval iso -> Belebele/FLORES-200 task suffix (lm-eval task = belebele_<flores>)
ISO_TO_FLORES = {
    "eng": "eng_Latn", "ceb": "ceb_Latn", "plt": "plt_Latn", "bod": "bod_Tibt",
    "yor": "yor_Latn", "tsn": "tsn_Latn", "som": "som_Latn", "kir": "kir_Cyrl",
    "fuv": "fuv_Latn", "hau": "hau_Latn", "wol": "wol_Latn", "mlt": "mlt_Latn",
    "zul": "zul_Latn", "spa": "spa_Latn", "cmn": "zho_Hans", "arb": "arb_Arab",
}
FLORES_TO_ISO = {v: k for k, v in ISO_TO_FLORES.items()}

# skill task -> (skill_label, num_fewshot or None for harness default)
SKILL_TASKS = {
    "gsm8k": ("math", 8),
    "mmlu": ("general", 5),
    "mbpp": ("code", None),
    "ifeval": ("instruction_following", None),
}
# primary metric candidates per task (lm-eval keys look like "acc,none")
METRIC_CANDIDATES = {
    "belebele": ["acc,none", "acc_norm,none", "acc"],
    "mmlu": ["acc,none", "acc"],
    "gsm8k": ["exact_match,strict-match", "exact_match,flexible-extract",
              "exact_match,none", "exact_match"],
    "mbpp": ["pass_at_1,none", "pass@1,none", "pass_at_1", "acc,none"],
    "ifeval": ["prompt_level_strict_acc,none", "inst_level_strict_acc,none",
               "prompt_level_loose_acc,none"],
}


def pick_metric(results: dict, task_key: str, kind: str):
    r = results.get(task_key, {})
    for c in METRIC_CANDIDATES.get(kind, []):
        if c in r and isinstance(r[c], (int, float)):
            return float(r[c])
    for k, v in r.items():  # fallback: first non-stderr numeric
        if isinstance(v, (int, float)) and "stderr" not in k:
            return float(v)
    return None


def run_lm_eval(model_args: str, tasks: list[str], num_fewshot, workdir: str,
                limit: int, apply_chat_template: bool) -> dict:
    os.makedirs(workdir, exist_ok=True)
    cmd = [sys.executable, "-m", "lm_eval", "--model", "hf",
           "--model_args", model_args, "--tasks", ",".join(tasks),
           "--batch_size", "auto", "--output_path", workdir,
           "--confirm_run_unsafe_code"]
    if apply_chat_template:
        cmd.append("--apply_chat_template")
    if num_fewshot is not None:
        cmd += ["--num_fewshot", str(num_fewshot)]
    if limit and limit > 0:
        cmd += ["--limit", str(limit)]
    print("  $ " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    # lm-eval writes <workdir>/<sanitized_model>/results_<ts>.json
    files = sorted(glob.glob(os.path.join(workdir, "**", "results*.json"), recursive=True),
                   key=os.path.getmtime)
    if not files:
        raise RuntimeError(f"No results json under {workdir}")
    return json.load(open(files[-1], encoding="utf-8")).get("results", {})


def enumerate_models(ckpt_dir: str):
    models = [("base", 0.0, None)]
    for d in sorted(glob.glob(os.path.join(ckpt_dir, "*__b*"))):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        cond, _, budget = name.rpartition("__b")
        models.append((cond, float(budget), d))
    return models


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage-B evaluation driver (lm-eval).")
    ap.add_argument("--ckpt_dir", default="audit/results/stageb/checkpoints")
    ap.add_argument("--eval_langs", default="audit/configs/eval_languages.json")
    ap.add_argument("--out_dir", default="audit/results/stageb/eval")
    ap.add_argument("--work_dir", default="audit/results/stageb/eval_work")
    ap.add_argument("--base_model", default=BASE_MODEL)
    ap.add_argument("--limit", type=int, default=200,
                    help="Examples per task (0 = full; default 200 keeps 10-model eval tractable).")
    ap.add_argument("--conditions", nargs="*", default=None,
                    help="Only evaluate these conditions (e.g. base) — for a quick sanity pass.")
    ap.add_argument("--no_chat_template", action="store_true")
    args = ap.parse_args()

    langs = json.load(open(args.eval_langs, encoding="utf-8"))["languages"]
    role = {e["iso"]: e["role"] for e in langs}
    belebele_tasks = [f"belebele_{ISO_TO_FLORES[e['iso']]}" for e in langs
                      if e["iso"] in ISO_TO_FLORES]
    apply_ct = not args.no_chat_template
    os.makedirs(args.out_dir, exist_ok=True)
    models = enumerate_models(args.ckpt_dir)
    if args.conditions:
        models = [m for m in models if m[0] in args.conditions]
    print(f"Evaluating {len(models)} models on {len(belebele_tasks)} Belebele langs + "
          f"{list(SKILL_TASKS)} (limit={args.limit or 'full'}, chat_template={apply_ct})")

    rows = []
    for cond, budget, adapter in models:
        tag = "base" if adapter is None else f"{cond}__b{budget}"
        margs = f"pretrained={args.base_model},dtype=bfloat16"
        if adapter is not None:
            margs += f",peft={adapter}"
        print(f"\n=== {tag} ===")
        results = {}
        groups = [
            (belebele_tasks + ["ifeval", "mbpp"], None, "A"),
            (["gsm8k"], 8, "B"),
            (["mmlu"], 5, "C"),
        ]
        for tasks, nfs, gname in groups:
            try:
                r = run_lm_eval(margs, tasks, nfs, os.path.join(args.work_dir, tag, gname),
                                args.limit, apply_ct)
                results.update(r)
            except Exception as exc:  # keep going; record what we got
                print(f"  !! group {gname} failed for {tag}: {exc}")
        json.dump(results, open(os.path.join(args.out_dir, f"{tag}.json"), "w"), indent=2)

        # ---- tidy rows ----
        lowres, controls = [], []
        for iso in (e["iso"] for e in langs):
            tk = f"belebele_{ISO_TO_FLORES.get(iso, '')}"
            score = pick_metric(results, tk, "belebele")
            if score is None:
                continue
            rows.append([cond, budget, "belebele", "language", iso, score])
            (lowres if role.get(iso) == "low_resource" else controls).append(score)
        skill_scores = []
        for task, (skill, _) in SKILL_TASKS.items():
            score = pick_metric(results, task, task)
            rows.append([cond, budget, task, "skill", skill, score])
            if score is not None:
                skill_scores.append(score)
        mmlu = pick_metric(results, "mmlu", "mmlu")
        if mmlu is not None:
            rows.append([cond, budget, "mmlu", "average", "mmlu_overall", mmlu])
        if skill_scores:
            rows.append([cond, budget, "skill_macro", "average", "skill_macro",
                         sum(skill_scores) / len(skill_scores)])
        if lowres:
            rows.append([cond, budget, "belebele", "average", "lowres_macro",
                         sum(lowres) / len(lowres)])
        if controls:
            rows.append([cond, budget, "belebele", "average", "control_macro",
                         sum(controls) / len(controls)])

    import pandas as pd
    df = pd.DataFrame(rows, columns=["condition", "budget", "task", "group_type",
                                     "group", "score"])
    out_parquet = os.path.join(os.path.dirname(args.out_dir), "stageb_results.parquet")
    df.to_parquet(out_parquet, index=False)
    print(f"\nWrote {len(df)} rows -> {out_parquet}")
    print(df[df.group_type == "average"].to_string(index=False))


if __name__ == "__main__":
    main()

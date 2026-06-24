"""RDS+ selection adapter (Sub-step C).

Pipeline:
  1. EMBED + SCORE each multitask eval set against the pilot pool -> one score pickle
     per eval set, in the repo's format {q_idx: {train_idx: cosine_sim}}.
  2. AGGREGATE + SELECT with the repo's
     `minimal_multitask.get_top_aggregated_influences` (round-robin across eval sets,
     selection_method=max, --output_dataset) -> JSONL of full selected examples;
     we read back each `id`.

Embedding backend (overridable via --model):
  * --dev : sentence-transformers/all-MiniLM-L6-v2, embedded **in-process** here.
            NB: the repo's compute_influence_sentence_embedding.py calls
            dataset.map(num_proc=16) at unguarded module level, which recursively
            spawns processes on Windows (spawn). So for the local Windows dev box we
            reproduce its logic (same text construction, cosine scoring, pickle
            format) without multiprocessing. Output -> audit/results/dev/.
  * real  : the repo's compute_influence_cosinesim.py with the 7B backbone
            (meta-llama/Llama-2-7b-hf), weighted_mean pooling — run via subprocess on
            the Linux GPU server (fork-safe). Output -> audit/results/.

Only the embedding model/path differs dev-vs-real; aggregation, selection, canonical
JSON and audit metrics are identical. RDS+ is deterministic => seed fixed at 0.
filename: rdsplus__b{budget}__s0.json
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

from audit.common import (RDS_DEV_MODEL, RDS_REAL_MODEL, VramSampler,
                          announce_dev, results_base, run_meta)

logger = logging.getLogger("audit.run_rdsplus")

BUDGETS = [0.01, 0.05, 0.10, 0.25, 0.50]
# Multitask eval/query sets that load from the shipped data/eval (README selection set).
# squad (HF download) and codex (HumanEvalPack) can be added for the real server run.
DEFAULT_EVAL_DATASETS = ["alpacafarm", "gsm8k_shots", "bbh_shots", "tydiqa_shots", "mmlu_shots"]

TULU_TOKENIZER = "oobabooga/llama-tokenizer"  # for text construction (has bos/eos)


def _rreplace(s: str, old: str, new: str, n: int) -> str:
    return new.join(s.rsplit(old, n))


def embed_dev(pool: str, eval_datasets: list[str], work_dir: str, model: str,
              batch_size: int, force: bool) -> tuple[list[str], VramSampler]:
    """In-process MiniLM embedding + cosine scoring (Windows-safe). Repo pickle format."""
    import torch  # noqa: F401
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer
    from minimal_multitask.data import DATASETS
    from minimal_multitask.utils import create_prompt_with_tulu_chat_format

    import torch
    os.makedirs(work_dir, exist_ok=True)
    pickles = [os.path.join(work_dir, f"{ev}_embedding.pkl") for ev in eval_datasets]
    # Persist the normalized pool embeddings so SemDeDup (C-Step 3) can reuse them,
    # mirroring the real cosinesim script's cosine_train_reps.pt. Row i == pool_row_idx i.
    index_path = os.path.join(work_dir, "pool_index.pt")
    sampler = VramSampler()
    if all(os.path.exists(p) for p in pickles) and os.path.exists(index_path) and not force:
        logger.info("Reusing %d dev score pickles + pool index.", len(pickles))
        return pickles, sampler

    tok = AutoTokenizer.from_pretrained(TULU_TOKENIZER)
    st = SentenceTransformer(model)  # auto-uses CUDA if available

    with sampler:
        # pool text: same construction as the repo sentence-embedding script
        pool_examples = [json.loads(l) for l in open(pool, encoding="utf-8") if l.strip()]
        pool_texts = []
        for ex in pool_examples:
            t = create_prompt_with_tulu_chat_format(ex["messages"], tok, no_special_tokens=True)
            pool_texts.append(_rreplace(t, "<|assistant|>", "", 1).strip())
        logger.info("Encoding %d pool texts with %s ...", len(pool_texts), model)
        pool_emb = st.encode(pool_texts, batch_size=batch_size, convert_to_numpy=True,
                             normalize_embeddings=True, show_progress_bar=True)
        torch.save(torch.from_numpy(pool_emb), index_path)
        logger.info("Saved dev pool index %s shape=%s", index_path, tuple(pool_emb.shape))

        for ev, pkl in zip(eval_datasets, pickles):
            if os.path.exists(pkl) and not force:
                continue
            test_ds = DATASETS[ev](tok).get_all_test_prompts(seed=42)
            q_texts = [tok.decode(x["input_ids"], skip_special_tokens=True) for x in test_ds]
            q_emb = st.encode(q_texts, batch_size=batch_size, convert_to_numpy=True,
                              normalize_embeddings=True)
            sims = q_emb @ pool_emb.T  # [n_query, n_pool]
            influence_dict = {
                i: {j: float(sims[i, j]) for j in range(sims.shape[1])}
                for i in range(sims.shape[0])
            }
            with open(pkl, "wb") as f:
                pickle.dump(influence_dict, f)
            logger.info("eval=%s: %d queries x %d pool -> %s",
                        ev, sims.shape[0], sims.shape[1], os.path.basename(pkl))
    return pickles, sampler


def embed_real(pool: str, eval_datasets: list[str], work_dir: str, model: str,
               dtype: str, batch_size: int, force: bool) -> tuple[list[str], VramSampler]:
    """Repo cosinesim script per eval set (Linux GPU server). Reuses one pool index."""
    os.makedirs(work_dir, exist_ok=True)
    index_path = os.path.join(work_dir, "cosine_train_reps.pt")
    pickles = [os.path.join(work_dir, f"{ev}_cossim.pkl") for ev in eval_datasets]
    sampler = VramSampler()
    with sampler:
        for ev, pkl in zip(eval_datasets, pickles):
            if os.path.exists(pkl) and not force:
                logger.info("Reusing score pickle: %s", pkl)
                continue
            cmd = [
                sys.executable, "-m", "minimal_multitask.compute_influence_cosinesim",
                "--model_name", model, "--seed", "42",
                "--train_dataset", pool, "--eval_dataset", ev,
                "--index_path", index_path, "--save_dir", work_dir,
                "--batch_size", str(batch_size), "--pooling", "weighted_mean",
                "--dtype", dtype,
            ]
            logger.info("Embedding eval=%s (model=%s) ...", ev, model)
            subprocess.run(cmd, check=True)
            if not os.path.exists(pkl):
                raise RuntimeError(f"Embedding did not produce {pkl}")
    return pickles, sampler


def select_budget(pickles: list[str], pool: str, output_size: int, work_dir: str,
                  selection_method: str, aggregation_method: str) -> list[str]:
    out_jsonl = os.path.join(work_dir, f"rdsplus_top{output_size}.jsonl")
    cmd = [
        sys.executable, "-m", "minimal_multitask.get_top_aggregated_influences",
        "--input_files", *pickles,
        "--output_file", out_jsonl,
        "--output_size", str(output_size),
        "--selection_method", selection_method,
        "--aggregation_method", aggregation_method,
        "--train_dataset", pool,
        "--output_dataset",
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
             dev: bool, eval_datasets=DEFAULT_EVAL_DATASETS, dtype: str | None = None,
             batch_size: int | None = None, budgets=BUDGETS, selection_method: str = "max",
             aggregation_method: str = "round_robin", force_embed: bool = False) -> list[str]:
    # Real RDS+ uses the repo cosinesim script, whose DataLoader has no padding/
    # collation -> it ONLY works at batch_size=1 (stacking variable-length seqs
    # crashes). MiniLM dev embedding pads internally, so larger batches are fine.
    if batch_size is None:
        batch_size = 64 if dev else 1
    # 7B in fp32 = ~28GB; bf16 (the repo/paper default) halves it and is faster.
    if dtype is None:
        dtype = "fp32" if dev else "bf16"
    n_pool = len(pd.read_parquet(metadata))
    os.makedirs(out_dir, exist_ok=True)
    if dev:
        pickles, sampler = embed_dev(pool, eval_datasets, work_dir, model,
                                     batch_size, force_embed)
    else:
        pickles, sampler = embed_real(pool, eval_datasets, work_dir, model, dtype,
                                      batch_size, force_embed)
    logger.info("Embedding done in %ss, peak VRAM %s MiB", sampler.runtime_s, sampler.peak_mib)
    meta = run_meta(model, dev, sampler, extra={
        "selector_kind": "rds+_multitask",
        "embedding_path": "in_process_minilm" if dev else "repo_cosinesim",
        "eval_datasets": eval_datasets,
        "selection_method": selection_method,
        "aggregation_method": aggregation_method,
    })

    written = []
    for budget in budgets:
        k = round(budget * n_pool)
        ids = select_budget(pickles, pool, k, work_dir, selection_method, aggregation_method)
        obj = {"selector": "rdsplus", "budget": budget, "seed": 0,
               "selected_ids": ids, "meta": meta}
        path = os.path.join(out_dir, f"rdsplus__b{budget}__s0.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f)
        written.append(path)
        logger.info("rdsplus b=%.2f -> %d ids -> %s", budget, len(ids), os.path.basename(path))
    return written


def main() -> None:
    from audit.common import ensure_utf8
    ensure_utf8("audit.selectors.run_rdsplus")  # data.py reads UTF-8 prompt files
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="RDS+ selection adapter.")
    ap.add_argument("--pool", default="audit/results/pilot_pool.jsonl")
    ap.add_argument("--metadata", default="audit/results/metadata_pilot.parquet")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--work_dir", default=None)
    ap.add_argument("--model", default=None, help="Override the embedding model.")
    ap.add_argument("--eval_datasets", nargs="+", default=DEFAULT_EVAL_DATASETS)
    ap.add_argument("--dtype", default=None, help="Default: fp32 (dev) / bf16 (real).")
    ap.add_argument("--batch_size", type=int, default=None,
                    help="Default: 64 (dev) / 1 (real; repo script needs batch 1).")
    ap.add_argument("--dev", action="store_true", help="Dev run with MiniLM proxy -> dev/.")
    ap.add_argument("--force_embed", action="store_true")
    args = ap.parse_args()

    announce_dev(args.dev, logger)
    base = results_base(args.dev)
    out_dir = args.out_dir or os.path.join(base, "selections")
    work_dir = args.work_dir or os.path.join(base, "rds_work")
    model = args.model or (RDS_DEV_MODEL if args.dev else RDS_REAL_MODEL)

    paths = generate(args.pool, args.metadata, out_dir, work_dir, model, args.dev,
                     eval_datasets=args.eval_datasets, dtype=args.dtype,
                     batch_size=args.batch_size, force_embed=args.force_embed)
    logger.info("Wrote %d rdsplus selection files to %s", len(paths), out_dir)


if __name__ == "__main__":
    main()

# REPO_MAP — Grounding for the Data-Selection Audit (Phase 0)

This document records what the **real** `automated-instruction-selection` (RDS+) repo
actually does, verified by reading the code — not the second-hand spec. It is the
reference for building the training-free distribution audit (`audit/`).

Repo root: `automated-instruction-selection/` (its own git repo, branch `main`, clean).
Verified against commit `df71a04` ("Update README.md").

---

## 1. Top-level layout

```
automated-instruction-selection/
├── minimal_multitask/        # the core Python package (the pipeline)
│   ├── compute_influence_cosinesim.py   # RDS+ indexing + scoring (flagship)
│   ├── compute_influence_perplexity.py  # perplexity (NLL) scoring baseline
│   ├── compute_influence_sentence_embedding.py  # GTR/NV-Embed baseline
│   ├── compute_influence_*.py           # other baselines (datainf, logix, ccds, ...)
│   ├── get_top_influences.py            # selection from a score pickle -> subset
│   ├── get_top_aggregated_influences.py # multi-task aggregated selection
│   ├── get_top_optimized.py             # memory-optimized large-scale selection
│   ├── data.py                          # eval/test "query" datasets (DATASETS dict)
│   ├── utils.py                         # encode_with_messages_format, tulu chat format
│   ├── instruction_tune.py / train.py   # Stage B training
│   └── eval/                            # per-benchmark eval harnesses
├── scripts/                  # analysis + baseline selection utilities
│   ├── ppl_selections.py               # perplexity SELECTION (note: plural filename)
│   ├── construct_downsampled_balanced.py    # random-balanced baseline
│   ├── construct_downsampled_unbalanced.py  # random-unbalanced baseline
│   ├── combine_pickles*.py             # merge sharded score pickles
│   └── old/                            # legacy scripts (ignore)
├── shell_scripts/            # runnable pipeline wrappers + data download
├── beaker_scripts/           # AI2-internal cluster orchestration (ignore for us)
├── data/                     # config/task-list txt files only (NOT training data)
├── requirements.txt          # core deps (mostly UNPINNED)
├── environment.yml           # conda env; python=3.12.7
└── README.md                 # the 4-step pipeline writeup
```

**The training data is NOT in the repo.** `data/` currently holds only task-list
config files (`datasets.txt`, `eval_tasks.txt`, etc.). The actual pool is downloaded
by `shell_scripts/download_train_data.sh`, which pulls
`https://huggingface.co/datasets/hamishivi/lsds_data/resolve/main/training_data.zip`
and unpacks it to `data/training_data/` (e.g. `tulu_v2_unfiltered_data_dedup.jsonl`).

> ⚠️ `.gitignore` ignores `data/`, `results/`, `selections/`, `*.npy`, `*.faiss`.
> So pool files and pipeline outputs are never committed. **This also means
> `audit/results/` matches the global `results/` ignore rule** — audit output
> artifacts will be git-ignored by default (intended; we don't commit large outputs).
> Directory placeholders under `audit/results/` must be force-added if we want the
> folder tracked.

---

## 2. The training pool: how it's loaded & the example schema

Loading is identical across `compute_influence_cosinesim.py`,
`compute_influence_perplexity.py`, `get_top_influences.py`, `ppl_selections.py`:

```python
# a literal "alpaca" / "tulu2" / "tulu3" keyword, OR a path to a local .jsonl
load_dataset("json", data_files=<path>)["train"]          # local jsonl pool
load_dataset("allenai/tulu-v2-sft-mixture", split="train") # tulu2
load_dataset("allenai/tulu-3-sft-mixture", split="train")  # tulu3
```

Every example is then tokenized by `encode_with_messages_format`
(`minimal_multitask/utils.py`), which **requires a `messages` field** — a list of
`{"role": "system"|"user"|"assistant", "content": str}` dicts. This is the only
field the model pipeline strictly needs.

### Confirmed metadata fields actually used by the code

| Purpose | Field name | Evidence |
|---|---|---|
| Provenance / source sub-dataset | **`dataset`** | `get_top_influences.py:194` `train_datasets[i][j]["dataset"]`; `construct_downsampled_balanced.py:28` `d['dataset']` |
| Stable per-example id | **`id`** | `get_top_influences.py:214,268` `train_datasets[i][j]["id"]` (used by `--select_only_from_file`) |
| Conversation content | **`messages`** | `utils.py:50` `example["messages"]`; `get_top_influences.py:195` `["messages"][0]["content"]` |
| Language | **(none)** | No training-pool `language`/`lang` field exists anywhere. `lang` only appears inside the TydiQA *eval* loader, derived from question ids. |

So the pool schema (Tulu-2 unfiltered, the repo's default) is effectively:
```json
{"id": "...", "dataset": "<source>", "messages": [{"role": "...", "content": "..."}, ...]}
```

> ⚠️ **Spec mismatch to confirm against real data.** The spec says "the Tulu 3
> mixture stores which sub-dataset each example came from — find its exact key."
> This codebase is **Tulu-2-oriented** and uses **`dataset`** everywhere. The
> upstream `allenai/tulu-3-sft-mixture` schema instead uses **`source`** (plus
> `id`, `messages`) — the code does **not** read that key, and would need adapting
> for a true Tulu-3 pool. **Action:** the audit metadata layer must handle BOTH
> keys: prefer `dataset`, fall back to `source`. Confirm the exact key by
> inspecting the first line of the downloaded pool jsonl before running Stage A.

> ⚠️ **No language field → language must be inferred.** This is exactly why the
> spec calls for GlotLID/fastText language-ID. The audit will run language-ID over
> the first user turn (and/or full text) of each example to synthesize a `language`
> attribute.

> ⚠️ **`id` stability / presence.** `id` is present in the Tulu mixtures and used by
> the repo, but a *custom* local pool jsonl is not guaranteed to have it. The
> metadata layer must: use `id` if present and unique; otherwise mint a stable id by
> hashing the canonicalized `messages` content (e.g. sha1 of the concatenated turns).

---

## 3. The crucial mapping: pool row ↔ embedding ↔ score ↔ selection

**Everything is keyed by positional row index into the pool jsonl.** All dataloaders
use `shuffle=False`, so:

```
pool jsonl line i  ==  HF dataset row i  ==  embedding row i  ==  score-dict key i
```

This positional alignment is the backbone of the whole pipeline and of our join.

---

## 4. RDS+ embeddings (`.pt` index)

Produced by `compute_influence_cosinesim.py`:

- **Written to:** the `--index_path` (e.g. `<save_dir>/cosine_train_reps.pt`), via
  `torch.save`. Reused on subsequent runs if the file already exists.
- **How built:** forward pass each train example, take last-layer hidden states,
  pool them (`--pooling weighted_mean` for RDS+, the position-weighted SGPT mean;
  also `mean` / `none`=last-token), then **L2-normalize each row**.
- **Tensor shape:** `[N_pool, H]` — one row per pool example, `H` = model hidden
  size (4096 for Llama-2-7B). `dtype` follows `--dtype` (default `bf16`).
- **Row alignment:** row `i` ↔ pool example `i` (built with `shuffle=False`,
  appended in order). No id is stored in the tensor — alignment is purely positional.

### Score pickles (the per-eval-task output of the same script)
- **Written to:** `<save_dir>/<eval_basename>_cossim.pkl`
  (or `..._cossim_promptonly.pkl` / `..._cossim_labelonly.pkl`).
- **Format:** a nested dict `{ test_idx: { train_idx: cosine_sim_float } }`, with
  `test_idx` and `train_idx` both ints. `train_idx` = pool row index.

---

## 5. Selection scripts — exact output formats

### `get_top_influences.py` (single-task selection)
Inputs: one or more score pickles/jsons + the matching pool dataset(s).
Selection methods: `max` (RDS+ default), `min`, `mean`, `mean_min/max`,
`normalized_mean_*`. `max` does **round-robin across eval/test instances**, each
round taking the highest-scoring not-yet-picked train example, until `--output_size`.

Two output modes:
- **`--output_dataset` set → JSONL of full selected examples.** Each line is the
  original pool example dict (`id`, `dataset`, `messages`, ...) **plus an added
  `influence_score`**. Required when multiple input files are given.
- **default (flag absent) → JSON list of integer indices.** A plain
  `json.dump([int, ...])` of pool row indices (single pool only). Maps back to the
  pool by positional index (see §3).

### `get_top_aggregated_influences.py` (multi-task / round-robin across datasets)
- `--output_dataset` → **JSONL of full examples** (`+ influence_score`).
- default → **JSON list of integer indices** (`saved_instances`, ints).
  Same positional mapping.

### `ppl_selections.py` (perplexity baseline selection)  ⚠ filename is *plural*
> README references `scripts/ppl_selection.py`; the real file is
> `scripts/ppl_selections.py`.
- Reads perplexity pickle(s) `nlls.pkl` (see §6), sorts by NLL **descending**.
  Default = top-perplexity; `--mid_ppl` = the 33–66th percentile slice.
- **Output: always JSONL of full examples** (`+ influence_score` = the NLL), written
  to `--output_file_path`. Keyed internally by `(file_idx, data_idx)`; `data_idx` =
  pool row index.

### Random baselines
- `construct_downsampled_balanced.py <in.jsonl> <out.jsonl> --seed --num_samples`:
  round-robin even sampling across `dataset` values (collapses `science*` into one
  bucket, drops `hard_coded`). **Output: JSONL of full examples** (no score field).
- `construct_downsampled_unbalanced.py`: flat random subsample. Output: JSONL.
- README also documents a pure `shuf`/`sort -R` one-liner for random-unbalanced.

**Summary of join strategy for the audit (Stage A):**
- If a selector emits **JSONL** (most do via `--output_dataset` / by default for
  ppl & random): join to pool metadata by **`id`** (present in each line), or by
  content-hash if `id` is absent.
- If a selector emits a **JSON int list** (RDS+/aggregated without
  `--output_dataset`): join by **positional pool row index**.
- The audit's selection-result reader must accept both shapes.

---

## 6. Perplexity scoring (`compute_influence_perplexity.py`)
- Forward pass each pool example with `labels=input_ids`; record `outputs.loss`.
- **Output:** `<save_dir>/nlls.pkl` = dict `{ row_idx: nll_float }`. Row index =
  pool row (built `shuffle=False`).

---

## 7. Implications / decisions for the audit package

1. **Stage A needs no model and no training.** It needs only: (a) the pool jsonl,
   (b) a metadata table (id, source, inferred language, length, ...), and (c) each
   selector's selected-id (or index) set. Representation ratio = (share of a group
   in the selected subset) / (share of that group in the pool).
2. **Selection-result reader** must normalize JSONL-of-examples, JSON-int-list, and
   raw score pickles into a common `selected_ids` (or `selected_indices`) form.
3. **Metadata layer** keys on `id` (or content-hash); records `source` from
   `dataset`→`source` fallback; infers `language` via GlotLID/fastText; stores to a
   columnar file (parquet via pyarrow) for fast repeated audits.
4. **To expose RDS+ selection additively** (the one allowed core touch): the cleanest
   hook is to always be able to run `get_top_influences.py` with `--output_dataset`
   (already supported) so selections carry `id`. No core rewrite required for Phase 0.

---

## 8. `audit/` package layout (created in Phase 0)

```
audit/
├── __init__.py
├── REPO_MAP.md          # this file
├── requirements-audit.txt  # pinned audit deps (kept separate from core requirements)
├── metadata/   __init__.py   # pool metadata build (id, source, language, length)
├── metrics/    __init__.py   # representation ratio & distribution metrics
├── selectors/  __init__.py   # selection-result readers + extra/rarity-aware selectors
├── experiments/ __init__.py  # Stage A driver, noise-disentanglement, etc.
├── configs/    .gitkeep      # YAML run configs
└── results/    .gitkeep      # run outputs + logs (git-ignored by repo's results/ rule)
```

---

## 9. Join keys: `pool_row_idx` (pilot) vs `id` (scaling)  [Phase 1/2 note]

The pilot metadata parquet has `pool_row_idx` 0–9999, reflecting positions in the
*saved pilot file* `audit/results/pool_pilot.jsonl`.

- **Within the pilot:** when selectors and embeddings run over that exact pilot file
  (Phase 3), positional alignment is internally consistent —
  `pool_row_idx` is a valid, cheap join key.
- **When scaling to the full pool:** `pool_row_idx` is *file-specific* (it depends on
  the reservoir sample + seed). It is **not** portable across pool files. So at full
  scale **`id` is the primary join key**, and `pool_row_idx` is only a convenience
  alias valid for one specific materialized pool file.

Therefore the canonical selection format (Phase 3) and `audit/metrics/run_audit.py`
key on **`id`** first; `pool_row_idx` / `selected_indices` is supported only as a
fallback that is resolved to `id` *via the same pool file's metadata*.

### Canonical selection JSON (defined here, produced in Phase 3)
```json
{
  "selector": "rds_plus",        // string name of the selection method
  "budget": 1000,                 // intended number selected (int)
  "seed": 42,                     // selection seed (int; 0/None if N/A)
  "pool": "audit/results/pool_pilot.jsonl",   // which pool the indices refer to
  "selected_ids": ["<id>", ...]   // PREFERRED: list of metadata `id` strings
  // optional alternative to selected_ids:
  // "selected_indices": [0, 5, 9, ...]  // pool_row_idx into `pool`, mapped to id
}
```
`run_audit.py` reads `selected_ids` if present, else maps `selected_indices` through
the metadata's `pool_row_idx`→`id`. `selector`/`budget`/`seed` fall back to the
filename if absent from the JSON.

## 10. Environment flag: broken local torch  [Phase 3 blocker]

This machine's PyTorch install fails to load (`OSError: [WinError 1114] ...c10.dll`).
- **Phase 1/2 are unaffected** — they are pure pandas/numpy + `datasets` (streaming).
  We pass `USE_TORCH=0` so `datasets` does not import torch.
- **Phase 3 will need torch** (RDS+ embeddings, perplexity). The DLL issue must be
  fixed before Phase 3 (e.g. reinstall CPU `torch`, or run the embedding/selection
  steps in the repo's CUDA/Linux env). **Flagged, not addressed now.**

## 11. Two-machine workflow: dev (local) vs real (GPU server)  [Phase 3]

Set by the project owner:

| | LOCAL (dev) | GPU SERVER (real) |
|---|---|---|
| Role | write + test pipeline | all real experiments (M1+) |
| GPU | RTX 2050, 4 GB | 3 × 24 GB |
| Flag | `--dev` on every experiment script | (no flag) |
| Output | `audit/results/dev/` (never research results) | `audit/results/` |
| Perplexity model | Pythia-160m (proxy) | Qwen2.5-1.5B (multilingual, ≥1B) |
| RDS+ embedder | all-MiniLM-L6-v2 (proxy) | repo 7B (Llama-2-7b-hf) |

`--dev` (see `audit/common.py`) routes output to `audit/results/dev/`, prints
`DEV RUN — small proxy model, not for research results.`, and swaps in the proxy
models. **Only the model + output dir change** between dev and real; adapter logic,
canonical JSON, and metrics are identical. Every selection JSON carries a `meta`
block: model, device, gpu_name, runtime_s, vram_peak_mib.

### Windows-only blocker the dev box must work around
The repo's embedding scripts (`compute_influence_cosinesim.py`,
`compute_influence_sentence_embedding.py`) call `dataset.map(num_proc=8/16)` at
**unguarded module level**. On Windows (`spawn`), each worker re-imports the module
and re-runs it → recursive process spawn → hang (observed: 17 procs from one script).
On the Linux server (`fork`) this is fine.
- **Perplexity** is unaffected: its file-path branch uses `num_proc=1`.
- **RDS+ dev** therefore does the MiniLM embedding **in-process** in
  `audit/selectors/run_rdsplus.py::embed_dev` (same text construction, cosine
  scoring, and pickle format as the repo script), then still calls the repo's
  `get_top_aggregated_influences` for selection. Real RDS+ (server) uses the repo
  `compute_influence_cosinesim.py` unchanged.

## 12. Transfer to the GPU server
Push **code only**; the server regenerates all results from scratch (do not copy
`audit/results/`, which is git-ignored anyway). Eval data (55 MB) and the pilot pool
are recreated on the server via `shell_scripts/download_eval_data.sh` and
`audit.metadata.build_metadata` / `export_pilot_jsonl`. Then run, without `--dev`:
`python -m audit.experiments.run_stage_a` → `run_audit` → `make_m1_summary`.

### Dependency decision
The core `requirements.txt` is intentionally left untouched (ground rule: touch core
files only to expose selection results, additively). Audit deps live in
**`audit/requirements-audit.txt`**, pinned, installable alongside the core env.
Target interpreter: **Python 3.12** (per `environment.yml`).
</content>
</invoke>

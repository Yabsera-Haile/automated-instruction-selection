"""Shared helpers for audit experiment scripts: dev-mode routing, model defaults,
and best-effort GPU runtime/VRAM capture.

Two-machine convention (set by the project owner):
  * LOCAL (small GPU): development only. Pass --dev to every experiment script.
    Output is routed to audit/results/dev/ and a loud warning is printed. Proxy
    models stand in for the real ones so the pipeline can be confirmed end to end.
  * GPU SERVER (real): no --dev. Output goes to audit/results/. Real models.

Only the *model* (and output dir) differ between dev and real — adapter logic,
canonical JSON, and audit metrics are identical.
"""
from __future__ import annotations

import logging
import os
import subprocess
import threading
import time

# Force MKL's GNU threading layer to avoid the Linux crash:
#   "MKL_THREADING_LAYER=INTEL is incompatible with libgomp ... library"
# (MKL numpy defaults to the INTEL layer, which clashes with the GNU OpenMP
# libgomp that torch/sklearn load). Set before numpy/torch import; this module is
# imported by every entry point, and child processes inherit os.environ. We only
# override when unset or INTEL, preserving a deliberate GNU/SEQUENTIAL/TBB choice.
if os.environ.get("MKL_THREADING_LAYER", "").upper() in ("", "INTEL"):
    os.environ["MKL_THREADING_LAYER"] = "GNU"

DEV_WARNING = "DEV RUN — small proxy model, not for research results."


def ensure_utf8(module: str) -> None:
    """On Windows, re-exec under PYTHONUTF8=1 so the repo's unencoded ``open()`` calls
    (e.g. data.py reading UTF-8 BBH/MMLU prompt files) don't crash under cp1252.

    No-op on non-Windows or once PYTHONUTF8 is already set. ``module`` is the dotted
    module path so the re-exec preserves the ``python -m`` form (needed for the
    ``audit`` package to import).
    """
    import os
    import sys
    if os.name != "nt" or os.environ.get("PYTHONUTF8") == "1":
        return
    os.environ["PYTHONUTF8"] = "1"
    os.execv(sys.executable, [sys.executable, "-m", module] + sys.argv[1:])

# Model defaults per selector x mode. All overridable via each script's --model.
PPL_DEV_MODEL = "EleutherAI/pythia-160m"      # GPT-2-class; 2048 ctx (GPT-2's 1024
#                                               crashes the repo's 2048-token script)
PPL_REAL_MODEL = "Qwen/Qwen2.5-1.5B"          # multilingual >=1B base LM. Chosen over an
#   English-centric LM (e.g. Pythia) on purpose: Pythia's English bias is exactly the
#   bias we audit, so it would confound low-resource perplexity. Qwen2.5's multilingual
#   pretraining makes the low-resource result representative and harder to dismiss.
RDS_DEV_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # sentence-embedding path
RDS_REAL_MODEL = "meta-llama/Llama-2-7b-hf"   # repo/paper RDS+ 7B backbone (cosinesim)
# LLM-quality judge (Ask-LLM/QuRating style). Instruct models.
QUALITY_DEV_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"   # fits the 4GB dev GPU
QUALITY_REAL_MODEL = "Qwen/Qwen2.5-7B-Instruct"    # strong judge on the server (bf16)

RESULTS_DIR = "audit/results"
DEV_RESULTS_DIR = "audit/results/dev"


def results_base(dev: bool) -> str:
    """Output root: audit/results/dev when dev else audit/results."""
    return DEV_RESULTS_DIR if dev else RESULTS_DIR


def model_slug(model: str) -> str:
    """Filesystem-safe tag for a model id, used to route Stage-B outputs to a
    per-model subtree so multiple target models don't collide.
    'Qwen/Qwen2.5-1.5B' -> 'qwen2.5-1.5b'; 'Qwen/Qwen2.5-7B' -> 'qwen2.5-7b'."""
    return model.split("/")[-1].lower().replace("_", "-")


# Gemma base (`-pt`) tokenizers ship WITHOUT a chat template; the -it variants do. This is
# Gemma's OWN official format (roles user/model, turns wrapped in <start_of_turn>..<end_of_turn>,
# BOS prepended once) applied via apply_chat_template -- NOT a hardcoded Llama/Qwen format.
GEMMA_CHAT_TEMPLATE = (
    "{{ bos_token }}{% for message in messages %}"
    "{{ '<start_of_turn>' + (message['role'] if message['role'] != 'assistant' else 'model') "
    "+ '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}"
    "{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"
)


# Llama-3 user/assistant format (the official turn structure, minus system/tools headers we
# don't use — a leading system turn is folded into the first user turn before templating). The
# special tokens (<|begin_of_text|> = bos, <|start_header_id|>, <|end_header_id|>, <|eot_id|>)
# are in the Llama-3.1 tokenizer's vocab even on the -pt base, which ships no chat_template.
LLAMA3_CHAT_TEMPLATE = (
    "{{ bos_token }}{% for message in messages %}"
    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' "
    "+ message['content'] | trim + '<|eot_id|>' }}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
)


def ensure_chat_template(tok, model_id: str):
    """If the tokenizer has no chat template (Gemma `-pt` and Llama-3 `-pt` bases don't), supply
    the model family's official one so apply_chat_template works identically for train and eval.
    Returns the tokenizer. Only fills a MISSING template; never overrides an existing one (so an
    instruct base like Aya keeps its own template)."""
    if getattr(tok, "chat_template", None):
        return tok
    mid = model_id.lower()
    if "gemma" in mid:
        tok.chat_template = GEMMA_CHAT_TEMPLATE
    elif "llama" in mid:
        tok.chat_template = LLAMA3_CHAT_TEMPLATE
    return tok


def announce_dev(dev: bool, logger: logging.Logger) -> None:
    if dev:
        logger.warning(DEV_WARNING)


def gpu_name() -> str | None:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None


def device_str() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class VramSampler:
    """Context manager that polls nvidia-smi to record peak GPU memory.used (MiB).

    Coarse (whole-GPU, includes other processes) but honest and dependency-free.
    ``peak_mib`` is None if nvidia-smi is unavailable. Also times the with-block.
    """

    def __init__(self, interval: float = 1.0) -> None:
        self.interval = interval
        self.peak_mib = None
        self.runtime_s = None
        self._stop = threading.Event()
        self._t0 = None
        self._thread = None

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    stderr=subprocess.DEVNULL, timeout=5,
                ).decode().strip().splitlines()
                used = max(int(x) for x in out if x.strip())
                self.peak_mib = used if self.peak_mib is None else max(self.peak_mib, used)
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self) -> "VramSampler":
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        self.runtime_s = round(time.time() - self._t0, 1)


def run_meta(model: str, dev: bool, sampler: "VramSampler | None" = None,
             extra: dict | None = None) -> dict:
    """Provenance block embedded in every result file (spec: model, VRAM, runtime)."""
    meta = {
        "model": model,
        "dev": dev,
        "device": device_str(),
        "gpu_name": gpu_name(),
        "runtime_s": sampler.runtime_s if sampler else None,
        "vram_peak_mib": sampler.peak_mib if sampler else None,
    }
    if extra:
        meta.update(extra)
    return meta

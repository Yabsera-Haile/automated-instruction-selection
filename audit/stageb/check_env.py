"""B-Step 0 / R2-Step 0 [SERVER]: verify the GPU server can load a Stage-B target model.

Confirms CUDA is available, reports per-GPU VRAM, loads the target model (BASE, not
Instruct) in bf16 onto a single card, and runs a tiny forward pass to confirm the
weights are usable. The target model is a CLI argument (--model) so the same check
serves every round:
    Round 1 (saturated): Qwen/Qwen2.5-7B   (default)
    Round 2 (movable)  : Qwen/Qwen2.5-1.5B  -- less saturated, so training-data amount
                         and selection can actually move it; same family, already on the
                         server (it was the perplexity scorer); Apache-2.0; fits trivially.

The base (not Instruct) is required: SFT cannot teach a language the base never saw, so
the target must have genuine multilingual pretraining.

Run on the server:
    python -m audit.stageb.check_env --model Qwen/Qwen2.5-1.5B
"""
from __future__ import annotations

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

GB = 1024 ** 3


def main() -> None:
    ap = argparse.ArgumentParser(description="Confirm a Stage-B target model loads on the server.")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B", help="HF model id (BASE, not Instruct).")
    args = ap.parse_args()

    print(f"torch {torch.__version__} | cuda build {torch.version.cuda}")
    print("cuda available:", torch.cuda.is_available())
    assert torch.cuda.is_available(), "CUDA not available on this machine."

    n = torch.cuda.device_count()
    print(f"device count: {n}")
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        print(f"  GPU{i}: {p.name}  total={total/GB:.1f} GiB  free={free/GB:.1f} GiB")

    kwargs = {"torch_dtype": torch.bfloat16}
    if os.getenv("HF_TOKEN"):
        kwargs["token"] = os.getenv("HF_TOKEN")

    print(f"\nLoading {args.model} in bf16 on cuda:0 (single-card fit for LoRA) ...")
    tok = AutoTokenizer.from_pretrained(args.model, **{k: v for k, v in kwargs.items() if k == "token"})
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map={"": 0}, **kwargs).eval()

    n_params = sum(p.numel() for p in model.parameters())
    alloc = torch.cuda.memory_allocated(0) / GB
    reserved = torch.cuda.memory_reserved(0) / GB
    free, total = torch.cuda.mem_get_info(0)
    print(f"\nLoaded: {n_params/1e9:.2f}B params | dtype={next(model.parameters()).dtype}")
    print(f"GPU0 after load: allocated={alloc:.1f} GiB  reserved={reserved:.1f} GiB  "
          f"free={free/GB:.1f} GiB / {total/GB:.1f} GiB")

    ids = tok("The capital of France is", return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**ids).logits
    nxt = tok.decode(logits[0, -1].argmax())
    print(f"forward OK | logits {tuple(logits.shape)} | next-token guess: {nxt!r}")

    headroom = (total - reserved * GB) / GB
    print(f"\nHEADROOM after weights: ~{headroom:.1f} GiB free on a {total/GB:.0f} GiB card "
          f"(LoRA + activations need single-digit GiB with grad checkpointing).")
    print(f"OK: {args.model} base loads in bf16 within {total/GB:.0f}GB.")


if __name__ == "__main__":
    main()

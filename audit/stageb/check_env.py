"""B-Step 0 [SERVER]: verify the GPU server can load the Stage-B target model.

Confirms CUDA is available, reports per-GPU VRAM, and loads Qwen2.5-7B (the BASE model,
not Instruct) in bf16 onto a single 24GB card to confirm headroom for LoRA training.
A tiny forward pass confirms the weights are usable.

The base (not Instruct) is required: SFT cannot teach a language the base never saw, so
the target must have genuine multilingual pretraining. Qwen2.5-7B is Apache-2.0,
multilingual, and fits 24GB in bf16 with LoRA.

Run on the server:
    python -m audit.stageb.check_env
"""
from __future__ import annotations

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-7B"
GB = 1024 ** 3


def main() -> None:
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

    print(f"\nLoading {MODEL} in bf16 on cuda:0 (single-card fit for LoRA) ...")
    tok = AutoTokenizer.from_pretrained(MODEL, **{k: v for k, v in kwargs.items() if k == "token"})
    model = AutoModelForCausalLM.from_pretrained(MODEL, device_map={"": 0}, **kwargs).eval()

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
    print("OK: Qwen2.5-7B base loads in bf16 within 24GB.")


if __name__ == "__main__":
    main()

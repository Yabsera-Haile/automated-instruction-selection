"""B-Step 0 / R2-Step 0 / R3-Step 0 [SERVER]: verify the server can load a target model.

Confirms CUDA, reports per-GPU VRAM, loads the target model (BASE, not Instruct) in bf16
on a single card, verifies its CHAT TEMPLATE (renders a couple of examples so the turn
format is eyeballed before any training), and runs a tiny forward pass. The target model
is a CLI arg (--model) so the same check serves every round:
    Round 1 (saturated) : Qwen/Qwen2.5-7B   (default)
    Round 2 (movable)   : Qwen/Qwen2.5-1.5B
    Round 3 (movable +  : google/gemma-3-1b-pt , google/gemma-3-4b-pt  (gated; needs
     multilingual)        HF_TOKEN + accepted license). 4B/12B are vision-text models
                          (Gemma3ForConditionalGeneration), so the loader falls back to
                          the image-text auto class when plain CausalLM doesn't apply.

The base (not Instruct) is required: SFT cannot teach a language the base never saw, so
the target must have genuine multilingual pretraining. The chat template MUST come from
the tokenizer (apply_chat_template) -- never a hardcoded Llama/Qwen format.

Run on the server (Round 3):
    HF_TOKEN=hf_xxx python -m audit.stageb.check_env --model google/gemma-3-1b-pt
    HF_TOKEN=hf_xxx python -m audit.stageb.check_env --model google/gemma-3-4b-pt
"""
from __future__ import annotations

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

GB = 1024 ** 3

# A tiny two-turn conversation to render the chat template (Gemma has no system role).
SAMPLE = [
    {"role": "user", "content": "Translate to French: good morning."},
    {"role": "assistant", "content": "Bonjour."},
]


def load_model(model_id: str, kwargs: dict):
    """Load for text generation, tolerating Gemma-3 vision-text checkpoints (4B/12B)."""
    try:
        m = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": 0}, **kwargs)
        return m.eval(), "AutoModelForCausalLM"
    except (ValueError, KeyError, OSError) as e:
        try:  # multimodal Gemma-3 -> Gemma3ForConditionalGeneration
            from transformers import AutoModelForImageTextToText
            m = AutoModelForImageTextToText.from_pretrained(model_id, device_map={"": 0}, **kwargs)
            return m.eval(), "AutoModelForImageTextToText"
        except Exception:
            raise e


def check_chat_template(tok) -> None:
    tmpl = getattr(tok, "chat_template", None)
    if not tmpl:
        print("chat_template: MISSING on this tokenizer -- must be supplied before "
              "training (do NOT hardcode a Llama/Qwen format).")
        return
    print("chat_template: present.")
    full = tok.apply_chat_template(SAMPLE, tokenize=False, add_generation_prompt=False)
    prompt = tok.apply_chat_template(SAMPLE[:1], tokenize=False, add_generation_prompt=True)
    print("--- rendered (full 2-turn, training form) ---")
    print(repr(full))
    print("--- rendered (user-only + generation prompt, eval form) ---")
    print(repr(prompt))


def main() -> None:
    ap = argparse.ArgumentParser(description="Confirm a Stage-B target model loads on the server.")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B", help="HF model id (BASE, not Instruct).")
    args = ap.parse_args()

    print(f"torch {torch.__version__} | cuda build {torch.version.cuda}")
    import transformers
    print(f"transformers {transformers.__version__}")
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
    else:
        print("WARNING: HF_TOKEN not set -- gated models (Gemma) will 401.")

    print(f"\nLoading {args.model} in bf16 on cuda:0 ...")
    tok = AutoTokenizer.from_pretrained(args.model, **{k: v for k, v in kwargs.items() if k == "token"})
    check_chat_template(tok)
    model, loaded_via = load_model(args.model, kwargs)

    n_params = sum(p.numel() for p in model.parameters())
    reserved = torch.cuda.memory_reserved(0) / GB
    free, total = torch.cuda.mem_get_info(0)
    print(f"\nLoaded via {loaded_via}: {n_params/1e9:.2f}B params | "
          f"dtype={next(model.parameters()).dtype} | class={type(model).__name__}")
    print(f"GPU0 after load: reserved={reserved:.1f} GiB  free={free/GB:.1f} GiB / {total/GB:.1f} GiB")

    try:  # best-effort text forward (works for CausalLM; text path of the multimodal model)
        ids = tok("The capital of France is", return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**ids)
        logits = out.logits
        print(f"forward OK | logits {tuple(logits.shape)} | "
              f"next-token guess: {tok.decode(logits[0, -1].argmax())!r}")
    except Exception as e:
        print(f"forward pass skipped/failed (non-fatal for a load check): {type(e).__name__}: {e}")

    headroom = (total - reserved * GB) / GB
    print(f"\nHEADROOM after weights: ~{headroom:.1f} GiB free on a {total/GB:.0f} GiB card.")
    print(f"OK: {args.model} loads in bf16; chat template checked above.")


if __name__ == "__main__":
    main()

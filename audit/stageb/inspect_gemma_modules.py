"""R3-Step 3 [SERVER, READ-ONLY]: introspect Gemma-3-4B for language-tower-only LoRA.

google/gemma-3-4b-pt is packaged as a vision-text model (Gemma3ForConditionalGeneration:
a SigLIP vision tower + a language tower). Our task is TEXT-ONLY, so it needs the
image-text loader just to *open*, and LoRA must target ONLY the language tower's
projections -- never the vision tower's. (The SigLIP attention also has q_proj/k_proj/
v_proj, so a plain suffix list would wrongly adapt vision modules.)

This script loads the model and PRINTS (no training, no writes):
  1. the top-level class and where the language / vision towers live,
  2. every q/k/v/o/gate/up/down_proj module path, collapsed by layer index and tagged
     language vs vision, with counts,
  3. suggested target_modules regexes, each VERIFIED to match all language-tower
     projections and ZERO vision-tower projections (PEFT uses re.fullmatch).

Run on the server:
    HF_TOKEN=hf_xxx python -m audit.stageb.inspect_gemma_modules --model google/gemma-3-4b-pt
"""
from __future__ import annotations

import argparse
import os
import re
from collections import Counter

import torch

PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def load(model_id: str):
    kwargs = {"torch_dtype": torch.bfloat16, "low_cpu_mem_usage": True}
    if os.getenv("HF_TOKEN"):
        kwargs["token"] = os.getenv("HF_TOKEN")
    else:
        print("WARNING: HF_TOKEN not set -- gated Gemma will 401.")
    from transformers import AutoModelForImageTextToText, AutoModelForCausalLM
    try:  # 4B/12B are vision-text (Gemma3ForConditionalGeneration)
        return AutoModelForImageTextToText.from_pretrained(model_id, **kwargs), "AutoModelForImageTextToText"
    except Exception as e:
        print(f"(image-text loader failed: {type(e).__name__}; trying CausalLM)")
        return AutoModelForCausalLM.from_pretrained(model_id, **kwargs), "AutoModelForCausalLM"


def is_vision(name: str) -> bool:
    return "vision" in name.lower()


def main() -> None:
    ap = argparse.ArgumentParser(description="Read-only Gemma-3 module introspection.")
    ap.add_argument("--model", default="google/gemma-3-4b-pt")
    args = ap.parse_args()

    model, via = load(args.model)
    model.eval()

    # (1) top-level structure
    print("=" * 78)
    print("TOP-LEVEL CLASS:", type(model).__name__, f"(loaded via {via})")
    print("TOP-LEVEL CHILDREN:")
    for name, child in model.named_children():
        print(f"  {name:28} {type(child).__name__}")
    print("\nTOWERS (where language / vision / projector live):")
    seen = set()
    for name, mod in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in ("language_model", "vision_tower", "multi_modal_projector") and leaf not in seen:
            seen.add(leaf)
            print(f"  {leaf:22} path='{name}'  class={type(mod).__name__}")

    # (2) every projection module, tagged by tower
    lang_paths, vis_paths = [], []
    patterns = Counter()
    proj_leaf = Counter()
    for name, _ in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in PROJ:
            (vis_paths if is_vision(name) else lang_paths).append(name)
            tower = "vision" if is_vision(name) else "language"
            patterns[(tower, re.sub(r"\.\d+\.", ".N.", name))] += 1
            proj_leaf[(tower, leaf)] += 1
    print("\n" + "=" * 78)
    print(f"PROJECTION MODULES: language={len(lang_paths)}  vision={len(vis_paths)}")
    print("\nDISTINCT PATH PATTERNS (layer index -> N):")
    for (tower, pat), c in sorted(patterns.items()):
        print(f"  [{tower:8}] {pat}   x{c}")
    print("\nPROJ LEAF NAMES BY TOWER:")
    for (tower, leaf), c in sorted(proj_leaf.items()):
        print(f"  [{tower:8}] {leaf:10} x{c}")
    if lang_paths:
        print("\nSAMPLE language-tower paths:")
        for n in lang_paths[:4]:
            print("   ", n)
    if vis_paths:
        print("SAMPLE vision-tower paths (must NOT be adapted):")
        for n in vis_paths[:4]:
            print("   ", n)

    # (3) suggested regexes, each verified against every proj module (PEFT uses fullmatch)
    proj_alt = "|".join(PROJ)
    candidates = []
    for key in ("language_model", "text_model"):
        if lang_paths and all(key in n for n in lang_paths) and not any(key in n for n in vis_paths):
            candidates.append(f".*{key}.*\\.({proj_alt})")
    candidates.append(f"(?!.*vision).*\\.({proj_alt})")   # vision-excluding fallback
    print("\n" + "=" * 78)
    print("SUGGESTED target_modules (PEFT matches via re.fullmatch on each module name):")
    for rgx in candidates:
        lang_hit = sum(1 for n in lang_paths if re.fullmatch(rgx, n))
        vis_hit = sum(1 for n in vis_paths if re.fullmatch(rgx, n))
        ok = (vis_hit == 0 and lang_hit == len(lang_paths) and lang_hit > 0)
        print(f"  regex: {rgx!r}")
        print(f"     -> language matched {lang_hit}/{len(lang_paths)}, "
              f"vision matched {vis_hit}/{len(vis_paths)}   {'*** USE THIS ***' if ok else 'check'}")
    print("\nNo training, no writes performed (read-only introspection).")


if __name__ == "__main__":
    main()

"""C2-Step 12: assemble the Wave-1 necessity matrix -> stagec/phase2/results.parquet.

Unifies the two-axis eval outputs on the matrix checkpoints with the base/full anchors and the
reused Phase-1 / C2-9 arm cells. Each cell is parsed into (selector, axis, floor, budget, seed);
each metric is tagged RECOVERY (the cell's own-axis primary), COST (own-axis preserved
capability), or CROSS (the other axis's headline). Movable/saturated skill handling
(C2-8/anchors): only math (GSM8K) and IFEval carry skill recovery; general/science are MMLU-
saturated -> reported by Stage-A representation-ratio only (flagged, not a benchmark recovery);
multilingual skill recovers via held-out perplexity.

Sources (each with its own cell-naming convention):
  matrix language : results/stagec/phase2/matrix/eval/results.parquet
  matrix skill    : results/stagec/phase2/matrix/eval_skill/skill_cells_results.parquet
  anchors         : results/stagec/phase2/anchors/anchors.json
  reuse (perplexity-low + refs): phase1/eval/results.parquet (language@10%),
                    phase2/necessity/armA_langtight/eval (language@2%),
                    phase2/necessity/armC_if_skill/eval (skill-IF@10%, IFEval)

    python -m audit.experiments.assemble_matrix
"""
from __future__ import annotations

import argparse
import json
import os
import re

DECISIVE = ["ceb", "hau", "kir", "mlt", "plt", "som", "zul"]
CONTROL = ["eng", "spa", "cmn", "arb"]
MOVABLE_SKILLS = ["math", "instruction_following", "multilingual"]
SKILL_BENCH = {"math": "gsm8k", "instruction_following": "ifeval"}   # movable native benchmarks
SATURATED_SKILLS = ["code", "general", "science/stem"]              # ratio-only (C2-8)


def parse_matrix(cell):
    """{selector}__{axis}__{floor}__b{bp}__s{seed}  or  {selector}__none__b{bp}__s{seed}."""
    m = re.match(r"^(.*?)__(none|(?:language|skill)__(?:proportional|absolute|hybrid))__b(\d+)__s(\d+)$", cell)
    if not m:
        return None
    sel, mid, bp, seed = m.groups()
    if mid == "none":
        axis, floor = "-", "none"
    else:
        axis, floor = mid.split("__")
    return dict(selector=sel, axis=axis, floor=floor, budget=int(bp) / 100.0, seed=int(seed))


def parse_reuse(cell, axis, budget):
    """Phase-1 / arm naming: {selector}__{floor}__s{seed}, full__s0, random__s0, base."""
    if cell == "base":
        return dict(selector="base", axis="-", floor="base", budget=0.0, seed=0)
    m = re.match(r"^(.*?)__(none|proportional|absolute|hybrid|proportional-nogate)__s(\d+)$", cell)
    if m:
        sel, floor, seed = m.groups()
        return dict(selector=sel, axis=("-" if floor == "none" else axis), floor=floor,
                    budget=budget, seed=int(seed))
    m = re.match(r"^(full|random)__s(\d+)$", cell)
    if m:
        return dict(selector=m.group(1), axis="-", floor=m.group(1), budget=budget, seed=int(m.group(2)))
    return None


def add(rows, meta, kind, group, metric, value):
    if value is not None and meta is not None:
        rows.append({**meta, "kind": kind, "group": group, "metric": metric, "value": float(value)})


def from_language_parquet(path, parser, rows):
    """run_stagec_eval results.parquet -> heldout_ppl (decisive=RECOVERY, control=COST),
    chrf/belebele (decisive=RECOVERY-secondary), mmlu/control-belebele (COST)."""
    import pandas as pd
    if not os.path.exists(path):
        return
    df = pd.read_parquet(path)
    for cond in df.condition.unique():
        meta = parser(cond)
        if meta is None:
            continue
        sub = df[df.condition == cond]
        for _, r in sub.iterrows():
            g, met, v = r["group"], r["metric"], r["value"]
            if met == "heldout_ppl":
                kind = "recovery" if g in DECISIVE else ("cost" if g in CONTROL else "other")
                add(rows, meta, kind, g, "heldout_ppl", v)
            elif met in ("chrf_eng_to_xx", "chrf_xx_to_eng"):
                add(rows, meta, "recovery-secondary", g, met, v)
            elif met == "belebele":
                add(rows, meta, "recovery-secondary" if g in DECISIVE else "cost", g, "belebele", v)
            elif met == "mmlu":
                add(rows, meta, "cost", "mmlu", "mmlu", v)


def from_skill_parquet(path, parser, rows):
    """run_skill_cells_eval skill_cells_results.parquet -> per movable skill on its benchmark
    (RECOVERY if the cell's audited skill, else COST/other-skill), mmlu/mmlu_stem = COST."""
    import pandas as pd
    if not os.path.exists(path):
        return
    df = pd.read_parquet(path)
    for cell in df.cell.unique():
        meta = parser(cell)
        if meta is None:
            continue
        sub = df[df.cell == cell]
        for _, r in sub.iterrows():
            b, v = r["benchmark"], r["value"]
            if b == "gsm8k":
                add(rows, meta, "skill", "math", "gsm8k", v)
            elif b == "ifeval":
                add(rows, meta, "skill", "instruction_following", "ifeval", v)
            elif b == "mbpp":
                add(rows, meta, "skill-saturated", "code", "mbpp", v)
            elif b in ("mmlu", "mmlu_stem"):
                add(rows, meta, "cost", b, b, v)


def from_anchors(path, rows):
    if not os.path.exists(path):
        return
    a = json.load(open(path, encoding="utf-8"))["anchors"]
    key2 = {"decisive_ppl": ("recovery", "decisive_macro", "heldout_ppl"),
            "control_ppl": ("cost", "control_macro", "heldout_ppl"),
            "chrf_eng_xx": ("recovery-secondary", "decisive_macro", "chrf_eng_to_xx"),
            "belebele_dec": ("recovery-secondary", "decisive_macro", "belebele"),
            "mmlu_lang": ("cost", "mmlu", "mmlu"),
            "gsm8k": ("skill", "math", "gsm8k"), "ifeval": ("skill", "instruction_following", "ifeval"),
            "mbpp": ("skill-saturated", "code", "mbpp"), "mmlu": ("cost", "mmlu", "mmlu"),
            "mmlu_stem": ("cost", "mmlu_stem", "mmlu_stem")}
    for tag in ("base", "full"):
        meta = dict(selector=tag, axis="-", floor=tag, budget=(0.0 if tag == "base" else 1.0), seed=0)
        for col, val in a.get(tag, {}).items():
            if col in key2:
                kind, g, met = key2[col]
                add(rows, meta, kind, g, met, val)


def main():
    ap = argparse.ArgumentParser(description="Assemble the Wave-1 necessity matrix.")
    ap.add_argument("--root", default="audit/results/stagec")
    ap.add_argument("--out", default="audit/results/stagec/phase2/results.parquet")
    args = ap.parse_args()
    import pandas as pd
    R, P1, P2 = args.root, f"{args.root}/phase1", f"{args.root}/phase2"
    rows = []
    # new matrix cells (both axes)
    from_language_parquet(f"{P2}/matrix/eval/results.parquet", parse_matrix, rows)
    from_skill_parquet(f"{P2}/matrix/eval_skill/skill_cells_results.parquet", parse_matrix, rows)
    # anchors
    from_anchors(f"{P2}/anchors/anchors.json", rows)
    # reused perplexity-low + refs
    from_language_parquet(f"{P1}/eval/results.parquet",
                          lambda c: parse_reuse(c, "language", 0.10), rows)
    from_language_parquet(f"{P2}/necessity/armA_langtight/eval/results.parquet",
                          lambda c: parse_reuse(c, "language", 0.02), rows)
    from_skill_parquet(f"{P2}/necessity/armC_if_skill/eval/skill_cells_results.parquet",
                       lambda c: parse_reuse(c, "skill", 0.10), rows)

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No eval rows found — run the eval passes first (see the step commands).")
    df = df.drop_duplicates(subset=["selector", "axis", "floor", "budget", "seed", "kind",
                                    "group", "metric"], keep="last")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"Assembled {len(df)} rows -> {args.out}\n")

    def table(title, mask, group, metric, agg_groups=None):
        sub = df[mask & (df.metric == metric)]
        if agg_groups:
            sub = sub[sub.group.isin(agg_groups)]
        if sub.empty:
            return
        g = sub.groupby(["selector", "axis", "floor", "budget", "seed"])["value"].mean().reset_index()
        piv = g.groupby(["selector", "axis", "floor", "budget"])["value"].agg(["mean", "std", "count"])
        print(f"=== {title} ===")
        print(piv.round(4).to_string()); print()

    print("################  RECOVERY  ################\n")
    table("LANGUAGE recovery — decisive macro held-out perplexity (lower=better)",
          df.kind == "recovery", "group", "heldout_ppl", DECISIVE + ["decisive_macro"])
    table("SKILL recovery — math (GSM8K, higher=better)", df.group == "math", "group", "gsm8k")
    table("SKILL recovery — instruction_following (IFEval, higher=better)",
          df.group == "instruction_following", "group", "ifeval")
    print("################  COST  ################\n")
    table("COST — control perplexity (lower=better)", df.kind == "cost", "group", "heldout_ppl",
          CONTROL + ["control_macro"])
    table("COST — MMLU (higher=better)", df.metric == "mmlu", "group", "mmlu")
    print("################  CROSS-COST  ################")
    print("(language cells' skill benchmarks + skill cells' decisive perplexity are in the "
          "parquet; read per (selector,axis) to see if flooring one axis starved the other.)")
    print(f"\nSATURATED skills (ratio-only, no benchmark recovery): {SATURATED_SKILLS}")


if __name__ == "__main__":
    main()

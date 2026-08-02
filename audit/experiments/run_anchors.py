"""C2-Step 11 anchors: consolidate base + full-data anchors on the COMPLETE Phase-2 metric
suite (both axes, identical columns) and fold in the movability prune.

Phase-2 pool == Phase-1 pool (no new pool was built), so the full-data anchor IS Phase-1
full__s0 (trained on the exact Phase-2 pool) and base is the no-SFT model -- neither needs a new
run. Every matrix cell's recovery is reported relative to these two anchors, so this fixes the
column set once. Reads existing evals:

  language  results/stagec/phase1/eval/results.parquet
            -> decisive-macro heldout_ppl, chrF++ eng->xx, Belebele; control heldout_ppl; MMLU
  skill     results/stagec/phase2/movability/metrics/{base,full__s0}.json
            -> GSM8K (strict+flexible), MBPP, IFEval, MMLU, MMLU-STEM

Movability (folded in): a skill benchmark is MOVABLE iff full - base clears the noise band
(reuses movability.json's per-benchmark random-seed SD when present, else a 0.02 floor).
Saturated skill-axis necessity cells are pruned to ratio-only (no seeds spent).

    python -m audit.experiments.run_anchors
"""
from __future__ import annotations

import argparse
import json
import os

from audit.experiments.run_movability_check import BENCHMARKS, bench_metric

DECISIVE = ["ceb", "hau", "kir", "mlt", "plt", "som", "zul"]
CONTROL = ["eng", "spa", "cmn", "arb"]
SKILL_OF = {"gsm8k": "math", "mbpp": "code", "ifeval": "instruction_following",
            "mmlu": "general", "mmlu_stem": "science/stem"}


def lang_anchors(parquet_path):
    """base + full language macros from the Phase-1 two-axis eval parquet."""
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    out = {}
    for cond, tag in (("base", "base"), ("full__s0", "full")):
        sub = df[df.condition == cond]
        if sub.empty:
            continue
        def macro(metric, groups):
            s = sub[(sub.metric == metric) & (sub.group.isin(groups))]["value"]
            return round(float(s.mean()), 4) if len(s) else None
        out[tag] = {
            "decisive_ppl": macro("heldout_ppl", DECISIVE),
            "chrf_eng_xx": macro("chrf_eng_to_xx", DECISIVE),
            "belebele_dec": macro("belebele", DECISIVE),
            "control_ppl": macro("heldout_ppl", CONTROL),
            "mmlu_lang": (lambda v: round(float(v.iloc[0]), 4) if len(v) else None)(
                sub[sub.metric == "mmlu"]["value"]),
        }
    return out


def skill_anchors(metrics_dir):
    """base + full skill benchmarks from the C2-8 movability per-model jsons (+ GSM8K flexible)."""
    out = {}
    for fn, tag in (("base.json", "base"), ("full__s0.json", "full")):
        p = os.path.join(metrics_dir, fn)
        if not os.path.exists(p):
            continue
        res = json.load(open(p, encoding="utf-8"))
        d = {name: bench_metric(res, name) for name, *_ in BENCHMARKS}
        gv = res.get("gsm8k", {})
        d["gsm8k_flexible"] = gv.get("exact_match,flexible-extract")
        out[tag] = {k: (round(float(v), 4) if isinstance(v, (int, float)) else None)
                    for k, v in d.items()}
    return out


def movability(metrics_dir, skill):
    """Reuse the movability.json verdict if present; else derive full-base per skill benchmark."""
    mp = os.path.join(os.path.dirname(metrics_dir), "movability.json")
    if os.path.exists(mp):
        v = json.load(open(mp, encoding="utf-8")).get("benchmarks", {})
        return {name: d.get("label") for name, d in v.items()}
    verdict = {}
    for name in SKILL_OF:
        b, f = skill.get("base", {}).get(name), skill.get("full", {}).get(name)
        if b is None or f is None:
            verdict[name] = "INCOMPLETE"
        else:
            verdict[name] = "movable" if abs(f - b) > 0.02 else "saturated"
    return verdict


def main():
    ap = argparse.ArgumentParser(description="Consolidate Phase-2 base/full anchors (C2-Step 11).")
    ap.add_argument("--lang_parquet", default="audit/results/stagec/phase1/eval/results.parquet")
    ap.add_argument("--skill_metrics", default="audit/results/stagec/phase2/movability/metrics")
    ap.add_argument("--out_dir", default="audit/results/stagec/phase2/anchors")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    lang = lang_anchors(args.lang_parquet) if os.path.exists(args.lang_parquet) else {}
    skill = skill_anchors(args.skill_metrics)
    move = movability(args.skill_metrics, skill)

    anchors = {}
    for tag in ("base", "full"):
        anchors[tag] = {**lang.get(tag, {}), **skill.get(tag, {})}

    print("=== ANCHORS (base + full-data; complete Phase-2 suite; every cell reports vs these) ===")
    cols = ["decisive_ppl", "chrf_eng_xx", "belebele_dec", "control_ppl", "mmlu_lang",
            "gsm8k", "gsm8k_flexible", "mbpp", "ifeval", "mmlu", "mmlu_stem"]
    print(f"{'metric':16} {'base':>10} {'full':>10}")
    for c in cols:
        b, f = anchors["base"].get(c), anchors["full"].get(c)
        print(f"{c:16} {('' if b is None else f'{b:.4f}'):>10} "
              f"{('' if f is None else f'{f:.4f}'):>10}")

    print("\n=== MOVABILITY (folded in): full-base spread -> which skill-axis cells to keep ===")
    keep, prune = [], []
    for name, skl in SKILL_OF.items():
        lab = move.get(name, "?")
        (keep if lab == "movable" else prune).append(f"{skl}({name})")
        print(f"  {skl:22} {name:10} {lab}")
    print(f"\nKEEP skill-axis necessity cells (movable): {keep}")
    print(f"PRUNE to ratio-only (saturated; no seeds):  {prune}")
    print("multilingual: measured by held-out perplexity (movable) -> keep.")

    miss = [f"{tag}:{c}" for tag in ("base", "full") for c in cols if anchors[tag].get(c) is None]
    if miss:
        print(f"\nMISSING anchor metrics (fill before Wave 1): {miss}")
    else:
        print("\nAll anchor columns present for base and full.")

    json.dump({"pool": "phase2==phase1 (37090)", "anchors": anchors, "movability": move,
               "keep_skill_axis": keep, "prune_ratio_only": prune},
              open(os.path.join(args.out_dir, "anchors.json"), "w"), indent=2)
    print(f"\n-> {os.path.join(args.out_dir, 'anchors.json')}")


if __name__ == "__main__":
    main()

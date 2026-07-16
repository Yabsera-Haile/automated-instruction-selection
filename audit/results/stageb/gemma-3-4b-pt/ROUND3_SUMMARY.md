# Stage B — Round 3 (Gemma-3-4B): the decisive low-resource test

**Setup.** Base = `google/gemma-3-4b-pt` (vision-text; LoRA on the language tower only, 238 modules, 0 vision).
Same 13 SFT cells as Rounds 1–2 (full + {random, perplexity-low, quality, perplexity-high} × {1%, 5%, 10%}), fixed 3 epochs.
This is the first base that is **movable AND above chance on low-resource languages** (gate, R3-Step 1: 10/12 low-resource
languages above chance on full Belebele), so low-resource erosion finally has something to act on.

**Decisive metric** = macro over the gate's above-chance LOW-RESOURCE languages
(`bod, ceb, hau, kir, mlt, plt, som, tsn, wol, zul`), evaluated on **full Belebele** (n≈900/language → macro sampling SE ≈ 0.005).
SFT models evaluated with the Gemma chat template they were trained with; `base` = raw no-SFT probe (gate protocol).
MMLU + other languages at `--limit 200`. GSM8K/MBPP/IFEval = **[PENDING WAVE B]**.

---

## TL;DR — the decisive answers

1. **Low-resource erosion by the multilingual deleters (perplexity-low, quality): NOT CONFIRMED.** Effects vs random are
   small (|Δ| ≤ 0.05) and **reverse sign between adjacent budgets** (perplexity-low: −0.025 at 5%, **+0.042 at 10%**, both
   "significant" by sampling SE) — proof that training-run variance dominates. No consistent deficit exists.
2. **The multilingual direction-flip (perplexity-high above random on low-res): REFUTED.** The Stage-A multilingual
   *flooder* scores at-or-below random on the above-chance low-resource languages (−0.023 at 5%, +0.004 at 10%). Flooding
   multilingual *data* does not lift low-resource *performance*.
3. **Where erosion is real (consistent): high-resource control languages.** All 6 eroder cells are below random
   (−0.026…−0.053; 6/6 negative, 2 individually significant) — the selectors damage what SFT actually installs.
4. **Refined thesis:** the above-chance low-resource capability lives in *pretraining*; a few-hundred-to-thousand-example SFT
   teaches the *task format*, not language knowledge. So deleting (or flooding) multilingual data in the selection neither
   destroys nor builds low-resource performance — Stage-A data distortion does **not** automatically localize downstream to
   low-resource languages on a base that already knows them.

---

## Q1+Q2 — Decisive test with effect sizes (full-Belebele sampling SEs)

Macro over the 10 above-chance low-resource languages. Δ = selector − random; SE(Δ) = √(SE²+SE²) ≈ 0.007; z = Δ/SE(Δ).

| budget | random | perplexity-low (deleter) | quality (deleter) | perplexity-high (flooder) |
|---|---|---|---|---|
| 5% | 0.353 ± 0.005 | 0.328 · **Δ −0.025 ± 0.007 (z −3.5)** | 0.347 · Δ −0.006 (z −0.9, ns) | 0.330 · **Δ −0.023 ± 0.007 (z −3.3)** |
| 10% | 0.321 ± 0.005 | 0.363 · **Δ +0.042 ± 0.007 (z +6.1)** | 0.331 · Δ +0.010 (z +1.5, ns) | 0.324 · Δ +0.004 (z +0.6, ns) |
| pooled 5+10 | 0.337 | 0.346 | 0.339 | 0.327 |

(base raw = 0.359; full-data = 0.436; every 1% cell = 0.229, degenerate — see caveats.)

**Statistical honesty — why the z-scores above do NOT establish erosion.** The sampling SEs (binomial, n≈900/language) are
tiny (±0.005 on the macro), so ±0.02 effects appear "significant." But the **same selector flips sign across adjacent
budgets at |z| > 3 in both directions**, and the random baseline itself *drops* 0.353 → 0.321 (−0.032, >6 sampling-SEs) when
given **more** data. Both facts are impossible under sampling noise alone: they measure **between-training-run variance**
(single seed per cell) of roughly ±0.03 — the same order as every candidate effect. Conclusion: on the decisive metric,
**no claim of erosion (or protection) survives**; the honest verdict is *no detectable low-resource erosion at these budgets,
with a run-variance noise floor of ~0.03*.

Per-language view at 5% (strongest languages): random ≥ perplexity-low on ceb (0.482 vs 0.398), kir (0.413 vs 0.308),
mlt (0.472 vs 0.421) — the pattern that *suggests* erosion — but each reverses or vanishes at 10% (ceb 0.449 vs 0.479,
mlt 0.423 vs 0.499). The instability is the finding.

**Direction-flip verdict (Q2).** perplexity-high amplified low-resource data 5–12× in Stage A, yet is at-or-below random
here. Combined with the 1.5B round (where perplexity-high eroded everything movable), the flood is a *data-level*
phenomenon with no capability-level payoff: extra low-resource SFT text does not teach a language the base doesn't know,
and isn't needed for languages it does know.

---

## Q3 — Budget curve: is Gemma movable?

| metric | base (raw) | 1% | 5% | 10% | 100% |
|---|---|---|---|---|---|
| MMLU | 0.596 | 0.600–0.609 | 0.613–0.618 | 0.616–0.621 | 0.620 |
| Control Belebele | 0.632 | **0.255 (degenerate)** | 0.562–0.609 | 0.590–0.643 | 0.711 |
| Low-res above-chance | 0.359 | **0.229 (degenerate)** | 0.328–0.353 | 0.321–0.363 | 0.436 |

- **Partially movable.** Belebele has real headroom (control 0.63 → 0.71; decisive low-res 0.36 → 0.44 under full data), but
  **MMLU is saturated on the 4B** (spread ~0.02 across all conditions — like the 7B, unlike the 1.5B's 0.22 spread).
- **1% SFT is destructive, uniformly.** Every 1% cell collapses to an identical, below-chance constant (0.229 on all 10
  languages, all selectors; control 0.255): 6 optimizer steps + chat template produce a degenerate answer distribution.
  1% cells are breakage, not signal.

## Q4 — Selector-vs-random on the movable capabilities

**Control Belebele (n=200/language, macro SE ≈ 0.017):**

| budget | perplexity-low | quality | perplexity-high |
|---|---|---|---|
| 5% | Δ −0.046 ± 0.025 (z −1.9) | Δ −0.033 (z −1.3) | Δ −0.026 (z −1.1) |
| 10% | Δ −0.033 (z −1.3) | **Δ −0.051 ± 0.024 (z −2.1)** | **Δ −0.053 ± 0.024 (z −2.2)** |

Individually mostly non-significant, but **6/6 cells negative** — a consistent, modest (~3–5 pt) erosion of high-resource
languages by every non-random selector, unlike the sign-flipping low-res metric. This mirrors the 1.5B finding: selection
hurts what SFT is actually building. MMLU: no signal (saturated, all 0.60–0.62). **Math (GSM8K) — [PENDING WAVE B]**
(on the 1.5B, perplexity-high eroded math with budget; to be checked on the 4B).

---

## Q5 — Three-base contrast (the backbone table)

| | **Qwen2.5-7B** (R1) | **Qwen2.5-1.5B** (R2) | **Gemma-3-4B** (R3) |
|---|---|---|---|
| Regime | saturated | movable, no language capability | movable (Belebele), MMLU saturated |
| MMLU across conditions | 0.745–0.753 (spread 0.009) | 0.44 → 0.61 (spread 0.22) | 0.60–0.62 (spread ~0.02) |
| Control Belebele | 0.81–0.87 (flat) | 0.26 → 0.69 | 0.63 → 0.71 |
| Low-res above-chance set | (base capacity floor) | **empty** — untestable | **10/12 languages** — testable |
| Selection effect seen | none — distortion invisible | perplexity-high erodes MMLU/control/**math** (−0.03…−0.07) | consistent modest erosion on control (−0.03…−0.05); **low-res: no robust erosion** (within ±0.03 run noise) |

**The arc:** 7B — saturated, nothing measurable. 1.5B — movable, distortion visibly damages general capability, but
low-resource untestable (base at chance). Gemma-3-4B — the regime where low-resource erosion can finally be measured — and
when measured honestly, **it is not there**: the Stage-A multilingual deletion (perplexity-low keeps 0.00 multilingual)
does not produce a downstream low-resource deficit vs random, while the same selectors do consistently shave the
high-resource languages. Selection distortion is real and harmful — but its downstream harm lands on the capabilities SFT
installs, not on pretrained language knowledge.

---

## Caveats

- **Single seed per cell** — the decisive metric's run-variance floor (~0.03) is *estimated from* the random baseline's
  5%→10% swing and the sign reversals; a multi-seed replication (random + perplexity-low at 5%/10%, seeds 1–2) would put a
  proper band on it and is the highest-value robustness check before writing this up.
- **1% cells are degenerate** (constant output, below chance) — excluded from all conclusions.
- **Protocol**: base = raw no-template probe (gate protocol, matches the above-chance set definition); SFT models = Gemma
  chat template (as trained). Selector-vs-random comparisons are template-vs-template (clean); base-vs-SFT deltas are
  cross-protocol (caveated).
- **Wave B pending**: GSM8K (strict+flexible), MBPP, IFEval — the 4B math direction-flip check will complete Q4.
- Eval details: `--model hf` + `add_bos_token=True` + `attn_implementation=eager`; adapters evaluated with their saved
  tokenizer (`tokenizer={adapter}`); decisive languages at full Belebele, rest at limit 200; batch 4 (loglik is
  batch-invariant).

## Status & next

- Decisive question answered (erosion **not confirmed**; flip **refuted**) with effect sizes + SEs and an explicit
  run-variance argument. Budget curve, selector comparison, three-base contrast: done.
- Next: **Wave B** (slow group) for the 4B math/code/IF picture; then the **multi-seed robustness pass** on the decisive
  cells; then the cross-round writeup (average-vs-worst-group + Stage-A→Stage-B correlation across three bases).

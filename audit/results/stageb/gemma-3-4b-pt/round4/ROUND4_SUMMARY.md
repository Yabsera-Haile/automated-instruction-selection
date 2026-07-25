# Stage B — Round 4 (Gemma-3-4B, concentrated pool): low-resource erosion, finally measurable

**What Round 4 changed.** Round 3 returned a null ("no low-resource erosion detectable") for two reasons, both fixed here:
(1) a **sparse** pool (~5 examples per decisive language) gave no dynamic range — the best- and worst-possible per-language
outcome were nearly identical; (2) **Belebele** (4-way MC, floored at 0.25) measures reading-comprehension *format*, not
language ability. Round 4 concentrates density (**7 decisive languages × N=2000 native MURI examples**, pool = 24,477) and
switches the primary metric to **held-out FLORES-200 perplexity** (continuous, floorless, format-free). Base and training path
are unchanged from Round 3 (Gemma-3-4B, language-tower-only LoRA, same config/epochs), so R3→R4 is a controlled change of the
*pool*, not the model.

**Setup.** Selectors random / perplexity-low / quality / perplexity-high at budgets {10%, 25%} + full (1% and 5% dropped:
degenerate / below the learning threshold). `random` and `perplexity-low` at **3 seeds each** (random = 3 genuine selection
draws; perplexity-low's selection is deterministic, so 3 *training* seeds) — this measures the run-variance floor directly, the
thing Round 3 could only estimate. Metric = macro held-out perplexity over the 7 languages, full FLORES devtest (1012
sentences/lang). **Base (no-SFT) macro perplexity = 39.31.**

---

## TL;DR

- **Q1 — Erosion CONFIRMED, overwhelmingly.** `perplexity-low` and `quality` (the multilingual deleters) yield **+9.4 / +7.1
  perplexity** worse than `random` at 10% — **~17–18 σ beyond the multi-seed noise band** (random's σ ≈ 0.007–0.11 ppl). The
  effect holds at 25% and in **all 7 languages** individually.
- **Q2 — Direction-flip CONFIRMED.** `perplexity-high` (multilingual *flooder*) gives the *lowest* perplexity — it beats random
  at 25% (26.3 vs 27.6). The same method, opposite direction knob, swings downstream perplexity by ~9 ppl. This is the clean
  test Round 3 couldn't run.
- **Q3 — Dose-response restored.** With density above threshold, injected dose predicts the gain (kept-count vs gain r=0.74;
  gain vs base-headroom r=0.87; selector dose 0→flood is monotone) — vs Round 3's r≈0.24 null.
- **Q4 — Metric agreement confirms the Belebele diagnosis.** Perplexity shows +9.36; chrF++ and Belebele show ≈0 (−0.69,
  −0.024) — same *sign*, but only perplexity has the sensitivity to see it. Belebele is format-confounded, exactly as diagnosed.
- **Q5 — Two-regime.** Same Gemma-3-4B: **sparse pool (R3) → no erosion; dense pool (R4) → decisive erosion.** Erosion
  detectability is a property of **density**, not model.

---

## Q1 — Decisive erosion test (held-out perplexity, primary), with multi-seed bands

Macro perplexity over 7 decisive languages; Δ = selector − random (positive = worse = eroded). Bands are ±SD across seeds.

| budget | random | perplexity-low | quality | perplexity-high |
|---|---|---|---|---|
| **10%** | 27.94 ± **0.007** | 37.31 ± 0.88 · **Δ +9.36** | 35.01 · **Δ +7.07** | 29.18 · Δ +1.24 |
| **25%** | 27.58 ± 0.11 | 35.23 ± 0.79 · **Δ +7.65** | 30.12 · **Δ +2.54** | 26.33 · **Δ −1.25** |
| full (100%) | 28.17 | | | |

- **random vs perplexity-low: z ≈ +18 (10%), +17 (25%).** The eroder is ~7–9 ppl worse, ~17× the noise band. Round 3's
  "single-seed run variance dominates" problem is gone: random's measured run-variance floor is **σ ≈ 0.007–0.11 ppl** while the
  effect is ~9 ppl — a ~100× signal-to-noise ratio.
- **Consistent across every language** (Δ perplexity-low − random @10%): hau +21.2, som +10.9, zul +9.7, plt +7.0, mlt +6.5,
  ceb +6.4, kir +4.0 — worse in all 7. Higher-perplexity (harder) languages are hit hardest.
- **`quality` erodes too**, but less severely than `perplexity-low` (both keep ~0 decisive-language examples in Stage A; quality's
  other retained data transfers slightly better). Single-seed, but Δ ≫ random's band.
- **full (100%) = 28.17 ≈ random @10–25%** — the languages saturate at the fair dose; more data barely helps (matches the
  pilot's 250→1000 plateau).

**Answer: yes — perplexity-low and quality yield significantly worse perplexity than random, far beyond the seed-variance band.**

---

## Q2 — Direction-flip: does the multilingual flooder help?

`perplexity-high` over-selects the decisive languages (Stage A: it *floods* multilingual). Downstream it gets the **lowest**
perplexity of any selector: 29.18 (10%) / **26.33 (25%)**, beating random (27.58) at 25% by 1.25 ppl (~11σ). The two directions
of the *same* method sit ~9 ppl apart (ppl-high 26.3 vs ppl-low 35.2 at 25%). At 10% ppl-high is ~random (its flooded examples
are high-perplexity/noisier, so per-example value is lower; the advantage grows with budget).

**Answer: yes — the perplexity "direction knob" flips the downstream victim, cleanly. Low-perplexity selection deletes these
languages; high-perplexity selection over-serves them.** Round 3 couldn't test this (multilingual was at the capacity/format
floor); here it's unambiguous.

---

## Q3 — Per-language dose-response (density now above threshold)

Kept native examples per language (random @25%) vs perplexity gain over base:

| lang | kept | base ppl | gain |
|---|---|---|---|
| hau | 522 | 65.53 | 28.64 |
| som | 504 | 57.34 | 12.71 |
| zul | 500 | 42.16 | 11.78 |
| plt | 498 | 34.69 | 9.67 |
| mlt | 511 | 22.77 | 8.18 |
| ceb | 467 | 32.54 | 6.64 |
| kir | 466 | 20.13 | 4.46 |

- **r(kept-count, gain) = +0.74** (vs Round 3's r ≈ 0.24 null). Kept-count is near-uniform here (random is fair, ~500/lang), so
  the stronger axis is **r(base-headroom, gain) = +0.87** — with density above threshold, each language gains in proportion to
  how much room it has. And across selectors the dose→gain relation is monotone: **0 kept → +2 ppl (perplexity-low), ~500 kept →
  +12 (random), flood → +13 (perplexity-high).**

**Answer: yes — with density above threshold, dose predicts the gain; the Round-3 null was a floor artifact, not an absence of
signal.**

---

## Q4 — Metric agreement (and the Belebele format-confound, confirmed)

`perplexity-low` − `random` on the decisive languages @10%, per metric:

| metric | random | perplexity-low | Δ | erosion detectable? |
|---|---|---|---|---|
| **held-out perplexity** (↓) | 27.94 | 37.31 | **+9.36** | **YES** |
| chrF++ eng→xx (↑) | 30.78 | 30.09 | −0.69 | no (≈0, within noise) |
| chrF++ xx→eng (↑) | 50.18 | 50.07 | −0.11 | no (≈0) |
| Belebele (↑) | 0.422 | 0.397 | −0.024 | no (≈0) |

All four agree in *sign* (the eroder is never better), but **only perplexity has the sensitivity to detect the erosion beyond
noise.** chrF++ and Belebele are dominated by *format* acquisition (learned equally from any selector's mostly-English data) and
so are nearly flat between an eroder and random. This is the pilot's finding reproduced on the full matrix: **Belebele — Round
3's decisive metric — is structurally blind to language density.** Perplexity being floorless and format-free is why it works.
(MMLU base = 0.625, saturated reference — unchanged from R3; general capability is not the story.)

**Answer: they tell the same *directional* story, but only perplexity makes it visible — which confirms the Belebele
format-confound diagnosis.**

---

## Q5 — Four-base / two-regime contrast (the backbone table)

| | **Qwen2.5-7B** (R1) | **Qwen2.5-1.5B** (R2) | **Gemma-3-4B, sparse pool** (R3) | **Gemma-3-4B, dense pool** (R4) |
|---|---|---|---|---|
| Regime | saturated | movable, language-blind | movable, above-chance, **~5 ex/lang** | movable, above-chance, **2000 ex/lang** |
| Low-res metric | Belebele (floored) | Belebele (at chance) | Belebele (format-confounded) | **held-out perplexity** (floorless) |
| Dynamic range | none | none (base at chance) | **none (sparse density)** | **large (dense density)** |
| Low-res erosion | invisible (saturated) | untestable (at chance) | **not detected (null)** | **DECISIVE: +9 ppl, z≈18** |

**The headline reframe:** R3 and R4 are the *same model and training path* — the only change is pool density (and the metric
needed to read a floorless signal). So **"does selection erosion appear" is a property of density/dynamic-range, not of the
model.** Round 3's null was not evidence of no erosion; it was a measurement-power failure. Given a language present densely
enough to be learnable and a metric able to resolve it, **popular selection methods (low-perplexity filtering, LLM-quality
filtering) measurably and severely erode low-resource-language modeling downstream — invisibly to the multiple-choice benchmarks
practitioners actually report.**

---

## Caveats

- **`quality` / `perplexity-high` are single-seed** (only random + perplexity-low were replicated). Their Δ ≫ random's band so
  significance is not in doubt, but their exact magnitudes carry more uncertainty; a seed replicate would tighten them.
- **Concentrated regime is deliberately multilingual-heavy** (decisive langs ≈ 57% of the pool). This maximizes power and is the
  correct design for a detectability test; absolute magnitudes are specific to that density (the *existence* and *direction* of
  the effect are the robust claims).
- **chrF++ b0.1 for perplexity-high collapsed** (generation degeneracy at the smallest budget) — a known small-budget artifact;
  it does not affect the perplexity result, which is teacher-forced.
- 1% and 5% budgets were dropped by design (1% degenerate; 5% thins even a concentrated language below the learning threshold).

## Status

Round 4 answers the project's central question: **yes, multilingual-deleting selection erodes low-resource languages
downstream** — with multi-seed significance (~17σ), a clean direction-flip, a restored dose-response, and a metric-agreement
result that doubles as a methodological finding (standard MC benchmarks miss it). This is the publishable core:
*signal-dependent distortion (Stage A) → measurable downstream low-resource erosion (Stage B), hidden by average-only and
multiple-choice reporting, and gated by a data-density regime that prior work (and our own Round 3) lacked the range to see.*

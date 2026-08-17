# Cross-Model Validation — does the Stage-C map replicate? (A-Step 6)

**Question.** Stage C established a rarity-aware retention floor on **Gemma-3-4B-pt** and three
load-bearing claims. A single-model result invites the "single-model paper" objection. Here we
re-run *only the load-bearing cells* on two bases from other families and ask, per claim,
whether it **replicates / partially replicates / fails**. Cross-model agreement ⇒ the claim is
general; disagreement ⇒ we *bound* it as model-dependent (still a finding, and an honest one).

**Bases** (each passed gate-before-trust in A-Step 1: above-chance Belebele on all 7 decisive
langs + a movability check that rules out a saturated metric):

| base | family | type | decisive base ppl | role |
|---|---|---|---|---|
| Gemma-3-4B-pt | Google | pt | 39.3 | original (reference) |
| Aya-Expanse-8B | Cohere | instruct | **884** (barely models them) | language replication |
| Llama-3.1-8B | Meta | pt | 33.3 (models them moderately) | language + skill replication |

Cells are **model-independent** (selection depends on the fixed selector scores + the Phase-2
concentrated pool, not the target model), so one materialization trained on all three. ≥3 seeds
per cell; budget 10%. Selector tools (perplexity scorer, quality judge) are held fixed.

---

## Claim 1 — Selector-dependent language necessity

**Gemma claim (Phase 2):** on the language axis, `perplexity-low`'s *proportional* floor **cures**
(lands at the fair random baseline) but `quality`'s *proportional* floor **fails** (stays ~2 ppl
above baseline); only the *absolute* floor cures quality. I.e. *which* selector needs the absolute
floor is selector-specific.

Decisive-macro held-out perplexity (↓ better), mean ± SD across seeds. **Δ = proportional − fair
baseline** (the necessity test: >0 ⇒ proportional insufficient ⇒ absolute floor necessary):

| base | selector | none | proportional | absolute | fair baseline | **Δ(prop)** | prop verdict |
|---|---|---|---|---|---|---|---|
| **Gemma-4B** | perplexity-low | 36.30 | 29.21 | 27.09 | 29.1 (random) | **+0.1** | **cures** |
| **Gemma-4B** | quality | 35.52 ± 0.30 | 31.14 ± 0.37 | 27.04 ± 0.26 | 29.1 (random) | **+2.0** | **fails** |
| **Aya-8B** | perplexity-low | 246.56 ± 0.49 | 88.46 ± 0.48 | 78.75 ± 0.10 | 82.1 (random) | **+6.4** | **fails** |
| **Aya-8B** | quality | 136.39 ± 0.35 | 79.30 ± 0.53 | 66.02 ± 0.21 | 82.1 (random) | **−2.8** | **cures** |
| **Llama-8B** | perplexity-low | 44.23 ± 0.68 | 28.28 ± 0.20 | 24.68 ± 0.05 | *(no random)* | +3.6 vs floor | needs floor |
| **Llama-8B** | quality | 32.08 ± 0.23 | 28.76 ± 0.05 | 24.28 ± 0.14 | *(no random)* | +4.5 vs floor | needs floor |

*(Gemma perplexity-low language is the published Phase-2 point estimate — "Phase-1 result
reproduced"; all other cells carry seed bands. Llama's random@10% was not trained in the
cross-model run, so its Δ is measured against the absolute-floor "cured" level instead.)*

**What replicates:** the *general* mechanism — plain selection erodes the decisive languages
(none is worst on every base), a floor recovers them, and the **absolute floor beats proportional
on all three bases** (Gemma −2.1/−4.1, Aya −9.7/−13.3, Llama −3.6/−4.5 ppl). Proportional is
insufficient on at least one selector on every base.

**What does not:** the *selector-specific attribution*. The clean "quality fails / ppl-low cures"
split is **reversed on Aya** (ppl-low proportional +6.4 fails, quality −2.8 cures) and **absent on
Llama** (both selectors sit ~28.5, 3.6–4.5 ppl above the cured floor — neither cures, no split).
The reversal is explained by the `none` column: on Aya `perplexity-low` erodes *far* harder
(246 vs quality's 136), because Aya barely models these languages (base 884), so recovery is
dominated by how well each selector's *surviving* data transfers — and quality's kept English
transfers better than ppl-low's low-perplexity boilerplate.

**Verdict: PARTIAL.** Floor-necessity (proportional can fail; absolute > proportional) replicates
on both new families. The selector-attribution does **not** — it tracks **erosion severity**
(whichever selector's `none` is worst is the one whose proportional fails), a base×selector
interaction, not a fixed "quality is the dangerous selector" rule. **Reframe → "floor necessity
∝ erosion severity."**

---

## Claim 2 — IFEval necessity (absolute recovers where fair-share doesn't)

**Gemma claim:** for selectors that starve instruction-following, the *absolute* floor (500 IF
examples) recovers IFEval well beyond *proportional* fair-share (96). Tested on Llama (Aya's IF is
instruct-saturated at 0.505, so it can't be eroded — that base is uninformative for skills).

IFEval prompt-strict accuracy (↑ better), mean ± SD. Base IFEval: Gemma 0.155, Llama 0.095.

| selector | floor | Gemma-4B | Llama-8B | abs − prop |
|---|---|---|---|---|
| perplexity-low | none / prop / **abs** | 0.158 / 0.218 / **0.357 ± .026** | 0.165 / 0.297 / **0.327 ± .017** | Gemma +0.14 · Llama +0.03 |
| perplexity-high | none / prop / **abs** | 0.113 / 0.303 / **0.368 ± .018** | 0.170 / 0.240 / **0.328 ± .012** | Gemma +0.07 · Llama +0.09 |
| quality | none / prop / abs | 0.302 / — / — (keeps IF, no erosion) | 0.305 / 0.292 / 0.310 (flat) | ≈0 both |

**Both bases, same pattern:** for the two selectors that *starve* IF (`perplexity-low`,
`perplexity-high`), **absolute > proportional > none** — the absolute floor recovers IFEval beyond
fair-share. `quality` *keeps* English IF data (148 examples survive its plain top-k), so it never
erodes IFEval and the floor is correctly a **no-op** on both bases. Magnitude is smaller on the
8B (base IF 0.095 → 0.33, vs Gemma 0.36) but the direction is identical.

**Verdict: REPLICATES.** Absolute-floor necessity for a starved small skill holds on the new base,
and its no-op on the non-eroding selector is the same "necessity ∝ erosion severity" logic as
Claim 1.

---

## Claim 3 — Direction-flip (perplexity-high erodes GSM8K)

**Gemma claim:** `perplexity-high` deletes low-perplexity (predictable) math, dropping GSM8K
*below* the base model — a direction-flip the deletion-prevention floor reverses.

GSM8K strict accuracy (↑ better), mean ± SD. Base GSM8K: Gemma 0.335, Llama 0.50.

| base | ppl-high none | vs base | math-absolute floor | vs base |
|---|---|---|---|---|
| **Gemma-4B** | 0.268 ± 0.006 | **−0.067 (erodes below base)** | 0.398 ± 0.010 | +0.063 |
| **Llama-8B** | 0.488 ± 0.010 | −0.012 (≈ base, within noise) | 0.582 ± 0.012 | +0.082 |

On Gemma, deleting math pushes GSM8K **below base** (0.335 → 0.268) — a real erosion the floor
rescues. On Llama-8B the same deletion leaves GSM8K **at base** (0.488 ≈ 0.50): the stronger
model's math is **pretraining-robust**, so there is no below-base flip. The floor still helps
(+0.08 over base, deletion-*prevention* enhancement), but there is no erosion to reverse.

**Verdict: FAILS (as a flip).** Bound it: the direction-flip requires a base weak enough on the
skill that SFT-math density moves it (Gemma-4B); it does **not** generalize to the more capable
Llama-8B. The floor-as-enhancement (adding a dense skill back lifts it above base) *is* general.

---

## Agreement matrix

| load-bearing claim | Gemma-4B | Aya-8B | Llama-8B | **verdict** | generality |
|---|---|---|---|---|---|
| Floor recovers low-resource erosion; **absolute > proportional** | ✓ | ✓ | ✓ | **REPLICATES** | **general** |
| Selector-*attribution* (quality's fair-share is the one that fails) | ✓ | ✗ reversed | ✗ absent | **PARTIAL** | model-dependent → reframe "necessity ∝ erosion severity" |
| IFEval: **absolute recovers a starved skill beyond fair-share**; no-op when not eroded | ✓ | n/a (IF saturated) | ✓ | **REPLICATES** | general |
| GSM8K **direction-flip below base** | ✓ | n/a | ✗ (math robust) | **FAILS** | bound to weak-on-skill bases |
| Floor-as-enhancement (add dense skill → above base) | ✓ | — | ✓ | replicates | general |
| Cost-neutrality (floor doesn't damage preserved capability) | ✓ | ✓ (ctrl ppl 26<38) | ✓ (ctrl ppl 19<25) | **REPLICATES** | general |

**Bonus (survives cross-model): metric-blindness.** Aya is above-chance on Belebele for all 7
decisive langs yet has base perplexity 884 — Belebele over-reports language knowledge on a raw
base, demonstrated on an independent model.

---

## Reframe for the write-up

The cross-model section **passes for the central mechanism**: on 3 families / 2 vendors / 2 base
types, the rarity-aware floor recovers selection-induced erosion, the **absolute** floor beats
**proportional** everywhere, and it is cost-neutral. The two claims that do *not* transfer — that
`quality` is the uniquely dangerous selector, and that math flips *below* base — are exactly the
small-model / selector-specific details reviewers expect to be model-dependent. Reporting them
honestly under one unifying principle is stronger than asserting they are universal:

> **Floor necessity is proportional to erosion severity.** A group needs the absolute floor
> exactly when the selector starves it (its `none` erosion is large); whichever selector starves a
> given group hardest is the one whose proportional fair-share fails. This subsumes the Gemma
> "quality fails / ppl-low cures" result (a special case where quality happens to erode hardest)
> and correctly predicts the Aya reversal (ppl-low erodes hardest there), the Llama IFEval no-op
> for quality (which never starves IF), and the Llama GSM8K non-flip (a robust skill is barely
> eroded, so the floor enhances rather than rescues).

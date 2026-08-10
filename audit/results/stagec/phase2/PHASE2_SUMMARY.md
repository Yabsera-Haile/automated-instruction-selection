# Stage C — Phase 2: when is a rarity-aware floor actually necessary?

**What Phase 2 tested.** Phase 1 validated the *mechanism* (a retention floor recovers eroded
low-resource modeling; the quality gate is load-bearing) but its phase gate came back NO-GO on the
*necessity* discriminator: at a 10% budget on the language axis, a **proportional** (fair-share)
floor already cured the erosion, so an **absolute** floor was sufficient-but-not-necessary. Phase 2
re-scoped to the real question — **where does fair-share fail and a floor become required?** — with
proportional and random as first-class baselines, both axes (language / skill), both budgets
(10% / 2%), and a two-dimensional readout (modeling/native-benchmark **recovery** AND
format/cross-axis **cost**). Everything is Gemma-3-4B (artifact-verified target; Qwen/Llama are
selector tools only), multi-seed (≥3) on decisive cells.

Data: `results/stagec/phase2/results.parquet` (2,419 rows). Anchors: base (no-SFT) and full-data,
on the identical benchmark set. Movability pre-check (C2-8) restricts skill-*benchmark* necessity
claims to **math (GSM8K)** and **instruction_following (IFEval)**; code/general/science are
MMLU-saturated → ratio-only; multilingual → perplexity.

---

## TL;DR — the map

- **Necessity is SELECTOR-DEPENDENT on the language axis.** perplexity-low's fair-share cures the
  erosion at 10% (Phase-1 result reproduced) — but **quality's does not** (proportional 31.1 vs
  baseline ~29.1), so the absolute floor is **necessary** for quality even at a moderate budget.
  Both selectors need the floor at 2%.
- **The floor is necessary for small movable skills, across signal families.** IFEval (fair-share =
  96 examples) is recovered only by the absolute floor (500) — for perplexity-low, perplexity-high,
  AND quality. SemDeDup (a *redundancy* signal, not a score threshold) confirms the erosion→floor→
  recovery arc on GSM8K, so the map spans **signal families**, not just perplexity/quality.
- **Dense skills need only deletion-prevention — and even that is selector-dependent.** perplexity-
  high zeroes math (GSM8K below base); any floor restores it. SemDeDup *thins* math instead of
  zeroing it, so the small 150-floor doesn't trigger and only the fair share recovers GSM8K.
- **Every floor has a cost, and it is never general capability.** MMLU and control-perplexity are
  flat everywhere. The costs are **task format** (Belebele) and **cross-axis starvation** (a hard
  language floor drives IFEval *below base*).

---

## Q1 — Recovery-per-cost map (headline)

Recovery = the worst-eroded group's own metric (↓ perplexity / ↑ native benchmark); cost = format
(control-Belebele ↑) + general (MMLU, flat). Bands are ±SD over 3 seeds. **Winner** = best recovery
at acceptable cost.

### Language axis (recovery = decisive macro perplexity; base 39.31, baseline random/full ≈ 29.1)

| selector @ budget | none | proportional | absolute | control-Belebele (format) abs vs prop | **winner** |
|---|---|---|---|---|---|
| perplexity-low @10% | 36.30 | **29.21** | 27.09 | 0.55 vs 0.68 (−0.13) | **proportional** — cures at no format cost; absolute over-recovers 2 ppl but pays format |
| **quality @10%** | 35.52 | **31.14** | **27.04** | 0.58 vs 0.69 (−0.11) | **absolute** — proportional fails (+2 over baseline); floor required |
| perplexity-low @2% | 35.11 | 31.71 | 27.49 | — | **absolute** — proportional fails; floor required (overspends budget) |
| quality @2% | 34.82 | 31.70 | 26.78 | — | **absolute** — proportional fails; floor required |

(hybrid ≡ absolute where protected: perplexity-low hybrid@10% = 27.09, identical.)

### Skill axis (recovery = native benchmark ↑; movable only)

| selector | skill | none | proportional | absolute | **winner** |
|---|---|---|---|---|---|
| perplexity-low | IFEval (0.155→0.395) | 0.158 | 0.218 | **0.357** | **absolute** — fair-share (96) below threshold |
| perplexity-high | IFEval | 0.113 | 0.303 | **0.368** | **absolute** |
| quality | IFEval | 0.302 | 0.245 | **0.367** | **absolute** |
| perplexity-high | GSM8K (0.335→0.400) | **0.268** | 0.402 | 0.407 | **either** — dense skill; any floor prevents deletion |
| semdedup | GSM8K | **0.297** | **0.390** | 0.302 | **proportional** — 150-floor insufficient (thins-not-zeros) |

**Reading:** the winner is **proportional** in the loose-budget / cross-helped-language / dense-skill
regimes, and **absolute** in the hard-eroder / tight-budget / small-skill regimes — exactly the
partition Phase 2 predicted, now measured.

---

## Q2 — Necessity verdict

**Fair-share FAILS and a floor is REQUIRED in three regimes:**
1. **Hard eroders on the language axis, even at 10%.** quality's proportional floor lands at 31.14 —
   ~2 ppl *above* the fair (random) baseline of 29.1 — while absolute reaches 27.04 (gap +4.1 ppl,
   ~16σ). perplexity-low's proportional cures (29.21 ≈ baseline), so this is genuinely
   selector-dependent, not universal.
2. **Tight budgets (2%), all selectors.** proportional recovers only to ~31.7; the floor is required
   to reach ~27 (it must exceed the nominal budget to guarantee density).
3. **Small movable skills (IFEval), all selectors.** fair-share (96 examples) recovers IFEval only to
   0.22–0.30; the absolute floor (500) reaches ~0.36 (near full 0.395). Robust across perplexity-low,
   perplexity-high, and quality.

**Fair-share SUFFICES (floor is over-recovery/robustness, not necessity):**
- easy eroder (perplexity-low) at a moderate budget on cross-helping languages;
- dense skills where a selector merely deletes and the fair share is far above threshold (math under
  perplexity-high — proportional 0.402 ≈ absolute 0.407).

**A refinement Phase 2 adds:** deletion-prevention (the small N_abs=150 skill floor) is itself
selector-dependent — it rescues GSM8K when a selector *zeroes* math (perplexity-high) but not when a
selector *thins* it (SemDeDup: absolute 0.302 ≈ none, only proportional's fair share 0.390 recovers).

---

## Q3 — Cross-help magnitude (single-language arm, C2-9 arm B)

Protecting **only kir** (the other 6 decisive languages removed so proportional cannot cross-help),
kir dose held at Phase-1 levels: kir perplexity none 19.22 → proportional 17.18 → absolute 16.00, vs
cross-helped Phase-1 (proportional 16.72 / absolute 15.69). **Cross-help contributes only ~0.3–0.5
ppl** to per-language recovery, and its removal does **not** make proportional fail (isolated
proportional still recovers kir). So per-language recovery is **mostly standalone, not collective** —
cross-help is a minor bonus, not the reason Phase-1's fair-share worked. (kir was chosen as the
least-cross-helped; this bounds the effect from below.)

---

## Q4 — Neutrality check (floors don't distort a fair selector)

On **random** (a non-eroding selector), the floor is near-neutral: random `none` decisive ppl 29.10
≈ random+`proportional` 29.27 (both ≈ baseline), GSM8K 0.377 and IFEval 0.235 sit at the fair-share
level. And the **DRoP-analog** (`proportional-nogate`) matches `proportional` on the clean pool
(29.61 vs 29.21, within band). So the floors restore/keep pool shape without adding distortion to a
selector that isn't eroding — a method-safety result. (Full neutrality would add random+absolute and
perplexity-mid; not run, since Wave 2 was scoped to the one cross-family selector.)

---

## Q5 — Stage A → B → C consistency (the closing arc)

Per selector, on the same axes, what Stage A deleted predicts what eroded downstream and what the
floor repaired:

| selector | Stage A (representation) | Stage B / native (downstream erosion) | Stage C (floor repair) |
|---|---|---|---|
| perplexity-low | deletes decisive langs → 0, IF → 0 | decisive ppl 36.3; IFEval 0.16 | absolute → 27.1; IFEval 0.36 |
| quality | down-scores low-resource (judge bucket-1/2 ≈ 2–3 vs eng 4.0) → deletes them | decisive ppl 35.5 | absolute → 27.0 |
| perplexity-high | deletes low-perplexity math → 0 | GSM8K 0.27 (**below base**) | floor → 0.40 |
| semdedup | removes redundant math (thins) | GSM8K 0.30 (below base) | proportional → 0.39 |

The signal each selector disfavours in Stage A is exactly the capability that erodes in Stage B and
that the Stage-C floor restores — the audit's arc, closed on four selectors across two signal
families.

---

## Q6 — Metric-blindness, generalized

Recovery is visible on the sensitive/native metrics and **invisible or inverted** on the confounded
ones — the R4/Phase-1 finding, now general:
- **Language:** quality's absolute floor is best on perplexity (27.0) but **worst on Belebele**
  (0.352 vs proportional 0.458) and control-Belebele (0.578 vs 0.691) — Belebele tracks *format*, not
  the modeling recovery, and points the wrong way. chrF++ likewise flat/inverted.
- **Skill:** math and IF recover on GSM8K/IFEval, but **general and science are MMLU-saturated**
  (full − base ≈ 0.01), so their floors can only be reported by Stage-A representation-ratio
  restoration — a benchmark-recovery claim there would be an instrument null, not a finding.
- **MMLU and control-perplexity are flat across every floor** (0.60–0.62; ~28), confirming the floors
  touch neither general knowledge nor high-resource language *modeling* — the costs are format and
  cross-axis only.

---

## Cross-cost (two-axis trade-off)

At a fixed budget, flooring one axis hard costs the other. The **language absolute floor starves
skills**: quality-language absolute drives IFEval to **0.128 — below the base model's 0.155** — and
GSM8K to 0.31 (vs 0.41 at none), because 94% of the budget goes to decisive languages. Conversely,
**skill floors mildly cost languages** (quality-skill absolute pushes decisive ppl to 33.7 vs
proportional 31.2 — flooring IF steals from multilingual). So "which floor wins" is budget-bound: you
cannot maximize both axes at once.

---

## The bottom line

Phase 2 turns the Phase-1 "mechanism, not general method" into a **map with regimes**:

- Use **proportional** (fair-share) when the eroder is mild, the budget loose, and the groups
  cross-help or are dense — it un-erodes at no format cost and doesn't over-concentrate.
- Use **absolute** (or hybrid) when the eroder is aggressive (quality), the budget tight (≤2%), or
  the group is small enough that its fair share is below learnability (IFEval) — there the floor is
  *necessary*, and its format/cross-axis cost is the price of recovery.
- **Deletion-prevention** covers dense skills a selector zeroes, but is calibrated to zeroing, not
  thinning (SemDeDup needs the fair share).
- Across all of it, general capability is free; the honest costs are task-format (invisible to
  perplexity, visible on Belebele) and cross-axis starvation.

This resolves Phase 1 rather than contradicting it: fair-share *did* suffice in Phase-1's easy case,
and Phase 2 locates precisely the harder cases where it doesn't.

## Caveats

- **Skill-axis benchmark necessity is restricted to math/IFEval** (movable on 4B); code/general/
  science are ratio-only. multilingual-as-skill is read via perplexity.
- **Absolute at 2% overspends the budget** by design (500/lang guarantee > the 2% pie); the necessity
  claim there is "no within-2%-budget fair-share allocation recovers these languages."
- **Wave 2 was scoped to SemDeDup** (the one mechanistically-distinct selector); perplexity-mid / IFD
  / RDS+ are descriptive coverage that the gate judged would not change a conclusion. `perplexity-
  high__skill__absolute__s1` was reseeded (its inflated language-side SD is gone).

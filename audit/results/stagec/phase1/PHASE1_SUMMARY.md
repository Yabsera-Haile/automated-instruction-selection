# Stage C — Phase 1: the rarity-aware correction (retention floor + quality gate)

**What Phase 1 tested.** Round 4 established the disease: on a dense pool, `perplexity-low` (and `quality`)
delete low-resource languages and erode their downstream perplexity by ~+9 ppl (z≈18), invisibly to the
multiple-choice benchmarks practitioners report. Phase 1 asks whether a **rarity-aware wrapper** — which changes
only *how the budget is partitioned across groups* (a retention **floor**) and *which rows are valid* (a quality
**gate**), never the base selector's per-example score — can undo that erosion in the *easiest* case: a single
selector (`perplexity-low`), one budget (10%), the language axis, on a realistic pool where the 7 decisive
languages are a 20.8% minority.

**Design.** Pool = 37,090 rows (29,390 Tülu base + 7×1,100 native MURI). Base = `gemma-3-4b-pt`, language-tower
LoRA (238/0 vision), fixed 3 epochs — identical to R3/R4, so Stage C differs from Stage B *only* in which rows the
wrapper keeps. Floor modes materialized at b=10% (k=3,709), each vs `random`/`full`, multi-seed:

| condition | floor rule | decisive kept/lang | decisive % of budget |
|---|---|---|---|
| `none` (= plain `perplexity-low`) | pool-wide top-k | **0** | 0% |
| `proportional` | round(k·\|g\|/\|pool\|) | 110 | ~21% |
| `proportional-nogate` (DRoP-analog) | proportional, **gate off** | 110 | ~21% |
| `random` | — | ~110 | ~20% |
| `absolute` | min(N_abs=500, \|g\|) | **500** | ~94% |
| `hybrid` | min(max(prop, 500), \|g\|) | **500** | ~94% |
| `full` (100%) | — | 1,100 | ~21% |

Primary metric = held-out FLORES-200 perplexity (floorless, format-free), macro over the 7 decisive languages,
full devtest. Base (no-SFT) macro = **39.31**. Multi-seed bands are ±SD over 3 seeds.

---

## TL;DR — verdict: **PHASE GATE = NO-GO (as written) → report Phase 1 as the Stage-C result.**

The rarity-aware correction is a **validated mechanism**, but the gate's *specific* discriminator did not hold:

- **The floor recovers, beyond the seed band.** `absolute`/`hybrid` recover decisive perplexity to **27.1** — the
  best of *any* condition, below `random` (29.1) and even `full` (29.1). Recovery vs the eroded `none` (36.3) is
  −9.2 ppl (~30σ); vs `proportional` it is −2.1 ppl (~6σ), both ≫ the ±0.2 seed band.
- **But `proportional` did *not* fail** — the gate's discriminating prediction is **refuted**. `proportional`
  (110/lang, below the 250 "threshold") recovered to **29.2 ≈ random ≈ full**, fully curing the erosion. The naive
  "proportional stays eroded because 110 < 250" story is wrong: recovery is **continuous in the decisive fraction**,
  and 7 languages cross-help, so even the fair share cures it.
- **The quality gate is load-bearing** (proven on a noised pool: +3.58 ppl, ~20σ) — this component *is* clearly
  necessary.
- **`absolute`'s extra 2 ppl is bought with a task-format cost** `proportional`/`random` do not pay.

So in the easy case the *sophisticated* fix (absolute floor) does **not cleanly beat the trivial one**
(proportional / plain random): both un-erode the modeling metric; absolute over-recovers on perplexity but degrades
format (chrF++/Belebele/MMLU). Per the pre-registered gate — "proceed only if absolute recovers *while proportional
does not*" — this is the **fallback branch**: the correction is a **mechanism, not yet a general method**.

---

## Q1 — Recovery table (primary perplexity, with kept-counts)

Decisive macro perplexity; recovery is measured from the eroded `none` (36.30) toward the fair references
`random`/`full` (~29.1). ↓ better.

| condition | kept/lang | macro ppl ± SD | recovery vs `none` | vs `random` |
|---|---|---|---|---|
| base (no SFT) | — | 39.31 | — | — |
| **`none`** (the disease) | 0 | **36.30** | 0 (eroded) | +7.20 |
| `proportional-nogate` | 110 | 29.61 ± 0.18 | −6.69 | +0.51 |
| `proportional` | 110 | 29.21 ± 0.40 | −7.09 | +0.11 |
| `random` | ~110 | 29.10 ± 0.54 | −7.20 | 0 |
| `full` (100%) | 1,100 | 29.12 | −7.18 | +0.02 |
| **`absolute`** | 500 | **27.09 ± 0.19** | **−9.21** | **−2.01** |
| **`hybrid`** | 500 | **27.09 ± 0.16** | −9.22 | −2.01 |

**Reading.** Two robust facts, one refuted prediction:

1. **Recovery tracks the decisive *fraction*, not the raw count.** `random` (730 ex), `proportional` (770 ex) and
   `full` (**7,700** ex) all sit at ppl ≈ 29.1 despite a 10× spread in absolute count — because all three are ~21%
   decisive. Push the fraction to 94% (`absolute`/`hybrid`) and you reach 27.1. The floor's real lever is the
   *proportion* of the SFT mix spent on the rare languages.
2. **`absolute` uniquely over-recovers** — 2.0 ppl below `random`/`full`, beyond the seed band. This is the
   concentration effect (R4's `perplexity-high` flooder, now produced deliberately). `hybrid ≡ absolute` here by
   construction (max(110, 500) = 500), matching to 0.00 ppl — an internal consistency check.
3. **REFUTED: "proportional fails."** The pilot's 250-example threshold does not transfer — `proportional`
   (110/lang) fully cured the erosion. So an absolute floor is **not required** to un-erode; it only adds
   over-recovery (at a cost, below).

**Answer:** the floor recovers low-resource perplexity beyond the seed band — but *both* proportional and absolute
cure the erosion; only absolute over-recovers, and only via concentration.

---

## Q2 — Cost table (Axis 2: preserved capability)

Per condition (seed means). MMLU 5-shot; control = eng/spa/cmn/arb (high-resource, never floored). ↑ better except ppl.

| condition | MMLU(5s) | control ppl ↓ | control Belebele ↑ |
|---|---|---|---|
| base | 0.607 | 32.95 | 0.632 |
| `none` | 0.605 | 30.02 | 0.633 |
| `random` | 0.611 | 27.96 | 0.697 |
| `proportional` | 0.604 | 27.48 | 0.681 |
| `proportional-nogate` | 0.606 | 27.27 | 0.666 |
| `full` | 0.602 | 33.35 | 0.731 |
| **`absolute`** | **0.595** | 27.81 | **0.554** |
| **`hybrid`** | **0.596** | 27.77 | **0.544** |

**Modeling capability is ~free; task-format capability pays.** Control *perplexity* for `absolute` (27.8) ≈ `random`
(28.0) — the 94% reallocation does **not** hurt high-resource language modeling (pretraining-owned). MMLU dips only
~1.2 pts. But control *Belebele* drops **−0.14 vs random / −0.08 vs base**: with only ~209 non-decisive slots left,
`absolute` under-installs high-resource instruction/reading format.

---

## Q3 — The trade-off (the central Phase-1 finding)

The two metric families **dissociate**, exactly along the R4 fault line, and now *within* the fix:

| metric family | what it measures | `absolute` vs `proportional`/`random` |
|---|---|---|
| **held-out perplexity** (decisive + control) | language *modeling* | **absolute wins** (decisive −2 ppl; control flat) |
| **chrF++ / Belebele / MMLU** | task *format* / instruction-following | **absolute loses** (decisive chrF++ −4, Belebele −0.13; control Belebele −0.14) |

Decisive secondary metrics (macro): chrF++ eng→xx — `absolute` 28.5 vs `proportional` 32.4 vs `full` 33.4;
Belebele — `absolute` 0.317 vs `proportional` 0.416 vs `random` 0.449. `absolute` is the **worst** SFT condition on
both format metrics while being the **best** on perplexity.

**So there is no free lunch at an aggressive budget.** The absolute floor maximizes rare-language *modeling* by
concentrating 94% of the budget on 7 languages, but that starves the English/high-resource *format* data the task
metrics need. `proportional`/`random` are the balanced operating point: they reach full-data modeling (~29.1) **and**
keep format intact — they just don't reach absolute's concentrated modeling depth. Which you prefer is a choice
between modeling depth and task breadth, not a strict improvement.

---

## Q4 — Quality-gate ablation (C1-Step 5, noised pool)

The gate is the one component with a **clean, decisive** result. On a noised pool (30% of each decisive language
corrupted with a low-perplexity degenerate response — `"the "`×200, near-zero ppl), the *same* absolute floor with
the gate ON vs OFF:

| condition | corrupted rows admitted (of 2,310) | decisive macro ppl ± SD | control ppl |
|---|---|---|---|
| **`absolute` gate-OFF** | **2,310** (330/lang = 66% of decisive slots) | **30.90 ± 0.28** | 32.15 |
| **`absolute` gate-ON** | **0** | **27.32 ± 0.12** | 28.27 |

**Gap = +3.58 ppl, ≈ 20σ** (SE_diff ≈ 0.18) — far beyond the seed band. Because `perplexity-low` *prefers*
low-perplexity text, the ungated floor scoops up **every** degenerate row first, filling two-thirds of each
language's slots with `the`-repetition; the gate filters all of it and restores recovery to the clean `absolute`
level (27.1). The junk is **actively harmful, not merely wasted**: it also degrades *control* perplexity (gate-off
32.2 ≈ base, SFT benefit cancelled) — 2,310 repetition examples (6% of the set) poison general modeling. **The gate
is necessary.** (Aside: an earlier corruption that *truncated* responses raised perplexity, so `perplexity-low`
self-filtered it — the gate matters precisely against noise the selector would otherwise keep.)

---

## Q5 — Metric-blindness (perplexity moves; Belebele doesn't)

The recovery ordering is visible **only** on perplexity:

| condition | decisive perplexity ↓ | decisive Belebele ↑ |
|---|---|---|
| `none` | 36.30 | 0.405 |
| `proportional` | 29.21 | 0.416 |
| `absolute` | 27.09 | **0.317** |
| range across conditions | **9.2 ppl** | ±0.05, **wrong-signed for absolute** |

Perplexity spans 9+ ppl and cleanly orders the recovery; Belebele wobbles within ±0.05 and points the **wrong way**
for `absolute` (lowest, because it is format-confounded and `absolute` has the least format data). A practitioner
reading Belebele/chrF++ would conclude the absolute floor *hurts* the low-resource languages — the exact inversion
of the modeling truth. This reproduces R4's headline: **the standard multiple-choice metric is structurally blind
to the language-density recovery that perplexity resolves.**

---

## Q6 — DRoP-analog delta (does the gate matter on the *clean* pool?)

`proportional-nogate` = a proportional (class-ratio) floor with the quality gate **off** — an analog of
class-balancing methods like DRoP.

| metric | `proportional` (gate on) | `proportional-nogate` (gate off) | Δ |
|---|---|---|---|
| decisive ppl ↓ | 29.21 | 29.61 | +0.40 (within seed band) |
| chrF++ eng→xx ↑ | 32.4 | 32.6 | ~0 |
| decisive Belebele ↑ | 0.416 | 0.402 | ~0 |

**On the clean pool the gate is a no-op** (Δ 0.40 ppl ≈ the seed band): with no corrupted rows to filter, gate-on ≈
gate-off. The gate's value is **entirely** contingent on noise being present — which the Q4 noised ablation makes
explicit (Δ +3.58 there vs +0.40 here). This correctly locates the gate as a *robustness* component, not a
free-standing accuracy lever.

---

## PHASE GATE — go/no-go

**Pre-registered rule:** *proceed to Phase 2 only if the absolute floor recovers low-resource perplexity toward
random/full beyond the seed band **while proportional does not**. If the mechanism does not appear in this easy
case, stop and report Phase 1 as the Stage-C result.*

**Evaluation of the two conjuncts:**
- *Absolute recovers beyond the seed band* — **TRUE** (27.1 vs none 36.3, vs random 29.1; −2.0 ppl, ~6–30σ).
- *Proportional does not recover* — **FALSE.** `proportional` recovered to 29.2 ≈ random ≈ full, fully curing the
  erosion. The discriminating prediction is refuted.

**Verdict: NO-GO under the gate as written.** The floor is a real, beyond-seed-band **mechanism**, and the quality
gate is decisively necessary — but the easy case did **not** separate the sophisticated fix (absolute floor) from
the trivial one (proportional / plain random). Both un-erode the modeling metric; `absolute`'s extra 2 ppl is a
concentration effect that trades away task-format capability. This is exactly the contingency the gate anticipated:
**report Phase 1 as the Stage-C result — the rarity-aware correction as a validated mechanism, not yet a general
method that beats naive alternatives.**

**Defensible reframe (a deliberate PI choice, not an automatic pass).** If Phase 2 proceeds, it must be
*re-scoped*, because its original premise (absolute needed where proportional fails) is void:
1. carry `proportional` **and** plain `random` as first-class baselines, not strawmen;
2. lead with the **modeling-vs-format trade-off**, not a single-metric win;
3. treat recovery as **fraction-continuous** (no hard threshold);
4. hunt for a regime where the floor is genuinely *necessary* — harder selectors, tighter budgets, or the **skill
   axis** — since the language axis at 10% was too easy for proportional to fail.
Absent such a regime, the honest Stage-C claim is the mechanism + the gate + the trade-off + the metric-blindness,
which is a complete and publishable result on its own.

---

## Caveats

- **`quality`/`perplexity-high` not re-run in Stage C** — Phase 1 deliberately fixed the selector to `perplexity-low`
  (the cleanest eroder) to isolate the floor/gate. Generalizing across selectors is precisely what a re-scoped
  Phase 2 would add.
- **Single budget (10%).** The trade-off magnitude is budget-specific: at 10%, N_abs=500 consumes 94% of the pie, so
  the format cost is near its worst. A looser budget would shrink both the cost and the concentration bonus.
- **`hybrid ≡ absolute`** at this budget (proportional 110 < N_abs 500); they are not independent evidence here.
- **Cross-experiment comparisons** (e.g. gate-off 30.9 vs clean `random` 29.1) share the FLORES devtest metric but
  differ in pool; treat them as indicative, not controlled.

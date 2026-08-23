# Part B — Benchmark expansion: code + safety on the Gemma-4B skill map

Extends the Stage-C skill axis beyond math/IFEval to two new benchmarks, each **movability-gated**
first (B-Step 1) so a null can't be an instrument artifact. Pool = the Stage-C Phase-2 concentrated
pool (N=37,090); base = google/gemma-3-4b-pt; floors on the same rarity-aware wrapper; ≥3 seeds per
cell (39 training cells + base). Per-skill N_abs from `n_abs_by_axis.json`: **code = 150**
(dense/deletion-prevention, math-analog), **safety = 500** (rare/reach-threshold).

## Movability gate (B-Step 1)
- **HumanEval MOVABLE** (base 0.348 → full/random ~0.37–0.39, spread 0.041 > noise 0.031) → a native
  code instrument. **MBPP SATURATED** (base 0.485 → full 0.475, spread 0.010 < noise 0.021) →
  ratio-only reference; downstream MBPP recovery is unmeasurable on the 4B (not a null finding).
- **XSTest MOVABLE, both sub-scores, but NON-MONOTONIC**: SFT installs refusal as a package —
  unsafe-refusal 0→0.99 (safe) AND safe-over-refusal 0→0.42 (over-cautious) move together. Treated
  as a **refusal-calibration shift**, not an eroded→floor→recovered capability.

## Stage-A deletion analysis (who deletes what @10%, plain top-k)
Proportional share: code 451, safety 358. **perplexity-high** deletes BOTH (code 0, safety 1);
**perplexity-low** also deletes both (code 79, safety 1) — both perplexity *extremes* evict code and
safety because those skills sit mid-perplexity. **quality** over-keeps code (1.71×), keeps safety
0.85× (not flagged). **semdedup** thins code (0.64×) but over-keeps safety (1.73×). → code selectors
{semdedup, ppl-low, ppl-high}; safety selectors (data-driven <80% fair share) {ppl-low, ppl-high}.

## Code necessity — HumanEval (base 0.348), mean ± SD across 3 seeds
| selector | none | code·absolute (150) | code·proportional (452) | verdict |
|---|---|---|---|---|
| **perplexity-high** (deletes code→0) | **0.226 ± 0.017** | 0.268 ± 0.026 | **0.341 ± 0.010** | erodes −0.12; floor recovers (prop≈base). **Necessary.** |
| perplexity-low (keeps code) | 0.329 ± 0.044 | 0.360 ± 0.030 | 0.366 ± 0.018 | ≈base — no erosion; floor ~no-op. |
| semdedup (thins to 0.64×) | 0.358 ± 0.010 | 0.362 ± 0.006 | 0.362 ± 0.013 | ≈base — no erosion; floor no-op. |

- The code floor is necessary **exactly for the selector that deletes code** (perplexity-high). For a
  dense skill the absolute floor is deletion-prevention (150) so `proportional` (452) recovers more —
  the ladder is none < absolute < proportional, the inverse of the language/IF ordering.
- **MBPP flat** across all cells (~0.48–0.51) — saturated, ratio-only, as gated.

## Safety necessity — XSTest (base 0.000 / 0.000), unsafe-refusal ↑ / over-refusal ↓
| selector | floor | unsafe-refusal | over-refusal |
|---|---|---|---|
| **perplexity-high** (deletes safety→0) | none | **0.000** | 0.000 |
| | safety·proportional | 1.000 ± 0.000 | 0.595 ± 0.046 |
| | safety·absolute (500) | 1.000 ± 0.000 | **0.728 ± 0.014** |
| **perplexity-low** (deletes safety→0) | none | 0.043 ± 0.027 | 0.011 |
| | safety·proportional | 0.193 ± 0.088 | 0.017 |
| | safety·absolute (500) | **0.810 ± 0.053** | 0.275 ± 0.037 |
| semdedup (keeps safety 1.73×) | none | **0.880 ± 0.020** | 0.224 ± 0.034 |

- Both perplexity selectors **strip safety to near-zero** unsafe-refusal (0.000 / 0.043) — a total
  safety erosion. The floor restores it; **semdedup keeps safety (0.880) so its floor is a no-op** —
  necessity ∝ erosion severity again.
- **Floor strength is selector-dependent:** perplexity-high saturates unsafe-refusal at proportional
  already (1.000), while perplexity-low **needs the absolute floor** (proportional 0.193 leaves it
  unsafe; absolute 0.810 restores).
- **Non-monotonic cost:** restoring refusal buys over-refusal, and it scales with floor size —
  perplexity-high over-refusal 0.595 (prop) → 0.728 (abs); perplexity-low 0.017 (prop) → 0.275 (abs).
  The absolute floor over-installs refusal. Safety is a calibration trade, not a clean recovery.

## Cross-cost (safety floor vs code)
Forcing the safety floor starves code, worst for the code-deleter: `perplexity-high · safety·absolute`
HumanEval = **0.161** (below its own `none` 0.226 and base 0.348); `perplexity-low · safety·absolute`
= 0.333 (mild). A protected axis draws budget from an unprotected one — real cross-axis cost.

## Verdicts
- **Code:** the retention floor recovers a movable, dense skill (HumanEval) for the selector that
  deletes it; MBPP ratio-only. Extends the necessity map with a third movable skill.
- **Safety:** the floor **can** restore refusal, but the *absolute* floor overcorrects into
  over-refusal — report as a scope boundary of the floor (calibration, not capability), with the
  proportional floor as the gentler operating point.
- **Cross-cost** documents that aggressive single-axis flooring taxes other skills.

All cells floor=budget 10%, Gemma-3-4B, 0-vision LoRA (238/0). perplexity-low__none was re-run after
a disk-full save failure (os error 28); final matrix 39/39 cells trained, none baseline n=9.

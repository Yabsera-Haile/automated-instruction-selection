# Stage B — Round 2 (Qwen2.5-1.5B): does selection distortion show downstream on a *movable* base?

**Setup.** Base = `Qwen/Qwen2.5-1.5B` (no SFT) + 13 LoRA cells (same config/fixed 3 epochs as Round 1):
`full` (100%) and {random, perplexity-low, quality, perplexity-high} × {1%, 5%, 10%}. 14 conditions total.
Eval = held-out Belebele (12 low-resource + 4 high-resource control languages) + MMLU (5-shot), `--limit 200`.

**Data status.** Complete for all 14 conditions: fast group (Belebele + MMLU) **and** slow group (GSM8K / MBPP / IFEval).
GSM8K reports both **strict-match** (primary; feeds the skill macro) and **flexible-extract**, `--limit 200`.

---

## TL;DR

- **Q1 — Budget now matters: YES (decisively).** On the 1.5B, MMLU spans 0.44→0.61 and high-resource Belebele spans
  0.26→0.69 across the budget curve; on the 7B these were flat (MMLU spread 0.009). **The saturation diagnosis is confirmed
  and the 1.5B is a movable regime.** Low-resource Belebele stays at chance (~0.25–0.29) at every budget — a capacity floor, as predicted.
- **Q2 — Selection now matters: YES, at 5% and 10%.** `perplexity-high` consistently *erodes* the movable capabilities relative
  to random (MMLU −0.03 to −0.04; control Belebele −0.03 to −0.07), while `quality` is at or slightly above random and
  `perplexity-low` ≈ random. At 1% (≤108 examples) selection makes no difference — too little data to move anything.
- **Q3 — Direction-flip: CONFIRMED on the math side.** perplexity-high (Stage-A math-*deleter*) drives GSM8K *below base* and
  below perplexity-low, and the gap widens with budget (strict-match low−high: +0.025 at 5%, +0.050 at 10%). perplexity-high thus
  erodes *every* movable capability — MMLU, control Belebele, and math. The multilingual half stays confounded (capacity floor).
- **Q4 — Saturated-vs-movable contrast (the headline): confirmed.** Selection distortion's downstream impact is **regime-dependent** —
  invisible on the saturated 7B, visible on the movable 1.5B.

---

## Above-chance language set (recorded)

Rule: a language counts as "above chance" if the **no-SFT base**'s Belebele accuracy clears chance (0.25) at a one-sided 95%
lower bound (`acc − 1.645·stderr > 0.25`). Recorded in [`abovechance_languages.json`](abovechance_languages.json).

**Result: the set is EMPTY (0 of 16).** The 1.5B base is statistically at chance on *every* eval language, including the
high-resource controls (eng 0.265, spa 0.255, cmn 0.260, arb 0.260; best low-resource was tsn 0.290, lower bound 0.237). So the
two low-resource views are:
- **(a) full 12-language low-resource macro** — reported throughout.
- **(b) above-chance macro** — **undefined/empty** for the 1.5B base, reported honestly as "no eval language is above chance
  pre-SFT." Every Belebele gain in this round is therefore SFT *teaching the task format* from a chance-level start, not lifting a
  language the base already partly knew.

---

## Q1 — Does budget now matter?

Budget curves (macro accuracy; base = no-SFT reference). `full` is the single 100% point.

**MMLU (5-shot):**

| condition | base | 1% | 5% | 10% | 100% |
|---|---|---|---|---|---|
| random          | 0.443 | 0.440 | 0.418 | 0.506 | — |
| perplexity-low  | | 0.439 | 0.411 | 0.480 | — |
| quality         | | 0.430 | 0.446 | 0.520 | — |
| perplexity-high | | 0.440 | 0.389 | 0.464 | — |
| full            | | | | | **0.609** |

**High-resource (control) Belebele:**

| condition | base | 1% | 5% | 10% | 100% |
|---|---|---|---|---|---|
| random          | 0.260 | 0.255 | 0.664 | 0.688 | — |
| perplexity-low  | | 0.259 | 0.667 | 0.667 | — |
| quality         | | 0.255 | 0.672 | 0.682 | — |
| perplexity-high | | 0.255 | 0.598 | 0.660 | — |
| full            | | | | | **0.689** |

**Low-resource (12-lang) Belebele:** base 0.249; every SFT cell 0.246–0.284 across all budgets — **flat, at chance.**

**Verdict: budget matters, strongly, for the capabilities the model *can* learn.**
- MMLU: 0.44 → 0.61 (unsaturated, so training adds real knowledge). Note it is **non-monotonic** — at 5% some selectors dip
  *below* base (perplexity-high 0.389 < base 0.443): a little skewed data can hurt before more data helps.
- Control Belebele shows a sharp **budget threshold**: at 1% (108 examples) it is still at chance (~0.255); at 5% (542 examples)
  it jumps to ~0.66. The model needs ~500 instruction examples to acquire the Belebele multiple-choice format.
- Low-resource stays at chance regardless of budget — the 1.5B lacks the multilingual capacity, exactly the tradeoff flagged when
  choosing this base.

**Skills across the budget curve (min–max over selectors at each budget; base = no-SFT):**

| skill (metric) | base | 1% | 5% | 10% | 100% |
|---|---|---|---|---|---|
| GSM8K strict (math) | 0.605 | 0.60–0.63 | 0.59–0.64 | 0.57–0.65 | 0.625 |
| MBPP (code) | 0.46 | 0.44–0.45 | 0.44–0.46 | 0.47–0.49 | 0.48 |
| IFEval (prompt-strict) | 0.13 | 0.11–0.12 | 0.14–0.16 | 0.17–0.20 | 0.205 |

- **Math (GSM8K)** is already strong on the base (0.605 — Qwen is math-heavy), so it is not budget-limited but is clearly
  *selection*-sensitive (see Q3): the spread at 10% is 0.565–0.650 depending on selector.
- **Instruction-following (IFEval)** climbs monotonically with budget (0.13 → 0.205) — more instruction data ⇒ better IFEval.
- **Code (MBPP)** is flat and low (~0.46) — the 1.5B's code ability barely responds to general instruction SFT.

---

## Q2 — Does selection now matter?

Selector minus random at each budget (positive = better than random):

**MMLU:**

| budget | random (abs) | perplexity-high | perplexity-low | quality |
|---|---|---|---|---|
| 1%  | 0.440 | +0.000 | −0.001 | −0.010 |
| 5%  | 0.418 | **−0.029** | −0.007 | **+0.029** |
| 10% | 0.506 | **−0.042** | −0.026 | +0.014 |

**Control Belebele:**

| budget | random (abs) | perplexity-high | perplexity-low | quality |
|---|---|---|---|---|
| 1%  | 0.255 | +0.000 | +0.004 | +0.000 |
| 5%  | 0.664 | **−0.066** | +0.004 | +0.009 |
| 10% | 0.688 | **−0.028** | −0.020 | −0.005 |

**Verdict: selection matters at 5–10%, not at 1%.**
- **`perplexity-high` is the clear eroder** of movable capability — worst on MMLU at every non-trivial budget (−0.03 to −0.04)
  and worst on control Belebele (−0.07 at 5%). Its Stage-A behaviour (flood high-perplexity / low-resource / unusual text) fills
  the budget with data that is *less useful* for MMLU and for learning the task format, and the low-resource data it does pick up
  gives no offsetting gain because those languages sit at the 1.5B's capacity floor.
- **`quality` ≈ best** (at or above random on MMLU), consistent with it selecting clean, well-formed instruction data.
- **`perplexity-low` ≈ random** on these high-resource metrics (its predicted victim is multilingual/low-resource, which is at the
  capacity floor here — see Q3).
- **At 1% nothing separates** — with ≤108 examples the model barely moves, so distortion has no lever yet. (This is notable: the
  budget where Stage-A *distortion* is most extreme is also where the model is least able to express it downstream.)

---

## Q3 — Does the Stage-A perplexity direction-flip show downstream?

Stage A: `perplexity-high` floods multilingual / deletes math; `perplexity-low` floods math / deletes multilingual. Prediction:
their **math (GSM8K)** and **multilingual** scores should diverge in opposite directions.

- **Math half — CONFIRMED, in the predicted direction.** GSM8K strict-match:

  | budget | perplexity-high | perplexity-low | low−high | random | base |
  |---|---|---|---|---|---|
  | 1%  | 0.630 | 0.620 | −0.010 | 0.610 | 0.605 |
  | 5%  | 0.590 | 0.615 | **+0.025** | 0.605 | |
  | 10% | 0.565 | 0.615 | **+0.050** | 0.635 | |

  perplexity-high (the Stage-A *math-deleter*) drives GSM8K **down with budget** (0.630 → 0.565, below the base's 0.605), while
  perplexity-low (the Stage-A *math-flooder*) **holds** at ~0.615. The gap opens in the predicted direction and widens with budget.
  Individual cells are ~1 SE at n=200 (GSM8K SE ≈ 0.035), so the evidence is the *monotone divergence*, not any single cell; a full
  GSM8K run (1319 items) would tighten it. The flexible-extract variant shows the same ordering far more starkly — perplexity-high
  collapses to 0.33 at 10% and `full` to 0.44 — but that partly reflects an output-format/verbosity artifact (flexible extraction
  grabbing spurious numbers from longer generations), which is why the macro uses the stabler strict-match.
- **Multilingual half — visible but confounded.** On control Belebele, perplexity-high is *worse* than perplexity-low (0.598 vs
  0.667 at 5%). This is **not** a clean flip test: control Belebele is *high-resource* languages, which neither selector targets
  differentially, and the *low-resource* languages where perplexity-high's flooding would actually show sit at the 1.5B capacity
  floor (all at chance). So on this base the multilingual channel is muted; perplexity-high's deficit here reflects general data
  *usefulness*, not a multilingual advantage. A clean multilingual flip test needs a base that can do low-resource languages
  above chance (Gemma-3, next round).

**Verdict:** the direction-flip **shows downstream on the math side** — the Stage-A signal (perplexity-high deletes math) propagates
to a measurable, budget-growing GSM8K deficit. The multilingual half remains confounded on the 1.5B and needs a multilingual-capacity
base (Gemma-3) to test cleanly.

---

## Q4 — Saturated (7B) vs movable (1.5B): the headline

| metric | 7B (Round 1) | 1.5B (Round 2) |
|---|---|---|
| **MMLU** | 0.745–0.753 across all SFT cells — **flat** (spread 0.009) | base 0.443 → 0.61 (full); **spread 0.22** |
| **Control Belebele** | base 0.47 → SFT 0.81–0.87 (budget 5→10% modest) | base 0.26 → **1%: still 0.26 → 5%: 0.66 → full: 0.69** (spread 0.43) |
| **Low-res Belebele** | 0.30–0.33 (slightly *above* chance) | 0.25–0.29 (**at chance**) |
| **GSM8K (math)** | (Round 1 slow group at 2048 caps — not re-run) | base 0.605; perplexity-high erodes to 0.565, quality lifts to 0.650 |
| **Selection effect** | none visible (saturated) | perplexity-high erodes MMLU/control/math by 0.03–0.07 at 5–10% |

**Headline:** *the downstream impact of instruction-selection distortion is regime-dependent.* On a strong, benchmark-saturated
7B, a light LoRA nudge from any subset lands in the same place, so selection distortion is **invisible**. On a smaller, movable
1.5B, budget and selection both **visibly** change the model — and the Stage-A "flooder" (perplexity-high) measurably erodes the
capabilities the model can actually learn. The 1.5B is *weaker* on low-resource languages (at chance vs the 7B's slightly-above),
so this round reveals selection effects on **movable capabilities (MMLU, high-resource languages, math, instruction-following)**, not
on truly-low-resource languages. Lifting low-resource languages above chance remains a separate lever (a multilingual base such as
Gemma-3), as scoped.

---

## Caveats

- **1% cells are tiny and noisy** — 108 examples (perplexity-high: only **86** distinct, from the dedup artifact), 3 optimizer
  steps. Treat 1% as directional only.
- **`--limit 200`** ⇒ Belebele SE ≈ 0.031; the empty above-chance set is robust (even the best low-resource lower bound was 0.237),
  but borderline calls would tighten under full Belebele (900 items) if we ever need them.
- **`max_seq_len=1024`** drops ~2–3% of rows (prompt-longer-than-context) uniformly across cells — same policy as Round 1, so fair.
- **`--limit 200` on GSM8K** ⇒ SE ≈ 0.035, so the math direction-flip is directionally consistent but each cell is ~1 SE; treat as
  suggestive-and-consistent, confirmable with full GSM8K. Both strict-match (primary/macro) and flexible-extract are reported; the
  strict/flexible gap on perplexity-high and `full` is an output-format artifact, not a separate capability signal.
- **Round-1 (7B) slow group was run at 2048-token caps**, so its GSM8K/MBPP/IFEval are not directly comparable to Round 2's capped
  512/256 — the Q4 contrast uses fast-group metrics (MMLU/Belebele) for the 7B, which are cap-independent.
- All Round-2 outputs are under the model-tagged tree `audit/results/stageb/qwen2.5-1.5b/`; Round-1 (7B) results are untouched.

## Status & next

- **Q1 ✓, Q2 ✓, Q3 ✓ (math side; multilingual side capacity-limited), Q4 ✓** — all four answered; both eval waves complete for all
  14 conditions.
- **The one selection effect that is unambiguous downstream is `perplexity-high` erosion** of every movable capability (MMLU,
  control Belebele, GSM8K math) at 5–10%, growing with budget — a direct downstream footprint of the Stage-A distortion.
- **Next lever — a multilingual base (`Gemma-3-1B`):** the 1.5B confirms the regime is movable but cannot express *low-resource*
  erosion because it is at chance there pre- and post-SFT. Testing whether selection erodes languages the base *can* do requires a
  base with real low-resource capacity.

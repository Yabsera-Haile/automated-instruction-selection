# Stage B — Round 2 (Qwen2.5-1.5B): does selection distortion show downstream on a *movable* base?

**Setup.** Base = `Qwen/Qwen2.5-1.5B` (no SFT) + 13 LoRA cells (same config/fixed 3 epochs as Round 1):
`full` (100%) and {random, perplexity-low, quality, perplexity-high} × {1%, 5%, 10%}. 14 conditions total.
Eval = held-out Belebele (12 low-resource + 4 high-resource control languages) + MMLU (5-shot), `--limit 200`.

**Data status.** This summary covers the **fast group (Belebele + MMLU)**, which is complete for all 14 conditions.
The **slow group (GSM8K / MBPP / IFEval) has not run yet (Wave B)** — so the *skills* curve in Q1 and the *math* half of the
Q3 direction-flip are marked **[PENDING WAVE B]** below and will be filled in when those results land.

---

## TL;DR

- **Q1 — Budget now matters: YES (decisively).** On the 1.5B, MMLU spans 0.44→0.61 and high-resource Belebele spans
  0.26→0.69 across the budget curve; on the 7B these were flat (MMLU spread 0.009). **The saturation diagnosis is confirmed
  and the 1.5B is a movable regime.** Low-resource Belebele stays at chance (~0.25–0.29) at every budget — a capacity floor, as predicted.
- **Q2 — Selection now matters: YES, at 5% and 10%.** `perplexity-high` consistently *erodes* the movable capabilities relative
  to random (MMLU −0.03 to −0.04; control Belebele −0.03 to −0.07), while `quality` is at or slightly above random and
  `perplexity-low` ≈ random. At 1% (≤108 examples) selection makes no difference — too little data to move anything.
- **Q3 — Direction-flip: partially visible, math half [PENDING WAVE B].** The clean test is GSM8K (needs Wave B). The
  multilingual half is confounded on the 1.5B (see below).
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
  choosing this base. **[Skills curve (GSM8K/MBPP/IFEval) across budgets: PENDING WAVE B.]**

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

- **Math half — [PENDING WAVE B].** GSM8K is the clean discriminator (perplexity-low floods math ⇒ predict perplexity-low > perplexity-high
  on GSM8K). Not yet evaluated.
- **Multilingual half — visible but confounded.** On control Belebele, perplexity-high is *worse* than perplexity-low (0.598 vs
  0.667 at 5%). This is **not** a clean flip test: control Belebele is *high-resource* languages, which neither selector targets
  differentially, and the *low-resource* languages where perplexity-high's flooding would actually show sit at the 1.5B capacity
  floor (all at chance). So on this base the multilingual channel is muted; perplexity-high's deficit here reflects general data
  *usefulness*, not a multilingual advantage. A clean multilingual flip test needs a base that can do low-resource languages
  above chance (Gemma-3, next round).

**Verdict:** the direction-flip cannot be fully adjudicated until Wave B (math), and its multilingual half will remain confounded
until the multilingual-capacity lever is pulled.

---

## Q4 — Saturated (7B) vs movable (1.5B): the headline

| metric | 7B (Round 1) | 1.5B (Round 2) |
|---|---|---|
| **MMLU** | 0.745–0.753 across all SFT cells — **flat** (spread 0.009) | base 0.443 → 0.61 (full); **spread 0.22** |
| **Control Belebele** | base 0.47 → SFT 0.81–0.87 (budget 5→10% modest) | base 0.26 → **1%: still 0.26 → 5%: 0.66 → full: 0.69** (spread 0.43) |
| **Low-res Belebele** | 0.30–0.33 (slightly *above* chance) | 0.25–0.29 (**at chance**) |
| **Selection effect** | none visible (saturated) | perplexity-high erodes MMLU/control by 0.03–0.07 at 5–10% |

**Headline:** *the downstream impact of instruction-selection distortion is regime-dependent.* On a strong, benchmark-saturated
7B, a light LoRA nudge from any subset lands in the same place, so selection distortion is **invisible**. On a smaller, movable
1.5B, budget and selection both **visibly** change the model — and the Stage-A "flooder" (perplexity-high) measurably erodes the
capabilities the model can actually learn. The 1.5B is *weaker* on low-resource languages (at chance vs the 7B's slightly-above),
so this round reveals selection effects on **movable capabilities (MMLU, high-resource languages, and — pending — skills)**, not
on truly-low-resource languages. Lifting low-resource languages above chance remains a separate lever (a multilingual base such as
Gemma-3), as scoped.

---

## Caveats

- **1% cells are tiny and noisy** — 108 examples (perplexity-high: only **86** distinct, from the dedup artifact), 3 optimizer
  steps. Treat 1% as directional only.
- **`--limit 200`** ⇒ Belebele SE ≈ 0.031; the empty above-chance set is robust (even the best low-resource lower bound was 0.237),
  but borderline calls would tighten under full Belebele (900 items) if we ever need them.
- **`max_seq_len=1024`** drops ~2–3% of rows (prompt-longer-than-context) uniformly across cells — same policy as Round 1, so fair.
- **Skills (GSM8K/MBPP/IFEval) and the Q3 math test are not yet in** — Wave B.
- All Round-2 outputs are under the model-tagged tree `audit/results/stageb/qwen2.5-1.5b/`; Round-1 (7B) results are untouched.

## Status & next

- **Q1 ✓ (MMLU/Belebele), Q2 ✓, Q4 ✓** answered. **Q1-skills and Q3-math ⇒ run Wave B** (slow group: IFEval 512 / MBPP 256 /
  GSM8K 512 both strict+flexible, batch 8, 3 GPUs) over all 14 conditions, then this summary's two [PENDING] blocks get filled.
- After Wave B: the low-resource-erosion question specifically needs a multilingual base (**Gemma-3-1B**) — the 1.5B confirms the
  regime is movable but cannot express low-resource erosion because it is at chance there pre- and post-SFT.

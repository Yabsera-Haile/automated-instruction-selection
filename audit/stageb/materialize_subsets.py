"""B-Step 1 [LOCAL]: materialize the 9 Stage-B training subsets.

Joins each Stage-A selection's `selected_ids` back to the enriched pilot pool to produce
a training jsonl that preserves the `messages` field the trainer needs. The 9 cells:

    full                          (the whole enriched pilot)            budget 1.00
    random,         b 0.05 / 0.10 (fair baseline, seed 0)
    perplexity-low, b 0.05 / 0.10 (practitioner-default eroder)
    quality,        b 0.05 / 0.10 (practitioner-default eroder)
    perplexity-high,b 0.05 / 0.10 (flooder contrast)

Reads the enriched pool + selection JSONs from audit/stageb/data/ (committed from the
server). Writes audit/results/stageb/subsets/<condition>__b<budget>.jsonl and checks
each row count against round(budget * pool_size).

Run locally:
    python -m audit.stageb.materialize_subsets
"""
from __future__ import annotations

import argparse
import json
import logging
import os

logger = logging.getLogger("audit.stageb.materialize_subsets")

# (condition, selection filename or None for full, budget)
CELLS = [
    ("full", None, 1.0),
    ("random", "random__b0.05__s0.json", 0.05),
    ("random", "random__b0.1__s0.json", 0.10),
    ("perplexity-low", "perplexity-low__b0.05__s0.json", 0.05),
    ("perplexity-low", "perplexity-low__b0.1__s0.json", 0.10),
    ("quality", "quality__b0.05__s0.json", 0.05),
    ("quality", "quality__b0.1__s0.json", 0.10),
    ("perplexity-high", "perplexity-high__b0.05__s0.json", 0.05),
    ("perplexity-high", "perplexity-high__b0.1__s0.json", 0.10),
]


def read_pool(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def valid_messages(row: dict) -> bool:
    m = row.get("messages")
    return isinstance(m, list) and len(m) > 0 and all(
        isinstance(x, dict) and x.get("content") for x in m)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Materialize Stage-B training subsets.")
    ap.add_argument("--pool", default="audit/stageb/data/pilot_pool.jsonl",
                    help="Enriched pilot pool jsonl (id + messages).")
    ap.add_argument("--selections_dir", default="audit/stageb/data/sel")
    ap.add_argument("--out_dir", default="audit/results/stageb/subsets")
    args = ap.parse_args()

    missing = []
    if not os.path.exists(args.pool):
        missing.append(args.pool)
    for _, fn, _ in CELLS:
        if fn and not os.path.exists(os.path.join(args.selections_dir, fn)):
            missing.append(os.path.join(args.selections_dir, fn))
    if missing:
        raise SystemExit(
            "Missing enriched Stage-B inputs (commit them from the server):\n  "
            + "\n  ".join(missing))

    pool = read_pool(args.pool)
    pool_size = len(pool)
    by_id = {str(r.get("id")): r for r in pool}
    if len(by_id) != pool_size:
        logger.warning("Pool has duplicate ids: %d rows, %d unique ids.",
                       pool_size, len(by_id))
    logger.info("Enriched pool: %d rows", pool_size)
    os.makedirs(args.out_dir, exist_ok=True)

    all_ok = True
    for condition, fn, budget in CELLS:
        if fn is None:  # full-data
            selected = pool
        else:
            obj = json.load(open(os.path.join(args.selections_dir, fn), encoding="utf-8"))
            sel_ids = [str(i) for i in obj["selected_ids"]]
            sel_set = set(sel_ids)
            missing_ids = sum(1 for i in sel_set if i not in by_id)
            if missing_ids:
                logger.warning("%s: %d selected ids not in pool.", fn, missing_ids)
            # pool order, restricted to the selected set (deterministic)
            selected = [r for r in pool if str(r.get("id")) in sel_set]

        out_path = os.path.join(args.out_dir, f"{condition}__b{budget}.jsonl")
        n_bad = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for r in selected:
                if not valid_messages(r):
                    n_bad += 1
                    continue
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n = len(selected) - n_bad
        expected = round(budget * pool_size)
        ok = (n == expected) if fn else (n == pool_size)
        flag = "" if ok else f"  <-- EXPECTED {expected}"
        if not ok:
            all_ok = False
        bad = f" ({n_bad} dropped: no valid messages)" if n_bad else ""
        logger.info("%-16s b=%.2f -> %5d rows%s%s -> %s",
                    condition, budget, n, bad, flag, os.path.basename(out_path))

    logger.info("=" * 60)
    logger.info("ACCEPTANCE: %s — 9 subsets written to %s",
                "PASS" if all_ok else "FAIL (row-count mismatch above)", args.out_dir)
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

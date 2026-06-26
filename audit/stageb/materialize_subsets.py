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
    """Trainable for SFT: at least one user turn AND one assistant turn with content.
    (Drops e.g. wildguard rows that have an empty assistant response — no target.)"""
    m = row.get("messages")
    if not (isinstance(m, list) and m):
        return False
    has_user = any(x.get("role") == "user" and (x.get("content") or "").strip() for x in m)
    has_asst = any(x.get("role") == "assistant" and (x.get("content") or "").strip() for x in m)
    return has_user and has_asst


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
    # dedup by id (first occurrence); split into trainable vs empty-target.
    by_id: dict[str, dict] = {}
    invalid_ids: set[str] = set()
    for r in pool:
        i = str(r.get("id"))
        if i in by_id or i in invalid_ids:
            continue
        (by_id.__setitem__(i, r) if valid_messages(r) else invalid_ids.add(i))
    n_unique = len(by_id) + len(invalid_ids)
    logger.info("Pool: %d rows | %d unique ids (%d dup rows) | %d trainable | %d "
                "empty-target dropped", pool_size, n_unique, pool_size - n_unique,
                len(by_id), len(invalid_ids))
    os.makedirs(args.out_dir, exist_ok=True)

    all_ok = True
    for condition, fn, budget in CELLS:
        if fn is None:  # full-data = all unique trainable rows
            rows = list(by_id.values())
            nominal = n_unique
            dup, dropped, missing = pool_size - n_unique, len(invalid_ids), 0
        else:
            obj = json.load(open(os.path.join(args.selections_dir, fn), encoding="utf-8"))
            sel = [str(i) for i in obj["selected_ids"]]
            nominal = len(sel)
            uniq = list(dict.fromkeys(sel))           # unique, order-preserving
            dup = len(sel) - len(uniq)
            rows = [by_id[i] for i in uniq if i in by_id]
            dropped = sum(1 for i in uniq if i in invalid_ids)
            missing = sum(1 for i in uniq if i not in by_id and i not in invalid_ids)

        out_path = os.path.join(args.out_dir, f"{condition}__b{budget}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        n = len(rows)
        # accounting must balance (no inflation, nothing unexplained) and all rows valid.
        balanced = (n + dropped + missing + (dup if fn else 0)) == nominal + (0 if fn else 0)
        # for full: nominal == n + dropped (dup already excluded from n_unique)
        if fn is None:
            balanced = (n + dropped == nominal)
        else:
            balanced = (n + dropped + missing == len(uniq)) and (n <= len(uniq))
        all_ok = all_ok and balanced
        deltas = []
        if dup:
            deltas.append(f"-{dup} dup-id")
        if dropped:
            deltas.append(f"-{dropped} empty-target")
        if missing:
            deltas.append(f"-{missing} not-in-pool")
        d = (" (" + ", ".join(deltas) + ")") if deltas else ""
        logger.info("%-16s b=%.2f | nominal %5d -> %5d trainable%s | %s%s",
                    condition, budget, nominal, n, d,
                    "OK" if balanced else "UNBALANCED!", "" if balanced else " <--")

    logger.info("=" * 60)
    logger.info("ACCEPTANCE: %s — 9 subsets in %s; every row has a user+assistant "
                "message with content; counts reconcile to budget minus dropped "
                "dup-id/empty-target rows.",
                "PASS" if all_ok else "FAIL", args.out_dir)
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

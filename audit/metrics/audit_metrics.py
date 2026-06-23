"""Audit metrics — pure functions, no I/O, no side effects.

Every function takes the same signature::

    (df, selected_ids, group_col)

where
  - ``df``          : metadata DataFrame (must contain columns ``id`` and ``group_col``)
  - ``selected_ids``: a set/list of ``id`` strings that were selected
  - ``group_col``   : column to group by (e.g. ``resource_bucket`` or ``skill_label``)

"Pool" = all rows of ``df``. "Kept"/"selected" = rows of ``df`` whose ``id`` is in
``selected_ids`` (ids not present in ``df`` are ignored, so counts are always
consistent with the metadata).

The headline metric is ``representation_ratio``: <1 means a group is under-selected
relative to its share of the pool.
"""
from __future__ import annotations

import math
from typing import Iterable

import pandas as pd


# --------------------------------------------------------------------------- #
# internal helpers
# --------------------------------------------------------------------------- #

def _counts(df: pd.DataFrame, selected_ids: Iterable[str], group_col: str):
    """Return (pool_counts, kept_counts) as pandas Series indexed by group.

    ``kept_counts`` is reindexed to the full set of pool groups (missing => 0).
    """
    selected_set = set(selected_ids)
    pool_counts = df[group_col].value_counts()
    kept_mask = df["id"].isin(selected_set)
    kept_counts = df.loc[kept_mask, group_col].value_counts()
    kept_counts = kept_counts.reindex(pool_counts.index, fill_value=0)
    return pool_counts, kept_counts


def _shannon_entropy(counts: pd.Series) -> float:
    """Shannon entropy (base 2, bits) of a count distribution. Empty => 0.0."""
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values:
        if c > 0:
            p = c / total
            ent -= p * math.log2(p)
    return ent


# --------------------------------------------------------------------------- #
# public metrics
# --------------------------------------------------------------------------- #

def representation_ratio(df, selected_ids, group_col) -> dict:
    """(kept_g / total_kept) / (pool_g / total_pool) per group. <1 = under-selected."""
    pool_counts, kept_counts = _counts(df, selected_ids, group_col)
    total_pool = float(pool_counts.sum())
    total_kept = float(kept_counts.sum())
    out = {}
    for g in pool_counts.index:
        pool_share = pool_counts[g] / total_pool if total_pool else 0.0
        if total_kept == 0 or pool_share == 0:
            out[g] = 0.0
        else:
            kept_share = kept_counts[g] / total_kept
            out[g] = kept_share / pool_share
    return out


def retention_rate(df, selected_ids, group_col) -> dict:
    """kept_g / pool_g per group."""
    pool_counts, kept_counts = _counts(df, selected_ids, group_col)
    return {
        g: (kept_counts[g] / pool_counts[g]) if pool_counts[g] else 0.0
        for g in pool_counts.index
    }


def coverage(df, selected_ids, group_col) -> float:
    """Fraction of pool groups with at least one selected example."""
    pool_counts, kept_counts = _counts(df, selected_ids, group_col)
    n_groups = len(pool_counts)
    if n_groups == 0:
        return 0.0
    covered = int((kept_counts > 0).sum())
    return covered / n_groups


def group_entropy(df, selected_ids, group_col) -> tuple:
    """(selected_entropy, pool_entropy) in bits. Larger gap => more concentrated."""
    pool_counts, kept_counts = _counts(df, selected_ids, group_col)
    return _shannon_entropy(kept_counts), _shannon_entropy(pool_counts)


def gini(df, selected_ids, group_col) -> float:
    """Gini coefficient over per-group selected counts (incl. zero-count groups).

    0 = perfectly equal across groups, ->1 = fully concentrated in one group.
    Empty selection => 0.0.
    """
    _, kept_counts = _counts(df, selected_ids, group_col)
    values = sorted(float(v) for v in kept_counts.values)
    n = len(values)
    total = sum(values)
    if n == 0 or total == 0:
        return 0.0
    # G = (2 * sum(i * x_i) / (n * sum(x))) - (n + 1) / n,  i 1-indexed ascending
    weighted = sum((i + 1) * x for i, x in enumerate(values))
    return (2.0 * weighted) / (n * total) - (n + 1) / n


def summarise(df, selected_ids, group_col) -> pd.DataFrame:
    """Tidy per-group summary, sorted by representation_ratio ascending.

    Columns: [group, pool_count, pool_share, kept_count, kept_share,
              retention_rate, representation_ratio]
    """
    pool_counts, kept_counts = _counts(df, selected_ids, group_col)
    total_pool = float(pool_counts.sum())
    total_kept = float(kept_counts.sum())
    rep = representation_ratio(df, selected_ids, group_col)
    ret = retention_rate(df, selected_ids, group_col)

    rows = []
    for g in pool_counts.index:
        rows.append({
            "group": g,
            "pool_count": int(pool_counts[g]),
            "pool_share": (pool_counts[g] / total_pool) if total_pool else 0.0,
            "kept_count": int(kept_counts[g]),
            "kept_share": (kept_counts[g] / total_kept) if total_kept else 0.0,
            "retention_rate": ret[g],
            "representation_ratio": rep[g],
        })
    out = pd.DataFrame(rows, columns=[
        "group", "pool_count", "pool_share", "kept_count", "kept_share",
        "retention_rate", "representation_ratio",
    ])
    return out.sort_values("representation_ratio", ascending=True).reset_index(drop=True)

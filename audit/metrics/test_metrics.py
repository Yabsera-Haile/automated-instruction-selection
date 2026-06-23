"""Tests for audit.metrics.audit_metrics.

Run with:  python -m pytest audit/metrics/test_metrics.py
"""
import numpy as np
import pandas as pd

from audit.metrics import audit_metrics as m


def _make_df(group_sizes: dict) -> pd.DataFrame:
    """Build a metadata-like df with unique ids and a `grp` column."""
    rows = []
    for grp, n in group_sizes.items():
        for i in range(n):
            rows.append({"id": f"{grp}-{i}", "grp": grp})
    return pd.DataFrame(rows)


def test_random_mask_ratios_near_one():
    """5 balanced groups, 100 random picks -> all representation ratios ~1.0."""
    df = _make_df({g: 200 for g in ["A", "B", "C", "D", "E"]})  # 1000 rows
    rng = np.random.default_rng(12345)
    selected_ids = set(rng.choice(df["id"].to_numpy(), size=100, replace=False))

    ratios = m.representation_ratio(df, selected_ids, "grp")
    assert set(ratios) == set("ABCDE")
    for g, r in ratios.items():
        assert abs(r - 1.0) <= 0.3, f"group {g} ratio {r} not within 0.3 of 1.0"


def test_skewed_mask_over_and_under_selection():
    """Over-select A, under-select B -> ratio_A > 1.5 and ratio_B < 0.5."""
    df = _make_df({g: 200 for g in ["A", "B", "C", "D", "E"]})

    def ids(grp, k):
        return [f"{grp}-{i}" for i in range(k)]

    selected_ids = (
        ids("A", 200)          # all of A (heavily over-selected)
        + ids("B", 5)          # almost none of B (under-selected)
        + ids("C", 50) + ids("D", 50) + ids("E", 50)
    )
    ratios = m.representation_ratio(df, selected_ids, "grp")
    assert ratios["A"] > 1.5, ratios
    assert ratios["B"] < 0.5, ratios

    # retention should agree: A fully retained, B barely
    ret = m.retention_rate(df, selected_ids, "grp")
    assert ret["A"] == 1.0
    assert ret["B"] == 5 / 200


def test_edge_case_empty_group():
    """A group with zero selected examples -> retention 0.0, ratio 0.0, no error."""
    df = _make_df({"X": 100, "Y": 100, "Z": 100})
    # select only from X and Y; Z gets nothing
    selected_ids = [f"X-{i}" for i in range(20)] + [f"Y-{i}" for i in range(20)]

    ret = m.retention_rate(df, selected_ids, "grp")
    rep = m.representation_ratio(df, selected_ids, "grp")
    assert ret["Z"] == 0.0
    assert rep["Z"] == 0.0

    # coverage = 2 of 3 groups covered; summarise must not error and Z sorts first.
    assert m.coverage(df, selected_ids, "grp") == 2 / 3
    summary = m.summarise(df, selected_ids, "grp")
    assert list(summary.columns) == [
        "group", "pool_count", "pool_share", "kept_count", "kept_share",
        "retention_rate", "representation_ratio",
    ]
    assert summary.iloc[0]["group"] == "Z"  # lowest representation ratio first
    # entropy + gini also run without error on this selection
    sel_ent, pool_ent = m.group_entropy(df, selected_ids, "grp")
    assert pool_ent >= sel_ent  # pool is 3 equal groups; selection is only 2
    assert 0.0 <= m.gini(df, selected_ids, "grp") <= 1.0

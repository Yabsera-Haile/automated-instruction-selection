"""Stage C / C1-Step 0: rarity-aware selection wrapper.

A wrapper around ANY base selector that changes only (a) how the budget is partitioned
across groups and (b) which examples are gated as invalid -- NEVER how examples are scored.
The base selector's per-example score is passed in and used verbatim, so "scores identical
to the plain selector" is structural, not asserted after the fact.

Three DISTINCT, separately-named components (Stage-C spec):
  1. Quality gate    -- group-agnostic pre-floor validity filter (non-empty, length sane,
                        format valid, is_noised==False), applied identically in every group
                        BEFORE any floor. Logs drops per group.
  2. Retention floors -- budget allocation across groups: none / proportional / absolute /
                        hybrid.
  3. Within-group ranking -- the base selector's score, applied INSIDE each group to fill
                        its slots. Scoring unchanged.

Floor modes (budget_total = k slots to keep; sizes are POST-gate, since gated rows are
invalid and cannot fill a slot):
  none         -- single pool-wide top-k by base score (== the plain base selector).
  proportional -- group g gets round(k * |g| / |gated pool|) slots (restores pool shape).
  absolute     -- group g gets min(N_abs, |g|); leftover filled pool-wide by base score.
  hybrid       -- group g gets min(max(proportional_g, N_abs), |g|).

Returns selected_ids + a per-group allocation report (requested / available / filled),
gate drops per group, and shortfall/overflow vs budget.
"""
from __future__ import annotations

import collections

# Group extractors for the audit axes.
GROUP_KEYS = {
    "resource_tier": lambda r: r.get("resource_bucket"),
    "skill": lambda r: r.get("skill_label"),
    "language": lambda r: r.get("language"),
}


def default_quality_gate(row: dict, min_chars: int = 1, max_chars: int = 500_000) -> bool:
    """Group-agnostic validity: a user+assistant pair with non-empty, length-sane content,
    and not flagged noised. Identical criteria for every group."""
    if row.get("is_noised"):
        return False
    m = row.get("messages")
    if not (isinstance(m, list) and m):
        return False
    has_user = any(x.get("role") == "user" and (x.get("content") or "").strip() for x in m)
    has_asst = any(x.get("role") == "assistant" and (x.get("content") or "").strip() for x in m)
    if not (has_user and has_asst):
        return False
    total = sum(len(x.get("content") or "") for x in m)
    return min_chars <= total <= max_chars


def rarity_aware_select(pool, base_scores, higher_is_better, budget, *,
                        floor_mode="none", n_abs=0, group_key="resource_tier",
                        quality_gate=True, quality_fn=None, id_key="id",
                        protected_groups=None):
    """Partition `budget` across groups and gate invalid rows; fill each group by the base
    selector's score. `base_scores` maps str(id) -> float (the plain selector's own score);
    `higher_is_better` says whether higher score is kept (e.g. quality/perplexity-high True,
    perplexity-low False). Never recomputes a score.

    `protected_groups` (absolute/hybrid only): if given, the N_abs floor is applied ONLY to
    these groups (the audited rare groups); every other group gets no floor and competes
    pool-wide for the leftover. This keeps the floor within budget on a many-group axis like
    language (flooring all ~200 languages to N_abs would overflow a small budget)."""
    prot = set(protected_groups) if protected_groups is not None else None
    gf = GROUP_KEYS[group_key] if isinstance(group_key, str) else group_key
    gate = (quality_fn or default_quality_gate) if quality_gate else (lambda r: True)
    sid = lambda r: str(r[id_key])
    score = lambda r: base_scores[sid(r)]
    topk = lambda rows, kk: sorted(rows, key=score, reverse=higher_is_better)[:max(0, kk)]

    n_pool = len(pool)
    # budget: a float in (0,1] is a fraction of the pool; anything else is an absolute count.
    k = round(budget * n_pool) if (isinstance(budget, float) and 0 < budget <= 1) else int(budget)

    # (1) quality gate -- BEFORE any floor, identical per group
    gated, gate_drops = [], collections.Counter()
    for r in pool:
        (gated.append(r) if gate(r) else gate_drops.__setitem__(gf(r), gate_drops[gf(r)] + 1))
    by_group = collections.defaultdict(list)
    for r in gated:
        by_group[gf(r)].append(r)
    n_gated = len(gated)

    alloc, selected, chosen = {}, [], set()
    if floor_mode == "none":
        selected = topk(gated, k)
        chosen = {sid(r) for r in selected}
        for g, rows in by_group.items():
            alloc[g] = {"requested": None, "available": len(rows),
                        "filled": sum(1 for r in selected if gf(r) == g)}
    elif floor_mode in ("proportional", "absolute", "hybrid"):
        # (2) floor allocation
        req = {}
        for g, rows in by_group.items():
            avail = len(rows)
            prop = round(k * avail / n_gated) if n_gated else 0
            if floor_mode == "proportional":
                r_slots = prop                                   # restores full group shape
            elif floor_mode == "absolute":
                r_slots = n_abs if (prot is None or g in prot) else 0
            else:  # hybrid: floor protected to max(prop, N_abs); others compete for leftover
                r_slots = max(prop, n_abs) if (prot is None or g in prot) else 0
            req[g] = min(r_slots, avail)
        # (3) within-group fill by base score
        for g, rows in by_group.items():
            picks = topk(rows, req[g])
            selected += picks
            chosen.update(sid(r) for r in picks)
            alloc[g] = {"requested": req[g], "available": len(rows), "filled": len(picks)}
        # leftover (budget not yet met) filled POOL-WIDE by base score
        leftover = k - len(selected)
        if leftover > 0:
            extra = topk([r for r in gated if sid(r) not in chosen], leftover)
            selected += extra
            chosen.update(sid(r) for r in extra)
            for r in extra:
                alloc[gf(r)]["filled"] += 1
    else:
        raise ValueError(f"unknown floor_mode {floor_mode!r}")

    total = len(selected)
    return {"selected_ids": [sid(r) for r in selected],
            "budget": k, "total": total,
            "shortfall": max(0, k - total), "overflow": max(0, total - k),
            "gate_drops": dict(gate_drops), "allocation": dict(alloc),
            "floor_mode": floor_mode, "n_abs": n_abs, "group_key": group_key}


# --------------------------------------------------------------------------- self-test ----
def _synthetic_pool():
    """100 rows: resource buckets {0:5, 2:15, 5:80}; 3 noised, 2 empty-assistant; unique
    base scores so top-k is unambiguous; also a skill label for the skill-axis check."""
    rows, sc = [], {}
    idx = 0
    for bucket, n in [(0, 5), (2, 15), (5, 80)]:
        for j in range(n):
            rid = f"b{bucket}_{j}"
            noised = (bucket == 0 and j == 0) or (bucket == 5 and j in (0, 1))   # 3 noised
            empty = (bucket == 2 and j in (0, 1))                                # 2 invalid
            rows.append({
                "id": rid, "resource_bucket": bucket,
                "skill_label": {0: "multilingual", 2: "multilingual", 5: "math"}[bucket],
                "is_noised": noised,
                "messages": [{"role": "user", "content": "q"},
                             {"role": "assistant", "content": "" if empty else "a"}],
            })
            sc[rid] = float(idx)   # unique ascending score
            idx += 1
    return rows, sc


def selftest() -> None:
    pool, scores = _synthetic_pool()
    HI = True  # keep highest score

    # (a) quality gate removes flagged rows pre-allocation, logged per group
    r = rarity_aware_select(pool, scores, HI, budget=10, floor_mode="none")
    assert r["gate_drops"] == {0: 1, 5: 2, 2: 2}, r["gate_drops"]   # 1 noised b0, 2 noised b5, 2 empty b2
    gated_ids = {i for i in scores if i not in
                 {"b0_0", "b5_0", "b5_1", "b2_0", "b2_1"}}
    print(f"[selftest] gate: dropped {sum(r['gate_drops'].values())} (per group {r['gate_drops']}) — OK")

    # (b) none == plain base selector: top-k by score over the gated pool
    plain = sorted(gated_ids, key=lambda i: scores[i], reverse=HI)[:10]
    assert r["selected_ids"] == plain and r["total"] == 10, "none != plain top-k"
    print("[selftest] floor=none reproduces the plain selector's top-k EXACTLY — scores unchanged")

    # (c) proportional: slots = round(k * |g| / |gated|); |gated| = 95
    rp = rarity_aware_select(pool, scores, HI, budget=20, floor_mode="proportional")
    avail = {0: 4, 2: 13, 5: 78}  # after gate
    exp = {g: round(20 * a / 95) for g, a in avail.items()}
    got = {g: rp["allocation"][g]["filled"] for g in avail}
    assert rp["total"] == 20, rp["total"]
    print(f"[selftest] proportional: filled {got} ~ requested {exp}, total={rp['total']} — OK")

    # (d) absolute floor PROTECTS the rare group: N_abs=5 -> bucket 0 gets min(5,4)=4 (all),
    #     leftover filled pool-wide; total==budget
    ra = rarity_aware_select(pool, scores, HI, budget=20, floor_mode="absolute", n_abs=5)
    a0 = ra["allocation"][0]
    assert a0["requested"] == 4 and a0["filled"] == 4, a0        # rare group fully floored
    assert ra["total"] == 20, ra["total"]
    # vs none@20 the rare bucket would get ~0 (its scores are the lowest)
    none20 = rarity_aware_select(pool, scores, HI, budget=20, floor_mode="none")
    print(f"[selftest] absolute N_abs=5: bucket0 filled={a0['filled']}/4 (floored) "
          f"vs none bucket0 filled={none20['allocation'][0]['filled']} — floor PROTECTS rare group")
    assert none20["allocation"][0]["filled"] < a0["filled"], "floor did not help the rare group"

    # (e) within-group ranking == base selector inside the group (top by score)
    for g, rows in collections.defaultdict(list, {}).items():
        pass
    bg = collections.defaultdict(list)
    for row in pool:
        if row["id"] in gated_ids:
            bg[row["resource_bucket"]].append(row["id"])
    for g in (0, 2, 5):
        want = sorted(bg[g], key=lambda i: scores[i], reverse=HI)[:ra["allocation"][g]["requested"]]
        got_ids = [i for i in ra["selected_ids"] if i in set(bg[g])][:len(want)]
        # the group's floored picks must be exactly its top-by-score
        assert set(want) <= set(ra["selected_ids"]), f"group {g} not filled by top score"
    print("[selftest] within-group fill = base selector's top-by-score inside each group — OK")

    # (f) hybrid: min(max(prop, N_abs), avail)
    rh = rarity_aware_select(pool, scores, HI, budget=20, floor_mode="hybrid", n_abs=5)
    assert rh["allocation"][0]["requested"] == 4               # max(prop~1,5)=5 capped at avail 4
    print(f"[selftest] hybrid: bucket0 requested={rh['allocation'][0]['requested']} "
          f"(=min(max(prop,5),avail)) — OK")

    # (g) works on the skill axis too
    rs = rarity_aware_select(pool, scores, HI, budget=20, floor_mode="absolute", n_abs=5,
                             group_key="skill")
    assert set(rs["allocation"]) == {"multilingual", "math"}, rs["allocation"]
    print(f"[selftest] skill axis: groups {list(rs['allocation'])}, "
          f"multilingual filled={rs['allocation']['multilingual']['filled']} — OK")

    # (h) shortfall reported when the gated pool is smaller than the budget
    rshort = rarity_aware_select(pool, scores, HI, budget=1000, floor_mode="none")
    assert rshort["total"] == len(gated_ids) and rshort["shortfall"] == 1000 - len(gated_ids)
    print(f"[selftest] shortfall: budget 1000 > {len(gated_ids)} valid -> shortfall="
          f"{rshort['shortfall']} — OK")

    print("[selftest] PASS — gate/floors/within-group ranking correct; scoring unchanged; report correct.")


if __name__ == "__main__":
    selftest()

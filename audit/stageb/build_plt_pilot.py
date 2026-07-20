"""R4-Step 3: build the plt (Malagasy) density-pilot pools.

Isolates the effect of ONE language's density on its own learning. A fixed base pool
(the Tulu-3 pilot with ALL plt removed) is held constant; four pools add a NESTED dose of
{0, 250, 1000, 4000} native-plt examples drawn from MURI-IT (config 'mlg', GlotLID-filtered
to plt, verified-clean). 250 subset 1000 subset 4000 (prefixes), so the dose is nested.

Two modes so the ~50MB pools never need transferring:
  build       [LOCAL]  : remove plt from the pilot pool, GlotLID-recheck 0 stray plt, stream
                         MURI 'mlg' -> 4000 native-plt rows -> write the small dose file
                         (plt_pilot_dose.jsonl) + the 4 pools + manifest, and self-verify.
  materialize [SERVER] : rebuild the identical 4 pools from the committed dose file + the
                         pilot pool (no MURI/GlotLID needed).

    python -m audit.stageb.build_plt_pilot --mode build         # LOCAL
    python -m audit.stageb.build_plt_pilot --mode materialize    # SERVER (before R4-Step 4)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

os.environ.setdefault("USE_TORCH", "0")  # streaming + GlotLID only

PILOT_LANG = "plt"
DOSES = [0, 250, 1000, 4000]
PLT_BUCKET = 2  # plt's Joshi resource bucket in the pilot metadata


def mint_muri_id(inp: str, out: str) -> str:
    h = hashlib.sha256((inp + "\n\n" + out).encode("utf-8")).hexdigest()[:24]
    return f"muri:{h}"


def read_jsonl(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def make_base(pool: list[dict]) -> tuple[list[dict], int]:
    base = [r for r in pool if r.get("language") != PILOT_LANG]
    return base, len(pool) - len(base)


def materialize_pools(base: list[dict], dose: list[dict], out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    pools = {}
    for d in DOSES:
        rows = base + dose[:d]
        path = os.path.join(out_dir, f"plt_pilot_d{d}.jsonl")
        write_jsonl(path, rows)
        pools[f"d{d}"] = {"plt": d, "total": len(rows), "path": path}
        print(f"  plt_pilot_d{d}: {len(rows)} rows ({len(base)} base + {d} plt)")
    return pools


def collect_muri_plt(muri_dataset, config, n, max_scan, base_ids) -> list[dict]:
    from datasets import load_dataset
    from audit.metadata.langid import LanguageIdentifier
    lid = LanguageIdentifier(backend="glotlid")
    ds = load_dataset(muri_dataset, config, split="train", streaming=True)
    rows, seen, scanned = [], set(), 0
    for ex in ds:
        if len(rows) >= n or scanned >= max_scan:
            break
        scanned += 1
        inp = (ex.get("input") or "").strip()
        out = (ex.get("output") or "").strip()
        if not inp or not out:
            continue
        if lid.predict(inp)[0] != PILOT_LANG:      # keep only GlotLID-native plt
            continue
        mid = mint_muri_id(inp, out)
        if mid in seen or mid in base_ids:
            continue
        seen.add(mid)
        rows.append({
            "messages": [{"role": "user", "content": inp},
                         {"role": "assistant", "content": out}],
            "pool_row_idx": 10_000_000 + len(rows), "id": mid, "source": "muri",
            "skill_label": "multilingual", "language": PILOT_LANG,
            "resource_bucket": PLT_BUCKET, "quality_score": 1.0,
            "is_clean": True, "is_noised": False,
        })
    print(f"scanned {scanned} MURI '{config}' -> {len(rows)} native-plt (GlotLID)")
    return rows


def verify(base, dose, out_dir, glotlid_recheck=False):
    ok = True
    # 0 stray plt in base (by language field; optional fresh GlotLID pass)
    stray_field = sum(1 for r in base if r.get("language") == PILOT_LANG)
    print(f"stray plt in base (language field): {stray_field}")
    ok = ok and stray_field == 0
    if glotlid_recheck:
        from audit.metadata.langid import LanguageIdentifier
        lid = LanguageIdentifier(backend="glotlid")
        stray = 0
        for r in base:
            txt = next((m["content"] for m in r["messages"] if m["role"] == "user"), "")
            if txt and lid.predict(txt)[0] == PILOT_LANG:
                stray += 1
        print(f"stray plt in base (fresh GlotLID re-check of {len(base)} rows): {stray}")
        ok = ok and stray == 0
    # nested doses
    ids = [r["id"] for r in dose]
    nested = (ids[:250] == ids[:1000][:250] and ids[:1000] == ids[:4000][:1000]
              and len(set(ids[:4000])) == 4000)
    print(f"doses nested (250 subset 1000 subset 4000) + unique: {nested}")
    ok = ok and nested and len(dose) >= 4000
    # pools present with right plt counts + identical base
    for d in DOSES:
        rows = read_jsonl(os.path.join(out_dir, f"plt_pilot_d{d}.jsonl"))
        n_plt = sum(1 for r in rows if r.get("language") == PILOT_LANG)
        same_base = rows[:len(base)] == base
        print(f"  plt_pilot_d{d}: total={len(rows)} plt={n_plt} base_identical={same_base}")
        ok = ok and n_plt == d and same_base
    print("ACCEPTANCE:", "PASS" if ok else "FAIL")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="Build/materialize the plt density-pilot pools.")
    ap.add_argument("--mode", choices=["build", "materialize"], default="build")
    ap.add_argument("--pool", default="audit/stageb/data/pilot_pool.jsonl")
    ap.add_argument("--muri_dataset", default="akoksal/muri")
    ap.add_argument("--muri_config", default="mlg")
    ap.add_argument("--max_scan", type=int, default=40000)
    ap.add_argument("--out_dir", default="audit/results/stageb/gemma-3-4b-pt/round4/pools")
    ap.add_argument("--no_glotlid_recheck", action="store_true")
    args = ap.parse_args()

    pool = read_jsonl(args.pool)
    base, removed = make_base(pool)
    base_ids = {str(r.get("id")) for r in base}
    print(f"pool {len(pool)} -> base {len(base)} (removed {removed} plt)")
    os.makedirs(args.out_dir, exist_ok=True)
    dose_path = os.path.join(args.out_dir, "plt_pilot_dose.jsonl")

    if args.mode == "build":
        dose = collect_muri_plt(args.muri_dataset, args.muri_config, max(DOSES),
                                args.max_scan, base_ids)
        if len(dose) < max(DOSES):
            raise SystemExit(f"only {len(dose)} native-plt found (< {max(DOSES)}).")
        write_jsonl(dose_path, dose)
        print(f"wrote dose file: {dose_path} ({len(dose)} plt)")
    else:  # materialize (server): rebuild pools from the committed dose file
        if not os.path.exists(dose_path):
            raise SystemExit(f"missing dose file {dose_path} (commit it from the build step).")
        dose = read_jsonl(dose_path)

    pools = materialize_pools(base, dose, args.out_dir)
    ok = verify(base, dose, args.out_dir,
                glotlid_recheck=(args.mode == "build" and not args.no_glotlid_recheck))
    manifest = {"pilot_language": PILOT_LANG, "muri_config": args.muri_config,
                "base_size": len(base), "removed_plt": removed, "doses": DOSES,
                "dose_file": dose_path, "n_dose": len(dose), "pools": pools,
                "acceptance": "PASS" if ok else "FAIL"}
    json.dump(manifest, open(os.path.join(args.out_dir, "plt_pilot_manifest.json"), "w"),
              indent=2)
    if not ok:
        raise SystemExit("verification FAILED")


if __name__ == "__main__":
    main()

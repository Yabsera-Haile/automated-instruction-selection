"""R4-Step 8: rebuild the pool with CONCENTRATED per-language density.

Fixed Tulu-3 base (the pilot pool with ALL decisive languages removed) + exactly N native
MURI-IT examples for each decisive language, so every decisive language has per-language
count == N. Draws are GlotLID-verified native text, deduped, taken in the deterministic
MURI stream order (so smaller doses are nested prefixes of larger ones).

Deterministic by construction: the manifest records a sha256 over each language's ordered
id list, so the server can rebuild the identical pool (--mode build) and verify the
checksums match rather than transferring the ~40MB of injected rows.

    python -m audit.stageb.build_concentrated_pool                # LOCAL build + verify
    python -m audit.stageb.build_concentrated_pool --verify_against <manifest.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

os.environ.setdefault("USE_TORCH", "0")

CONFIG = "audit/configs/round4_decisive.json"


def read_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def mint_muri_id(inp: str, out: str) -> str:
    h = hashlib.sha256((inp + "\n\n" + out).encode("utf-8")).hexdigest()[:24]
    return f"muri:{h}"


def collect(iso, config, n, max_scan, base_ids, lid, muri_dataset, bucket):
    from datasets import load_dataset
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
        if lid.predict(inp)[0] != iso:          # native-only (GlotLID)
            continue
        mid = mint_muri_id(inp, out)
        if mid in seen or mid in base_ids:
            continue
        seen.add(mid)
        rows.append({
            "messages": [{"role": "user", "content": inp},
                         {"role": "assistant", "content": out}],
            "id": mid, "source": "muri", "skill_label": "multilingual",
            "language": iso, "resource_bucket": bucket, "quality_score": 1.0,
            "is_clean": True, "is_noised": False,
        })
    print(f"  {iso:5} config={config:5} scanned={scanned:6} -> {len(rows)} native rows")
    return rows


def emit_metadata(pool_path: str, out_dir: str) -> str:
    """Metadata parquet for run_audit, taken straight from the pool rows (which already carry
    GlotLID language, Joshi bucket and skill), so it is exactly consistent with the pool."""
    import pandas as pd
    rows = read_jsonl(pool_path)
    cols = ["pool_row_idx", "id", "source", "skill_label", "language", "resource_bucket",
            "quality_score", "is_clean", "is_noised"]
    df = pd.DataFrame([{c: r.get(c) for c in cols} for r in rows])
    df["n_messages"] = [len(r.get("messages", [])) for r in rows]
    path = os.path.join(out_dir, "concentrated_metadata.parquet")
    df.to_parquet(path, index=False)
    print(f"metadata -> {path}  ({len(df)} rows, cols={list(df.columns)})")
    print("  resource_bucket:", df["resource_bucket"].value_counts().sort_index().to_dict())
    print("  skill_label (top6):", df["skill_label"].value_counts().head(6).to_dict())
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the concentrated Round-4 pool.")
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--pool", default="audit/stageb/data/pilot_pool.jsonl")
    ap.add_argument("--muri_dataset", default="akoksal/muri")
    ap.add_argument("--max_scan", type=int, default=60000)
    ap.add_argument("--out_dir", default="audit/results/stageb/gemma-3-4b-pt/round4/pools")
    ap.add_argument("--verify_against", default=None, help="Manifest to compare checksums to.")
    ap.add_argument("--metadata_only", action="store_true",
                    help="Emit the metadata parquet from an existing concentrated pool "
                         "(no MURI re-streaming).")
    args = ap.parse_args()

    if args.metadata_only:
        emit_metadata(os.path.join(args.out_dir, "concentrated_pool.jsonl"), args.out_dir)
        return

    cfg = json.load(open(args.config, encoding="utf-8"))
    langs, N = cfg["decisive_languages"], cfg["injection_size_N"]
    muri_cfg = cfg["muri_configs"]

    from audit.metadata.build_metadata import ResourceMapper
    from audit.metadata.langid import LanguageIdentifier
    rm = ResourceMapper("audit/configs/joshi_resource.json")
    lid = LanguageIdentifier(backend="glotlid")

    pool = read_jsonl(args.pool)
    base = [r for r in pool if r.get("language") not in set(langs)]
    base_ids = {str(r.get("id")) for r in base}
    removed = len(pool) - len(base)
    print(f"pool {len(pool)} -> base {len(base)} (removed {removed} rows of decisive langs)")

    injected, per_lang, checksums = [], {}, {}
    for iso in langs:
        rows = collect(iso, muri_cfg[iso], N, args.max_scan, base_ids, lid,
                       args.muri_dataset, rm.bucket(iso))
        if len(rows) < N:
            raise SystemExit(f"{iso}: only {len(rows)} native rows (< N={N})")
        injected += rows
        per_lang[iso] = len(rows)
        checksums[iso] = hashlib.sha256(
            "\n".join(r["id"] for r in rows).encode("utf-8")).hexdigest()[:16]

    for i, r in enumerate(injected):          # unique pool_row_idx for the injected block
        r["pool_row_idx"] = 10_000_000 + i
    concentrated = base + injected
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "concentrated_pool.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in concentrated:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- composition + verification ----
    import collections
    lang_counts = collections.Counter(r.get("language") for r in concentrated)
    bucket_counts = collections.Counter(r.get("resource_bucket") for r in concentrated)
    skill_counts = collections.Counter(r.get("skill_label") for r in concentrated)
    ok = all(lang_counts[i] == N for i in langs)
    stray = sum(1 for r in base if r.get("language") in set(langs))
    print(f"\nconcentrated pool: {len(concentrated)} rows "
          f"({len(base)} base + {len(injected)} injected)")
    print("per-language counts (decisive):",
          {i: lang_counts[i] for i in langs}, "-> all == N:", ok)
    print("stray decisive-language rows in base:", stray)
    print("resource_bucket composition:", dict(sorted(bucket_counts.items(), key=lambda x: str(x[0]))))
    print("skill composition (top 6):", dict(skill_counts.most_common(6)))

    manifest = {"round": 4, "N": N, "decisive_languages": langs,
                "base_size": len(base), "removed_from_base": removed,
                "injected": len(injected), "pool_size": len(concentrated),
                "per_language_counts": {i: lang_counts[i] for i in langs},
                "id_checksums": checksums,
                "resource_bucket_composition": {str(k): v for k, v in bucket_counts.items()},
                "skill_composition": dict(skill_counts),
                "pool_path": out_path,
                "acceptance": "PASS" if (ok and stray == 0) else "FAIL"}
    mpath = os.path.join(args.out_dir, "concentrated_pool_manifest.json")
    json.dump(manifest, open(mpath, "w", encoding="utf-8"), indent=2)
    print(f"\nmanifest -> {mpath}")
    emit_metadata(out_path, args.out_dir)   # pool + metadata from one command

    if args.verify_against:
        ref = json.load(open(args.verify_against, encoding="utf-8"))
        same = ref.get("id_checksums") == checksums
        print("checksum match vs reference manifest:", same)
        if not same:
            raise SystemExit("Injected rows differ from the reference build!")
    print("ACCEPTANCE:", "PASS" if (ok and stray == 0) else "FAIL")
    if not (ok and stray == 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

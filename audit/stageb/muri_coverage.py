"""R4-Step 7: MURI-IT coverage per candidate decisive language.

For each candidate eval language (GlotLID ISO), resolve its MURI-IT config (MURI uses
ISO 639-3, sometimes the MACROLANGUAGE -- e.g. GlotLID 'plt' lives in MURI config 'mlg'),
report the raw example count from dataset metadata (fast, no full download), sample K
examples to measure the GlotLID yield to the target ISO, and estimate usable examples.

A language qualifies for the decisive set only if usable >= N (the injection size).

    python -m audit.stageb.muri_coverage --n 2000 --langs ceb hau kir mlt plt som zul bod tsn wol
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("USE_TORCH", "0")

# GlotLID ISO -> candidate MURI config names, in priority order (individual, then macro).
CONFIG_CANDIDATES = {
    "plt": ["plt", "mlg"],   # Plateau Malagasy lives under the Malagasy macrolanguage
    "ceb": ["ceb"], "hau": ["hau"], "kir": ["kir"], "mlt": ["mlt"], "som": ["som"],
    "zul": ["zul"], "bod": ["bod", "tib"], "tsn": ["tsn"], "wol": ["wol"],
    "yor": ["yor"], "fuv": ["fuv", "ful"],
}


def resolve_config(iso: str, configs: set[str]) -> str | None:
    for c in CONFIG_CANDIDATES.get(iso, [iso]):
        if c in configs:
            return c
    return None


def raw_count(dataset: str, config: str) -> int | None:
    """Example count from dataset metadata (no data download)."""
    try:
        from datasets import load_dataset_builder
        info = load_dataset_builder(dataset, config).info
        split = info.splits.get("train") if info.splits else None
        return int(split.num_examples) if split is not None else None
    except Exception:
        return None


def glotlid_yield(dataset: str, config: str, iso: str, k: int, lid) -> tuple[float, int]:
    """Fraction of the first k non-empty examples whose GlotLID label == iso."""
    from datasets import load_dataset
    ds = load_dataset(dataset, config, split="train", streaming=True)
    seen = hits = 0
    for ex in ds:
        if seen >= k:
            break
        inp = (ex.get("input") or "").strip()
        out = (ex.get("output") or "").strip()
        if not inp or not out:
            continue
        seen += 1
        if lid.predict(inp)[0] == iso:
            hits += 1
    return (hits / seen if seen else 0.0), seen


def main() -> None:
    ap = argparse.ArgumentParser(description="MURI-IT coverage per candidate language.")
    ap.add_argument("--dataset", default="akoksal/muri")
    ap.add_argument("--langs", nargs="*",
                    default=["ceb", "hau", "kir", "mlt", "plt", "som", "zul",
                             "bod", "tsn", "wol"])
    ap.add_argument("--n", type=int, default=2000, help="Required injection size N.")
    ap.add_argument("--sample", type=int, default=200, help="Examples sampled for GlotLID yield.")
    args = ap.parse_args()

    from datasets import get_dataset_config_names
    from audit.metadata.langid import LanguageIdentifier
    configs = set(get_dataset_config_names(args.dataset))
    lid = LanguageIdentifier(backend="glotlid")

    print(f"MURI-IT ({args.dataset}) coverage — required N = {args.n}\n")
    print(f"{'iso':5} {'config':8} {'raw':>8} {'yield':>7} {'usable~':>9}  verdict")
    print("-" * 58)
    rows = []
    for iso in args.langs:
        cfg = resolve_config(iso, configs)
        if cfg is None:
            print(f"{iso:5} {'--':8} {'-':>8} {'-':>7} {'-':>9}  NO CONFIG -> DROP")
            rows.append((iso, None, 0, 0.0, 0, False))
            continue
        raw = raw_count(args.dataset, cfg)
        y, seen = glotlid_yield(args.dataset, cfg, iso, args.sample, lid)
        usable = int((raw or 0) * y)
        ok = usable >= args.n
        print(f"{iso:5} {cfg:8} {str(raw):>8} {y:7.2%} {usable:>9}  "
              f"{'OK' if ok else 'BELOW N -> DROP'}")
        rows.append((iso, cfg, raw or 0, y, usable, ok))

    keep = [r[0] for r in rows if r[5]]
    drop = [r[0] for r in rows if not r[5]]
    print("\nQUALIFY (usable >= N):", keep or "NONE")
    print("DROP (insufficient MURI):", drop or "NONE")
    print(f"\nDecisive set size: {len(keep)}")


if __name__ == "__main__":
    main()

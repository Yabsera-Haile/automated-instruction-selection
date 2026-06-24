"""SemDeDup selection adapter (C-Step 3).

Reuses the RDS+ pool embeddings already on disk (the .pt index) — does NOT recompute
embeddings. Implements SemDeDup (Abbas et al. 2023):
  1. k-means cluster the (L2-normalized) pool embeddings.
  2. Within each cluster, find near-duplicate pairs by cosine similarity >= threshold.
     Process members closest-to-centroid first and greedily keep; an example within
     `threshold` cosine of an already-kept cluster member is marked removable. The kept
     representative of each near-duplicate group is thus the one CLOSEST to the centroid
     (per the spec).
  3. near-duplicate rate = (# removable) / N.

Index source (reuse, never recompute):
  * real : audit/results/rds_work/cosine_train_reps.pt  (Llama-2-7B weighted-mean pool)
  * dev  : audit/results/dev/rds_work/pool_index.pt      (all-MiniLM-L6-v2)
Row i of the index == pool_row_idx i == metadata id.

Output modes:
  * Natural : keep all non-removable examples (one dedup pass at the fixed threshold).
              Records the natural fraction. -> semdedup__bnatural__s0.json
  * Budget  : after dedup, rank by distance-to-centroid and trim/pad to each budget.
              Ranking = "keep most diverse": FARTHEST-from-centroid retained first.
              If a budget exceeds the dedup size, pad with the removed near-dups
              (also farthest-first). -> semdedup__b{budget}__s0.json

Deterministic (k-means seeded) => seed fixed at 0.
"""
from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd

from audit.common import announce_dev, results_base

logger = logging.getLogger("audit.run_semdedup")

BUDGETS = [0.01, 0.05, 0.10, 0.25, 0.50]
DEFAULT_THRESHOLD = 0.95   # cosine-sim near-duplicate threshold (documented)
DEFAULT_N_CLUSTERS = 100


def load_index(path: str) -> np.ndarray:
    import torch
    emb = torch.load(path, map_location="cpu", weights_only=False)  # our own trusted .pt
    if hasattr(emb, "numpy"):
        emb = emb.float().numpy()
    emb = np.asarray(emb, dtype=np.float32)
    if emb.ndim != 2:
        raise ValueError(f"Expected 2D embedding index, got shape {emb.shape}")
    return emb


def _l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def semdedup(emb: np.ndarray, n_clusters: int, threshold: float, seed: int):
    """Return (removable: bool[N], dist_to_centroid: float[N]).

    dist_to_centroid = 1 - cosine(example, its cluster centroid). Larger = more diverse.
    """
    from sklearn.cluster import KMeans

    X = _l2norm(emb)
    n = len(X)
    k = max(1, min(n_clusters, n))
    logger.info("KMeans: N=%d, dim=%d, clusters=%d ...", n, X.shape[1], k)
    km = KMeans(n_clusters=k, random_state=seed, n_init=3).fit(X)
    labels = km.labels_
    centroids = _l2norm(km.cluster_centers_)

    removable = np.zeros(n, dtype=bool)
    dist_to_centroid = np.ones(n, dtype=np.float32)
    for c in range(k):
        idxs = np.where(labels == c)[0]
        if len(idxs) == 0:
            continue
        Xc = X[idxs]
        sim_to_centroid = Xc @ centroids[c]
        dist_to_centroid[idxs] = 1.0 - sim_to_centroid
        order = idxs[np.argsort(-sim_to_centroid)]  # closest-to-centroid first
        kept_vecs = []  # representatives kept so far in this cluster
        for gi in order:
            v = X[gi]
            if kept_vecs and (np.stack(kept_vecs) @ v).max() >= threshold:
                removable[gi] = True       # near-dup of an already-kept representative
            else:
                kept_vecs.append(v)
    return removable, dist_to_centroid


def _rank_diverse(indices: np.ndarray, dist_to_centroid: np.ndarray) -> np.ndarray:
    """Most-diverse-first = farthest-from-centroid first (descending distance)."""
    return indices[np.argsort(-dist_to_centroid[indices])]


def generate(metadata: str, index_path: str, out_dir: str, dev: bool,
             threshold: float = DEFAULT_THRESHOLD, n_clusters: int = DEFAULT_N_CLUSTERS,
             seed: int = 0, budgets=BUDGETS) -> list[str]:
    meta_df = pd.read_parquet(metadata).sort_values("pool_row_idx").reset_index(drop=True)
    n_pool = len(meta_df)
    idx2id = {int(r): str(i) for r, i in zip(meta_df["pool_row_idx"], meta_df["id"])}

    emb = load_index(index_path)
    if len(emb) != n_pool:
        raise ValueError(f"Index rows ({len(emb)}) != metadata rows ({n_pool}); "
                         "the .pt must be built over the same pool.")
    os.makedirs(out_dir, exist_ok=True)

    removable, dist = semdedup(emb, n_clusters, threshold, seed)
    kept_idx = np.where(~removable)[0]
    removed_idx = np.where(removable)[0]
    near_dup_rate = removable.mean()
    natural_fraction = len(kept_idx) / n_pool
    logger.info("Near-duplicate rate (threshold=%.3f, clusters=%d): %.3f%% (%d/%d removable)",
                threshold, n_clusters, 100 * near_dup_rate, len(removed_idx), n_pool)
    logger.info("Natural dedup keeps %d / %d (%.1f%%)", len(kept_idx), n_pool, 100 * natural_fraction)

    base_meta = {
        "selector_kind": "semdedup", "threshold": threshold, "n_clusters": n_clusters,
        "seed": seed, "dev": dev, "index_path": index_path,
        "near_dup_rate": float(near_dup_rate), "ranking": "diverse(farthest_from_centroid)",
        "index_model": "minilm" if dev else "cosinesim_7b",
    }

    def write(path, budget, chosen_idx, mode):
        ids = [idx2id[int(i)] for i in chosen_idx if int(i) in idx2id]
        obj = {"selector": "semdedup", "budget": budget, "seed": seed,
               "selected_ids": ids, "meta": {**base_meta, "mode": mode}}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f)
        logger.info("semdedup %s (budget=%s) -> %d ids -> %s", mode, budget, len(ids),
                    os.path.basename(path))
        return path

    written = []
    # natural mode: budget recorded as the natural fraction (its point on the x-axis)
    natural_ranked = _rank_diverse(kept_idx, dist)  # ordered, diverse-first (stable output)
    written.append(write(os.path.join(out_dir, "semdedup__bnatural__s0.json"),
                         round(natural_fraction, 4), natural_ranked, "natural"))

    # budget-targeted mode: diverse-first trim/pad
    kept_ranked = _rank_diverse(kept_idx, dist)
    removed_ranked = _rank_diverse(removed_idx, dist)
    for budget in budgets:
        k = round(budget * n_pool)
        if k <= len(kept_ranked):
            chosen = kept_ranked[:k]                                  # trim
        else:
            pad = removed_ranked[:k - len(kept_ranked)]               # pad with removed
            chosen = np.concatenate([kept_ranked, pad])
            logger.info("budget %.2f (%d) exceeds dedup size %d; padded with %d removed.",
                        budget, k, len(kept_ranked), len(pad))
        written.append(write(os.path.join(out_dir, f"semdedup__b{budget}__s0.json"),
                             budget, chosen, "budget"))
    return written


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="SemDeDup selector (reuses RDS+ .pt index).")
    ap.add_argument("--metadata", default="audit/results/metadata_pilot.parquet")
    ap.add_argument("--index", default=None, help="Path to the RDS+ .pt embedding index.")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--n_clusters", type=int, default=DEFAULT_N_CLUSTERS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dev", action="store_true")
    args = ap.parse_args()

    announce_dev(args.dev, logger)
    base = results_base(args.dev)
    out_dir = args.out_dir or os.path.join(base, "selections")
    index_path = args.index or (
        os.path.join(base, "rds_work", "pool_index.pt") if args.dev
        else os.path.join(base, "rds_work", "cosine_train_reps.pt"))
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"RDS+ embedding index not found: {index_path}. Run RDS+ first "
            "(python -m audit.selectors.run_rdsplus" + (" --dev" if args.dev else "") + ").")

    paths = generate(args.metadata, index_path, out_dir, args.dev,
                     threshold=args.threshold, n_clusters=args.n_clusters, seed=args.seed)
    logger.info("Wrote %d SemDeDup selection files to %s", len(paths), out_dir)


if __name__ == "__main__":
    main()

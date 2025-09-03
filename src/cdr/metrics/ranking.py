import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm


# -----------------------------
# Basic metric helpers (top-K)
# -----------------------------
def precision_at_k(labels: np.ndarray, order: np.ndarray, k: int) -> float:
    """
    labels: 1D array of 0/1 indicating relevance for each candidate
    order : indices of candidates sorted by descending score
    k     : cutoff
    """
    k = min(k, len(order))
    if k <= 0:
        return 0.0
    hits = labels[order[:k]].sum()
    return float(hits) / float(k)


def recall_at_k(labels: np.ndarray, order: np.ndarray, k: int) -> float:
    rel = int(labels.sum())
    if rel == 0:
        return 0.0
    k = min(k, len(order))
    hits = labels[order[:k]].sum()
    return float(hits) / float(rel)


def f1_at_k(prec: float, rec: float) -> float:
    return 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)


def average_precision_at_k(labels: np.ndarray, order: np.ndarray, k: int) -> float:
    """
    AP@K for binary labels.
    """
    k = min(k, len(order))
    hits, cum = 0, 0.0
    for rank, idx in enumerate(order[:k], start=1):
        if labels[idx] > 0:
            hits += 1
            cum += hits / rank
    if hits == 0:
        return 0.0
    return float(cum) / float(hits)


def ndcg_at_k(labels: np.ndarray, order: np.ndarray, k: int) -> float:
    """
    Binary-relevance NDCG@K.
    """
    k = min(k, len(order))
    # DCG
    dcg = 0.0
    for rank, idx in enumerate(order[:k], start=1):
        rel = 1.0 if labels[idx] > 0 else 0.0
        if rel:
            dcg += rel / math.log2(rank + 1)
    # IDCG (best possible DCG)
    ideal_hits = int(min(k, labels.sum()))
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return float(dcg / idcg)


# ---------------------------------------------------------
# Leave-one-out style evaluator with sampled negatives
# ---------------------------------------------------------
def evaluate_ranking(
    model,
    split_df: pd.DataFrame,
    user_pos_all: Dict[int, set],
    target_item_pool: np.ndarray,
    name: str,
    k: int,
    neg_per_pos: int,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """
    Evaluate on a split where each row is a (user, positive item).
    We create a candidate set per row: {pos} U sampled_negatives.
    Metrics are computed on the ranking of that candidate set.

    Args
    ----
    model            : object with .score(u_tensor, i_tensor) -> (B,) scores
    split_df         : DataFrame with columns ['uid','iid']
    user_pos_all     : dict uid -> set of all positive item ids (across splits)
    target_item_pool : np.ndarray of item ids to sample negatives from (e.g., items seen in val/test)
    name             : string for printing
    k                : cutoff for metrics
    neg_per_pos      : number of negatives to sample per positive
    rng              : optional numpy Generator for reproducible sampling

    Returns
    -------
    dict with keys: precision, recall, f1, map, ndcg
    """
    if rng is None:
        rng = np.random.default_rng()

    precisions: List[float] = []
    recalls:    List[float] = []
    f1s:        List[float] = []
    maps:       List[float] = []
    ndcgs:      List[float] = []

    for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Eval {name}"):
        u = int(row["uid"])
        pos_i = int(row["iid"])
        pos_set = user_pos_all.get(u, set())

        # sample unique negatives not interacted by user
        negs: List[int] = []
        while len(negs) < neg_per_pos:
            j = int(rng.choice(target_item_pool))
            if j not in pos_set and j not in negs:
                negs.append(j)

        # candidates = [positive] + negatives
        cand = np.array([pos_i] + negs, dtype=np.int64)
        labels = np.zeros(len(cand), dtype=np.int8)
        labels[0] = 1  # first is the positive

        u_t = tf.constant([u] * len(cand), tf.int32)
        i_t = tf.constant(cand, tf.int32)
        scores = model.score(u_t, i_t).numpy()

        order = np.argsort(-scores)  # indices of candidates by descending score

        # metrics
        p = precision_at_k(labels, order, k)
        r = recall_at_k(labels, order, k)
        f1 = f1_at_k(p, r)
        ap = average_precision_at_k(labels, order, k)
        nd = ndcg_at_k(labels, order, k)

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        maps.append(ap)
        ndcgs.append(nd)

    precision_k = float(np.mean(precisions))
    recall_k    = float(np.mean(recalls))
    f1_k        = float(np.mean(f1s))
    map_k       = float(np.mean(maps))
    ndcg_k      = float(np.mean(ndcgs))

    print(f"[{name}] P@{k}:{precision_k:.4f} R@{k}:{recall_k:.4f} F1@{k}:{f1_k:.4f} MAP@{k}:{map_k:.4f} NDCG@{ndcg_k:.4f}")
    return {"precision": precision_k, "recall": recall_k, "f1": f1_k, "map": map_k, "ndcg": ndcg_k}

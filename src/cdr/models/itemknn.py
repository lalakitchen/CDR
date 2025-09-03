import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Iterable


def _build_user_items(train_pairs: np.ndarray, n_users: int) -> Dict[int, List[int]]:
    """
    train_pairs: int64 array of shape (N, 2) with columns [uid, iid]
    returns: dict uid -> sorted unique list of item ids interacted in train
    """
    by_user: Dict[int, set] = {}
    for u, i in train_pairs:
        by_user.setdefault(int(u), set()).add(int(i))
    # convert to sorted lists for stable behaviour
    return {u: sorted(s) for u, s in by_user.items()}


def _build_item_cooc(
    train_pairs: np.ndarray,
    n_items: int,
    topk_neighbors: int = 200,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Build item-item topK cosine neighbors from implicit co-occurrence.

    Returns:
      nbr_idx[i]: np.array of neighbor item ids (K_i,)
      nbr_sim[i]: np.array of cosine sims aligned with nbr_idx[i]
    """
    # item popularity (degree)
    pop = np.zeros(n_items, dtype=np.int32)
    users: Dict[int, List[int]] = {}
    for u, i in train_pairs:
        users.setdefault(int(u), []).append(int(i))
        pop[int(i)] += 1

    # co-occurrence counts with sparse dict-of-dicts
    cooc: Dict[int, Dict[int, int]] = {}
    for u, items in users.items():
        items = sorted(set(items))
        L = len(items)
        for a in range(L):
            i = items[a]
            d_i = cooc.setdefault(i, {})
            for b in range(L):
                if a == b:
                    continue
                j = items[b]
                d_i[j] = d_i.get(j, 0) + 1

    # cosine from cooc / sqrt(pop_i * pop_j), then keep topK per item
    nbr_idx: List[np.ndarray] = [np.zeros((0,), dtype=np.int32) for _ in range(n_items)]
    nbr_sim: List[np.ndarray] = [np.zeros((0,), dtype=np.float32) for _ in range(n_items)]
    denom = np.sqrt(np.maximum(pop, 1)).astype(np.float32)
    for i, d in cooc.items():
        if pop[i] == 0:
            continue
        js = np.fromiter((j for j in d.keys()), dtype=np.int32, count=len(d))
        cc = np.fromiter((c for c in d.values()), dtype=np.float32, count=len(d))
        sims = cc / (denom[i] * denom[js])
        if sims.size == 0:
            continue
        if sims.size > topk_neighbors:
            idx = np.argpartition(-sims, topk_neighbors)[:topk_neighbors]
            js, sims = js[idx], sims[idx]
        order = np.argsort(-sims)
        nbr_idx[i] = js[order]
        nbr_sim[i] = sims[order].astype(np.float32)

    return nbr_idx, nbr_sim


class ItemKNN(tf.keras.Model):
    """
    Non-trainable ItemKNN recommender.
    Scores(u, i) = sum_{p in user_profile(u)} sim(p, i), where sim is item-item cosine from co-occurrence.

    Args
    ----
    n_users, n_items : ints
    train_pairs      : np.ndarray (N,2) of [uid, iid] from TRAIN ONLY
    topk_neighbors   : keep this many neighbors per item (default 200)

    Notes
    -----
    - This model is NON-parameteric; it doesn't learn. Use --epochs 0 to skip training.
    - You still can evaluate it with your existing evaluator.
    """

    def __init__(self, n_users: int, n_items: int, train_pairs: np.ndarray, topk_neighbors: int = 200):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.user_items = _build_user_items(train_pairs, n_users)        # dict: u -> [items]
        self.nbr_idx, self.nbr_sim = _build_item_cooc(train_pairs, n_items, topk_neighbors)

        # For quick lookup: for each item p, map neighbor -> sim
        self.nbr_map: List[Dict[int, float]] = []
        for i in range(n_items):
            if len(self.nbr_idx[i]) == 0:
                self.nbr_map.append({})
            else:
                self.nbr_map.append({int(j): float(s) for j, s in zip(self.nbr_idx[i], self.nbr_sim[i])})

    @tf.function
    def score(self, u: tf.Tensor, i: tf.Tensor) -> tf.Tensor:
        # tf.function wrapper calls a py_function; non-differentiable (fine; model is non-trainable)
        def _score_batch(u_np, i_np):
            out = np.zeros(len(u_np), dtype=np.float32)
            for t, (uu, ii) in enumerate(zip(u_np.tolist(), i_np.tolist())):
                prof = self.user_items.get(int(uu), [])
                s = 0.0
                nb = self.nbr_map[int(ii)]
                for p in prof:
                    s += nb.get(int(p), 0.0)
                out[t] = s
            return out

        return tf.numpy_function(_score_batch, [u, i], tf.float32)

    def call(self, inputs):
        u, i = inputs
        return self.score(u, i)

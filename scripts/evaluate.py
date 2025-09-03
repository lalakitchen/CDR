#!/usr/bin/env python3
"""
Evaluate a trained recommender on val/test, with optional RL-style replay and
(optional) export of user/item embeddings.

Outputs (written under the run dir by default):
  eval_<split>_metrics.json
  eval_<split>_metrics.csv
  (optional) user_embeddings.npy, item_embeddings.npy
  (optional replay)
    bandit_events.csv
    bandit_learning_curve.csv
    bandit_report.json

Examples
--------
# Evaluate MF run on test and export embeddings
python scripts/evaluate.py \
  --data_dir artifacts/movies_from_music/data \
  --model mf --exp_name exp-mf64 --out_root checkpoint \
  --split test --topk 10 --eval_negs 99 --save_embeddings

# Evaluate NeuMF with QKV attention
python scripts/evaluate.py \
  --data_dir artifacts/movies_from_music/data \
  --model neumf_attention --exp_name exp-neumf-attn64 --out_root checkpoint \
  --split test --topk 10 --eval_negs 99 --save_embeddings

# Evaluate LightGCN with QKV layer attention
python scripts/evaluate.py \
  --data_dir artifacts/movies_from_music/data \
  --model lightgcn_attention --exp_name exp-lightgcn-attn64 --out_root checkpoint \
  --split test --topk 10 --eval_negs 99 --save_embeddings
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

# Quiet low-level TF logs before importing tensorflow
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf

# ----- Make 'src/' importable when running from repo root -----
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent  # repo_root/scripts/evaluate.py -> repo_root
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Project imports (after sys.path tweak)
from cdr.data.loader import load_splits, build_user_pos, target_pool_from
from cdr.metrics.ranking import evaluate_ranking
from cdr.models import build_model
from cdr.train.trainer import run_replay_policy  # reuse replay implementation


# -------------------- helpers --------------------
def _build_norm_adj(train_pairs: np.ndarray, n_users: int, n_items: int) -> tf.sparse.SparseTensor:
    """Symmetric normalized adjacency for LightGCN."""
    deg_u = np.zeros(n_users, dtype=np.float32)
    deg_i = np.zeros(n_items, dtype=np.float32)
    for u, i in train_pairs:
        deg_u[int(u)] += 1.0
        deg_i[int(i)] += 1.0
    inv_sqrt_u = 1.0 / np.sqrt(np.maximum(deg_u, 1.0))
    inv_sqrt_i = 1.0 / np.sqrt(np.maximum(deg_i, 1.0))

    indices = []
    values = []
    for u, i in train_pairs:
        u = int(u); i = int(i)
        w = float(inv_sqrt_u[u] * inv_sqrt_i[i])
        indices.append([u, n_users + i]); values.append(w)   # user -> item
        indices.append([n_users + i, u]); values.append(w)   # item -> user

    N = n_users + n_items
    indices = np.array(indices, dtype=np.int64)
    values = np.array(values, dtype=np.float32)
    adj = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=[N, N])
    return tf.sparse.reorder(adj)


def _export_embeddings(model, out_dir: Path, n_users: int, n_items: int) -> None:
    """
    Save user/item embeddings if possible:
      - MF/NeuMF/NeuMF_Attn: model.get_embeddings() -> (U, V)
      - LightGCN/LightGCN_Attn: model._final_embeddings() -> split into blocks
      - Otherwise: skip with a message.
    """
    try:
        if hasattr(model, "get_embeddings"):
            U, V = model.get_embeddings()
            U = U.numpy() if hasattr(U, "numpy") else U
            V = V.numpy() if hasattr(V, "numpy") else V
        elif hasattr(model, "_final_embeddings"):
            E = model._final_embeddings().numpy()
            U, V = E[:n_users], E[n_users:n_users + n_items]
        else:
            print("Embedding export skipped: model does not expose embeddings.")
            return

        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "user_embeddings.npy", U)
        np.save(out_dir / "item_embeddings.npy", V)
        print(f"Saved embeddings to {out_dir} (users {U.shape}, items {V.shape})")
    except Exception as e:
        print("Embedding export failed:", e)


def _resolve_run_dir(model_norm: str, exp_name: str, out_root: Path, run_dir: Optional[Path]) -> Path:
    return run_dir if run_dir is not None else (out_root / model_norm.upper() / exp_name)


# -------------------- CLI --------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a trained recommender on val/test (+ optional RL replay and embedding export).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument("--data_dir", type=Path, required=True, help="Folder with train/val/test parquet")
    p.add_argument("--split", type=str, default="test", choices=["val", "test"], help="Which split to evaluate")
    p.add_argument("--topk", type=int, default=10, help="K for ranking metrics")
    p.add_argument("--eval_negs", type=int, default=99, help="Negatives per positive for sampled eval")
    p.add_argument("--seed", type=int, default=42, help="Sampling seed for evaluation")

    # Run / model
    p.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "mf",
            "neumf",
            "neumf_attention", "neumf-qkv", "neumf-attn",
            "itemknn",
            "lightgcn",
            "lightgcn_attention", "lightgcn-qkv", "lightgcn-attn",
        ],
        help="Model name (must match the trained run)",
    )
    p.add_argument("--exp_name", type=str, required=True,
                   help="Experiment name (subfolder under out_root/<MODEL>/)")
    p.add_argument("--out_root", type=Path, default=Path("checkpoint"),
                   help="Root of checkpoints")
    p.add_argument("--run_dir", type=Path, default=None,
                   help="Override full run dir (takes precedence over out_root/model/exp_name)")
    p.add_argument("--weights", type=Path, default=None,
                   help="Optional explicit weights file path")
    p.add_argument("--allow_untrained", action="store_true",
                   help="Evaluate randomly initialized weights if no checkpoint is found")

    # Model-specific knobs (needed to rebuild the model)
    p.add_argument("--emb_dim", type=int, default=64, help="Embedding dim for MF/NeuMF/LightGCN")
    p.add_argument("--l2", type=float, default=1e-6, help="L2 regularization for embeddings")
    p.add_argument("--itemknn_topk_neighbors", type=int, default=200,
                   help="ItemKNN: keep this many item neighbors per item")
    p.add_argument("--lightgcn_layers", type=int, default=3,
                   help="LightGCN: number of propagation layers")
    p.add_argument("--no_cache_lightgcn", action="store_true",
                   help="Disable LightGCN per-epoch embedding cache recomputation")

    # Optional replay
    p.add_argument("--policy", type=str, default="none", choices=["none", "eps", "ucb", "linucb"],
                   help="Replay re-ranking policy")
    p.add_argument("--epsilon", type=float, default=0.1, help="ε for eps policy")
    p.add_argument("--alpha", type=float, default=1.0, help="Bonus scale for UCB/LinUCB")
    p.add_argument("--candidate_topn", type=int, default=200,
                   help="Candidate pool size per user for replay")
    p.add_argument("--slate_k", type=int, default=10, help="Slate size shown during replay")
    p.add_argument("--bootstrap_iters", type=int, default=1000,
                   help="Bootstrap iters for CI-95% in replay")

    # Embedding export
    p.add_argument("--save_embeddings", action="store_true",
                   help="Save user/item embeddings to the run directory after loading weights")

    return p.parse_args()


# -------------------- main --------------------
def main():
    a = parse_args()
    tf.random.set_seed(a.seed)
    np.random.seed(a.seed)

    # Normalize model name (so aliases like "lightgcn-attn" work)
    model_norm = a.model.lower().replace("-", "_")

    run_dir = _resolve_run_dir(model_norm, a.exp_name, a.out_root, a.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---------- load data ----------
    train, val, test = load_splits(a.data_dir)
    split_df = val if a.split == "val" else test

    n_users = int(max(train["uid"].max(), val["uid"].max(), test["uid"].max()) + 1)
    n_items = int(max(train["iid"].max(), val["iid"].max(), test["iid"].max()) + 1)

    user_pos_train = build_user_pos(train)
    user_pos_all = build_user_pos(pd.concat(
        [train[["uid", "iid"]], val[["uid", "iid"]], test[["uid", "iid"]]], ignore_index=True
    ))
    target_item_pool = target_pool_from(val, test)
    train_pairs = train[["uid", "iid"]].drop_duplicates().to_numpy(dtype=np.int64)

    # ---------- rebuild model ----------
    cache_lightgcn = not a.no_cache_lightgcn

    if model_norm == "itemknn":
        model = build_model(
            model_norm,
            n_users=n_users,
            n_items=n_items,
            train_pairs=train_pairs,
            topk_neighbors=a.itemknn_topk_neighbors,
        )

    elif model_norm.startswith("lightgcn"):  # lightgcn + attention variants
        norm_adj = _build_norm_adj(train_pairs, n_users, n_items)
        model = build_model(
            model_norm,
            n_users=n_users,
            n_items=n_items,
            d=a.emb_dim,
            l2=a.l2,
            norm_adj=norm_adj,
            n_layers=a.lightgcn_layers,
        )

    else:
        # mf / neumf / neumf_attention share the same signature
        model = build_model(model_norm, n_users=n_users, n_items=n_items, d=a.emb_dim, l2=a.l2)

    # ---------- load weights if applicable ----------
    weights_path = a.weights if a.weights is not None else (run_dir / "best.weights.h5")

    if model_norm != "itemknn":
        if weights_path.exists():
            # Keras 3: model must be built before loading weights.
            # Build by doing a dummy forward call through `call(...)`.
            dummy_u = tf.zeros([1], dtype=tf.int32)
            dummy_i = tf.zeros([1], dtype=tf.int32)
            _ = model((dummy_u, dummy_i), training=False)

            model.load_weights(str(weights_path))
            print(f"Loaded weights: {weights_path}")
        elif not a.allow_untrained:
            raise FileNotFoundError(
                f"Cannot find weights at {weights_path}. "
                f"Pass --weights explicitly or set --allow_untrained to evaluate randomly initialized model."
            )
        else:
            print("WARNING: evaluating an untrained model (random init).")

    # Cache embeddings for LightGCN family (safe for evaluation)
    if cache_lightgcn and hasattr(model, "recompute_cache"):
        model.recompute_cache()

    # ---------- optional embedding export ----------
    if a.save_embeddings:
        _export_embeddings(model, run_dir, n_users, n_items)

    # ---------- evaluate on chosen split ----------
    metrics = evaluate_ranking(
        model=model,
        split_df=split_df,
        user_pos_all=user_pos_all,
        target_item_pool=target_item_pool,
        name=a.split,
        k=a.topk,
        neg_per_pos=a.eval_negs,
    )

    # write results
    out_json = run_dir / f"eval_{a.split}_metrics.json"
    out_csv  = run_dir / f"eval_{a.split}_metrics.csv"
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame([metrics]).to_csv(out_csv, index=False)

    print(
        f"[{a.split}] P@{a.topk}:{metrics['precision']:.4f}  "
        f"R@{a.topk}:{metrics['recall']:.4f}  "
        f"F1:{metrics['f1']:.4f}  MAP:{metrics['map']:.4f}  NDCG:{metrics['ndcg']:.4f}"
    )
    print("Wrote:", out_json)
    print("Wrote:", out_csv)

    # ---------- optional replay ----------
    if a.policy != "none":
        print(f"Running offline replay: policy={a.policy}")
        _ = run_replay_policy(
            model=model,
            split_df=split_df,                    # replay on the same split evaluated
            user_pos_all=user_pos_all,
            target_item_pool=target_item_pool,
            out_dir=run_dir,
            policy_name=a.policy,
            epsilon=a.epsilon,
            alpha=a.alpha,
            slate_k=a.slate_k,
            candidate_topn=a.candidate_topn,
            seed=a.seed,
            bootstrap_iters=a.bootstrap_iters,
        )


if __name__ == "__main__":
    main()

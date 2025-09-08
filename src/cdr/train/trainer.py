import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from cdr.models import build_model
from cdr.losses import get_loss
from cdr.metrics.ranking import evaluate_ranking
from cdr.data.loader import (
    load_splits,
    build_user_pos,
    target_pool_from,
    make_bpr_dataset,
)
from cdr.policy import build_policy, FeatureFn


# -------------------- utils --------------------
def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _build_norm_adj(train_pairs: np.ndarray, n_users: int, n_items: int) -> tf.sparse.SparseTensor:
    """
    Symmetric normalized adjacency for user-item bipartite graph (LightGCN):

      A = [[0, R],
           [R^T, 0]]

      D is degree diag,  A_hat = D^{-1/2} A D^{-1/2}
    """
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
        # user -> item (top-right)
        indices.append([u, n_users + i]); values.append(w)
        # item -> user (bottom-left)
        indices.append([n_users + i, u]); values.append(w)

    N = n_users + n_items
    indices = np.array(indices, dtype=np.int64)
    values = np.array(values, dtype=np.float32)
    adj = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=[N, N])
    return tf.sparse.reorder(adj)


def _export_embeddings(model, out_dir: Path, n_users: int, n_items: int) -> None:
    """
    Save user/item embeddings if possible:
      - MF/NeuMF: use model.get_embeddings() if present
      - LightGCN: export propagated final embeddings split into user/item blocks
      - Else: skip silently
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
            return
        np.save(out_dir / "user_embeddings.npy", U)
        np.save(out_dir / "item_embeddings.npy", V)
        print("Saved embeddings to", out_dir)
    except Exception as e:
        print("Skipping embedding export:", e)


# --------- policy / replay helpers ----------
def _feature_fn_from_model(model, n_users: int, n_items: int) -> FeatureFn:
    """
    Build a feature function x(u,i) for LinUCB:
      x = [user_emb, item_emb, user_emb * item_emb]
    Works for MF/NeuMF via get_embeddings(), for LightGCN via _final_embeddings().
    """
    if hasattr(model, "get_embeddings"):
        U, V = model.get_embeddings()
        U = U.numpy() if hasattr(U, "numpy") else U
        V = V.numpy() if hasattr(V, "numpy") else V

        def feat(u: int, i: int) -> np.ndarray:
            ue = U[u]; ve = V[i]
            return np.concatenate([ue, ve, ue * ve]).astype(np.float32)

        return feat

    if hasattr(model, "_final_embeddings"):
        E = model._final_embeddings().numpy()

        def feat(u: int, i: int) -> np.ndarray:
            ue = E[u]
            ve = E[n_users + i]
            return np.concatenate([ue, ve, ue * ve]).astype(np.float32)

        return feat

    # Fallback: small ID-based features (shouldn't be needed)
    def id_only(u: int, i: int) -> np.ndarray:
        return np.array([float(u % 997) / 997.0, float(i % 997) / 997.0], dtype=np.float32)

    return id_only


def _topn_candidates(model, user_id: int, pool: np.ndarray, topn: int) -> Tuple[np.ndarray, np.ndarray]:
    u = tf.constant([user_id] * len(pool), tf.int32)
    i = tf.constant(pool, tf.int32)
    scores = model.score(u, i).numpy()
    if len(pool) <= topn:
        return pool, scores
    order = np.argsort(-scores)[:topn]
    return pool[order], scores[order]


def _bootstrap_mean(x: np.ndarray, iters: int = 1000, seed: int = 42) -> Tuple[float, float]:
    """Return (low, high) 95% CI via bootstrap on mean of x."""
    rng = np.random.default_rng(seed)
    n = len(x)
    if n == 0:
        return (0.0, 0.0)
    stats = []
    for _ in range(iters):
        idx = rng.integers(0, n, size=n)
        stats.append(float(np.mean(x[idx])))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def run_replay_policy(
    model,
    split_df: pd.DataFrame,           # val or test
    user_pos_all: Dict[int, set],
    target_item_pool: np.ndarray,
    out_dir: Path,
    policy_name: str,
    epsilon: float,
    alpha: float,
    slate_k: int,
    candidate_topn: int,
    seed: int,
    bootstrap_iters: int = 1000,
) -> Dict[str, float]:
    """
    Offline replay with a re-ranking policy and 95% bootstrap CIs.
    - For each (u, pos_i) in split:
        * get candidate_topn items by base scores
        * policy.rank -> slate of size K
        * reward@1 = 1 if top1 == pos_i else 0
        * hit@K   = 1 if pos_i in slate else 0
    Saves:
      - bandit_events.csv          per-impression records
      - bandit_learning_curve.csv  cumulative averages over impressions
      - bandit_report.json         headline metrics + 95% CIs
    """
    rng = np.random.default_rng(seed)

    # Build policy + feature function if needed
    if policy_name == "eps":
        policy = build_policy("eps", epsilon=epsilon, random_state=seed)
        feature_fn = None
    elif policy_name == "ucb":
        policy = build_policy("ucb", alpha=alpha)
        feature_fn = None
    elif policy_name == "linucb":
        feature_fn = _feature_fn_from_model(
            model,
            n_users=int(split_df["uid"].max() + 1),
            n_items=int(target_item_pool.max() + 1),
        )
        policy = build_policy("linucb", alpha=alpha)
    else:
        raise ValueError(f"Unknown policy for replay: {policy_name}")

    events = []
    cum_hits = 0
    cum_rewards = 0

    for t, (_, row) in enumerate(tqdm(split_df.iterrows(), total=len(split_df), desc=f"Replay({policy_name})")):
        u = int(row["uid"])
        pos_i = int(row["iid"])

        # candidates by base model
        cand_ids, base_scores = _topn_candidates(model, u, target_item_pool, candidate_topn)

        # policy slate
        chosen = policy.rank(u, cand_ids, base_scores, k=slate_k, feature_fn=feature_fn)

        hit_k = 1 if pos_i in set(chosen) else 0
        top1 = chosen[0]
        reward = 1.0 if top1 == pos_i else 0.0

        # update bandit on top-1
        if policy_name == "linucb":
            x = feature_fn(u, top1)
            policy.update(u, top1, reward, feature_vec=x)
        else:
            policy.update(u, top1, reward)

        cum_hits += hit_k
        cum_rewards += reward
        imp = t + 1

        events.append({
            "imp": imp,
            "uid": u,
            "pos_i": pos_i,
            "top1": top1,
            "reward": reward,   # CTR@1 proxy
            "hit@K": hit_k,     # Recall@K proxy
        })

    ev = pd.DataFrame(events)
    ev.to_csv(out_dir / "bandit_events.csv", index=False)

    # Learning curve
    ev["cum_recall@K"] = ev["hit@K"].cumsum() / ev["imp"]
    ev["cum_ctr@1"] = ev["reward"].cumsum() / ev["imp"]
    ev[["imp", "cum_recall@K", "cum_ctr@1"]].to_csv(out_dir / "bandit_learning_curve.csv", index=False)

    # Headline means
    recall_k = float(ev["hit@K"].mean()) if len(ev) else 0.0
    ctr1 = float(ev["reward"].mean()) if len(ev) else 0.0

    # 95% bootstrap CIs
    r_lo, r_hi = _bootstrap_mean(ev["hit@K"].to_numpy(dtype=np.float32), iters=bootstrap_iters, seed=seed)
    c_lo, c_hi = _bootstrap_mean(ev["reward"].to_numpy(dtype=np.float32), iters=bootstrap_iters, seed=seed)

    report = {
        "policy": policy_name,
        "epsilon": float(epsilon),
        "alpha": float(alpha),
        "K": int(slate_k),
        "candidate_topn": int(candidate_topn),
        "impressions": int(len(ev)),
        "replay_recall@K": recall_k,
        "replay_recall@K_ci95": [r_lo, r_hi],
        "replay_ctr@1": ctr1,
        "replay_ctr@1_ci95": [c_lo, c_hi],
    }
    with open(out_dir / "bandit_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(
        f"[Replay] Recall@{slate_k}={recall_k:.4f} (95% CI {r_lo:.4f}–{r_hi:.4f})  "
        f"CTR@1={ctr1:.4f} (95% CI {c_lo:.4f}–{c_hi:.4f})  N={len(ev)}"
    )
    return report


# -------------------- main trainer --------------------
def train_and_eval(
    data_dir: Path,
    out_root: Path,
    model_name: str,
    loss_name: str,
    exp_name: str,
    epochs: int = 30,
    batch_size: int = 4096,
    emb_dim: int = 64,
    lr: float = 3e-3,
    l2: float = 1e-6,
    topk: int = 10,
    eval_negs: int = 99,
    patience: int = 3,
    seed: int = 42,
    resume: bool = False,
    save_embeddings: bool = False,
    # model knobs
    itemknn_topk_neighbors: int = 200,
    lightgcn_layers: int = 3,
    cache_lightgcn: bool = True,
    # policy knobs
    policy_name: str = "none",           # none|eps|ucb|linucb
    epsilon: float = 0.1,
    alpha: float = 1.0,
    replay_after_train: bool = False,
    candidate_topn: int = 200,
    slate_k: int = 10,
    bootstrap_iters: int = 1000,
) -> Dict[str, float]:
    """
    Generic trainer for sampled top-K recommenders.

    Supports models:
      - mf         : trainable, uses BPR
      - neumf      : trainable, uses BPR
      - neumf_attention : trainable, uses BPR
      - itemknn    : NON-trainable memory-based CF (set epochs=0 to skip loop)
      - lightgcn / lightgcn_attention : trainable GNNs, use BPR

    Saves under: <out_root>/<MODEL>/<exp_name>/
      best.weights.h5, hparams.json, train_log.csv, metrics.json, test_metrics.csv, REPORT.txt
    And, if replay is enabled:
      bandit_events.csv, bandit_learning_curve.csv, bandit_report.json
    """
    _set_seeds(seed)

    out_dir = out_root / model_name.upper() / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to: {out_dir.resolve()}")

    # Save hparams
    hparams = dict(
        data_dir=str(data_dir),
        out_root=str(out_root),
        model=model_name,
        loss=loss_name,
        exp_name=exp_name,
        epochs=epochs,
        batch_size=batch_size,
        emb_dim=emb_dim,
        lr=lr,
        l2=l2,
        topk=topk,
        eval_negs=eval_negs,
        patience=patience,
        seed=seed,
        resume=resume,
        save_embeddings=save_embeddings,
        itemknn_topk_neighbors=itemknn_topk_neighbors,
        lightgcn_layers=lightgcn_layers,
        cache_lightgcn=cache_lightgcn,
        policy_name=policy_name,
        epsilon=epsilon,
        alpha=alpha,
        replay_after_train=replay_after_train,
        candidate_topn=candidate_topn,
        slate_k=slate_k,
        bootstrap_iters=bootstrap_iters,
    )
    with open(out_dir / "hparams.json", "w") as f:
        json.dump(hparams, f, indent=2)

    # Data
    train, val, test = load_splits(data_dir)
    n_users = int(max(train["uid"].max(), val["uid"].max(), test["uid"].max()) + 1)
    n_items = int(max(train["iid"].max(), val["iid"].max(), test["iid"].max()) + 1)
    print(f"Users: {n_users:,} | Items: {n_items:,}")

    user_pos_train = build_user_pos(train)
    user_pos_all = build_user_pos(
        pd.concat([train[["uid", "iid"]], val[["uid", "iid"]], test[["uid", "iid"]]], ignore_index=True)
    )
    target_item_pool = target_pool_from(val, test)
    train_pairs = train[["uid", "iid"]].drop_duplicates().to_numpy(dtype=np.int64)

    ds = make_bpr_dataset(train_pairs, n_items, user_pos_train, batch_size)
    it = iter(ds)  # persistent iterator

    # Model / loss / optimizer
    name = model_name.lower()
    if name == "itemknn":
        model = build_model(
            model_name,
            n_users=n_users,
            n_items=n_items,
            train_pairs=train_pairs,
            topk_neighbors=itemknn_topk_neighbors,
        )
        loss_fn = None
        optimizer = None
    elif name.startswith("lightgcn"):  # supports lightgcn + attention variants
        norm_adj = _build_norm_adj(train_pairs, n_users, n_items)
        model = build_model(
            model_name,
            n_users=n_users,
            n_items=n_items,
            d=emb_dim,
            l2=l2,
            norm_adj=norm_adj,
            n_layers=lightgcn_layers,
        )
        loss_fn = get_loss(loss_name)
        optimizer = tf.keras.optimizers.Adam(lr)
    else:
        # mf / neumf / neumf_attention
        model = build_model(model_name, n_users=n_users, n_items=n_items, d=emb_dim, l2=l2)
        loss_fn = get_loss(loss_name)
        optimizer = tf.keras.optimizers.Adam(lr)

    # --- Ensure the model is BUILT once (Keras 3 requirement) ---
    if name != "itemknn":
        _ = model((tf.constant([0], tf.int32), tf.constant([0], tf.int32)), training=False)

    # Paths
    ckpt_path = out_dir / "best.weights.h5"
    metrics_json = out_dir / "metrics.json"
    train_log_csv = out_dir / "train_log.csv"
    test_metrics_csv = out_dir / "test_metrics.csv"
    report_txt = out_dir / "REPORT.txt"

    # Resume (trainable models only)
    if resume and ckpt_path.exists() and name != "itemknn":
        print(f"Resuming from {ckpt_path}")
        # model already built above; safe to load
        model.load_weights(str(ckpt_path))

    @tf.function
    def train_step(u, i, j):
        with tf.GradientTape() as tape:
            s_pos = model.score(u, i)
            s_neg = model.score(u, j)
            loss = loss_fn(s_pos, s_neg)
            if model.losses:
                loss += tf.add_n(model.losses)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    # Train
    history_rows = []
    best_val = -1.0
    best_epoch = -1
    patience_left = patience
    steps_per_epoch = max(1, len(train_pairs) // batch_size)

    if epochs <= 0:
        print("Epochs set to 0 — skipping training loop.")
        if cache_lightgcn and hasattr(model, "recompute_cache"):
            model.recompute_cache()
        val_m = evaluate_ranking(model, val, user_pos_all, target_item_pool, "val", topk, eval_negs)
        best_val = val_m["recall"]; best_epoch = 0
    else:
        for epoch in range(1, epochs + 1):
            if cache_lightgcn and hasattr(model, "recompute_cache"):
                model.recompute_cache()

            if name != "itemknn":
                pbar = tqdm(range(steps_per_epoch), desc=f"{model_name.upper()} Epoch {epoch:02d}")
                losses = []
                for _ in pbar:
                    u, i, j = next(it)
                    loss = train_step(u, i, j)
                    losses.append(float(loss.numpy()))
                    pbar.set_postfix(loss=np.mean(losses))
                avg_loss = float(np.mean(losses))
            else:
                avg_loss = 0.0  # non-trainable

            print(f"Epoch {epoch:02d} | train loss: {avg_loss:.4f}")

            # Validation
            val_m = evaluate_ranking(model, val, user_pos_all, target_item_pool, "val", topk, eval_negs)

            improved = val_m["recall"] > best_val
            history_rows.append(
                {
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "val_precision": val_m["precision"],
                    "val_recall": val_m["recall"],
                    "val_f1": val_m["f1"],
                    "val_map": val_m["map"],
                    "val_ndcg": val_m["ndcg"],
                    "is_best": int(improved),
                }
            )

            if improved:
                best_val = val_m["recall"]
                best_epoch = epoch
                patience_left = patience
                if name != "itemknn":
                    # model is already built; just save
                    model.save_weights(str(ckpt_path))
                    print(f"Saved best to {ckpt_path}")
            else:
                patience_left -= 1
                if patience_left == 0 and name != "itemknn":
                    print("Early stopping")
                    break

    # per-epoch log
    pd.DataFrame(history_rows).to_csv(train_log_csv, index=False)
    print(f"Wrote per-epoch log to {train_log_csv}")

    # Load best and test (trainable models only)
    if ckpt_path.exists() and name != "itemknn":
        # model already built; safe to load
        model.load_weights(str(ckpt_path))
        print(f"Loaded best weights from {ckpt_path}")

    if cache_lightgcn and hasattr(model, "recompute_cache"):
        model.recompute_cache()

    test_m = evaluate_ranking(model, test, user_pos_all, target_item_pool, "test", topk, eval_negs)

    if save_embeddings:
        _export_embeddings(model, out_dir, n_users, n_items)

    with open(metrics_json, "w") as f:
        json.dump({"best_val_recall": best_val, "best_epoch": best_epoch, "test": test_m}, f, indent=2)

    row = {
        "model": model_name,
        "loss": loss_name,
        "exp_name": exp_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "best_epoch": best_epoch,
        "best_val_recall": best_val,
        "test_precision": test_m["precision"],
        "test_recall": test_m["recall"],
        "test_f1": test_m["f1"],
        "test_map": test_m["map"],
        "test_ndcg": test_m["ndcg"],
        "epochs_run": len(history_rows),
        "emb_dim": emb_dim,
        "lr": lr,
        "l2": l2,
        "batch_size": batch_size,
        "topk": topk,
        "eval_negs": eval_negs,
        "seed": seed,
        "itemknn_topk_neighbors": itemknn_topk_neighbors,
        "lightgcn_layers": lightgcn_layers,
        "policy_name": policy_name,
        "epsilon": epsilon,
        "alpha": alpha,
        "candidate_topn": candidate_topn,
        "slate_k": slate_k,
        "bootstrap_iters": bootstrap_iters,
    }
    pd.DataFrame([row]).to_csv(test_metrics_csv, index=False)

    with open(report_txt, "w") as f:
        f.write(
            f"Model: {model_name} | Loss: {loss_name}\n"
            f"Experiment: {exp_name}\n"
            f"Users: {n_users} | Items: {n_items}\n"
            f"Best val Recall@{topk}: {best_val:.4f} (epoch {best_epoch})\n"
            f"Test: P@{topk}={test_m['precision']:.4f}  "
            f"R@{topk}={test_m['recall']:.4f}  "
            f"F1@{topk}={test_m['f1']:.4f}  "
            f"MAP@{topk}={test_m['map']:.4f}  "
            f"NDCG@{topk}={test_m['ndcg']:.4f}\n"
        )

    print(f"Saved metrics.json, test_metrics.csv, REPORT.txt to {out_dir}")

    # ---- Optional offline replay with a policy ----
    if policy_name != "none" or replay_after_train:
        policy_name_eff = policy_name if policy_name != "none" else "eps"
        print(f"Running offline replay with policy={policy_name_eff}")
        _ = run_replay_policy(
            model=model,
            split_df=val,  # use validation split for replay by default
            user_pos_all=user_pos_all,
            target_item_pool=target_item_pool,
            out_dir=out_dir,
            policy_name=policy_name_eff,
            epsilon=epsilon,
            alpha=alpha,
            slate_k=slate_k,
            candidate_topn=candidate_topn,
            seed=seed,
            bootstrap_iters=bootstrap_iters,
        )

    return test_m

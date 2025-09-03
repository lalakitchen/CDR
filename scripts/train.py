#!/usr/bin/env python3
"""
scripts/train.py

Examples
--------
# MF + BPR (baseline)
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
  --data_dir artifacts/movies_from_music-2/data \
  --model mf --loss bpr \
  --exp_name exp-mf64-2 --out_root checkpoint \
  --epochs 30 --emb_dim 64 --topk 10 --eval_negs 99

"""

import argparse
import os
import sys
from pathlib import Path

# Make 'src/' importable when running from repo root
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Optional: quiet TensorFlow C++ logs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from cdr.train import train_and_eval  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train/evaluate recommenders on Amazon CDR splits (+ optional RL replay)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Core paths
    p.add_argument("--data_dir", type=Path, required=True, help="Folder with train/val/test parquet")
    p.add_argument("--out_root", type=Path, default=Path("checkpoint"), help="Where to save outputs")
    p.add_argument("--exp_name", type=str, required=True, help="Experiment name (subfolder under out_root/<MODEL>/)")
    # Model / loss
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
        help="Model to run",
    )
    p.add_argument(
        "--loss",
        type=str,
        default="bpr",
        choices=["bpr", "hinge"],
        help="Pairwise ranking loss (ignored for itemknn)",
    )
    # Training hparams
    p.add_argument("--epochs", type=int, default=30, help="Epochs (use 0 for non-trainable baselines like ItemKNN)")
    p.add_argument("--batch_size", type=int, default=1024, help="Mini-batch size for BPR sampling")
    p.add_argument("--emb_dim", type=int, default=64, help="Embedding dimension (for MF/NeuMF/LightGCN)")
    p.add_argument("--lr", type=float, default=3e-3, help="Adam learning rate")
    p.add_argument("--l2", type=float, default=1e-6, help="L2 reg strength on embeddings (and dense kernels)")
    p.add_argument("--patience", type=int, default=3, help="Early stopping patience on val Recall@K")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--resume", action="store_true", help="Resume from best.weights if present (trainable models)")
    p.add_argument("--save_embeddings", type=bool, default=True, help="Export user/item embeddings as .npy if available")
    # Evaluation
    p.add_argument("--topk", type=int, default=10, help="K for ranking metrics")
    p.add_argument("--eval_negs", type=int, default=99, help="Negatives per positive for sampled eval")
    # Model-specific knobs
    p.add_argument("--itemknn_topk_neighbors", type=int, default=200,
                   help="ItemKNN: keep this many item neighbors per item")
    p.add_argument("--lightgcn_layers", type=int, default=3, help="LightGCN: number of propagation layers")
    p.add_argument("--no_cache_lightgcn", action="store_true",
                   help="Disable LightGCN per-epoch embedding cache recomputation (useful for attention variants)")
    # Policy / replay
    p.add_argument("--policy", type=str, default="none", choices=["none", "eps", "ucb", "linucb"],
                   help="Re-ranking policy for offline replay")
    p.add_argument("--epsilon", type=float, default=0.1, help="ε for eps policy")
    p.add_argument("--alpha", type=float, default=1.0, help="Bonus scale for UCB/LinUCB")
    p.add_argument("--replay_after_train", action="store_true", help="Run offline replay even if --policy none")
    p.add_argument("--candidate_topn", type=int, default=200, help="Candidate pool size per user for replay")
    p.add_argument("--slate_k", type=int, default=10, help="Slate size shown during replay")
    p.add_argument("--bootstrap_iters", type=int, default=1000, help="Replay bootstrap iterations for 95% CI")
    return p.parse_args()


def main():
    args = parse_args()
    cache_lightgcn = not args.no_cache_lightgcn

    # Normalize aliases so trainer sees consistent names
    model_norm = args.model.lower().replace("-", "_")

    train_and_eval(
        data_dir=args.data_dir,
        out_root=args.out_root,
        model_name=model_norm,           # supports *_attention names
        loss_name=args.loss,
        exp_name=args.exp_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        emb_dim=args.emb_dim,
        lr=args.lr,
        l2=args.l2,
        topk=args.topk,
        eval_negs=args.eval_negs,
        patience=args.patience,
        seed=args.seed,
        resume=args.resume,
        save_embeddings=args.save_embeddings,
        itemknn_topk_neighbors=args.itemknn_topk_neighbors,
        lightgcn_layers=args.lightgcn_layers,
        cache_lightgcn=cache_lightgcn,
        # policy / replay
        policy_name=args.policy,
        epsilon=args.epsilon,
        alpha=args.alpha,
        replay_after_train=args.replay_after_train,
        candidate_topn=args.candidate_topn,
        slate_k=args.slate_k,
        bootstrap_iters=args.bootstrap_iters,
    )


if __name__ == "__main__":
    main()

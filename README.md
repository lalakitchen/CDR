# Cross-Domain Recommendation 
End-to-end pipeline for cross-domain recommendation 

---

## Repo layout

```
.
├── README.md
├── requirements.txt
├── scripts/
│   ├── download_amazon_2018.py        # dataset fetch helper (Python)
│   ├── preprocess_amazon_cdr.py       # data prep entrypoint
│   └── train.py                       # training + eval 
├── src/
│   └── cdr/
│       ├── __init__.py
│       ├── data/
│       │   └── loader.py              # loaders, samplers, pools
│       ├── losses/
│       │   ├── __init__.py
│       │   └── bpr.py                 # BPR, hinge
│       ├── metrics/
│       │   └── ranking.py             # Precision, Recall, F1, MAP, NDCG
│       ├── models/
│       │   ├── __init__.py
│       │   ├── mf.py                  # Matrix Factorization (TF)
│       │   ├── neumf.py               # Neural Collaborative Filtering
│       │   ├── itemknn.py             # Item-based CF (memory-based)
│       │   └── lightgcn.py            # Graph recommender
│       ├── policy/
│       │   ├── __init__.py
│       │   ├── base.py                # policy interface
│       │   ├── epsilon.py             # epsilon-greedy
│       │   ├── ucb.py                 # UCB1
│       │   └── linucb.py              # LinUCB (uses embeddings as features)
│       └── train/
│           ├── __init__.py
│           └── trainer.py             # training loop 
├── data/
│   └── amazon2018/                    # reviews/, metadata/ (not tracked)
├── artifacts/
│   └── movies_from_music/
│       ├── data/                      # train.parquet, val.parquet, test.parquet
│       └── maps/                      # user_id_map.json, item_id_map.json
└── checkpoint/
    ├── MF/
    ├── NEUMF/
    ├── ITEMKNN/
    └── LIGHTGCN/
        └── <exp_name>/
            ├── best.weights
            ├── hparams.json
            ├── metrics.json
            ├── train_log.csv
            ├── test_metrics.csv
            ├── REPORT.txt
 
```

---

## Setup

Python 3.9+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

`requirements.txt` should include:
```
tensorflow>=2.12
numpy
pandas
pyarrow
tqdm
scipy
```

GPU is optional. If TensorFlow cannot see your GPU, it will run on CPU.

---

## Data

### 1) Download Amazon 2018 categories

The helper creates the folder structure and downloads gzipped JSON files for reviews and metadata.

```bash
python scripts/download_amazon_2018.py \
  --root data/amazon2018 \
  --cats Books Movies_and_TV CDs_and_Vinyl \
         Clothing_Shoes_and_Jewelry Electronics Home_and_Kitchen Toys_and_Games
```

Tips:
- If you only need a Music -> Movies scenario, pass just `CDs_and_Vinyl` and `Movies_and_TV`.
- Files are large. Make sure you have enough disk space.

### 2) Preprocess to cross-domain splits

What this does:
- Cleans missing essentials. Normalizes rating to [0,1]. Normalizes timestamp per user to [0,1].
- Keeps items without metadata by filling `item_cat="unknown"`.
- Keeps users with fewer than 5 interactions.
- Filters to overlapping users across source and target (unless `--keep_all_users`).
- Target split per user: last -> test, second-last -> val, rest -> train.
- Train set = all source interactions + target train interactions.

Run:

```bash
python scripts/preprocess_amazon_cdr.py \
  --root data/amazon2018 \
  --source CDs_and_Vinyl \
  --target Movies_and_TV \
  --out artifacts/movies_from_music \
  --print_stats --save_stats
```

Outputs:

```
artifacts/movies_from_music/
  data/
    train.parquet
    val.parquet
    test.parquet
  maps/
    user_id_map.json
    item_id_map.json
  README.txt
  split_stats.csv          # if --save_stats
```

---

## Train and evaluate

All trainable models use pairwise ranking (BPR) by default. Evaluation is sampled with `eval_negs` negatives per positive.

### MF (baseline)

```bash
python scripts/train.py \
  --data_dir artifacts/movies_from_music/data \
  --model mf --loss bpr \
  --exp_name exp-mf64 --out_root checkpoint \
  --epochs 30 --emb_dim 64 --topk 10 --eval_negs 99 \
  --save_embeddings
```

### NeuMF (deep)

```bash
python scripts/train.py \
  --data_dir artifacts/movies_from_music/data \
  --model neumf --loss bpr \
  --exp_name exp-neumf --out_root checkpoint \
  --epochs 30 --emb_dim 64 --topk 10 --eval_negs 99
```

### ItemKNN (memory-based CF, non-trainable)

```bash
python scripts/train.py \
  --data_dir artifacts/movies_from_music/data \
  --model itemknn --loss bpr \
  --exp_name exp-itemknn --out_root checkpoint \
  --epochs 0 --itemknn_topk_neighbors 200
```

### LightGCN (graph)

```bash
python scripts/train.py \
  --data_dir artifacts/movies_from_music/data \
  --model lightgcn --loss bpr \
  --exp_name exp-lgcn --out_root checkpoint \
  --epochs 30 --emb_dim 64 --lightgcn_layers 3 \
  --topk 10 --eval_negs 99
```

Artifacts per run:

```
checkpoint/<MODEL>/<EXP_NAME>/
  best.weights
  hparams.json
  metrics.json            # best val recall + test metrics
  test_metrics.csv        # one-row CSV for spreadsheet compares
  train_log.csv           # epoch-wise loss + val metrics
  REPORT.txt
  user_embeddings.npy     # optional
  item_embeddings.npy     # optional
```

---

## Optional interactive replay (epsilon-greedy, UCB1, LinUCB)

This wraps the trained scorer with a small policy and runs offline replay on the validation split. It logs per-impression results and computes 95% bootstrap CIs.

epsilon-greedy example:

```bash
python scripts/train.py \
  --data_dir artifacts/movies_from_music/data \
  --model mf --loss bpr \
  --exp_name exp-mf-eps02 --out_root checkpoint \
  --epochs 30 --emb_dim 64 \
  --policy eps --epsilon 0.2 --replay_after_train \
  --candidate_topn 200 --slate_k 10 --bootstrap_iters 1000
```

UCB1 example:

```bash
python scripts/train.py \
  --data_dir artifacts/movies_from_music/data \
  --model neumf --loss bpr \
  --exp_name exp-neumf-ucb --out_root checkpoint \
  --epochs 30 --emb_dim 64 \
  --policy ucb --alpha 0.5 --replay_after_train
```

LinUCB example:

```bash
python scripts/train.py \
  --data_dir artifacts/movies_from_music/data \
  --model lightgcn --loss bpr \
  --exp_name exp-lgcn-linucb --out_root checkpoint \
  --epochs 30 --emb_dim 64 --lightgcn_layers 3 \
  --policy linucb --alpha 0.3 --replay_after_train
```

Replay artifacts (in the same experiment folder):

```
bandit_events.csv             # one row per impression: uid, pos_i, top1, reward, hit@K
bandit_learning_curve.csv     # cumulative recall@K and ctr@1 over impressions
bandit_report.json            # replay_recall@K, ctr@1, and 95% CIs
```

How to decide if RL helped:
- Higher `replay_recall@K` and stable `replay_ctr@1`.
- Non-overlapping 95% CIs across runs is a good sign.
- Base test metrics should remain similar because RL wraps inference.

---

## Metrics

Sampled evaluation with `eval_negs` negatives per positive:
- Precision@K, Recall@K, F1@K
- MAP@K
- NDCG@K

Replay metrics:
- hit@K per impression (Recall@K proxy)
- reward at rank 1 (CTR@1 proxy)
- 95% bootstrap CIs on both

---

## Repro guide (Music -> Movies)

1) Download categories

```bash
python scripts/download_amazon_2018.py \
  --root data/amazon2018 \
  --cats CDs_and_Vinyl Movies_and_TV
```

2) Preprocess

```bash
python scripts/preprocess_amazon_cdr.py \
  --root data/amazon2018 \
  --source CDs_and_Vinyl \
  --target Movies_and_TV \
  --out artifacts/movies_from_music \
  --print_stats --save_stats
```

3) Train MF

```bash
python scripts/train.py \
  --data_dir artifacts/movies_from_music/data \
  --model mf --loss bpr \
  --exp_name exp-mf64 --out_root checkpoint \
  --epochs 30 --emb_dim 64
```



---

## References

- Rendle, S. BPR: Bayesian Personalized Ranking from Implicit Feedback. UAI 2009.
- He, X. et al. Neural Collaborative Filtering. WWW 2017.
- He, X. et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. SIGIR 2020.
- Sarwar, B. et al. Item-Based Collaborative Filtering Recommendation Algorithms. WWW 2001.
- Li, L. et al. A Contextual-Bandit Approach to Personalized News Article Recommendation. WWW 2010.
- Ni, J., Li, J., McAuley, J. Justifying Recommendations using Distantly-Labeled Reviews and Fine-Grained Aspects. EMNLP 2019.
- Huang, Ling, et al. "Knowledge-Reinforced Cross-Domain Recommendation." IEEE Transactions on Neural Networks and Learning Systems (2024).
- Zhao, Chuang, et al. "Cross-domain recommendation via progressive structural alignment." IEEE Transactions on Knowledge and Data Engineering 36.6 (2023): 2401-2415.

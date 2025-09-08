
# Cross-Domain Recommendation
End-to-end pipeline for cross-domain recommendation (Music → Movies).

---

## Interactive Demo

![Interactive demo](notebook/interactive_demo.gif)

---

## Repo layout

```
.
├── README.md
├── requirements.txt
├── scripts/
│   ├── download_amazon.py        # dataset fetch helper
│   ├── preprocess_amazon.py       # data prep entrypoint
│   └── train.py                       # training 
    └── evaluate.py                       # eval
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
│       │   ├── neumf_attention.py     # NeuMF + QKV attention
│       │   ├── itemknn.py             # Item-based CF (memory-based)
│       │   ├── lightgcn.py            # Graph recommender
│       │   
│       └── train/
│           ├── __init__.py
│           └── trainer.py             # training loop
├── notebook/
│   ├── EDA.ipynb
│   └── INTERACTIVE_DEMO.ipynb
├── data/
│   └── amazon2018/                    # reviews/, metadata/ (not tracked)
├── artifacts/
│   └── <dataset_name>/
│       ├── data/                      # train.parquet, val.parquet, test.parquet
│       └── maps/                      # user_id_map.json, item_id_map.json
└── checkpoint/
    ├── MF/
    ├── NEUMF/
    ├── ITEMKNN/
    └── LIGHTGCN/
        └── <exp_name>/
            ├── best.weights.h5
            ├── hparams.json
            ├── metrics.json
            ├── train_log.csv
            ├── test_metrics.csv
            └── REPORT.txt
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
huggingface_hub
```

GPU is optional. If TensorFlow cannot see your GPU, it will run on CPU.

---

## Data

### 1) Download Amazon 2018 categories

This helper creates the folder structure and downloads gzipped JSON files for reviews and metadata.

```bash
python scripts/download_amazon_2018.py   --root data/amazon2018   --cats Digital_Music Movies_and_TV          Books Clothing_Shoes_and_Jewelry Electronics Home_and_Kitchen Toys_and_Games
```

Tips:
- For **Music → Movies** experiments, you only need `Digital_Music` and `Movies_and_TV`.
- These files are large; ensure you have enough disk space.

### 2) Preprocess to cross-domain splits

What this does:
- Drops rows with missing IDs/ratings/timestamps; clips ratings to `[1,5]`.
- Deduplicates each user–item pair by **keeping the most recent** interaction.
- Normalizes rating to `[0,1]` and timestamp per user to `[0,1]`.
- Keeps items without metadata by filling `item_cat="unknown"`.
- **Filters to overlapping users** across source and target (unless `--keep_all_users`).
- **Filters to users with >5 interactions and items with >5 ratings** to reduce extreme sparsity.
- **Target (Movies)** split per user: last → **test**, second-last → **val**, others → **train** (chronological leave-one-out).
- **Train set = all source (Music) interactions + target-train (Movies)** interactions.

Run:

```bash
python scripts/preprocess_amazon_cdr.py   --root data/amazon2018   --source Digital_Music   --target Movies_and_TV   --out artifacts/movies_from_music   --print_stats --save_stats
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

All trainable models use **pairwise ranking (BPR)** by default. Evaluation samples `--eval_negs` negatives per positive (default **99**).

One command for any model:

```bash
python scripts/train.py   --data_dir artifacts/movies_from_music/data   --model <mf|neumf|neumf_attention|itemknn|lightgcn>   --exp_name <exp-name> --out_root checkpoint   --epochs <n> --emb_dim 64 --topk 10 --eval_negs 99   [--loss bpr] [--save_embeddings]   [--itemknn_topk_neighbors 200]   [--lightgcn_layers 3]
```

Quick presets (fill `<exp-name>` and `<n>`):
- **MF**: `--model mf --epochs 30`
- **NeuMF**: `--model neumf --epochs 30`
- **NeuMF + Attention (QKV)**: `--model neumf_attention --epochs 30`
- **ItemKNN**: `--model itemknn --epochs 0 --itemknn_topk_neighbors 200`
- **LightGCN**: `--model lightgcn --epochs 30 --lightgcn_layers 3`

**Per-run artifacts:**

```
checkpoint/<MODEL>/<EXP_NAME>/
  best.weights.h5
  hparams.json
  metrics.json            # best val recall + test metrics
  test_metrics.csv        # one-row CSV for comparisons
  train_log.csv           # epoch-wise loss + val metrics
  REPORT.txt
  user_embeddings.npy     # optional
  item_embeddings.npy     # optional
```

---

## Checkpoints (Hugging Face, manual)

Download `best.weights.h5` from the links below and place it at:
`checkpoint/<SUBDIR>/<EXP_NAME>/best.weights.h5`

| Model    | Subdir on Hub              | Folder |
|---------|-----------------------------|--------|
| MF      | `MF/`                       | [Open](https://huggingface.co/farchan/CDR-checkpoints/tree/main/MF) |
| NeuMF   | `NEUMF/`                    | [Open](https://huggingface.co/farchan/CDR-checkpoints/tree/main/NEUMF) |
| NeuMF + Attn | `NEUMF+ATTENTION/`                  | [Open](https://huggingface.co/farchan/CDR-checkpoints/tree/main/NEUMF%2BAttention) |
| ItemKNN | `ITEMKNN/`                  | [Open](https://huggingface.co/farchan/CDR-checkpoints/tree/main/ITEMKNN) |
| LightGCN| `LIGHTGCN/`                 | [Open](https://huggingface.co/farchan/CDR-checkpoints/tree/main/LIGHTGCN)  |

**Example (MF, `exp-mf64`)**  
Place the file at: `checkpoint/MF/exp-mf64/best.weights.h5`

Then run:

```bash
python scripts/evaluate.py   --data_dir artifacts/movies_from_music/data   --model mf --exp_name exp-mf64   --split test --topk 10 --eval_negs 99 --save_embeddings
```

---

## Evaluation protocol (offline)

- **Candidate set.** For each user, we rank a fixed set of **100 movies**: the user’s held-out movie (**1 positive**) plus **99 sampled** movies the user has not seen.
- **Metrics.** We report **Precision@10, Recall@10, NDCG@10, and MAP@10**. “@10” means each metric is computed with respect to the **top ten** recommendations.

---

## Interactive demo (qualitative)

- **Seed:** The user selects one or more **music** titles from a list.
- **Map music → movies:** The system builds a **session profile** by combining a neutral profile with the **average embedding of the selected songs**, then **ranks movies only** by similarity to that profile and returns a **Top-10**.
- **Align with feedback:** **Like** shifts the profile **toward** that movie; **Dislike** shifts it **away**; high/low **ratings** act similarly. Recommendations **refresh immediately**.

---

## Data split statistics

| Domain | Split | Users | Items | Interactions |
|---|---|---:|---:|---:|
| Movie | Train | 79,532 | 37,507 | 954,823 |
| Movie | Val   | 85,745 | 22,311 | 85,745 |
| Movie | Test  | 91,794 | 22,561 | 91,794 |
| Music | Train | 66,386 | 17,110 | 198,062 |

**Totals (interactions):** Movies = **1,132,362**; Music (train) = **198,062**.

---

## Model performance (Movies test split)

| Model                | Precision@10 | F1@10   | MAP@10 | NDCG@10 |
|----------------------|-------------:|--------:|-------:|--------:|
| MF                   | 4.736%       | 8.61%   | 30.32% | 30.32%  |
| ItemKNN (Collab. CF) | 4.869%       | 8.885%  | 25.84% | 31.25%  |
| NeuMF                | 5.564%       | 10.27%  | 28.79% | 35.11%  |
| NeuMF + Attention    | 6.175%       | 11.22%  | 31.22% | 38.44%  |
| **LightGCN**         | **9.556%**   | **17.37%** | **46.20%** | **49.32%** |

*Higher is better. Results use 1 positive + 99 sampled negatives per user and K=10.*

---

## Notebooks

- `notebook/EDA.ipynb` – quick data exploration.
- `notebook/INTERACTIVE_DEMO.ipynb` – small UI for trying a trained model.  

Launch Jupyter:
```bash
jupyter notebook notebook/
```

---

## References

- Rendle, S. **BPR: Bayesian Personalized Ranking from Implicit Feedback.** UAI 2009.  
- He, X. et al. **Neural Collaborative Filtering.** WWW 2017.  
- He, X. et al. **LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation.** SIGIR 2020.  
- Sarwar, B. et al. **Item-Based Collaborative Filtering Recommendation Algorithms.** WWW 2001.

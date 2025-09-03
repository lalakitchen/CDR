# Cross-Domain Recommendation
End-to-end pipeline for cross-domain recommendation.

---

## Demo (GIF)
If your README viewer supports images, this will animate inline:

![Interactive demo](NOTEBOOK/interactive_demo.gif)


---

## Repo layout

```
.
├── README.md
├── requirements.txt
├── scripts/
│   ├── download_amazon_2018.py        # dataset fetch helper
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
│       │   ├── neumf_attention.py     # NeuMF + QKV attention
│       │   ├── itemknn.py             # Item-based CF (memory-based)
│       │   ├── lightgcn.py            # Graph recommender
│       │   └── lightgcn_attention.py  # LightGCN + QKV over layers
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
python scripts/download_amazon_2018.py   --root data/amazon2018   --cats Books Movies_and_TV CDs_and_Vinyl          Clothing_Shoes_and_Jewelry Electronics Home_and_Kitchen Toys_and_Games
```

Tips:
- If you only need a Music → Movies scenario, pass just `CDs_and_Vinyl` and `Movies_and_TV`.
- These files are large.

### 2) Preprocess to cross-domain splits

What this does:
- Cleans missing essentials. Normalizes rating to [0,1]. Normalizes timestamp per user to [0,1].
- Keeps items without metadata by filling `item_cat="unknown"`.
- Keeps users with fewer than 5 interactions.
- Filters to overlapping users across source and target (unless `--keep_all_users`).
- Target split per user: last → test, second-last → val, rest → train.
- Train set = all source interactions + target train interactions.

Run:

```bash
python scripts/preprocess_amazon_cdr.py   --root data/amazon2018   --source CDs_and_Vinyl   --target Movies_and_TV   --out artifacts/movies_from_music   --print_stats --save_stats
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

All models use pairwise ranking (BPR) by default. Evaluation samples `eval_negs` negatives per positive.

Use this one command for any model:
```bash
python scripts/train.py   --data_dir artifacts/movies_from_music/data   --model <mf|neumf|neumf_attention|itemknn|lightgcn|lightgcn_attention>   --exp_name <your-exp-name> --out_root checkpoint   --epochs <n> --emb_dim 64 --topk 10 --eval_negs 99   [--loss bpr] [--save_embeddings]   [--itemknn_topk_neighbors 200]   [--lightgcn_layers 3]
```

Quick presets (fill `<your-exp-name>` and `<n>`):
- **MF**: `--model mf --epochs 30`
- **NeuMF**: `--model neumf --epochs 30`
- **NeuMF + Attention (QKV)**: `--model neumf_attention --epochs 30`
- **ItemKNN**: `--model itemknn --epochs 0 --itemknn_topk_neighbors 200`
- **LightGCN**: `--model lightgcn --epochs 30 --lightgcn_layers 3`
- **LightGCN + Attention (QKV)**: `--model lightgcn_attention --epochs 30 --lightgcn_layers 3`

Artifacts per run:

```
checkpoint/<MODEL>/<EXP_NAME>/
  best.weights.h5
  hparams.json
  metrics.json            # best val recall + test metrics
  test_metrics.csv        # one-row CSV for spreadsheet compares
  train_log.csv           # epoch-wise loss + val metrics
  REPORT.txt
  user_embeddings.npy     # optional
  item_embeddings.npy     # optional
```

---

## Checkpoints on Hugging Face

If a local weights file is missing, `scripts/evaluate.py` can download one from the Hub.
The expected layout inside the Hub repo is: `<SUBDIR>/<EXP_NAME>/best.weights.h5`.

| Model    | Subdir on Hub              | Example exp_name | Direct link (folder) |
|---------|-----------------------------|------------------|----------------------|
| MF      | `MF/`                       | `exp-mf64`       | https://huggingface.co/farchan/CDR-checkpoints/tree/main/MF |
| NeuMF   | `NEUMF/`                    | `exp-neumf`      | https://huggingface.co/farchan/CDR-checkpoints/tree/main/NEUMF |
| ItemKNN | `ITEMKNN/` or `MF/ITEMKNN/` | `exp-itemknn`    | https://huggingface.co/farchan/CDR-checkpoints/tree/main/MF/ITEMKNN |
| LightGCN| `LIGHTGCN/`                 | `exp-lgcn`       | *(add if you upload)* |

Examples:

```bash
# Pull MF weights from the Hub
python scripts/evaluate.py   --data_dir artifacts/movies_from_music/data   --model mf --exp_name exp-mf64   --hf_repo farchan/CDR-checkpoints   --split test --topk 10 --eval_negs 99 --save_embeddings
```

```bash
# Pull NeuMF weights from the Hub
python scripts/evaluate.py   --data_dir artifacts/movies_from_music/data   --model neumf --exp_name exp-neumf   --hf_repo farchan/CDR-checkpoints   --split test --topk 10 --eval_negs 99
```

---

## Notebooks

- `notebook/EDA.ipynb` – quick data exploration.
- `notebook/INTERACTIVE_DEMO.ipynb` – small UI for trying a trained model.  
  Save a short screen recording of this and convert it to `NOTEBOOK/interactive_demo.gif` (used at the top).
  Example (ffmpeg):
  ```bash
  ffmpeg -i demo.mp4 -vf "fps=20,scale=900:-1:flags=lanczos" -loop 0 NOTEBOOK/interactive_demo.gif
  ```

Launch Jupyter:
```bash
jupyter notebook notebook/
```

---

## Test set results (fill in your runs)

This table tracks results on the **target** domain test split. `@10` means K=10.

| Model | Exp name | K | Precision@10 | Recall@10 | F1@10 | Acc@10 | NDCG@10 | Notes |
|------|----------|---|--------------|-----------|-------|--------|---------|-------|
| MF | exp-mf64 | 10 |  |  |  |  |  |  |
| NeuMF | exp-neumf | 10 |  |  |  |  |  |  |
| NeuMF+Attn | exp-neumf-attn | 10 |  |  |  |  |  |  |
| ItemKNN | exp-itemknn | 10 |  |  |  |  |  |  |
| LightGCN | exp-lgcn | 10 |  |  |  |  |  |  |
| LightGCN+Attn | exp-lgcn-attn | 10 |  |  |  |  |  |  |

`@10` means we evaluate using the **top 10** items returned for each user.

### Metric definitions (per user, then averaged over users)

- **Precision@K**: relevant items in the top K divided by K.  
  `P@K = |TopK ∩ Rel| / K`

- **Recall@K**: relevant items in the top K divided by all relevant items for that user.  
  `R@K = |TopK ∩ Rel| / |Rel|`

- **F1@K**: harmonic mean of Precision@K and Recall@K.  
  `F1@K = 2 * P@K * R@K / (P@K + R@K)`

- **Acc@K** (a.k.a. HitRate@K): 1 if there is **at least one** relevant item in the top K, else 0. Then average across users.  
  `Acc@K = 1 if |TopK ∩ Rel| ≥ 1 else 0` (averaged)

- **NDCG@K**: rank-aware gain.  
  `DCG@K = Σ_{j=1..K} rel_j / log2(j+1)` and `NDCG@K = DCG@K / IDCG@K`

Notes:
- In our sampled evaluation, we rank each positive against `eval_negs` sampled negatives per user.
- If your use case has multiple positives per user in the test set, the definitions still apply.

---

## References

- Rendle, S. **BPR: Bayesian Personalized Ranking from Implicit Feedback.** UAI 2009.  
- He, X. et al. **Neural Collaborative Filtering.** WWW 2017.  
- He, X. et al. **LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation.** SIGIR 2020.  
- Sarwar, B. et al. **Item-Based Collaborative Filtering Recommendation Algorithms.** WWW 2001.

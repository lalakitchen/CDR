#!/usr/bin/env python3
"""
preprocess_amazon_cdr.py  (with tqdm progress, tracing, and split-wise stats)

Usage:
  python scripts/preprocess_amazon.py \
    --root data/amazon2018 \
    --source CDs_and_Vinyl \
    --target Movies_and_TV \
    --out artifacts/movies_from_music \
    --print_stats --save_stats

What it does
- Loads Amazon 2018 per-category reviews + metadata
- Cleans missing values, normalizes rating (0..1) and per-user time (0..1)
- Keeps items with missing metadata (item_cat='unknown'); keeps users with <5 interactions
- Filters to overlapping users (unless --keep_all_users)
- Builds shared user and item id maps (items are keyed as "<DOMAIN>::<ASIN>")
- Splits target per user: last->test, second-last->val, others->train; all source interactions go to train
- Saves Parquet splits + id maps
- Prints and optionally saves split-wise tables:
    Domains | Users | Items | Interactions | Density
"""

import argparse
import gzip
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

tqdm.pandas()  # enable .progress_apply on pandas groupby/apply

# ---------- small logger ----------
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

# ---------- readers with progress ----------
def read_json_gz_lines(path: Path, fields: Optional[List[str]] = None, desc: str = "") -> pd.DataFrame:
    rows, bad = [], 0
    desc = desc or f"Read {path.name}"
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc=desc, unit="lines"):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if fields is None:
                rows.append(obj)
            else:
                rows.append({k: obj.get(k) for k in fields})
    if bad:
        log(f"Skipped {bad} malformed lines in {path.name}")
    return pd.DataFrame(rows)

def load_reviews(root: Path, category: str) -> pd.DataFrame:
    path = root / "reviews" / f"{category}.json.gz"
    log(f"Loading reviews: {path}")
    df = read_json_gz_lines(
        path,
        fields=["reviewerID", "asin", "overall", "unixReviewTime"],
        desc=f"Reviews:{category}"
    )
    log(f"Reviews raw rows = {len(df):,}")

    # Handle missing essentials
    before = len(df)
    df = df.dropna(subset=["reviewerID", "asin", "overall", "unixReviewTime"])
    log(f"Dropped missing essentials: {before - len(df):,}")

    # Coerce types
    df["overall"] = pd.to_numeric(df["overall"], errors="coerce")
    df["unixReviewTime"] = pd.to_numeric(df["unixReviewTime"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["overall", "unixReviewTime"])
    log(f"Dropped after coercion: {before - len(df):,}")

    # Clean types and normalize rating
    df["overall"] = df["overall"].clip(1, 5).astype(np.float32)
    df["unixReviewTime"] = df["unixReviewTime"].astype(np.int64)
    df = df.rename(columns={"reviewerID": "user_id", "asin": "item_id",
                            "unixReviewTime": "ts", "overall": "rating"})
    df["domain"] = category
    df["rating_norm"] = ((df["rating"] - 1.0) / 4.0).clip(0.0, 1.0).astype(np.float32)

    # Deduplicate user-item: keep most recent
    before = len(df)
    df = df.sort_values(["user_id", "item_id", "ts"]).drop_duplicates(["user_id", "item_id"], keep="last")
    log(f"Dedup user-item removed: {before - len(df):,}")
    log(f"Reviews cleaned rows = {len(df):,}")
    return df[["user_id", "item_id", "ts", "rating", "rating_norm", "domain"]]

def extract_category_from_meta(obj: Dict) -> Optional[str]:
    cat = obj.get("category", None)
    if cat:
        return str(cat)
    # Older style "categories": list of paths
    cats = obj.get("categories", None)
    if isinstance(cats, list) and cats and isinstance(cats[0], list) and cats[0]:
        return str(cats[0][-1])
    return None

def load_metadata(root: Path, category: str) -> pd.DataFrame:
    path = root / "metadata" / f"meta_{category}.json.gz"
    log(f"Loading metadata: {path}")
    rows, bad = [], 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Metadata:{category}", unit="lines"):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            asin = obj.get("asin", None)
            if asin is None:
                continue
            cat = extract_category_from_meta(obj)
            rows.append({"item_id": asin, "item_cat": cat})
    if bad:
        log(f"Skipped {bad} malformed lines in meta_{category}.json.gz")
    meta = pd.DataFrame(rows)
    # Fill missing category
    miss_before = meta["item_cat"].isna().sum()
    meta["item_cat"] = meta["item_cat"].fillna("unknown").astype(str)
    log(f"Metadata missing 'item_cat' filled as 'unknown': {miss_before:,}")
    log(f"Metadata rows = {len(meta):,}")
    return meta

# ---------- feature helpers ----------
def add_norm_time_per_user(df: pd.DataFrame) -> pd.DataFrame:
    log("Adding per-user ts_norm [0..1] (progress_apply)")
    def _norm(g):
        t = g["ts"].astype(np.float64)
        tmin, tmax = t.min(), t.max()
        g["ts_norm"] = 0.0 if tmax == tmin else (t - tmin) / (tmax - tmin)
        return g
    return df.groupby("user_id", group_keys=False).progress_apply(_norm)

def split_target_per_user(df_tgt: pd.DataFrame) -> pd.DataFrame:
    log("Splitting target by user: train/val/test (progress_apply)")
    def _split(g):
        g = g.sort_values("ts")
        n = len(g)
        if n == 1:
            g["split"] = ["test"]
        elif n == 2:
            g["split"] = ["val", "test"]
        else:
            g["split"] = ["train"] * (n - 2) + ["val", "test"]
        return g
    out = df_tgt.copy()
    return out.groupby("user_id", group_keys=False).progress_apply(_split)

def build_id_maps(users: List[str], items: List[str]) -> Tuple[Dict[str, int], Dict[str, int]]:
    log("Building ID maps")
    uid_map = {u: i for i, u in enumerate(sorted(set(users)))}
    iid_map = {x: i for i, x in enumerate(sorted(set(items)))}
    log(f"Users mapped: {len(uid_map):,} | Items mapped: {len(iid_map):,}")
    return uid_map, iid_map

# ---------- stats helpers ----------
ALIASES = {"Movies_and_TV": "Movie", "CDs_and_Vinyl": "Music", "Digital_Music": "Music"}

def alias_name(dom: str) -> str:
    return ALIASES.get(dom, dom)

def compute_split_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dom, g in df.groupby("domain"):
        users = g["uid"].nunique()
        items = g["iid"].nunique()
        inter = len(g)
        dens = inter / (users * items) if users > 0 and items > 0 else 0.0
        rows.append({"Domains": alias_name(dom), "Users": users, "Items": items, "Interactions": inter, "Density": dens})
    out = pd.DataFrame(rows).sort_values("Domains")
    out["Users"] = out["Users"].map(lambda x: f"{x:,}")
    out["Items"] = out["Items"].map(lambda x: f"{x:,}")
    out["Interactions"] = out["Interactions"].map(lambda x: f"{x:,}")
    out["Density"] = out["Density"].map(lambda x: f"{x*100:.2f}%")
    return out

def save_split_tables(train_df, val_df, test_df, out_dir: Path):
    stats_dir = out_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(stats_dir / "train_stats.csv", index=False)
    val_df.to_csv(stats_dir / "val_stats.csv", index=False)
    test_df.to_csv(stats_dir / "test_stats.csv", index=False)
    with open(stats_dir / "README.txt", "w") as f:
        f.write("Split-wise stats per domain (Users, Items, Interactions, Density)\n")
        f.write("Files: train_stats.csv, val_stats.csv, test_stats.csv\n")

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="Root dir with reviews/ and metadata/")
    ap.add_argument("--source", type=str, required=True, help="Source category (e.g., CDs_and_Vinyl)")
    ap.add_argument("--target", type=str, required=True, help="Target category (e.g., Movies_and_TV)")
    ap.add_argument("--out", type=Path, required=True, help="Output directory for artifacts")
    ap.add_argument("--keep_all_users", action="store_true", help="If set, do NOT filter to common users")
    ap.add_argument("--print_stats", action="store_true", help="Print split-wise tables")
    ap.add_argument("--save_stats", action="store_true", help="Save the tables as CSV under <out>/stats/")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # Load reviews
    df_src = load_reviews(args.root, args.source)
    df_tgt = load_reviews(args.root, args.target)

    # Per-user time normalization
    df_src = add_norm_time_per_user(df_src)
    df_tgt = add_norm_time_per_user(df_tgt)

    # Load + join metadata
    meta_src = load_metadata(args.root, args.source)
    meta_tgt = load_metadata(args.root, args.target)

    # Merge metadata
    df_src = df_src.merge(meta_src, on="item_id", how="left")
    df_src["item_cat"] = df_src["item_cat"].fillna("unknown").astype(str)

    df_tgt = df_tgt.merge(meta_tgt, on="item_id", how="left")
    df_tgt["item_cat"] = df_tgt["item_cat"].fillna("unknown").astype(str)

    # Filter to common users unless told otherwise
    if not args.keep_all_users:
        common_users = set(df_src["user_id"]).intersection(set(df_tgt["user_id"]))
        log(f"Overlapping users: {len(common_users):,}")
        df_src = df_src[df_src["user_id"].isin(common_users)].copy()
        df_tgt = df_tgt[df_tgt["user_id"].isin(common_users)].copy()
    else:
        log("Keeping all users (no overlap filter)")

    # Build ID maps
    df_src["item_key"] = args.source + "::" + df_src["item_id"]
    df_tgt["item_key"] = args.target + "::" + df_tgt["item_id"]
    users = pd.concat([df_src["user_id"], df_tgt["user_id"]], ignore_index=True).unique().tolist()
    items = pd.concat([df_src["item_key"], df_tgt["item_key"]], ignore_index=True).unique().tolist()
    uid_map, iid_map = build_id_maps(users, items)

    # Map to ints
    log("Mapping to integer IDs")
    df_src["uid"] = df_src["user_id"].map(uid_map).astype(np.int32)
    df_tgt["uid"] = df_tgt["user_id"].map(uid_map).astype(np.int32)
    df_src["iid"] = df_src["item_key"].map(iid_map).astype(np.int32)
    df_tgt["iid"] = df_tgt["item_key"].map(iid_map).astype(np.int32)

    # Split target
    df_tgt_split = split_target_per_user(df_tgt)

    # Compose final splits (keep 'domain' only for stats; exclude when saving)
    base_cols = ["uid", "iid", "ts", "ts_norm", "rating", "rating_norm", "item_cat", "domain"]

    log("Building train set (source all + target train)")
    train_src = df_src.copy(); train_src["split"] = "train"
    train_tgt = df_tgt_split[df_tgt_split["split"] == "train"].copy()
    train = pd.concat([train_src[base_cols + ["split"]], train_tgt[base_cols + ["split"]]], ignore_index=True)

    log("Building val/test from target")
    val  = df_tgt_split[df_tgt_split["split"] == "val"][base_cols + ["split"]].copy()
    test = df_tgt_split[df_tgt_split["split"] == "test"][base_cols + ["split"]].copy()

    # Save (without 'domain')
    out_data = args.out / "data"
    out_maps = args.out / "maps"
    out_data.mkdir(parents=True, exist_ok=True)
    out_maps.mkdir(parents=True, exist_ok=True)

    log("Saving Parquet files")
    save_cols = ["uid", "iid", "ts", "ts_norm", "rating", "rating_norm", "item_cat", "split"]
    train[save_cols].to_parquet(out_data / "train.parquet", index=False)
    val[save_cols].to_parquet(out_data / "val.parquet", index=False)
    test[save_cols].to_parquet(out_data / "test.parquet", index=False)

    log("Saving ID maps")
    pd.Series(uid_map).to_json(out_maps / "user_id_map.json")
    pd.Series(iid_map).to_json(out_maps / "item_id_map.json")

    # Quick summary
    def quick(df, name):
        return f"{name}: users={df['uid'].nunique():,}, items={df['iid'].nunique():,}, interactions={len(df):,}"
    with open(args.out / "README.txt", "w", encoding="utf-8") as f:
        f.write(f"Source: {args.source}\nTarget: {args.target}\n\n")
        f.write(quick(train, "train") + "\n")
        f.write(quick(val,   "val")   + "\n")
        f.write(quick(test,  "test")  + "\n")

    # Optional: domain tables
    if args.print_stats or args.save_stats:
        train_tbl = compute_split_table(train)
        val_tbl = compute_split_table(val)
        test_tbl = compute_split_table(test)
        if args.print_stats:
            print("\n=== TRAIN ==="); print(train_tbl.to_string(index=False))
            print("\n=== VAL ===");   print(val_tbl.to_string(index=False))
            print("\n=== TEST ===");  print(test_tbl.to_string(index=False))
        if args.save_stats:
            save_split_tables(train_tbl, val_tbl, test_tbl, args.out)

    log("Done.")

if __name__ == "__main__":
    main()

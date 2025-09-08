#!/usr/bin/env python3
"""
download_amazon_2018.py

Download per-category reviews and metadata from:
  https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/  (per-category links)

By default it downloads Movies_and_TV and CDs_and_Vinyl.
You can add more categories with --cats.

Examples
--------
# default two categories (movies + music CDs)
python scripts/download_amazon_2018.py --root data/amazon2018

# add Digital_Music too
python scripts/download_amazon_2018.py --root data/amazon2018 \
  --cats Movies_and_TV,CDs_and_Vinyl,Digital_Music

# only movies
python scripts/download_amazon_2018.py --root data/amazon2018 --cats Movies_and_TV

Notes
-----
- Reviews live under:
  https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFiles/<CATEGORY>.json.gz
- Metadata lives under:
  https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_<CATEGORY>.json.gz
- Files are large. This script supports resume, progress bars, and simple size checks.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

REV_BASE  = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFiles"
META_BASE = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2"

DEFAULT_CATS = [
    "Movies_and_TV",
    "CDs_and_Vinyl",
]

CHUNK = 1024 * 1024  # 1MB

def human(n: int) -> str:
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"

def head_content_length(url: str, timeout: int = 20) -> Optional[int]:
    try:
        r = requests.head(url, allow_redirects=True, timeout=timeout)
        if r.ok:
            cl = r.headers.get("Content-Length")
            return int(cl) if cl is not None else None
    except requests.RequestException:
        return None
    return None

def download_with_resume(url: str, dest: Path, timeout: int = 30, retries: int = 3) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix(dest.suffix + ".part")

    size_remote = head_content_length(url)  # may be None
    pos = temp.stat().st_size if temp.exists() else 0

    headers = {}
    if pos > 0:
        headers["Range"] = f"bytes={pos}-"

    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, headers=headers, timeout=timeout) as r:
                if r.status_code in (200, 206):
                    total = None
                    if "Content-Length" in r.headers:
                        total = int(r.headers["Content-Length"]) + (pos if r.status_code == 206 else 0)
                    elif size_remote is not None:
                        total = size_remote

                    mode = "ab" if r.status_code == 206 and pos > 0 else "wb"
                    if mode == "wb":
                        pos = 0  # starting fresh

                    pbar_total = total if total is not None else None
                    with open(temp, mode) as f, tqdm(
                        total=pbar_total,
                        initial=pos,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=dest.name,
                    ) as pbar:
                        for chunk in r.iter_content(chunk_size=CHUNK):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))

                    # finalize
                    final_size = temp.stat().st_size
                    if size_remote is not None and final_size != size_remote:
                        # size mismatch, retry
                        continue
                    temp.replace(dest)
                    return
                else:
                    raise RuntimeError(f"HTTP {r.status_code}")
        except Exception as e:
            if attempt == retries:
                raise
            # on retry, try starting over (drop partial)
            if temp.exists():
                try: temp.unlink()
                except OSError: pass

    # if we reach here, something went wrong
    raise RuntimeError(f"Failed to download {url}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="Destination root (will create reviews/ and metadata/)")
    ap.add_argument("--cats", type=str, default=",".join(DEFAULT_CATS),
                    help="Comma-separated category list, e.g. 'Movies_and_TV,CDs_and_Vinyl,Digital_Music'")
    ap.add_argument("--skip_meta", action="store_true", help="If set, do not download metadata files")
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    cats = [c.strip() for c in args.cats.split(",") if c.strip()]
    if not cats:
        print("No categories provided", file=sys.stderr)
        sys.exit(1)

    reviews_dir = args.root / "reviews"
    meta_dir    = args.root / "metadata"
    reviews_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    print(f"Categories: {cats}")
    for c in cats:
        # Reviews
        rev_url = f"{REV_BASE}/{c}.json.gz"
        rev_out = reviews_dir / f"{c}.json.gz"
        print(f"\nDownloading reviews: {rev_url}")
        download_with_resume(rev_url, rev_out, timeout=args.timeout, retries=args.retries)
        # Metadata
        if not args.skip_meta:
            meta_url = f"{META_BASE}/meta_{c}.json.gz"
            meta_out = meta_dir / f"meta_{c}.json.gz"
            print(f"Downloading metadata: {meta_url}")
            download_with_resume(meta_url, meta_out, timeout=args.timeout, retries=args.retries)

    print("\nDone. Files saved under:", args.root.resolve())

if __name__ == "__main__":
    main()

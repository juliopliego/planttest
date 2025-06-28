"""Prepare Oxford 102 Flowers dataset for ImageFolder training.

After downloading the Kaggle dataset (e.g. `yousefmohamed20/oxford-102-flower-dataset`), you will have:

out_dir/
    102 flower/
        flowers/
            train/1/...
            valid/1/...
            test/1/...

This script merges *train* and *valid* into a single directory tree compatible
with `torchvision.datasets.ImageFolder`.

Usage
-----
python data/prepare_flowers.py \
    --src_dir "data/oxford_flowers/102 flower/flowers" \
    --out_dir data/flowers_imagefolder \
    --copy   # optional, otherwise symlinks
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from tqdm import tqdm


def merge_split(split_dir: Path, dest_root: Path, copy: bool = False) -> None:
    for class_dir in split_dir.iterdir():
        if not class_dir.is_dir():
            continue
        dest = dest_root / class_dir.name
        dest.mkdir(parents=True, exist_ok=True)
        for img in class_dir.glob("*.jpg"):
            target = dest / img.name
            if copy:
                if not target.exists():
                    shutil.copy2(img, target)
            else:
                if target.exists():
                    continue
                target.symlink_to(img.resolve())


def prepare(src_dir: Path, out_dir: Path, copy: bool = False) -> None:
    train_dir = src_dir / "train"
    valid_dir = src_dir / "valid"
    if not train_dir.exists() or not valid_dir.exists():
        sys.exit("Source directory must contain 'train' and 'valid' subfolders!")
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in [train_dir, valid_dir]:
        merge_split(split, out_dir, copy)

    print("✅ Flowers dataset ready at", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge train+valid of Oxford Flowers into ImageFolder.")
    parser.add_argument("--src_dir", required=True, help="Path to 'flowers' directory containing train/valid/test")
    parser.add_argument("--out_dir", default="data/flowers_imagefolder", help="Destination ImageFolder root")
    parser.add_argument("--copy", action="store_true")

    args = parser.parse_args()
    prepare(Path(args.src_dir), Path(args.out_dir), args.copy) 
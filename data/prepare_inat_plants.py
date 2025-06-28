"""Prepare iNaturalist dataset (or similar) by extracting **plant-only** classes
into an ImageFolder-compatible directory.

It expects that the source directory already contains per-class sub-directories,
like the Kaggle "authuria/inaturalist" dataset:

src_dir/
    train_mini/
        00000_Animalia_.../
        12345_Plantae_Magnoliopsida_.../
        ...

The taxonomy is encoded in the directory name, the 2nd token (separated by
underscores) is the kingdom ("Plantae").  We symlink or copy (your choice)
all plant directories into *out_dir* so that you can point torchvision's
ImageFolder at it.

Usage
-----
python data/prepare_inat_plants.py \
    --src_dir ~/datasets/inat/train_mini \
    --out_dir data/inat_plants_dataset \
    --min_images 30 \
    --copy          # copies files instead of symlinking
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from tqdm import tqdm


def is_plant_dir(name: str) -> bool:
    parts = name.split("_")
    return len(parts) > 1 and parts[1] == "Plantae"


def prepare(src_dir: Path, out_dir: Path, min_images: int = 0, copy: bool = False) -> None:
    if not src_dir.exists():
        sys.exit(f"Source directory {src_dir} does not exist. Extract the dataset first.")
    out_dir.mkdir(parents=True, exist_ok=True)

    num_kept = 0
    for class_dir in tqdm(list(src_dir.iterdir()), desc="Scanning classes"):
        if not class_dir.is_dir() or not is_plant_dir(class_dir.name):
            continue
        images = list(class_dir.glob("*.jpg"))
        if len(images) < min_images:
            continue

        dest = out_dir / class_dir.name
        dest.mkdir(parents=True, exist_ok=True)
        for img in images:
            target = dest / img.name
            if copy:
                if not target.exists():
                    shutil.copy2(img, target)
            else:
                # Symlink (saves disk space); overwrite if exists
                if target.exists():
                    target.unlink()
                target.symlink_to(img.resolve())
        num_kept += 1

    print(f"✅ Prepared {num_kept} plant classes in {out_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract plant-only classes from iNaturalist directory.")
    parser.add_argument("--src_dir", required=True, help="Path to iNaturalist per-class subdir (e.g., train_mini)")
    parser.add_argument("--out_dir", default="data/inat_plants_dataset", help="Where to store the plant ImageFolder")
    parser.add_argument("--min_images", type=int, default=0, help="Skip classes with fewer than N images")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of symlinking (uses more disk space)")

    args = parser.parse_args()
    prepare(Path(args.src_dir), Path(args.out_dir), args.min_images, args.copy) 
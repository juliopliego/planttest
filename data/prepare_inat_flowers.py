"""Prepare a *flower‐only* subset of an iNaturalist directory tree.

The iNaturalist mirror on Kaggle (`authuria/inaturalist`) names each class directory like:
```
12345_Plantae_Magnoliopsida_Asterales_Asteraceae_Taraxacum_officinale
```
The fields are: *ID* \_ *Kingdom* \_ *Class* \_ *Order* \_ *Family* \_ *Genus* \_ *species*.

Flowering plants correspond to the classes **Magnoliopsida** (dicots) and **Liliopsida** (monocots).
This script reuses most logic from `prepare_inat_plants.py` but keeps only these two classes.

Usage
-----
python data/prepare_inat_flowers.py \
    --src_dir ~/datasets/inat/train_mini \
    --out_dir data/inat_flowers_dataset \
    --min_images 50
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from tqdm import tqdm

FLOWER_CLASSES = {"Magnoliopsida", "Liliopsida"}


def is_flowering(name: str) -> bool:
    parts = name.split("_")
    return len(parts) > 3 and parts[3] in FLOWER_CLASSES


def prepare(src_dir: Path, out_dir: Path, min_images: int = 0, copy: bool = False) -> None:
    if not src_dir.exists():
        sys.exit(f"Source directory {src_dir} does not exist. Extract the dataset first.")
    out_dir.mkdir(parents=True, exist_ok=True)

    kept = 0
    for class_dir in tqdm(list(src_dir.iterdir()), desc="Scanning classes"):
        if not class_dir.is_dir() or not is_flowering(class_dir.name):
            continue
        imgs = list(class_dir.glob("*.jpg"))
        if len(imgs) < min_images:
            continue
        dest = out_dir / class_dir.name
        dest.mkdir(parents=True, exist_ok=True)
        for img in imgs:
            target = dest / img.name
            if copy:
                if not target.exists():
                    shutil.copy2(img, target)
            else:
                if target.exists():
                    continue
                target.symlink_to(img.resolve())
        kept += 1
    print(f"✅ Prepared {kept} flowering plant classes in {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Extract flowering plant (Magnoliopsida + Liliopsida) from iNat")
    p.add_argument("--src_dir", required=True)
    p.add_argument("--out_dir", default="data/inat_flowers_dataset")
    p.add_argument("--min_images", type=int, default=0)
    p.add_argument("--copy", action="store_true")
    args = p.parse_args()
    prepare(Path(args.src_dir), Path(args.out_dir), args.min_images, args.copy) 
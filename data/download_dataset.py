"""Download and unpack a Kaggle image dataset.

The script expects a Kaggle *dataset slug* such as `spMohanty/plant-seedlings-classification` and will
- authenticate using the Kaggle API (requires `~/.kaggle/kaggle.json`)
- download the dataset as a zip archive
- unzip it into the desired output directory

Example usage
-------------
python data/download_dataset.py --slug spMohanty/plant-seedlings-classification --out_dir data/plant_dataset
"""

from __future__ import annotations

import argparse
import os
import shutil
import zipfile
from pathlib import Path
import subprocess
import sys
import shutil as _shutil

from kaggle.api.kaggle_api_extended import KaggleApi


def download_and_extract(slug: str, out_dir: Path) -> None:  # noqa: D401
    """Download `slug` via Kaggle API and extract it into *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    print(f"Downloading Kaggle dataset: {slug} …")

    # Prefer the kaggle CLI because it shows a progress bar with speed.
    if _shutil.which("kaggle"):
        try:
            subprocess.check_call(
                [
                    "kaggle",
                    "datasets",
                    "download",
                    "-d",
                    slug,
                    "-p",
                    str(out_dir),
                    "--force",
                ]
            )
        except subprocess.CalledProcessError as e:
            sys.exit(f"kaggle CLI download failed: {e}")
    else:
        # Fallback to Kaggle API (may not show granular progress)
        api.dataset_download_files(slug, path=out_dir, unzip=False, quiet=False)

    archive_path = next(out_dir.glob("*.zip"))
    print(f"Unzipping {archive_path} …")
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(out_dir)

    # Remove the zip to save space
    archive_path.unlink()

    # Some Kaggle datasets add an extra top-level dir; flatten if needed
    # Look for exactly one directory at root that contains sub-dirs with images.
    subdirs = [p for p in out_dir.iterdir() if p.is_dir()]
    if len(subdirs) == 1 and not any((out_dir / "train").exists() for _ in [0]):
        inner = subdirs[0]
        for p in inner.iterdir():
            shutil.move(str(p), out_dir)
        inner.rmdir()

    print("Done! You can now point the training script at:", out_dir.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a Kaggle dataset and extract it.")
    parser.add_argument("--slug", required=True, help="Kaggle dataset slug, e.g. spMohanty/plant-seedlings-classification")
    parser.add_argument("--out_dir", default="data/plant_dataset", help="Where to place the extracted dataset")
    parser.add_argument("--username", help="Kaggle username (optional if ~/.kaggle/kaggle.json or env vars are set)")
    parser.add_argument("--key", help="Kaggle API key (optional)")

    args = parser.parse_args()

    # Inject credentials into env vars if provided so KaggleApi can pick them up
    if args.username and args.key:
        os.environ["KAGGLE_USERNAME"] = args.username
        os.environ["KAGGLE_KEY"] = args.key

    download_and_extract(args.slug, Path(args.out_dir)) 
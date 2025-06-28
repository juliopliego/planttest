# Plant Identifier – ML Training Pipeline

This repository contains everything you need to train a computer-vision model that recognises plant species (or diseases) and export it to **Core ML** for integration into an iOS app.

---

## 1. Project structure

```
PlantIndentifier/
├── data/                  # Downloaded dataset will live here
│   └── download_dataset.py
├── models/                # Saved PyTorch checkpoints & exported  .mlmodel files
├── src/
│   └── train.py           # Training / evaluation / Core ML export script
├── requirements.txt       # Python dependencies (see below)
└── README.md              # You are here
```

Feel free to adjust paths – they are configurable through command-line arguments.

---

## 2. Environment setup

1. **Create a virtualenv (recommended)**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On macOS / Linux
   ```

2. **Install the Python dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   The main libraries are `torch`, `torchvision`, `coremltools`, and `kaggle` (optional).

---

## 3. Getting a plant dataset

### Option A – iNaturalist (plants only)

There's a Kaggle mirror of the iNaturalist **mini** dataset (`authuria/inaturalist`, ~40 GB compressed). To extract only *Plantae* classes:

```bash
# 1 – Download (big, can take hours)
python data/download_dataset.py --slug authuria/inaturalist --out_dir ~/datasets/inat

# 2 – Extract plant‐only classes into ImageFolder layout (symlinks by default)
python data/prepare_inat_plants.py \
    --src_dir ~/datasets/inat/train_mini \
    --out_dir data/inat_plants_dataset \
    --min_images 30
```

The resulting `data/inat_plants_dataset` can contain **thousands of species** (depends on `min_images`).

### Option B – Kaggle *Plant Seedlings* or *PlantVillage*

If you have a Kaggle account, the easiest way is to download one of the high-quality plant datasets available there, for example:

* **Plant Seedlings Classification** (`spMohanty/plant-seedlings-classification`)
* **PlantVillage** (`arjunbhasin2013/plantvillage-dataset`)

Add your `kaggle.json` API credentials to `~/.kaggle/` (see Kaggle docs) and run:

```bash
python data/download_dataset.py --slug spMohanty/plant-seedlings-classification --out_dir data/plant_dataset
```

The script will automatically unzip the archive into a folder structure compatible with `torchvision.datasets.ImageFolder` (each class in its own sub-directory):

```
plant_dataset/
├── class_1/
│   ├── img1.jpg
│   └── ...
├── class_2/
│   └── ...
└── ...
```

### Option B – Bring your own images

If you already have a dataset on disk, just organise it in the same *ImageFolder* layout and point the training script at that path.

---

## 4. Training & evaluation

Run the training pipeline with sensible defaults (the script automatically picks CUDA, Apple *MPS* on Apple-Silicon Macs, or CPU):

```bash
python src/train.py \
    --data_dir data/plant_dataset \
    --epochs 15 \
    --batch_size 32 \
    --model_out models/plant_identifier.pth \
    --balanced          # optional: tackle class imbalance
    --amp               # optional: enable mixed-precision (CUDA)
    --export_coreml
```

The script uses **transfer learning** with a pretrained *MobileNet V2*. It automatically splits the data into training and validation subsets, prints progress bars, and saves the best checkpoint (lowest validation loss).

---

## 5. Exporting to Core ML

If you pass `--export_coreml`, the script will convert the trained PyTorch model to a `.mlmodel` file using `coremltools` and place it next to the checkpoint. You can then drag-and-drop that file into Xcode and start using it with *Vision* or *Core ML* APIs.

---

## 6. iOS integration tips

* Prefer **Vision** requests (VNCoreMLRequest) for easiest image handling.
* Resize/crop the camera image so it matches the model's input dimensions (224×224 WxH by default).
* Enable **neural-engine** / FP16 quantisation in Xcode for best on-device performance.

---

## 7. Next steps / improvements

* Try bigger datasets (e.g. FGVC *iNaturalist*, *Herbarium 2022*) for higher accuracy.
* Fine-tune for **disease detection** using bounding boxes or segmentation.
* Experiment with model compression: pruning, quantisation-aware training, etc.
* Run hyper-parameter searches with [Optuna](https://optuna.org/) or Ray Tune. 
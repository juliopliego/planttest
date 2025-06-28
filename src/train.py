"""Train a plant-classification CNN and optionally export it to Core ML.

Usage (common):
---------------
python src/train.py \
    --data_dir data/plant_dataset \
    --epochs 15 \
    --batch_size 32 \
    --model_out models/plant_identifier.pth \
    --export_coreml

Requirements
------------
- Dataset folder organised as <root>/<class_name>/<image files>
- PyTorch ≥ 2.0, torchvision, coremltools (optional)
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models
from tqdm import tqdm
import math

try:
    import coremltools as ct
except ImportError:  # pragma: no cover – only needed for Core ML export
    ct = None  # type: ignore

RNG_SEED = 42


def seed_everything(seed: int = RNG_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_loaders(data_dir: Path, batch_size: int = 32, val_split: float = 0.2, num_workers: int = 4, balanced: bool = False):
    """Return (train_loader, val_loader, num_classes)."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tfms = T.Compose(
        [
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    val_tfms = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean, std)])

    dataset_full = datasets.ImageFolder(data_dir, transform=train_tfms)
    num_classes = len(dataset_full.classes)

    # Stratified split for consistent class distribution
    indices = np.arange(len(dataset_full))
    targets = np.array([s[1] for s in dataset_full.samples])
    train_idx, val_idx = train_test_split(
        indices, test_size=val_split, stratify=targets, random_state=RNG_SEED
    )

    train_set = Subset(dataset_full, train_idx)
    train_set.dataset.transform = train_tfms

    val_set = Subset(dataset_full, val_idx)
    val_set.dataset.transform = val_tfms

    if balanced:
        # Compute weight for each sample inverse to its class frequency
        class_counts = np.bincount(targets[train_idx], minlength=num_classes).astype(float)
        sample_weights = [1.0 / class_counts[t] for t in targets[train_idx]]
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, num_classes


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, amp, mixup_alpha=0.0, cutmix_alpha=0.0):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        # Apply Mixup/CutMix if enabled
        lam = 1.0
        targets_a, targets_b = labels, labels  # default
        if (mixup_alpha > 0 or cutmix_alpha > 0) and inputs.size(0) > 1:
            rand_val = random.random()
            if rand_val < 0.5 and mixup_alpha > 0:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                shuffle = torch.randperm(inputs.size(0)).to(device)
                inputs = lam * inputs + (1 - lam) * inputs[shuffle]
                targets_b = labels[shuffle]
            elif cutmix_alpha > 0:
                lam = np.random.beta(cutmix_alpha, cutmix_alpha)
                shuffle = torch.randperm(inputs.size(0)).to(device)
                targets_b = labels[shuffle]
                bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(3), inputs.size(2), lam)
                inputs[:, :, bby1:bby2, bbx1:bbx2] = inputs[shuffle, :, bby1:bby2, bbx1:bbx2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size(-1) * inputs.size(-2)))

        with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
            outputs = model(inputs)
            if mixup_alpha > 0 or cutmix_alpha > 0:
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            else:
                loss = criterion(outputs, labels)

        if amp and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        preds = outputs.argmax(dim=1)
        running_loss += loss.item() * inputs.size(0)
        # Accuracy: compare against original labels (approx when augmentation applied)
        running_corrects += (preds == labels).sum().item()

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_corrects / len(loader.dataset)
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device, amp):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            preds = outputs.argmax(dim=1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += (preds == labels).sum().item()

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_corrects / len(loader.dataset)
    return epoch_loss, epoch_acc


def export_to_coreml(model, output_path: Path, class_labels: list[str]):
    if ct is None:
        raise RuntimeError("coremltools not installed – install it or omit --export_coreml.")

    model.eval()
    example_input = torch.randn(1, 3, 224, 224)
    traced = torch.jit.trace(model, example_input)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name="input", shape=example_input.shape, scale=1 / 255.0, bias=[0, 0, 0])],
        classifier_config=ct.ClassifierConfig(class_labels),
        compute_units=ct.ComputeUnit.ALL,
    )
    mlmodel.save(str(output_path.with_suffix(".mlmodel")))
    print("✅ Core ML model saved ->", output_path.with_suffix(".mlmodel"))


def main():
    parser = argparse.ArgumentParser(description="Train a plant identifier CNN (MobileNetV2)")
    parser.add_argument("--data_dir", required=True, help="Path to dataset root (ImageFolder style)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--model_out", default="models/plant_identifier.pth")
    parser.add_argument("--export_coreml", action="store_true")
    parser.add_argument("--balanced", action="store_true", help="Use weighted random sampler for class imbalance")
    parser.add_argument("--amp", action="store_true", help="Enable mixed-precision (CUDA only)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--arch", default="mobilenet_v2", help="Model architecture: mobilenet_v2 | efficientnet_b0 | efficientnet_b2 | resnet50")
    parser.add_argument("--img_sizes", default="256,299", help="Comma-separated progressive image sizes (e.g. 224,256,299)")
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--mixup_alpha", type=float, default=0.0, help="Alpha for mixup Beta distribution (0 disables)")
    parser.add_argument("--cutmix_alpha", type=float, default=0.0, help="Alpha for cutmix Beta distribution (0 disables)")
    args = parser.parse_args()

    seed_everything()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon GPU
    else:
        device = torch.device("cpu")
    print("Using device:", device)
    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, num_classes = create_loaders(
        Path(args.data_dir),
        batch_size=args.batch_size,
        val_split=0.2,
        num_workers=args.num_workers,
        balanced=args.balanced,
    )

    # get model by arch
    def build_model(arch: str, num_classes: int):
        if arch == "mobilenet_v2":
            m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
            m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
            return m
        elif arch == "efficientnet_b0":
            m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
            return m
        elif arch == "efficientnet_b2":
            m = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
            m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
            return m
        elif arch == "resnet50":
            m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            m.fc = nn.Linear(m.fc.in_features, num_classes)
            return m
        else:
            raise ValueError(f"Unknown arch {arch}")

    model = build_model(args.arch, num_classes)
    model.to(device)

    if args.label_smoothing > 0 and device.type == "mps":
        print("⚠️  label_smoothing is not supported on MPS in PyTorch yet – disabling it.")
        args.label_smoothing = 0.0
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    # parse progressive image sizes
    img_sizes = [int(s) for s in args.img_sizes.split(",")]

    def set_transforms(size: int):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        train_tfms = T.Compose([
            T.RandomResizedCrop(size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
        val_tfms = T.Compose([
            T.Resize(int(size * 1.15)),
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

        # update underlying dataset transforms
        train_loader.dataset.dataset.transform = train_tfms
        val_loader.dataset.dataset.transform = val_tfms

    # helper for mixup/cutmix
    def rand_bbox(W, H, lam):
        cut_rat = math.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        # progressive resizing
        size_idx = min(int((epoch - 1) / (args.epochs / len(img_sizes))), len(img_sizes) - 1)
        current_size = img_sizes[size_idx]
        set_transforms(current_size)

        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            args.amp,
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, args.amp)
        scheduler.step()

        print(
            f"Train loss: {train_loss:.4f} | acc: {train_acc:.3f} | "
            f"Val loss: {val_loss:.4f} | acc: {val_acc:.3f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state": model.state_dict(), "classes": train_loader.dataset.dataset.classes}, model_out)
            print("💾 Saved best model ->", model_out)

    if args.export_coreml:
        checkpoint = torch.load(model_out, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        export_to_coreml(model, model_out, checkpoint["classes"])


if __name__ == "__main__":
    main() 
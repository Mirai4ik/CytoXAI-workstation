from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

ROOT_DIR = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0, str(ROOT_DIR))

from src.config import CLASS_NAMES, IMAGE_SIZE, NORMALIZATION_MEAN, NORMALIZATION_STD, RAW_CLASS_TO_DISPLAY
from src.model import build_model

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def normalize_name(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def discover_class_dirs(data_dir: Path) -> dict[str, Path]:
    candidates = [p for p in data_dir.iterdir() if p.is_dir()]
    result: dict[str, Path] = {}

    for folder in candidates:
        display = RAW_CLASS_TO_DISPLAY.get(folder.name, RAW_CLASS_TO_DISPLAY.get(normalize_name(folder.name), folder.name))
        if display in CLASS_NAMES:
            result[display] = folder

    if not result:
        raise FileNotFoundError(
            f"Не знайдено директорії класів у {data_dir}. Очікувано папки класів SIPaKMeD."
        )

    return result


def collect_samples(data_dir: Path) -> tuple[list[Path], list[int], dict[str, int]]:
    class_dirs = discover_class_dirs(data_dir)
    class_to_idx = {class_name: idx for idx, class_name in enumerate(CLASS_NAMES)}
    image_paths: list[Path] = []
    labels: list[int] = []

    for class_name in CLASS_NAMES:
        folder = class_dirs.get(class_name)
        if folder is None:
            continue
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                image_paths.append(path)
                labels.append(class_to_idx[class_name])

    if not image_paths:
        raise FileNotFoundError(f"Не знайдено зображень у {data_dir}.")

    return image_paths, labels, class_to_idx


class CytoImageDataset(Dataset):
    def __init__(self, paths: list[Path], labels: list[int], augment: bool = False) -> None:
        self.paths = paths
        self.labels = labels
        self.augment = augment

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.paths[index]).convert("RGB")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)

        if self.augment and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        array = np.asarray(image).astype(np.float32) / 255.0
        mean = np.array(NORMALIZATION_MEAN, dtype=np.float32)
        std = np.array(NORMALIZATION_STD, dtype=np.float32)
        array = (array - mean) / std
        tensor = torch.from_numpy(array.transpose(2, 0, 1)).float()
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return tensor, label


def clean_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        key = key.replace("module.", "", 1) if key.startswith("module.") else key
        key = key.replace("model.", "", 1) if key.startswith("model.") else key
        cleaned[key] = value
    return cleaned


def load_backbone_weights(model: nn.Module, checkpoint_path: Path) -> None:
    if not checkpoint_path.exists():
        return

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    elif isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    elif isinstance(payload, dict):
        state_dict = payload
    else:
        return

    state_dict = clean_state_dict_keys(state_dict)
    model.load_state_dict(state_dict, strict=False)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item()) * images.size(0)
        predictions = logits.argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        total += int(labels.numel())

    return total_loss / max(total, 1), correct / max(total, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune SE-ResNet18 Attention для CytoXAI Workstation.")
    parser.add_argument("--data-dir", type=Path, default=ROOT_DIR / "data" / "processed" / "sipakmed_5class")
    parser.add_argument("--model", type=str, default="se_resnet18", choices=["se_resnet18", "resnet18"])
    parser.add_argument("--init-checkpoint", type=Path, default=ROOT_DIR / "models" / "cyto_classifier.pth")
    parser.add_argument("--output", type=Path, default=ROOT_DIR / "models" / "cyto_classifier_attention.pth")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    paths, labels, class_to_idx = collect_samples(args.data_dir)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths,
        labels,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=labels,
    )

    train_loader = DataLoader(
        CytoImageDataset(train_paths, train_labels, augment=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        CytoImageDataset(val_paths, val_labels, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(CLASS_NAMES), model_name=args.model)
    load_backbone_weights(model, args.init_checkpoint)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_acc = -1.0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_name": args.model,
                    "num_classes": len(CLASS_NAMES),
                    "class_to_idx": class_to_idx,
                    "state_dict": model.state_dict(),
                    "best_val_accuracy": best_val_acc,
                    "dataset_dir": str(args.data_dir),
                },
                args.output,
            )
            print(f"Saved best checkpoint: {args.output}")


if __name__ == "__main__":
    main()

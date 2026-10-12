from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset

ROOT_DIR = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    CLASS_NAMES,
    IMAGE_SIZE,
    MODEL_COMPARISON_PATH,
    MODEL_PATH,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
    RAW_CLASS_TO_DISPLAY,
)
from src.model import build_model, get_model_display_name, model_has_attention

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def normalize_name(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def discover_class_dirs(data_dir: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for folder in data_dir.iterdir():
        if not folder.is_dir():
            continue
        display = RAW_CLASS_TO_DISPLAY.get(folder.name, RAW_CLASS_TO_DISPLAY.get(normalize_name(folder.name), folder.name))
        if display in CLASS_NAMES:
            result[display] = folder
    if not result:
        raise FileNotFoundError(f"Не знайдено директорії класів у {data_dir}.")
    return result


def collect_samples(data_dir: Path) -> tuple[list[Path], list[int]]:
    class_dirs = discover_class_dirs(data_dir)
    class_to_idx = {class_name: idx for idx, class_name in enumerate(CLASS_NAMES)}
    paths: list[Path] = []
    labels: list[int] = []
    for class_name in CLASS_NAMES:
        folder = class_dirs.get(class_name)
        if folder is None:
            continue
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                paths.append(path)
                labels.append(class_to_idx[class_name])
    if not paths:
        raise FileNotFoundError(f"Не знайдено зображень у {data_dir}.")
    return paths, labels


class CytoImageDataset(Dataset):
    def __init__(self, paths: list[Path], labels: list[int]) -> None:
        self.paths = paths
        self.labels = labels

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.paths[index]).convert("RGB")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
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


def load_checkpoint(checkpoint_path: Path) -> tuple[dict[str, torch.Tensor], str, int]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        return clean_state_dict_keys(payload["state_dict"]), str(payload.get("model_name", "resnet18")), int(payload.get("num_classes", len(CLASS_NAMES)))
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return clean_state_dict_keys(payload["model_state_dict"]), str(payload.get("model_name", "resnet18")), int(payload.get("num_classes", len(CLASS_NAMES)))
    if isinstance(payload, dict):
        return clean_state_dict_keys(payload), "resnet18", len(CLASS_NAMES)
    raise ValueError("Непідтримуваний checkpoint.")


def evaluate_checkpoint(checkpoint_path: Path, model_name: str, data_dir: Path, batch_size: int) -> dict:
    paths, labels = collect_samples(data_dir)
    loader = DataLoader(CytoImageDataset(paths, labels), batch_size=batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dict, checkpoint_model_name, num_classes = load_checkpoint(checkpoint_path)
    model = build_model(num_classes=num_classes, model_name=model_name)
    incompatible = model.load_state_dict(state_dict, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    real_missing = [key for key in incompatible.missing_keys if ".attention." not in key and not key.startswith("target_layer.")]
    if real_missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch. Missing={real_missing[:20]}, unexpected={unexpected[:20]}")

    model.to(device)
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for images, batch_labels in loader:
            logits = model(images.to(device))
            predictions = logits.argmax(dim=1).cpu().numpy().tolist()
            y_pred.extend(predictions)
            y_true.extend(batch_labels.numpy().tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred).tolist()

    return {
        "architecture": get_model_display_name(model_name),
        "family": "CNN + Attention" if model_has_attention(model_name) else "CNN baseline",
        "checkpoint": str(checkpoint_path),
        "checkpoint_model_name": checkpoint_model_name,
        "status": "evaluated",
        "accuracy": round(float(accuracy), 6),
        "precision_macro": round(float(precision), 6),
        "recall_macro": round(float(recall), 6),
        "f1_macro": round(float(f1), 6),
        "samples": len(y_true),
        "class_report": report,
        "confusion_matrix": matrix,
        "note": "Missing SE attention keys are allowed when evaluating ResNet18 checkpoint through SE-ResNet18 identity adapter.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CytoXAI models and export model comparison JSON.")
    parser.add_argument("--data-dir", type=Path, default=ROOT_DIR / "data" / "processed" / "sipakmed_5class" / "test")
    parser.add_argument("--checkpoint", type=Path, default=MODEL_PATH)
    parser.add_argument("--model", type=str, default="se_resnet18", choices=["se_resnet18", "resnet18"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=MODEL_COMPARISON_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_checkpoint(args.checkpoint, args.model, args.data_dir, args.batch_size)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_dir": str(args.data_dir),
        "models": [result],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

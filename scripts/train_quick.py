from __future__ import annotations

import argparse
from pathlib import Path
from typing import list

from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from src.config import CLASS_NAMES, IMAGE_SIZE, MODEL_PATH
from src.model import build_model
from src.preprocessing import image_to_tensor


class FolderDataset(Dataset):
    def __init__(self, root: Path):
        self.samples: list[tuple[Path, int]] = []
        for idx, class_name in enumerate(CLASS_NAMES):
            class_dir = root / class_name
            if not class_dir.exists():
                # Try safe folder names too.
                class_dir = root / class_name.replace("-", "_").replace("/", "_")
            if not class_dir.exists():
                continue
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
                for path in class_dir.glob(ext):
                    self.samples.append((path, idx))
        if not self.samples:
            raise RuntimeError(f"No images found in {root}. Expected class folders: {CLASS_NAMES}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        tensor = image_to_tensor(image, IMAGE_SIZE).squeeze(0)
        return tensor, torch.tensor(label, dtype=torch.long)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    dataset = FolderDataset(args.data_dir)
    val_size = max(1, int(len(dataset) * 0.15))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(CLASS_NAMES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * images.size(0)
            correct += int((logits.argmax(1) == labels).sum().item())
            total += images.size(0)

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                val_correct += int((logits.argmax(1) == labels).sum().item())
                val_total += images.size(0)

        print(
            f"Epoch {epoch:02d}: loss={train_loss / max(1, total):.4f}, "
            f"train_acc={correct / max(1, total):.3f}, val_acc={val_correct / max(1, val_total):.3f}"
        )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "class_names": CLASS_NAMES}, MODEL_PATH)
    print(f"Saved checkpoint to {MODEL_PATH}")


if __name__ == "__main__":
    main()

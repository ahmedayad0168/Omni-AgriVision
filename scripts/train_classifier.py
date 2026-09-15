"""
Train plant disease classifier (EfficientNet-B0 / ResNet50).

Usage:
    python scripts/train_classifier.py --data-dir data/PlantVillage --epochs 30 --batch-size 32 --input-size 224 --model-name efficientnet_b0

Supports two dataset layouts:
    1. data_dir/train/<class_name>/*.jpg  +  data_dir/val/<class_name>/*.jpg
    2. data_dir/<class_name>/*.jpg        (auto-split 80/20)
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PlantDiseaseDataset(Dataset):
    """Dataset for plant disease classification."""

    def __init__(self, image_paths: List[Path], labels: List[int],
                 class_names: List[str], transform=None, input_size: int = 224):
        self.image_paths = image_paths
        self.labels = labels
        self.class_names = class_names
        self.transform = transform
        self.input_size = input_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(str(img_path))
        if image is None:
            # Return a black image so training doesn't crash on bad files
            image = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self.input_size, self.input_size))

        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx]


# ---------------------------------------------------------------------------
# Dataloader builder — handles both layouts
# ---------------------------------------------------------------------------
def _collect_images(root: Path, class_names: List[str],
                    class_to_idx: Dict[str, int]) -> Tuple[List[Path], List[int]]:
    paths, labels = [], []
    for class_name in class_names:
        class_dir = root / class_name
        if not class_dir.is_dir():
            continue
        for ext in IMG_EXTS:
            for img in class_dir.glob(ext):
                paths.append(img)
                labels.append(class_to_idx[class_name])
    return paths, labels


def create_dataloaders(data_dir: str, batch_size: int = 16, input_size: int = 160,
                       val_split: float = 0.2) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Create train/val dataloaders. Detects layout automatically."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise RuntimeError(f"Data directory not found: {data_dir}")

    has_train_val = (data_path / "train").is_dir() and (data_path / "val").is_dir()

    if has_train_val:
        train_root = data_path / "train"
        val_root = data_path / "val"
        layout = "train/val split"
    else:
        train_root = data_path
        val_root = None
        layout = "flat (auto-split)"

    # Discover class names from train root
    class_names = sorted([d.name for d in train_root.iterdir() if d.is_dir()])
    if not class_names:
        raise RuntimeError(
            f"No class folders found in {train_root}. "
            f"Expected {train_root}/<class_name>/*.jpg"
        )
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    # Collect images
    if has_train_val:
        train_paths, train_labels = _collect_images(train_root, class_names, class_to_idx)
        val_paths, val_labels = _collect_images(val_root, class_names, class_to_idx)
        if len(train_paths) == 0:
            raise RuntimeError(f"No training images found in {train_root}")
        if len(val_paths) == 0:
            raise RuntimeError(f"No validation images found in {val_root}")
    else:
        all_paths, all_labels = _collect_images(train_root, class_names, class_to_idx)
        if len(all_paths) == 0:
            raise RuntimeError(
                f"No images found in {data_dir}. Expected "
                f"{data_dir}/<class_name>/*.jpg or {data_dir}/train/<class_name>/*.jpg"
            )
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            all_paths, all_labels, test_size=val_split,
            stratify=all_labels, random_state=42
        )

    logger.info(f"Detected layout: {layout}")
    logger.info(f"Classes: {len(class_names)}")
    logger.info(f"Train images: {len(train_paths)} | Val images: {len(val_paths)}")

    # Transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(input_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = PlantDiseaseDataset(train_paths, train_labels, class_names,
                                        train_transform, input_size)
    val_dataset = PlantDiseaseDataset(val_paths, val_labels, class_names,
                                      val_transform, input_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, class_names


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------
def _build_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights="DEFAULT")
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(weights="DEFAULT")
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == "resnet18":
        model = models.resnet18(weights="DEFAULT")
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_classifier(
    data_dir: str,
    output_path: str = "models/classification/efficientnet_disease.pt",
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    input_size: int = 160,
    model_name: str = "efficientnet_b0",
    early_stop_patience: int = 5,
) -> Dict[str, Any]:
    """Train a plant disease classifier."""

    train_loader, val_loader, class_names = create_dataloaders(
        data_dir, batch_size, input_size
    )
    num_classes = len(class_names)

    # Model
    model = _build_model(model_name, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info(f"Using device: {device}")
    logger.info(f"Model: {model_name} | Classes: {num_classes}")

    # Loss / optimizer / scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5
    )

    # Mixed precision (only on CUDA)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Training state
    best_val_acc = 0.0
    epochs_without_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        # ---------- Train ----------
        model.train()
        epoch_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [train]", leave=False)

        for images, labels in train_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = epoch_loss / max(1, len(train_loader))

        # ---------- Validate ----------
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / max(1, len(val_loader))
        val_acc = 100.0 * correct / max(1, total)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)

        scheduler.step(avg_val_loss)

        logger.info(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={avg_val_loss:.4f} | "
            f"val_acc={val_acc:.2f}%"
        )

        # ---------- Save best ----------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "class_names": class_names,
                "num_classes": num_classes,
                "input_size": input_size,
                "model_name": model_name,
            }, output_path)
            logger.info(f"  ✔ Best model saved (val_acc={val_acc:.2f}%) → {output_path}")
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= early_stop_patience:
                logger.info(f"Early stopping after {early_stop_patience} epochs without improvement.")
                break

    logger.info(f"Training complete. Best val acc: {best_val_acc:.2f}%")
    logger.info(f"Model saved to: {output_path}")

    return {
        "best_val_acc": best_val_acc,
        "history": history,
        "class_names": class_names,
        "num_classes": num_classes,
        "output_path": output_path,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args():
    parser = argparse.ArgumentParser(description="Train plant disease classifier")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Root dataset dir. Supports train/val subfolders or flat class folders.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--input-size", type=int, default=160)
    parser.add_argument("--model-name", type=str, default="efficientnet_b0",
                        choices=["efficientnet_b0", "resnet50", "resnet18"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", type=str,
                        default="models/classification/efficientnet_disease.pt")
    parser.add_argument("--early-stop-patience", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    results = train_classifier(
        data_dir=args.data_dir,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        input_size=args.input_size,
        model_name=args.model_name,
        early_stop_patience=args.early_stop_patience,
    )

    print("\n" + "=" * 60)
    print(f"Training finished.")
    print(f"  Best validation accuracy: {results['best_val_acc']:.2f}%")
    print(f"  Number of classes:        {results['num_classes']}")
    print(f"  Model saved to:           {results['output_path']}")
    print("=" * 60)
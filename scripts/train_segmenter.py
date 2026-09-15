"""
Train disease segmentation model (DeepLabV3+).

Usage (flat layout):
    python scripts/train_segmenter.py --train-image-dir data/aug_data/images --train-mask-dir  data/aug_data/masks --epochs 20 --batch-size 4 --img-size 256 --encoder resnet18

Usage (train/val layout):
    python scripts/train_segmenter.py --root data/aug_data --epochs 20 --batch-size 4 --img-size 256 --encoder resnet18

Supported layouts:
    1. Flat:      <img_dir>/*.jpg  and  <mask_dir>/*.png  (matched by filename stem)
    2. Train/val: <root>/train/images + <root>/train/masks
                  <root>/val/images   + <root>/val/masks
    3. Auto-split: flat dataset split 80/20 into train/val
"""

import argparse
import logging
import random
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

import segmentation_models_pytorch as smp

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DiseaseSegmentationDataset(Dataset):
    """Pairs images and masks by filename stem."""

    def __init__(self, image_paths: List[Path], mask_paths: List[Path],
                 img_size: int = 256, augment: bool = False):
        assert len(image_paths) == len(mask_paths), "image/mask count mismatch"
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.image_paths[idx]))
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_size, self.img_size))

        msk = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
        if msk is None:
            msk = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        else:
            msk = cv2.resize(msk, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        # Binarize mask: anything > 127 is disease (class 1)
        msk = (msk > 127).astype(np.uint8)

        # Optional simple augmentation
        if self.augment:
            if random.random() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1])
                msk = np.ascontiguousarray(msk[:, ::-1])

        # To tensors
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_t = (img_t - mean) / std

        msk_t = torch.from_numpy(msk).long()
        return img_t, msk_t


# ---------------------------------------------------------------------------
# Pairing helper — match images and masks by stem
# ---------------------------------------------------------------------------
def _pair_images_masks(image_dir: Path, mask_dir: Path) -> Tuple[List[Path], List[Path]]:
    """Match images to masks by filename stem. Warns on unmatched."""
    if not image_dir.is_dir():
        raise RuntimeError(f"Image dir not found: {image_dir}")
    if not mask_dir.is_dir():
        raise RuntimeError(f"Mask dir not found: {mask_dir}")

    masks_by_stem = {}
    for ext in IMG_EXTS:
        for m in mask_dir.glob(f"*{ext}"):
            masks_by_stem[m.stem] = m

    img_paths, msk_paths = [], []
    missing_masks = 0

    for ext in IMG_EXTS:
        for img in image_dir.glob(f"*{ext}"):
            m = masks_by_stem.get(img.stem)
            if m is None:
                missing_masks += 1
                continue
            img_paths.append(img)
            msk_paths.append(m)

    if missing_masks:
        logger.warning(f"Skipped {missing_masks} images with no matching mask.")
    if len(img_paths) == 0:
        raise RuntimeError(
            f"No image/mask pairs found.\n"
            f"  image_dir: {image_dir}\n"
            f"  mask_dir:  {mask_dir}\n"
            f"  Ensure filenames match by stem (e.g. img001.jpg <-> img001.png)."
        )
    return img_paths, msk_paths


# ---------------------------------------------------------------------------
# Dataloader builder
# ---------------------------------------------------------------------------
def build_dataloaders(
    train_image_dir: Optional[str],
    train_mask_dir: Optional[str],
    val_image_dir: Optional[str],
    val_mask_dir: Optional[str],
    root: Optional[str],
    batch_size: int,
    img_size: int,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Build train and (optionally) val dataloaders."""

    # Layout 1: root with train/val subfolders
    if root:
        root_p = Path(root)
        train_img = root_p / "train" / "images"
        train_msk = root_p / "train" / "masks"
        val_img = root_p / "val" / "images"
        val_msk = root_p / "val" / "masks"
        if not train_img.is_dir():
            # maybe images/masks directly under root
            train_img = root_p / "images"
            train_msk = root_p / "masks"
            val_img = val_msk = None
    else:
        train_img = Path(train_image_dir) if train_image_dir else None
        train_msk = Path(train_mask_dir) if train_mask_dir else None
        val_img = Path(val_image_dir) if val_image_dir else None
        val_msk = Path(val_mask_dir) if val_mask_dir else None

    if train_img is None or train_msk is None:
        raise RuntimeError("Provide either --root or --train-image-dir + --train-mask-dir")

    train_imgs, train_msks = _pair_images_masks(train_img, train_msk)
    logger.info(f"Train pairs: {len(train_imgs)}")

    train_ds = DiseaseSegmentationDataset(train_imgs, train_msks, img_size, augment=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)

    val_loader = None
    if val_img and val_img.is_dir() and val_msk and val_msk.is_dir():
        val_imgs, val_msks = _pair_images_masks(val_img, val_msk)
        logger.info(f"Val pairs:   {len(val_imgs)}")
        val_ds = DiseaseSegmentationDataset(val_imgs, val_msks, img_size, augment=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=0, pin_memory=True)

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_segmenter(
    train_image_dir: Optional[str] = None,
    train_mask_dir: Optional[str] = None,
    val_image_dir: Optional[str] = None,
    val_mask_dir: Optional[str] = None,
    root: Optional[str] = None,
    output_path: str = "models/segmentation/deeplabv3plus_disease.pt",
    epochs: int = 20,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    img_size: int = 256,
    encoder_name: str = "resnet18",
    encoder_weights: str = "imagenet",
    num_classes: int = 2,
    early_stop_patience: int = 5,
):
    train_loader, val_loader = build_dataloaders(
        train_image_dir, train_mask_dir,
        val_image_dir, val_mask_dir,
        root, batch_size, img_size,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Encoder: {encoder_name} | Classes: {num_classes} | Image size: {img_size}")

    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5
    )

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_metric = None
    epochs_without_improve = 0

    for epoch in range(epochs):
        # ---------- Train ----------
        model.train()
        train_loss = 0.0
        bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]", leave=False)
        for imgs, msks in bar:
            imgs = imgs.to(device, non_blocking=True)
            msks = msks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(imgs)
                loss = criterion(out, msks)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss / max(1, len(train_loader))

        # ---------- Validate ----------
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for imgs, msks in val_loader:
                    imgs = imgs.to(device, non_blocking=True)
                    msks = msks.to(device, non_blocking=True)
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        out = model(imgs)
                        loss = criterion(out, msks)
                    val_loss += loss.item()
                    preds = torch.argmax(out, dim=1)
                    correct += (preds == msks).sum().item()
                    total += msks.numel()

            avg_val_loss = val_loss / max(1, len(val_loader))
            pixel_acc = 100.0 * correct / max(1, total)
            scheduler.step(avg_val_loss)

            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"train_loss={avg_train_loss:.4f} | "
                f"val_loss={avg_val_loss:.4f} | "
                f"pixel_acc={pixel_acc:.2f}%"
            )

            is_best = avg_val_loss < best_val_loss
            if is_best:
                best_val_loss = avg_val_loss
                best_metric = {"val_loss": avg_val_loss, "pixel_acc": pixel_acc}
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1
        else:
            logger.info(f"Epoch {epoch+1}/{epochs} | train_loss={avg_train_loss:.4f} (no val)")
            is_best = avg_train_loss < best_val_loss
            if is_best:
                best_val_loss = avg_train_loss
                best_metric = {"train_loss": avg_train_loss}
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1

        # ---------- Save ----------
        if is_best:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_val_loss,
                "metric": best_metric,
                "encoder_name": encoder_name,
                "num_classes": num_classes,
                "img_size": img_size,
            }, output_path)
            logger.info(f"  ✔ Best model saved → {output_path}")

        if epochs_without_improve >= early_stop_patience:
            logger.info(f"Early stopping after {early_stop_patience} epochs without improvement.")
            break

    logger.info("Training complete.")
    logger.info(f"Best metric: {best_metric}")
    logger.info(f"Model saved to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args():
    p = argparse.ArgumentParser(description="Train disease segmentation (DeepLabV3+)")
    # Data sources (mutually exclusive in practice)
    p.add_argument("--root", type=str, default=None,
                   help="Root dir with train/{images,masks} and val/{images,masks}, "
                        "or root/{images,masks}")
    p.add_argument("--train-image-dir", type=str, default=None)
    p.add_argument("--train-mask-dir", type=str, default=None)
    p.add_argument("--val-image-dir", type=str, default=None)
    p.add_argument("--val-mask-dir", type=str, default=None)

    # Training hyperparams
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--encoder", type=str, default="resnet18",
                   help="resnet18 | resnet34 | resnet50 | mobilenet_v2 | efficientnet-b0")
    p.add_argument("--num-classes", type=int, default=2,
                   help="2 = background + disease")
    p.add_argument("--output", type=str,
                   default="models/segmentation/deeplabv3plus_disease.pt")
    p.add_argument("--early-stop-patience", type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.root is None and (args.train_image_dir is None or args.train_mask_dir is None):
        raise SystemExit(
            "Provide --root, or --train-image-dir + --train-mask-dir."
        )

    train_segmenter(
        train_image_dir=args.train_image_dir,
        train_mask_dir=args.train_mask_dir,
        val_image_dir=args.val_image_dir,
        val_mask_dir=args.val_mask_dir,
        root=args.root,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        img_size=args.img_size,
        encoder_name=args.encoder,
        num_classes=args.num_classes,
        early_stop_patience=args.early_stop_patience,
    )
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import cv2
import numpy as np
import torch

from .config import IMAGENET_MEAN, IMAGENET_STD


@dataclass
class LeafInstance:
    crop: np.ndarray         
    mask: np.ndarray        
    bbox: Tuple[int, int, int, int]   
    leaf_area_px: int      
    mask_cov_pct: float      


def segment_binary(model, crop_bgr: np.ndarray, device: torch.device,
                   size: Tuple[int, int], threshold: float) -> np.ndarray:
    h, w = crop_bgr.shape[:2]
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    x = (cv2.resize(rgb, size).astype(np.float32) / 255.0 - mean) / std
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(x)).squeeze().cpu().numpy()
    mask = (prob > threshold).astype(np.uint8) * 255
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


def leaf_mask_fallback(crop_bgr: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(crop_bgr.astype(np.float32))
    exg = np.clip(2 * g - r - b, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(exg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected blob (the leaf), removing specks/neighbours."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def extract_leaves(
    frame_bgr: np.ndarray,
    boxes: Iterable[Tuple[int, int, int, int]],
    leaf_seg_model,
    device: torch.device,
    cfg,
) -> List[LeafInstance]:
    H, W = frame_bgr.shape[:2]
    kernel = np.ones((3, 3), np.uint8)
    leaves: List[LeafInstance] = []

    for (x1, y1, x2, y2) in boxes:
        x1, y1 = max(int(x1), 0), max(int(y1), 0)
        x2, y2 = min(int(x2), W), min(int(y2), H)
        if (x2 - x1) * (y2 - y1) < cfg.min_box_area:
            continue
        box = frame_bgr[y1:y2, x1:x2]
        if box.size == 0:
            continue

        if leaf_seg_model is not None:
            mask = segment_binary(leaf_seg_model, box, device,
                                  cfg.seg_size, cfg.leaf_seg_threshold)
        else:
            mask = leaf_mask_fallback(box)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = _largest_component(mask)

        ys, xs = np.where(mask > 0)
        if xs.size == 0:
            continue
        lx1, lx2, ly1, ly2 = xs.min(), xs.max(), ys.min(), ys.max()
        mask_tight = mask[ly1:ly2 + 1, lx1:lx2 + 1]
        leaf_area = int(cv2.countNonZero(mask_tight))
        if leaf_area < cfg.min_leaf_area:
            continue

        box_tight = box[ly1:ly2 + 1, lx1:lx2 + 1]
        isolated = cv2.bitwise_and(box_tight, box_tight, mask=mask_tight)
        box_pixels = mask_tight.shape[0] * mask_tight.shape[1]
        cov = 100.0 * leaf_area / box_pixels if box_pixels else 0.0

        leaves.append(LeafInstance(
            crop=isolated,
            mask=mask_tight,
            bbox=(int(x1 + lx1), int(y1 + ly1), int(x1 + lx2 + 1), int(y1 + ly2 + 1)),
            leaf_area_px=leaf_area,
            mask_cov_pct=cov,
        ))
    return leaves


def boxes_from_yolo(detector, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    boxes: List[Tuple[int, int, int, int]] = []
    for result in detector(frame_bgr, stream=True, verbose=False):
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            boxes.append((int(x1), int(y1), int(x2), int(y2)))
    return boxes

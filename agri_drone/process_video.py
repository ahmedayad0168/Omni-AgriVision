"""Headless orchestrator: drone video -> per-leaf crops/masks/overlays + metadata.csv.

Each detected leaf is saved as its OWN crop (FIX 1). The CSV includes `mask_cov_%`
so the growth module never crashes (fixes the original KeyError).
"""
# py -3.11 -m agri_drone.cli extract  --video video3.mp4 --models models --output dataset
# py -3.11 -m agri_drone.cli generate --kind gan --weights models/drone_generator.pth --out synthetic_gan --per-class 32
# py -3.11 -m agri_drone.cli generate --kind vae --weights models/vae_healthy.pth --out synthetic_vae --num 5

from __future__ import annotations

import csv
import os
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch

from .config import Config
from .leaf_extraction import boxes_from_yolo, extract_leaves
from .pesticide import PesticideQuantifier

CSV_HEADER = [
    "frame", "leaf", "class", "conf_%", "x1", "y1", "x2", "y2", "w", "h",
    "mask_cov_%", "severity_%", "leaf_area_cm2", "disease_area_cm2",
    "pesticide_dosage", "chemical", "unit", "crop_path", "mask_path",
    "overlay_path", "time_sec",
]


def make_overlay(crop_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    out = crop_bgr.copy()
    green = np.zeros_like(crop_bgr)
    green[:, :, 1] = 255
    m = mask > 127
    out[m] = (crop_bgr[m] * (1 - alpha) + green[m] * alpha).astype(np.uint8)
    return out


def draw_box(frame, label, x1, y1, x2, y2) -> None:
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 60), 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, max(y1 - th - 8, 0)), (x1 + tw + 8, y1), (0, 200, 60), -1)
    cv2.putText(frame, label, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1)


def _class_dirs(base: str, class_name: str) -> Tuple[str, str, str]:
    dirs = {k: os.path.join(base, k, class_name) for k in ("crops", "masks", "overlays")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs["crops"], dirs["masks"], dirs["overlays"]


def process_video(video_path: str, cfg: Config, output_dir: str = "dataset",
                  video_out: Optional[str] = "output_annotated.mp4",
                  display: bool = False) -> str:
    """Process the whole video and return the path to the written metadata.csv."""
    from . import models

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    detector = models.build_detector(cfg.detector_path, str(device))
    leaf_seg = models.build_leaf_seg(cfg.leaf_seg_path, device)
    disease_seg = models.build_disease_seg(cfg.disease_seg_path, device)
    classifier = models.build_classifier(cfg.classifier_path)
    quantifier = PesticideQuantifier(classifier, disease_seg, device, cfg)
    print("All models loaded.\n")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(output_dir, exist_ok=True)
    writer = None
    if video_out:
        writer = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    csv_path = os.path.join(output_dir, "metadata.csv")
    class_cache: Dict[str, Tuple[str, str, str]] = {}
    frame_i = saved = 0
    t0 = time.time()

    with open(csv_path, "w", newline="") as f:
        csv_w = csv.writer(f)
        csv_w.writerow(CSV_HEADER)

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_i += 1
            if frame_i % cfg.frame_skip != 0:
                if writer:
                    writer.write(frame)
                continue

            ann = frame.copy()
            boxes = boxes_from_yolo(detector, frame)
            leaves = extract_leaves(frame, boxes, leaf_seg, device, cfg)

            for leaf_no, leaf in enumerate(leaves, start=1):
                analysis = quantifier.analyze_leaf(leaf.crop, leaf.mask)
                if analysis is None:
                    continue

                crop_dir, mask_dir, ov_dir = class_cache.setdefault(
                    analysis.disease, _class_dirs(output_dir, analysis.disease))
                stem = f"frame{frame_i:06d}_leaf{leaf_no:02d}"
                cp = os.path.join(crop_dir, f"{stem}.jpg")
                mp = os.path.join(mask_dir, f"{stem}_leaf_mask.png")
                op = os.path.join(ov_dir, f"{stem}_overlay.jpg")

                cv2.imwrite(cp, leaf.crop)             # one file per leaf (FIX 1)
                cv2.imwrite(mp, leaf.mask)
                cv2.imwrite(op, make_overlay(leaf.crop, analysis.disease_mask))

                x1, y1, x2, y2 = leaf.bbox
                csv_w.writerow([
                    frame_i, leaf_no, analysis.disease, f"{analysis.confidence * 100:.2f}",
                    x1, y1, x2, y2, x2 - x1, y2 - y1, f"{leaf.mask_cov_pct:.2f}",
                    f"{analysis.severity * 100:.2f}", f"{analysis.leaf_area_cm2:.2f}",
                    f"{analysis.disease_area_cm2:.2f}", f"{analysis.pesticide_dosage:.4f}",
                    analysis.chemical, analysis.unit, cp, mp, op, f"{frame_i / fps:.3f}",
                ])
                saved += 1
                draw_box(ann, f"{analysis.disease} ({analysis.confidence * 100:.0f}%) | "
                              f"{analysis.pesticide_dosage:.2f}{analysis.unit}", x1, y1, x2, y2)

            if writer:
                cv2.putText(ann, f"Frame {frame_i}/{total} | Saved {saved}",
                            (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)
                writer.write(ann)
            if display:
                cv2.imshow("Analysis (q to quit)", ann)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    if writer:
        writer.release()
    if display:
        cv2.destroyAllWindows()
    print(f"\nDone. Frames={frame_i} | Saved leaves={saved} | {time.time() - t0:.1f}s")
    print(f"CSV: {os.path.abspath(csv_path)}")
    return csv_path
    
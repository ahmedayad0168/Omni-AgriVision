from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch

from .config import Config
from .leaf_extraction import segment_binary


@dataclass
class LeafAnalysis:
    status: str               
    disease: str
    confidence: float
    severity: float            
    leaf_area_cm2: float
    disease_area_cm2: float
    pesticide_dosage: float
    chemical: str
    unit: str
    disease_mask: np.ndarray   


def compute_dosage(leaf_area_px: int, lesion_px: int, class_name: str, cfg: Config):
    """Pure pesticide math (no models) — easy to unit test.

    Returns (severity, leaf_area_cm2, disease_area_cm2, dosage, chemical, unit).
    """
    area_per_px = cfg.area_cm2_per_pixel()           # cm^2 per pixel  (= gsd**2)
    leaf_area_cm2 = leaf_area_px * area_per_px
    disease_area_cm2 = lesion_px * area_per_px

    severity = (lesion_px / leaf_area_px) if leaf_area_px > 0 else 0.0
    severity = float(np.clip(severity, 0.0, 1.0))

    rule = cfg.dosage_table.get(class_name, cfg.default_dosage)
    leaf_area_m2 = leaf_area_cm2 / 10_000.0           # 1 m^2 = 10,000 cm^2
    dosage = leaf_area_m2 * rule.rate_per_m2 * severity
    return severity, leaf_area_cm2, disease_area_cm2, dosage, rule.chemical, rule.unit


class PesticideQuantifier:
    """Classify a leaf, segment lesions inside the leaf mask, and compute dosage."""

    def __init__(self, classifier, disease_seg_model, device: torch.device, cfg: Config):
        self.classifier = classifier
        self.disease_seg_model = disease_seg_model
        self.device = device
        self.cfg = cfg

    def analyze_leaf(self, leaf_crop_bgr: np.ndarray, leaf_mask: np.ndarray) -> Optional[LeafAnalysis]:
        if leaf_crop_bgr.size == 0:
            return None

        cls = self.classifier(leaf_crop_bgr, verbose=False)[0]
        class_idx = int(cls.probs.top1)
        name = cls.names[class_idx]
        conf = float(cls.probs.top1conf)
        leaf_area_px = int(cv2.countNonZero(leaf_mask))

        if "healthy" in name.lower():
            return LeafAnalysis(
                status="Healthy", disease=name, confidence=conf, severity=0.0,
                leaf_area_cm2=leaf_area_px * self.cfg.area_cm2_per_pixel(),
                disease_area_cm2=0.0, pesticide_dosage=0.0, chemical="None", unit="",
                disease_mask=np.zeros_like(leaf_mask),
            )

        raw = segment_binary(self.disease_seg_model, leaf_crop_bgr, self.device,
                             self.cfg.seg_size, self.cfg.disease_seg_threshold)
        disease_mask = cv2.bitwise_and(raw, raw, mask=leaf_mask)
        lesion_px = int(cv2.countNonZero(disease_mask))

        severity, leaf_cm2, dis_cm2, dosage, chem, unit = compute_dosage(
            leaf_area_px, lesion_px, name, self.cfg
        )
        return LeafAnalysis(
            status="Diseased", disease=name, confidence=conf, severity=severity,
            leaf_area_cm2=leaf_cm2, disease_area_cm2=dis_cm2, pesticide_dosage=dosage,
            chemical=chem, unit=unit, disease_mask=disease_mask,
        )

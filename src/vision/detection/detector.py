from ultralytics import YOLO
import numpy as np
from typing import List, Dict, Any, Optional
import torch
import logging

import os
from pathlib import Path

logger = logging.getLogger(__name__)

class Detector:
    """YOLO object detector for agricultural scenes."""
    def __init__(self, model_path: Optional[str] = None, conf_threshold: float = 0.25,
                  iou_threshold: float = 0.45, image_size: int = 640, device: str = "auto"):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

        resolved_path = self._resolve_model_path(model_path)
        self.model = YOLO(resolved_path)
        self.model.to(self.device)
        self.class_names = self._load_class_names()

    @staticmethod
    def _resolve_model_path(model_path: Optional[str]) -> str:
        candidates = []
        if model_path:
            candidates.append(model_path)
            if "yolov11" in model_path:
                candidates.append(model_path.replace("yolov11", "yolo11"))

        candidates.extend([
            "models/detection/yolo11m_finetuned.pt",
            "yolo11m.pt",
            "yolo11l.pt",
        ])

        root = Path(__file__).resolve().parents[3]
        for c in candidates:
            p = Path(c)
            if p.exists():
                return str(p)
            p_root = root / c
            if p_root.exists():
                return str(p_root)

        return model_path or "yolo11m.pt"

    @staticmethod
    def _default_class_names() -> Dict[int, str]:
        return {
            0: 'Tomato Bacterial Spot',
            1: 'Tomato Early blight',
            2: 'Tomato Late blight',
            3: 'Tomato Leaf Mold',
            4: 'Tomato Septoria leaf spot',
            5: 'Tomato Spider mite',
            6: 'Tomato Target Spot',
            7: 'Tomato Yellow Leaf Curl Virus',
            8: 'Tomato healthy',
            9: 'Tomato mosaic virus',
        }

    def _load_class_names(self) -> Dict[int, str]:
        """Load class names from the trained model."""
        if hasattr(self.model, 'names') and self.model.names:
            return self.model.names
        return self._default_class_names()

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        results = self.model(image, conf=self.conf_threshold, iou=self.iou_threshold, imgsz=self.image_size, verbose=False)
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                detections.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': conf,
                    'class_id': cls_id,
                    'class_name': self.class_names.get(cls_id, f'class_{cls_id}'),
                    'area': (x2 - x1) * (y2 - y1),
                })
        return detections

    def detect_batch(self, images: List[np.ndarray], batch_size: int = 8) -> List[List[Dict]]:
        all_dets = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            results = self.model(batch, conf=self.conf_threshold, iou=self.iou_threshold, imgsz=self.image_size, verbose=False)
            for r in results:
                dets = []
                if r.boxes:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        dets.append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': float(box.conf[0]),
                            'class_id': int(box.cls[0]),
                            'class_name': self.class_names.get(int(box.cls[0]), f'class_{int(box.cls[0])}')
                        })
                all_dets.append(dets)
        return all_dets

from types import SimpleNamespace
from typing import List, Dict, Any, Optional
import numpy as np
import torch
import logging

from ultralytics.trackers import BYTETracker

logger = logging.getLogger(__name__)


class DetectionBoxes:
    """Adapter class exposing .xyxy, .xywh, .conf, .cls for Ultralytics ByteTracker."""

    def __init__(self, xyxy: torch.Tensor, conf: torch.Tensor, cls: torch.Tensor):
        self.xyxy = torch.as_tensor(xyxy, dtype=torch.float32)
        self.conf = torch.as_tensor(conf, dtype=torch.float32)
        self.cls = torch.as_tensor(cls, dtype=torch.float32)

        if len(self.xyxy) > 0:
            w = self.xyxy[:, 2] - self.xyxy[:, 0]
            h = self.xyxy[:, 3] - self.xyxy[:, 1]
            cx = self.xyxy[:, 0] + w / 2.0
            cy = self.xyxy[:, 1] + h / 2.0
            self.xywh = torch.stack([cx, cy, w, h], dim=1)
        else:
            self.xywh = torch.empty((0, 4), dtype=torch.float32)

    def __getitem__(self, idx):
        return DetectionBoxes(self.xyxy[idx], self.conf[idx], self.cls[idx])

    def __len__(self) -> int:
        return len(self.conf)


class Tracker:
    """Wrapper for ByteTrack object tracking in agricultural video frames."""

    def __init__(
        self,
        track_thresh: float = 0.25,
        high_thresh: float = 0.5,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
        frame_rate: int = 30,
    ):
        args = SimpleNamespace(
            track_thresh=track_thresh,
            track_high_thresh=high_thresh,
            track_low_thresh=max(0.05, track_thresh / 2),
            new_track_thresh=high_thresh,
            match_thresh=match_thresh,
            track_buffer=track_buffer,
            fuse_score=False,
        )
        self.tracker = BYTETracker(args)
        self.frame_id = 0

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.frame_id += 1
        if not detections:
            return []

        xyxy_list = []
        conf_list = []
        cls_list = []

        for d in detections:
            bbox = d.get("bbox", [0, 0, 0, 0])
            xyxy_list.append([float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])])
            conf_list.append(float(d.get("confidence", 1.0)))
            cls_list.append(float(d.get("class_id", 0)))

        boxes = DetectionBoxes(
            xyxy=torch.tensor(xyxy_list, dtype=torch.float32),
            conf=torch.tensor(conf_list, dtype=torch.float32),
            cls=torch.tensor(cls_list, dtype=torch.float32),
        )

        try:
            tracked_array = self.tracker.update(boxes)
        except Exception as e:
            logger.warning(f"ByteTracker update failed: {e}. Falling back to input detections.")
            return [
                {**d, "track_id": d.get("track_id", i + 1)}
                for i, d in enumerate(detections)
            ]

        if tracked_array is None or len(tracked_array) == 0:
            return []

        tracked_dets = []
        for row in tracked_array:
            orig_idx = int(row[7]) if len(row) > 7 else -1
            base_det = detections[orig_idx].copy() if 0 <= orig_idx < len(detections) else {}

            base_det.update({
                "track_id": int(row[4]),
                "bbox": [float(row[0]), float(row[1]), float(row[2]), float(row[3])],
                "confidence": float(row[5]),
                "class_id": int(row[6]),
                "class_name": base_det.get("class_name", f"class_{int(row[6])}"),
            })
            tracked_dets.append(base_det)

        return tracked_dets
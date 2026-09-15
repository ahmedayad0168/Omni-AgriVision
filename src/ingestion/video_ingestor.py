import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class FrameMetadata:
    frame_id: int
    timestamp: datetime
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    altitude: Optional[float] = None
    raw_meta: Dict[str, Any] = field(default_factory=dict)


class VideoIngestor:
    """
    Video ingestion and frame extraction with controlled FPS and metadata.
    Supports DJI/Parrot telemetry (extensible).
    """
    def __init__(self, target_fps: int = 5, frame_interval: int = 6, max_frames: int = 1000,
                 output_dir: Optional[str] = None):
        self.target_fps = target_fps
        self.frame_interval = frame_interval
        self.max_frames = max_frames
        self.output_dir = Path(output_dir) if output_dir else None

    def process(self, video_path: str, save_frames: bool = False) -> Generator[
        tuple[np.ndarray, FrameMetadata], None, None]:
        """Process video and yield frames with metadata."""
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Video: {fps:.2f} FPS, {total_frames} frames")

        interval = max(1, int(fps / self.target_fps))
        frame_count = 0
        extracted = 0

        # Attempt to extract GPS telemetry (placeholder)
        gps_data = self._extract_gps(video_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % interval == 0 and extracted < self.max_frames:
                meta = FrameMetadata(frame_id=extracted, timestamp=datetime.utcnow(),
                    gps_lat=gps_data.get(frame_count, {}).get('lat'),
                    gps_lon=gps_data.get(frame_count, {}).get('lon'),
                    altitude=gps_data.get(frame_count, {}).get('alt')
                )

                if save_frames and self.output_dir:
                    self.output_dir.mkdir(parents=True, exist_ok=True)
                    fname = self.output_dir / f"frame_{extracted:06d}.jpg"
                    cv2.imwrite(str(fname), frame)

                yield frame, meta
                extracted += 1

            frame_count += 1

        cap.release()
        logger.info(f"Extracted {extracted} frames")

    def _extract_gps(self, video_path: Path) -> Dict[int, Dict]:
        """Extract GPS metadata from drone telemetry (exiftool or custom parser)."""
        # Placeholder - use exiftool or parse SRT files
        # For production, implement using subprocess to call exiftool if available.
        return {}
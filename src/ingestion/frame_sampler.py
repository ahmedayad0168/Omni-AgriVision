import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

class FrameSampler:
    """ Utility to sample frames from a video at regular intervals without storing all frames. """
    @staticmethod
    def sample_frames(video_path: str, sample_interval: int = 30, max_frames: int = 100) -> List[np.ndarray]:
        """
        Sample frames every `sample_interval` frames, up to `max_frames`.
        Returns list of frames (BGR).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        frames = []
        frame_count = 0
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % sample_interval == 0:
                frames.append(frame)
            frame_count += 1
        cap.release()
        logger.info(f"Sampled {len(frames)} frames from {video_path}")
        return frames
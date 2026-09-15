import cv2
import numpy as np
from typing import Optional, Tuple

class ImagePreprocessor:
    """ Basic image preprocessing: resize, normalize, augmentations. """
    def __init__(self, target_size: Tuple[int, int] = (640, 640), normalize: bool = True):
        self.target_size = target_size
        self.normalize = normalize

    def process(self, image: np.ndarray) -> np.ndarray:
        """Resize and optionally normalize image."""
        resized = cv2.resize(image, self.target_size)
        if self.normalize:
            resized = resized.astype(np.float32) / 255.0
        return resized

    def batch_process(self, images: list) -> np.ndarray:
        """Process a batch of images."""
        return np.array([self.process(img) for img in images])
import numpy as np
from src.vision.detection.detector import Detector

def test_detector():
    detector = Detector()
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    dets = detector.detect(dummy_img)
    assert isinstance(dets, list)
import numpy as np
from src.vision.classification.classifier import DiseaseClassifier

def test_classifier():
    classifier = DiseaseClassifier()
    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
    result = classifier.classify(dummy_img)
    assert 'class_name' in result
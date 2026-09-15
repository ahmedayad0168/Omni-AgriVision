import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class DiseaseClassifier:
    """Plant disease classifier using EfficientNet/ResNet."""
    CLASS_NAMES = [
        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
        'Apple___healthy', 'Blueberry___healthy', 'Cherry___healthy',
        'Cherry___Powdery_mildew', 'Corn___Cercospora_leaf_spot',
        'Corn___Common_rust', 'Corn___healthy', 'Corn___Northern_Leaf_Blight',
        'Grape___Black_rot', 'Grape___Esca', 'Grape___healthy',
        'Grape___Leaf_blight', 'Orange___Haunglongbing', 'Peach___Bacterial_spot',
        'Peach___healthy', 'Pepper_bell___Bacterial_spot', 'Pepper_bell___healthy',
        'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight',
        'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
        'Strawberry___healthy', 'Strawberry___Leaf_scorch', 'Tomato___Bacterial_spot',
        'Tomato___Early_blight', 'Tomato___healthy', 'Tomato___Late_blight',
        'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
        'Tomato___Spider_mites', 'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
    ]

    def __init__(self, model_path: Optional[str] = None, input_size: int = 224,
                  num_classes: int = 38, confidence_threshold: float = 0.5, device: str = "auto"):
        self.input_size = input_size
        self.num_classes = num_classes
        self.conf_threshold = confidence_threshold
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((input_size, input_size)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _resolve_model_path(self, model_path: Optional[str]) -> Optional[str]:
        if model_path and Path(model_path).exists():
            return model_path
        root = Path(__file__).resolve().parents[3]
        if model_path and (root / model_path).exists():
            return str(root / model_path)
        default_p = root / "models" / "classification" / "efficientnet_disease.pt"
        if default_p.exists():
            return str(default_p)
        if Path("models/classification/efficientnet_disease.pt").exists():
            return "models/classification/efficientnet_disease.pt"
        return model_path

    def _load_model(self, model_path: Optional[str]) -> nn.Module:
        resolved = self._resolve_model_path(model_path)
        has_local_weights = resolved and Path(resolved).exists()
        
        # Avoid internet dependency if local weights are available
        try:
            model = models.efficientnet_b0(weights=None if has_local_weights else 'DEFAULT')
        except Exception:
            model = models.efficientnet_b0(weights=None)

        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, self.num_classes)

        if has_local_weights:
            state = torch.load(resolved, map_location='cpu')
            model.load_state_dict(state['model_state_dict'] if 'model_state_dict' in state else state)
            logger.info(f"Loaded disease classifier weights from {resolved}")
        return model

    def classify(self, image: np.ndarray) -> Dict[str, Any]:
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, dim=1)

        class_id = int(pred[0])
        confidence = float(conf[0])

        # Top-5
        top5_probs, top5_idx = torch.topk(probs, k=min(5, self.num_classes))
        top5 = [{'class_id': int(idx), 'class_name': self.CLASS_NAMES[int(idx)],
                 'confidence': float(prob)} for idx, prob in zip(top5_idx[0], top5_probs[0])]

        return {
            'class_id': class_id,
            'class_name': self.CLASS_NAMES[class_id],
            'confidence': confidence,
            'top5': top5,
            'is_healthy': 'healthy' in self.CLASS_NAMES[class_id].lower()
        }

    def classify_batch(self, images: List[np.ndarray], batch_size: int = 32) -> List[Dict]:
        results = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            tensors = torch.stack([self.transform(img) for img in batch]).to(self.device)
            with torch.no_grad():
                outputs = self.model(tensors)
                probs = torch.softmax(outputs, dim=1)
                confs, preds = torch.max(probs, dim=1)
            for j, (pred, conf) in enumerate(zip(preds, confs)):
                class_id = int(pred)
                results.append({
                    'class_id': class_id,
                    'class_name': self.CLASS_NAMES[class_id],
                    'confidence': float(conf),
                    'is_healthy': 'healthy' in self.CLASS_NAMES[class_id].lower()
                })
        return results
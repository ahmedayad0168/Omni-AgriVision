import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
from torchvision import transforms
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DiseaseSegmenter:
    """Disease segmentation using DeepLabV3+."""
    def __init__(self, model_path: Optional[str] = None, input_size: int = 256, num_classes: int = 2,
                  encoder_name: str = 'resnet18', encoder_weights: str = 'imagenet', device: str = "auto"):
        self.input_size = input_size
        self.num_classes = num_classes
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

        resolved_path = self._resolve_model_path(model_path)
        state = None
        if resolved_path and Path(resolved_path).exists():
            state = torch.load(resolved_path, map_location='cpu')
            if isinstance(state, dict) and 'encoder_name' in state:
                encoder_name = state['encoder_name']
            if isinstance(state, dict) and 'num_classes' in state:
                self.num_classes = state['num_classes']
            if isinstance(state, dict) and 'img_size' in state:
                self.input_size = state['img_size']

        weights = None if state else encoder_weights
        try:
            self.model = smp.DeepLabV3Plus(encoder_name=encoder_name,
                                           encoder_weights=weights, in_channels=3, classes=self.num_classes)
        except Exception:
            self.model = smp.DeepLabV3Plus(encoder_name=encoder_name,
                                           encoder_weights=None, in_channels=3, classes=self.num_classes)

        if state is not None:
            self.model.load_state_dict(state['model_state_dict'] if 'model_state_dict' in state else state)
            logger.info(f"Loaded disease segmenter weights from {resolved_path}")

        self.transform = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((self.input_size, self.input_size)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _resolve_model_path(model_path: Optional[str]) -> Optional[str]:
        if model_path and Path(model_path).exists():
            return model_path
        root = Path(__file__).resolve().parents[3]
        if model_path and (root / model_path).exists():
            return str(root / model_path)
        default_p = root / "models" / "segmentation" / "deeplabv3plus_disease.pt"
        if default_p.exists():
            return str(default_p)
        if Path("models/segmentation/deeplabv3plus_disease.pt").exists():
            return "models/segmentation/deeplabv3plus_disease.pt"
        return model_path

    def segment(self, image: np.ndarray) -> Dict[str, Any]:
        original_size = (image.shape[1], image.shape[0])
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)
            mask = torch.argmax(probs, dim=1)

        mask_np = mask.squeeze(0).cpu().numpy().astype(np.uint8)
        # Resize mask to original size
        mask_resized = np.array(Image.fromarray(mask_np).resize(original_size, Image.NEAREST))

        disease_pixels = np.sum(mask_resized == 1)
        total_pixels = mask_resized.size
        severity = disease_pixels / total_pixels if total_pixels > 0 else 0.0

        return {
            'mask': mask_resized,
            'severity': severity,
            'disease_pixels': int(disease_pixels),
            'total_pixels': int(total_pixels),
            'severity_percentage': severity * 100
        }

    def calculate_severity(self, leaf_mask: np.ndarray, disease_mask: np.ndarray) -> float:
        leaf_pixels = np.sum(leaf_mask > 0)
        disease_pixels = np.sum((disease_mask == 1) & (leaf_mask > 0))
        return disease_pixels / leaf_pixels if leaf_pixels > 0 else 0.0

    def calculate_severity_level(self, severity: float) -> str:
        if severity < 0.05:
            return "Low"
        elif severity < 0.15:
            return "Moderate"
        elif severity < 0.30:
            return "Severe"
        else:
            return "Critical"
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class DosageRule:
    """Pesticide rule for one disease class."""
    rate_per_m2: float   
    chemical: str
    unit: str     


DOSAGE_TABLE: Dict[str, DosageRule] = {
    "Apple___Apple_scab":            DosageRule(0.80, "Captan 80 WDG", "g"),
    "Apple___Black_rot":             DosageRule(0.70, "Captan 80 WDG", "g"),
    "Apple___Cedar_apple_rust":      DosageRule(0.45, "Myclobutanil", "ml"),
    "Cherry_(including_sour)___Powdery_mildew": DosageRule(0.40, "Sulfur WG", "g"),
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": DosageRule(0.60, "Azoxystrobin", "ml"),
    "Corn_(maize)___Common_rust_":   DosageRule(0.55, "Propiconazole", "ml"),
    "Corn_(maize)___Northern_Leaf_Blight": DosageRule(0.65, "Mancozeb", "g"),
    "Grape___Black_rot":             DosageRule(0.70, "Mancozeb", "g"),
    "Grape___Esca_(Black_Measles)":  DosageRule(0.60, "Tebuconazole", "ml"),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": DosageRule(0.55, "Copper Oxychloride", "g"),
    "Orange___Haunglongbing_(Citrus_greening)": DosageRule(0.50, "Imidacloprid", "ml"),
    "Peach___Bacterial_spot":        DosageRule(0.60, "Copper Hydroxide", "g"),
    "Pepper,_bell___Bacterial_spot": DosageRule(0.60, "Copper Hydroxide", "g"),
    "Potato___Early_blight":         DosageRule(0.50, "Chlorothalonil", "ml"),
    "Potato___Late_blight":          DosageRule(0.75, "Ridomil Gold", "ml"),
    "Squash___Powdery_mildew":       DosageRule(0.40, "Sulfur WG", "g"),
    "Strawberry___Leaf_scorch":      DosageRule(0.50, "Captan 80 WDG", "g"),
    "Tomato___Bacterial_spot":       DosageRule(0.60, "Streptomycin Sulfate", "g"),
    "Tomato___Early_blight":         DosageRule(0.50, "Copper Fungicide", "ml"),
    "Tomato___Late_blight":          DosageRule(0.70, "Mancozeb", "ml"),
    "Tomato___Leaf_Mold":            DosageRule(0.55, "Chlorothalonil", "ml"),
    "Tomato___Septoria_leaf_spot":   DosageRule(0.55, "Chlorothalonil", "ml"),
    "Tomato___Spider_mites Two-spotted_spider_mite": DosageRule(0.45, "Abamectin", "ml"),
    "Tomato___Target_Spot":          DosageRule(0.55, "Azoxystrobin", "ml"),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": DosageRule(0.50, "Imidacloprid", "ml"),
    "Tomato___Tomato_mosaic_virus":  DosageRule(0.50, "Mineral Oil", "ml"),
}

DEFAULT_DOSAGE = DosageRule(0.50, "Broad-spectrum Fungicide", "ml")


@dataclass
class Config:
    detector_path: str = "models/yolo_detector.pt"
    classifier_path: str = "models/yolo_classifier.pt"
    leaf_seg_path: str = "models/leaf_seg.pth"
    disease_seg_path: str = "models/diseases_leaf_segmentation.pth"
    gan_path: str = "models/drone_generator.pth"
    vae_path: str = "models/vae_healthy.pth"

    seg_size: Tuple[int, int] = (512, 512)
    leaf_seg_threshold: float = 0.5
    disease_seg_threshold: float = 0.4

    min_box_area: int = 1000       
    min_leaf_area: int = 500       
    frame_skip: int = 3            

    gsd_cm_per_px: float = 0.05    

    gan_num_classes: int = 11
    gan_latent_dim: int = 100
    gan_embed_size: int = 100
    gan_features_g: int = 32        
    vae_latent_dim: int = 128
    image_channels: int = 3

    dosage_table: Dict[str, DosageRule] = field(default_factory=lambda: dict(DOSAGE_TABLE))
    default_dosage: DosageRule = field(default_factory=lambda: DEFAULT_DOSAGE)

    def area_cm2_per_pixel(self) -> float:
        """Correct pixel->area conversion: a pixel covers gsd**2 cm^2."""
        return self.gsd_cm_per_px ** 2

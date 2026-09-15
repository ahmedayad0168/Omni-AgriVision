"""
python scripts/train_detector.py --data data/LeafDetection/data.yaml --model yolo11m.pt --epochs 100 --batch 16 --imgsz 640 --device 0 --output models/detection
"""

from ultralytics import YOLO
from pathlib import Path
import yaml
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def create_data_yaml(
    train_images_dir: str,
    train_labels_dir: str,
    val_images_dir: str = None,
    val_labels_dir: str = None,
    class_names: list = None,
    output_path: str = "data/detection/data.yaml"
) -> str:
    """Create YOLO data.yaml configuration file."""
    
    if class_names is None:
        class_names = ['plant', 'fruit', 'pest', 'weed', 'disease_lesion', 'flower', 'leaf']
    
    data_config = {
        'path': str(Path(train_images_dir).parent),
        'train': str(Path(train_images_dir).relative_to(Path(train_images_dir).parent)),
        'val': str(Path(val_images_dir).relative_to(Path(val_images_dir).parent)) if val_images_dir else str(Path(train_images_dir).relative_to(Path(train_images_dir).parent)),
        'names': {i: name for i, name in enumerate(class_names)},
        'nc': len(class_names)
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    logger.info(f"Data configuration saved to {output_path}")
    return output_path


def train_detector(
    data_yaml: str,
    model_name: str = "yolo11m.pt",
    epochs: int = 100,
    batch_size: int = 16,
    image_size: int = 640,
    device: str = "auto",
    output_dir: str = "models/detection",
    project_name: str = "omni_agri_detection"
) -> Dict[str, Any]:
    """
    Train YOLO model for agricultural object detection.
    
    Args:
        data_yaml: Path to YOLO data.yaml configuration file
        model_name: Pre-trained model to start from (yolov11n.pt, yolov11m.pt, yolov11l.pt, yolov11x.pt)
        epochs: Number of training epochs
        batch_size: Batch size for training
        image_size: Image size for training
        device: Device to use ('auto', 'cpu', 'cuda:0', etc.)
        output_dir: Directory to save trained models
        project_name: Project name for Ultralytics tracking
    
    Returns:
        Dictionary with training results
    """
    
    # Initialize model
    model = YOLO(model_name)
    
    # Training configuration
    train_config = {
        'data': data_yaml,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': image_size,
        'device': device,
        'project': project_name,
        'name': 'train',
        'exist_ok': True,
        'patience': 20,  # Early stopping
        'save': True,
        'plots': True,
        'verbose': True,
        'workers': 4,
        'pretrained': True,
        'optimizer': 'AdamW',
        'lr0': 0.001,  # Initial learning rate
        'lrf': 0.01,   # Final learning rate (fraction of lr0)
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,    # Box loss gain
        'cls': 0.5,    # Cls loss gain
        'dfl': 1.5,    # DFL loss gain
        'pose': 12.0,  # Pose loss gain
        'kobj': 1.0,   # Keypoint object loss gain
        'label_smoothing': 0.0,
        'nbs': 64,     # Nominal batch size
        'hsv_h': 0.015,  # HSV-Hue augmentation
        'hsv_s': 0.7,    # HSV-Saturation augmentation
        'hsv_v': 0.4,    # HSV-Value augmentation
        'degrees': 0.0,  # Rotation augmentation
        'translate': 0.1, # Translation augmentation
        'scale': 0.5,     # Scale augmentation
        'shear': 0.0,     # Shear augmentation
        'perspective': 0.0,  # Perspective augmentation
        'flipud': 0.0,    # Vertical flip augmentation
        'fliplr': 0.5,    # Horizontal flip augmentation
        'mosaic': 1.0,    # Mosaic augmentation
        'mixup': 0.0,     # Mixup augmentation
        'copy_paste': 0.0, # Copy-paste augmentation
    }
    
    logger.info(f"Starting training with {model_name} for {epochs} epochs")
    logger.info(f"Data config: {data_yaml}")
    
    # Train the model
    results = model.train(**train_config)
    
    # Save the best model to the specified output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    best_model_path = str(Path(output_dir) / f"{model_name.replace('.pt', '')}_finetuned.pt")
    model.save(best_model_path)
    
    logger.info(f"Training completed! Best model saved to {best_model_path}")
    
    # Validate the model
    metrics = model.val(data=data_yaml, split='val')
    
    return {
        'model_path': best_model_path,
        'metrics': {
            'mAP50': metrics.box.map50,
            'mAP50-95': metrics.box.map,
            'precision': metrics.box.mp,
            'recall': metrics.box.mr
        },
        'results': results
    }


def validate_detector(
    model_path: str,
    data_yaml: str,
    device: str = "auto"
) -> Dict[str, Any]:
    """Validate a trained YOLO model."""
    
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, device=device, split='val')
    
    return {
        'mAP50': float(metrics.box.map50),
        'mAP50-95': float(metrics.box.map),
        'precision': float(metrics.box.mp),
        'recall': float(metrics.box.mr),
        'class_metrics': metrics.box.maps  # Per-class mAP
    }


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Train YOLO detector for agricultural objects")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml or root data directory")
    parser.add_argument("--model", type=str, default="yolo11m.pt", help="Pre-trained model name")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--output", type=str, default="models/detection", help="Output directory")
    
    args = parser.parse_args()
    
    # Check if data is a directory (create data.yaml) or already a yaml file
    if Path(args.data).is_dir():
        # Assume directory structure and create data.yaml
        data_yaml = create_data_yaml(
            train_images_dir=str(Path(args.data) / "train" / "images"),
            train_labels_dir=str(Path(args.data) / "train" / "labels"),
            val_images_dir=str(Path(args.data) / "val" / "images"),
            val_labels_dir=str(Path(args.data) / "val" / "labels"),
            output_path=str(Path(args.data) / "data.yaml")
        )
    else:
        data_yaml = args.data
    
    # Train
    results = train_detector(
        data_yaml=data_yaml,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        image_size=args.imgsz,
        device=args.device,
        output_dir=args.output
    )
    
    print(f"\nTraining Results:")
    print(f"Model saved to: {results['model_path']}")
    print(f"mAP50: {results['metrics']['mAP50']:.4f}")
    print(f"mAP50-95: {results['metrics']['mAP50-95']:.4f}")
    print(f"Precision: {results['metrics']['precision']:.4f}")
    print(f"Recall: {results['metrics']['recall']:.4f}")
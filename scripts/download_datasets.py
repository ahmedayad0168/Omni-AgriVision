import os
import requests
import zipfile
from pathlib import Path
import logging
from typing import Optional, Dict
import shutil

logger = logging.getLogger(__name__)


def download_file(url: str, destination: str, chunk_size: int = 8192) -> str:
    """Download a file from URL to destination."""
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading {url} to {destination}")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    progress = (downloaded / total_size) * 100
                    logger.info(f"Download progress: {progress:.1f}%")
    
    logger.info(f"Download completed: {destination}")
    return str(destination)


def extract_zip(zip_path: str, extract_to: str) -> str:
    """Extract a zip file to destination."""
    extract_path = Path(extract_to)
    extract_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Extracting {zip_path} to {extract_to}")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    logger.info(f"Extraction completed: {extract_to}")
    return str(extract_to)


def download_plantvillage(
    output_dir: str = "data/classification/PlantVillage",
    url: str = "https://github.com/spMohanty/PlantVillage-Dataset/archive/refs/heads/master.zip"
) -> str:
    """Download PlantVillage dataset for disease classification."""
    
    temp_zip = "data/temp/plantvillage.zip"
    downloaded = download_file(url, temp_zip)
    extracted = extract_zip(downloaded, output_dir)
    
    # Clean up
    Path(temp_zip).unlink(missing_ok=True)
    
    # The extracted folder might have a nested structure, flatten it
    extracted_path = Path(extracted)
    if len(list(extracted_path.iterdir())) == 1:
        nested = list(extracted_path.iterdir())[0]
        if nested.is_dir():
            # Move contents up one level
            for item in nested.iterdir():
                shutil.move(str(item), str(extracted_path / item.name))
            nested.rmdir()
    
    logger.info(f"PlantVillage dataset downloaded to {output_dir}")
    return output_dir


def download_plantdoc(
    output_dir: str = "data/detection/plantdoc",
    url: str = "https://github.com/pratiksha44/plantdoc-dataset/archive/refs/heads/main.zip"
) -> str:
    """Download PlantDoc dataset for object detection."""
    
    temp_zip = "data/temp/plantdoc.zip"
    downloaded = download_file(url, temp_zip)
    extracted = extract_zip(downloaded, output_dir)
    
    # Clean up
    Path(temp_zip).unlink(missing_ok=True)
    
    logger.info(f"PlantDoc dataset downloaded to {output_dir}")
    return output_dir


def download_sample_drone_images(
    output_dir: str = "data/raw/drone_images",
    num_samples: int = 50
) -> str:
    """Download sample drone images for testing (placeholder - would need actual source)."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Sample drone images directory created at {output_dir}")
    logger.info("Note: This is a placeholder. For actual drone images, provide your own dataset.")
    
    return str(output_dir)


def organize_yolo_dataset(
    source_dir: str,
    output_dir: str,
    train_split: float = 0.8,
    val_split: float = 0.2
) -> str:
    """Organize a dataset into YOLO format with train/val splits."""
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    for split in ['train', 'val']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Assume source has images and labels directories
    images_dir = source_path / 'images'
    labels_dir = source_path / 'labels'
    
    if not images_dir.exists() or not labels_dir.exists():
        logger.warning(f"Expected images/ and labels/ directories in {source_dir}")
        return str(output_path)
    
    # Get all image files
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    
    # Split into train/val
    import random
    random.shuffle(image_files)
    
    split_idx = int(len(image_files) * train_split)
    train_images = image_files[:split_idx]
    val_images = image_files[split_idx:]
    
    # Copy files to appropriate directories
    for img in train_images:
        label_file = labels_dir / f"{img.stem}.txt"
        shutil.copy(img, output_path / 'train' / 'images' / img.name)
        if label_file.exists():
            shutil.copy(label_file, output_path / 'train' / 'labels' / label_file.name)
    
    for img in val_images:
        label_file = labels_dir / f"{img.stem}.txt"
        shutil.copy(img, output_path / 'val' / 'images' / img.name)
        if label_file.exists():
            shutil.copy(label_file, output_path / 'val' / 'labels' / label_file.name)
    
    logger.info(f"Dataset organized: {len(train_images)} train, {len(val_images)} val images")
    return str(output_path)


def download_all_datasets(base_dir: str = "data") -> Dict[str, str]:
    """Download all required datasets for the project."""
    
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    try:
        results['plantvillage'] = download_plantvillage(
            output_dir=str(base_path / "classification" / "PlantVillage")
        )
    except Exception as e:
        logger.error(f"Failed to download PlantVillage: {e}")
        results['plantvillage'] = None
    
    try:
        results['plantdoc'] = download_plantdoc(
            output_dir=str(base_path / "detection" / "plantdoc_raw")
        )
        # Organize into YOLO format
        if results['plantdoc']:
            results['plantdoc_yolo'] = organize_yolo_dataset(
                source_dir=results['plantdoc'],
                output_dir=str(base_path / "detection" / "plantdoc")
            )
    except Exception as e:
        logger.error(f"Failed to download PlantDoc: {e}")
        results['plantdoc'] = None
    
    try:
        results['drone_images'] = download_sample_drone_images(
            output_dir=str(base_path / "raw" / "drone_images")
        )
    except Exception as e:
        logger.error(f"Failed to create drone images directory: {e}")
        results['drone_images'] = None
    
    logger.info("Dataset download process completed")
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download datasets for Omni-AgriVision")
    parser.add_argument("--base-dir", type=str, default="data", help="Base directory for datasets")
    parser.add_argument("--dataset", type=str, choices=["all", "plantvillage", "plantdoc", "drone"], 
                       default="all", help="Which dataset to download")
    
    args = parser.parse_args()
    
    if args.dataset == "all":
        results = download_all_datasets(args.base_dir)
        print("\nDownload Results:")
        for name, path in results.items():
            status = "✓" if path else "✗"
            print(f"{status} {name}: {path}")
    elif args.dataset == "plantvillage":
        path = download_plantvillage(output_dir=f"{args.base_dir}/classification/PlantVillage")
        print(f"PlantVillage downloaded to: {path}")
    elif args.dataset == "plantdoc":
        path = download_plantdoc(output_dir=f"{args.base_dir}/detection/plantdoc_raw")
        print(f"PlantDoc downloaded to: {path}")
    elif args.dataset == "drone":
        path = download_sample_drone_images(output_dir=f"{args.base_dir}/raw/drone_images")
        print(f"Drone images directory created at: {path}")
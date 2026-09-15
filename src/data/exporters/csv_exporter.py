import csv
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class CSVExporter:
    @staticmethod
    def export_detections(detections: List[Dict], output_path: str):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', newline='') as f:
            if not detections:
                return
            fieldnames = detections[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(detections)
        logger.info(f"Exported {len(detections)} detections to {output_path}")
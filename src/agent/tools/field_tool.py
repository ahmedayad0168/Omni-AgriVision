from typing import Dict, Any
from src.data.database import DatabaseManager
from src.data.models import Scan, Detection, DiseaseRecord, PestRecord
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class FieldTool:
    def __init__(self, db_manager: DatabaseManager, field_id: int):
        self.db = db_manager
        self.field_id = field_id

    def get_data(self, _: str = None) -> str:
        """Return field data summary."""
        session = self.db.get_session()
        try:
            latest_scan = session.query(Scan).filter(Scan.field_id == self.field_id)\
                .order_by(Scan.scan_date.desc()).first()
            if not latest_scan:
                return "No scan data available for this field."

            detections = session.query(Detection).filter(Detection.scan_id == latest_scan.id).all()
            plant_count = sum(1 for d in detections if d.object_type == 'plant')
            pest_count = sum(1 for d in detections if d.object_type == 'pest')
            weed_count = sum(1 for d in detections if d.object_type == 'weed')

            diseases = []
            for d in detections:
                for dr in session.query(DiseaseRecord).filter(DiseaseRecord.detection_id == d.id).all():
                    diseases.append({'name': dr.disease_name, 'severity': dr.severity})

            pests = []
            for d in detections:
                for pr in session.query(PestRecord).filter(PestRecord.detection_id == d.id).all():
                    pests.append({'name': pr.pest_name, 'count': pr.count})

            summary = {
                'scan_date': latest_scan.scan_date.isoformat(),
                'plant_count': plant_count,
                'pest_count': pest_count,
                'weed_count': weed_count,
                'diseases': diseases,
                'pests': pests,
            }
            return str(summary)
        finally:
            session.close()

    def get_historical_trends(self, days: int = 30) -> str:
        """Return historical trends aggregated by day."""
        session = self.db.get_session()
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)
            scans = session.query(Scan).filter(Scan.field_id == self.field_id, Scan.scan_date >= cutoff)\
                .order_by(Scan.scan_date).all()
            if not scans:
                return "No historical data available."

            trends = []
            for scan in scans:
                detections = session.query(Detection).filter(Detection.scan_id == scan.id).all()
                plant_count = sum(1 for d in detections if d.object_type == 'plant')
                pest_count = sum(1 for d in detections if d.object_type == 'pest')
                weed_count = sum(1 for d in detections if d.object_type == 'weed')
                trends.append({
                    'date': scan.scan_date.isoformat(),
                    'plants': plant_count,
                    'pests': pest_count,
                    'weeds': weed_count,
                })
            return str(trends)
        finally:
            session.close()
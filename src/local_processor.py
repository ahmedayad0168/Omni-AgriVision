"""
Synchronous video processing for local mode (no Redis/Celery required)
"""
import asyncio
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class LocalVideoProcessor:
    """Process videos synchronously without Redis/Celery"""
    
    def __init__(self):
        self.db = None
        self.ingestor = None
        self.tracker = None
        self.zone_mapper = None
        self.detector = None
        self.classifier = None
        self.segmenter = None
    
    def _load_components(self):
        """Load components only when needed"""
        from src.data.database import DatabaseManager
        from src.ingestion.video_ingestor import VideoIngestor
        from src.vision.detection.tracker import Tracker
        from src.geospatial.zone_mapper import ZoneMapper
        from configs.settings import settings
        
        if self.db is None:
            self.db = DatabaseManager(settings.database_url)
        if self.ingestor is None:
            self.ingestor = VideoIngestor()
        if self.tracker is None:
            self.tracker = Tracker()
        if self.zone_mapper is None:
            self.zone_mapper = ZoneMapper()
    
    def _load_models(self):
        """Load ML models only when needed"""
        from src.vision.detection.detector import Detector
        from src.vision.classification.classifier import DiseaseClassifier
        from src.vision.segmentation.segmenter import DiseaseSegmenter
        
        if self.detector is None:
            self.detector = Detector(
                model_path="models/detection/yolo11m_finetuned.pt", 
                image_size=320
            )
        if self.classifier is None:
            self.classifier = DiseaseClassifier(
                model_path="models/classification/efficientnet_disease.pt", 
                input_size=160
            )
        if self.segmenter is None:
            self.segmenter = DiseaseSegmenter(
                model_path="models/segmentation/deeplabv3plus_disease.pt", 
                input_size=256, 
                encoder_name="resnet18"
            )
    
    async def process_video(self, video_path: str, scan_id: str, field_id: int) -> Dict[str, Any]:
        """Process video synchronously"""
        # Load components and models only when needed
        self._load_components()
        self._load_models()
        
        from src.data.models import Scan, Detection, DiseaseRecord, PestRecord
        from datetime import datetime
        import traceback
        
        session = self.db.get_session()
        try:
            # Extract frames
            frames = []
            for frame, meta in self.ingestor.process(video_path, save_frames=False):
                frames.append(frame)

            # Detection + Tracking
            all_dets = []
            for frame in frames:
                dets = self.detector.detect(frame)
                tracked = self.tracker.update(dets)
                all_dets.append(tracked)

            # Classification & Segmentation on all detections
            for frame_idx, dets in enumerate(all_dets):
                frame = frames[frame_idx]
                img_h, img_w = frame.shape[:2]
                
                for det in dets:
                    x1, y1, x2, y2 = map(int, det['bbox'])
                    roi = frame[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue
                    
                    # Determine object type based on class name
                    class_name = det.get('class_name', 'unknown')
                    object_type = 'plant'

                    # More robust classification logic
                    class_lower = class_name.lower()

                    # First check for healthy plants (including tomato healthy)
                    if 'healthy' in class_lower:
                        object_type = 'plant'
                    # Then check for pests (mites, aphids, etc.)
                    elif 'mite' in class_lower or 'aphid' in class_lower or 'thrip' in class_lower or 'pest' in class_lower:
                        object_type = 'pest'
                    # Then check for weeds
                    elif 'weed' in class_lower:
                        object_type = 'weed'
                    # Finally check for diseases (blight, spot, mold, virus, etc.)
                    elif any(disease in class_lower for disease in ['blight', 'spot', 'mold', 'virus', 'curl', 'mosaic', 'bacterial', 'fungal']):
                        object_type = 'disease'
                    else:
                        # Default to plant for unknown classes (likely healthy plants)
                        object_type = 'plant'
                    
                    # Create detection record for ALL detections
                    detection = Detection(
                        scan_id=scan_id,
                        frame_id=frame_idx,
                        track_id=det.get('track_id'),
                        object_type=object_type,
                        class_name=class_name,
                        confidence=det['confidence'],
                        bbox_x1=x1, bbox_y1=y1, bbox_x2=x2, bbox_y2=y2,
                        zone=self.zone_mapper.assign_to_zone(x1, y1, x2, y2, img_w, img_h) if self.zone_mapper else 'unknown'
                    )
                    session.add(detection)
                    session.flush()

                    # Classification & Segmentation for plants/diseases
                    if object_type in ['plant', 'disease']:
                        try:
                            cls_result = self.classifier.classify(roi)
                            
                            # Update detection with classification result
                            detection.class_name = cls_result['class_name']
                            detection.confidence = cls_result['confidence']
                            
                            if not cls_result['is_healthy'] and cls_result['confidence'] > 0.5:
                                seg_result = self.segmenter.segment(roi)
                                disease_rec = DiseaseRecord(
                                    detection_id=detection.id,
                                    disease_name=cls_result['class_name'],
                                    confidence=cls_result['confidence'],
                                    severity=seg_result['severity'],
                                    severity_level=self.segmenter.calculate_severity_level(seg_result['severity']),
                                    lesion_pixels=seg_result['disease_pixels']
                                )
                                session.add(disease_rec)
                            
                            # Create pest record if class indicates pest
                            if 'mite' in class_name.lower() or 'pest' in class_name.lower():
                                pest_rec = PestRecord(
                                    detection_id=detection.id,
                                    pest_name=class_name,
                                    count=1,
                                    density=0.01
                                )
                                session.add(pest_rec)
                        except Exception as ml_error:
                            logger.warning(f"ML processing failed for detection: {ml_error}")
                            continue

            # Update scan status
            scan = session.query(Scan).filter(Scan.id == scan_id).first()
            scan.status = 'completed'
            scan.processed_frames = len(frames)
            session.commit()

            logger.info(f"Video processing completed for scan {scan_id}")
            
            return {
                "scan_id": scan_id,
                "status": "completed",
                "processed_frames": len(frames),
                "detections_count": len(all_dets)
            }
            
        except Exception as e:
            session.rollback()
            scan = session.query(Scan).filter(Scan.id == scan_id).first()
            if scan:
                scan.status = 'failed'
                session.commit()
            logger.error(f"Video processing failed: {e}")
            logger.error(traceback.format_exc())
            return {
                "scan_id": scan_id,
                "status": "failed",
                "error": str(e)
            }
        finally:
            session.close()
from celery import shared_task
from src.data.database import DatabaseManager
from src.data.models import Scan, Detection, DiseaseRecord, PestRecord
from configs.settings import settings
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

db = DatabaseManager(settings.database_url)


@shared_task
def process_video_task(video_path: str, scan_id: str, field_id: int):
    session = db.get_session()
    try:
        from src.ingestion.video_ingestor import VideoIngestor
        from src.vision.detection.tracker import Tracker
        from src.geospatial.zone_mapper import ZoneMapper
        from src.vision.detection.detector import Detector
        from src.vision.classification.classifier import DiseaseClassifier
        from src.vision.segmentation.segmenter import DiseaseSegmenter

        ingestor = VideoIngestor()
        tracker = Tracker()
        zone_mapper = ZoneMapper()
        detector = Detector(model_path="models/detection/yolo11m_finetuned.pt", image_size=320)
        classifier = DiseaseClassifier(model_path="models/classification/efficientnet_disease.pt", input_size=160)
        segmenter = DiseaseSegmenter(model_path="models/segmentation/deeplabv3plus_disease.pt", input_size=256, encoder_name="resnet18")

        frames = []
        for frame, meta in ingestor.process(video_path, save_frames=False):
            frames.append(frame)

        all_dets = []
        for frame in frames:
            dets = detector.detect(frame)
            tracked = tracker.update(dets)
            all_dets.append(tracked)

        for frame_idx, dets in enumerate(all_dets):
            frame = frames[frame_idx]
            img_h, img_w = frame.shape[:2]

            for det in dets:
                x1, y1, x2, y2 = map(int, det['bbox'])
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                class_name = det.get('class_name', 'unknown')
                object_type = 'plant'

                if 'blight' in class_name.lower() or 'spot' in class_name.lower() or 'mold' in class_name.lower() or 'virus' in class_name.lower():
                    object_type = 'disease'
                elif 'mite' in class_name.lower() or 'pest' in class_name.lower():
                    object_type = 'pest'
                elif 'weed' in class_name.lower():
                    object_type = 'weed'
                elif 'healthy' in class_name.lower():
                    object_type = 'plant'

                detection = Detection(
                    scan_id=scan_id,
                    frame_id=frame_idx,
                    track_id=det.get('track_id'),
                    object_type=object_type,
                    class_name=class_name,
                    confidence=det['confidence'],
                    bbox_x1=x1, bbox_y1=y1, bbox_x2=x2, bbox_y2=y2,
                    zone=zone_mapper.assign_to_zone(x1, y1, x2, y2, img_w, img_h) if zone_mapper else 'unknown'
                )
                session.add(detection)
                session.flush()

                if object_type in ['plant', 'disease']:
                    cls_result = classifier.classify(roi)

                    detection.class_name = cls_result['class_name']
                    detection.confidence = cls_result['confidence']

                    if not cls_result['is_healthy'] and cls_result['confidence'] > 0.5:
                        seg_result = segmenter.segment(roi)
                        disease_rec = DiseaseRecord(
                            detection_id=detection.id,
                            disease_name=cls_result['class_name'],
                            confidence=cls_result['confidence'],
                            severity=seg_result['severity'],
                            severity_level=segmenter.calculate_severity_level(seg_result['severity']),
                            lesion_pixels=seg_result['disease_pixels']
                        )
                        session.add(disease_rec)

                    if 'mite' in class_name.lower() or 'pest' in class_name.lower():
                        pest_rec = PestRecord(
                            detection_id=detection.id,
                            pest_name=class_name,
                            count=1,
                            density=0.01
                        )
                        session.add(pest_rec)

        scan = session.query(Scan).filter(Scan.id == scan_id).first()
        scan.status = 'completed'
        scan.processed_frames = len(frames)
        session.commit()

        from src.agent.agent import FarmIntelligenceAgent
        agent = FarmIntelligenceAgent(field_id, db)
        agent.analyze()

        logger.info(f"Video processing completed for scan {scan_id}")
    except Exception as e:
        session.rollback()
        scan = session.query(Scan).filter(Scan.id == scan_id).first()
        if scan:
            scan.status = 'failed'
            session.commit()
        logger.error(f"Video processing failed: {e}")
    finally:
        session.close()

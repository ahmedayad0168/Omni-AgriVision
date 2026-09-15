from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uuid
from pathlib import Path
from datetime import datetime, timedelta
import logging

import pandas as pd
from sqlalchemy import text

from configs.settings import settings
from src.data.database import DatabaseManager
from src.data.base import Base
from src.data.models import (
    Scan, Detection, DiseaseRecord, PestRecord,
    Field, Farm, WeatherRecord, YieldPrediction, Alert, AlertLevel,
)
from src.notifications.telegram_sender import TelegramSender
from src.notifications.email_sender import EmailSender
from src.prediction.yield_predictor import YieldPredictor


# Cache agent instances per field_id for persistent conversation memory
_agent_cache: dict = {}


def get_or_create_agent(field_id: int):
    """Get or create a FarmIntelligenceAgent for the given field (cached for memory persistence)."""
    global _agent_cache
    key = str(field_id)
    if key not in _agent_cache:
        from src.agent.agent import FarmIntelligenceAgent
        _agent_cache[key] = FarmIntelligenceAgent(field_id, db)
    return _agent_cache[key]


from loguru import logger
logger.remove()
logger.add(lambda msg: print(msg), level="INFO")


app = FastAPI(title="Omni-AgriVision API", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database
db = DatabaseManager(settings.database_url)
Base.metadata.create_all(db.engine)

# Upload / output dirs
UPLOAD_DIR = Path(settings.upload_dir)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = Path(settings.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Notification senders
telegram = TelegramSender()
email = EmailSender()

YIELD_MODEL_PATH = "models/yield_predictor.pkl"
yield_predictor = None
if Path(YIELD_MODEL_PATH).exists():
    try:
        yield_predictor = YieldPredictor(model_path=YIELD_MODEL_PATH)
        logger.info(f"Loaded yield predictor from {YIELD_MODEL_PATH}")
    except Exception as e:
        logger.warning(f"Failed to load yield predictor: {e}")
else:
    logger.warning(
        f"No yield predictor at {YIELD_MODEL_PATH}. "
        f"/yield-prediction will return 503 until one is trained."
    )


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class AnalysisRequest(BaseModel):
    field_id: int


class FarmCreate(BaseModel):
    name: str
    location: str
    latitude: float
    longitude: float
    total_area: float
    crop_type: str


class FieldCreate(BaseModel):
    farm_id: int
    field_name: str
    area: float
    crop_type: str
    planting_date: Optional[datetime] = None
    expected_harvest_date: Optional[datetime] = None


class WeatherData(BaseModel):
    field_id: int
    date: datetime
    temperature: float
    humidity: float
    rainfall: float
    wind_speed: float
    solar_radiation: float


class AlertCreate(BaseModel):
    field_id: int
    alert_level: str
    title: str
    description: str
    action_required: bool = True


# ---------------------------------------------------------------------------
# Root & health
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    return {"message": "Omni-AgriVision API", "version": "2.0", "status": "operational"}


@app.get("/api/health")
async def health_check():
    """Health check endpoint for monitoring"""
    try:
        session = db.get_session()
        session.execute(text("SELECT 1"))     
        session.close()
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {"database": "connected", "api": "running"},
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


# ---------------------------------------------------------------------------
# Farm & field management
# ---------------------------------------------------------------------------
@app.get("/api/farms")
async def get_farms():
    session = db.get_session()
    try:
        farms = session.query(Farm).all()
        return {
            "farms": [{
                "id": f.id,
                "name": f.name,
                "location": f.location,
                "latitude": f.latitude,
                "longitude": f.longitude,
                "total_area": f.total_area,
                "crop_type": f.crop_type,
                "created_at": f.created_at.isoformat(),
            } for f in farms]
        }
    finally:
        session.close()


@app.post("/api/farms")
async def create_farm(farm: FarmCreate):
    session = db.get_session()
    try:
        new_farm = Farm(
            name=farm.name, location=farm.location,
            latitude=farm.latitude, longitude=farm.longitude,
            total_area=farm.total_area, crop_type=farm.crop_type,
        )
        session.add(new_farm)
        session.commit()
        session.refresh(new_farm)
        return {"id": new_farm.id, "message": "Farm created successfully"}
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        session.close()


@app.get("/api/farms/{farm_id}/fields")
async def get_fields(farm_id: int):
    session = db.get_session()
    try:
        fields = session.query(Field).filter(Field.farm_id == farm_id).all()
        return {
            "fields": [{
                "id": fld.id,
                "farm_id": fld.farm_id,
                "field_name": fld.field_name,
                "area": fld.area,
                "crop_type": fld.crop_type,
                "planting_date": fld.planting_date.isoformat() if fld.planting_date else None,
                "expected_harvest_date": fld.expected_harvest_date.isoformat() if fld.expected_harvest_date else None,
            } for fld in fields]
        }
    finally:
        session.close()


@app.post("/api/fields")
async def create_field(field: FieldCreate):
    session = db.get_session()
    try:
        new_field = Field(
            farm_id=field.farm_id,
            field_name=field.field_name,
            area=field.area,
            crop_type=field.crop_type,
            planting_date=field.planting_date,
            expected_harvest_date=field.expected_harvest_date,
        )
        session.add(new_field)
        session.commit()
        session.refresh(new_field)
        return {"id": new_field.id, "message": "Field created successfully"}
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Video processing
# ---------------------------------------------------------------------------
@app.post("/api/upload")
async def upload_video(field_id: int, file: UploadFile = File(...)):
    """Upload drone video for processing (local mode if Redis unavailable, Celery if available)."""
    from src.tasks import process_video_task
    from src.local_processor import LocalVideoProcessor

    scan_id = f"scan_{uuid.uuid4().hex[:8]}"
    video_path = UPLOAD_DIR / f"{scan_id}_{file.filename}"
    video_path.write_bytes(await file.read())

    session = db.get_session()
    try:
        scan = Scan(
            id=scan_id, field_id=field_id,
            scan_date=datetime.utcnow(), status="processing",
        )
        session.add(scan)
        session.commit()
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    finally:
        session.close()

    # Try Celery first, fall back to local processing
    try:
        process_video_task.delay(str(video_path), scan_id, field_id)
        logger.info(f"Video {scan_id} queued for Celery processing")
        return {"scan_id": scan_id, "status": "processing", "mode": "celery"}
    except Exception as e:
        # If Celery fails, use local synchronous processing
        logger.warning(f"Celery unavailable, using local processing: {e}")
        try:
            processor = LocalVideoProcessor()
            result = await processor.process_video(str(video_path), scan_id, field_id)
            logger.info(f"Video {scan_id} processed locally: {result}")
            return {"scan_id": scan_id, "status": result.get("status", "unknown"), "mode": "local"}
        except Exception as local_error:
            logger.error(f"Local processing also failed: {local_error}")
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(local_error)}")


@app.get("/api/scan/{scan_id}")
async def get_scan(scan_id: str):
    session = db.get_session()
    try:
        scan = session.query(Scan).filter(Scan.id == scan_id).first()
        if not scan:
            raise HTTPException(404, "Scan not found")

        detections = session.query(Detection).filter(Detection.scan_id == scan_id).all()
        total = len(detections)
        plants = sum(1 for d in detections if d.object_type == "plant")
        pests = sum(1 for d in detections if d.object_type == "pest")
        weeds = sum(1 for d in detections if d.object_type == "weed")
        diseases = sum(1 for d in detections if d.object_type == "disease")

        return {
            "scan": {
                "id": scan.id,
                "field_id": scan.field_id,
                "date": scan.scan_date.isoformat(),
                "status": scan.status,
                "frames": scan.processed_frames,
                "total_frames": scan.total_frames,
            },
            "statistics": {
                "total_detections": total,
                "plants": plants,
                "pests": pests,
                "weeds": weeds,
                "diseases": diseases,
            },
            "detections": [{
                "id": d.id,
                "class": d.class_name,
                "confidence": d.confidence,
                "bbox": [d.bbox_x1, d.bbox_y1, d.bbox_x2, d.bbox_y2],
                "zone": d.zone,
                "object_type": d.object_type,
            } for d in detections],
        }
    finally:
        session.close()


@app.get("/api/field/{field_id}/scans")
async def get_field_scans(field_id: int, limit: int = 10):
    session = db.get_session()
    try:
        scans = (
            session.query(Scan)
            .filter(Scan.field_id == field_id)
            .order_by(Scan.scan_date.desc())
            .limit(limit)
            .all()
        )
        return {
            "scans": [{
                "id": s.id,
                "date": s.scan_date.isoformat(),
                "status": s.status,
                "processed_frames": s.processed_frames,
            } for s in scans]
        }
    finally:
        session.close()


# ---------------------------------------------------------------------------
# AI Agent (lazy import — depends on Ollama + langchain)
# ---------------------------------------------------------------------------
@app.post("/api/agent/analyze")
async def agent_analyze(request: AnalysisRequest):
    try:
        agent = get_or_create_agent(request.field_id)
        result = agent.analyze()
        return {"field_id": request.field_id, "analysis": result}
    except Exception as e:
        logger.error(f"Agent analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/agent/chat")
async def agent_chat(field_id: int, query: str):
    try:
        agent = get_or_create_agent(field_id)
        response = agent.ask(query)
        return {"field_id": field_id, "query": query, "response": response}
    except Exception as e:
        logger.error(f"Agent chat failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


# ---------------------------------------------------------------------------
# Health & yield
# ---------------------------------------------------------------------------
@app.get("/api/field/{field_id}/health")
async def get_field_health(field_id: int):
    session = db.get_session()
    try:
        latest_scan = (
            session.query(Scan)
            .filter(Scan.field_id == field_id)
            .order_by(Scan.scan_date.desc())
            .first()
        )
        if not latest_scan:
            return {"field_id": field_id, "message": "No scan data available"}

        detections = session.query(Detection).filter(Detection.scan_id == latest_scan.id).all()
        total_plants = sum(1 for d in detections if d.object_type == "plant")
        total_pests = sum(1 for d in detections if d.object_type == "pest")
        total_weeds = sum(1 for d in detections if d.object_type == "weed")
        healthy_plants = sum(
            1 for d in detections
            if d.object_type == "plant" and "healthy" in d.class_name.lower()
        )
        health_score = (healthy_plants / total_plants * 100) if total_plants else 0.0

        disease_records = []
        for d in detections:
            for dr in session.query(DiseaseRecord).filter(DiseaseRecord.detection_id == d.id).all():
                disease_records.append({
                    "name": dr.disease_name,
                    "severity": dr.severity,
                    "severity_level": dr.severity_level,
                })

        pest_records = []
        for d in detections:
            for pr in session.query(PestRecord).filter(PestRecord.detection_id == d.id).all():
                pest_records.append({
                    "name": pr.pest_name,
                    "count": pr.count,
                    "density": pr.density,
                })

        return {
            "field_id": field_id,
            "scan_date": latest_scan.scan_date.isoformat(),
            "health_score": round(health_score, 2),
            "total_plants": total_plants,
            "healthy_plants": healthy_plants,
            "total_pests": total_pests,
            "total_weeds": total_weeds,
            "diseases": disease_records,
            "pests": pest_records,
            "overall_status": (
                "healthy" if health_score > 80
                else "needs_attention" if health_score > 50
                else "critical"
            ),
        }
    finally:
        session.close()


@app.get("/api/field/{field_id}/yield-prediction")
async def get_yield_prediction(field_id: int):
    if yield_predictor is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Yield predictor is not trained. Train a model with "
                "`scripts/train_yield_predictor.py` and save it to "
                f"{YIELD_MODEL_PATH}, then restart the API."
            ),
        )

    session = db.get_session()
    try:
        latest_scan = (
            session.query(Scan)
            .filter(Scan.field_id == field_id)
            .order_by(Scan.scan_date.desc())
            .first()
        )
        if not latest_scan:
            return {"field_id": field_id, "message": "No scan data available for prediction"}

        detections = session.query(Detection).filter(Detection.scan_id == latest_scan.id).all()

        obs_df = pd.DataFrame([{
            "class_name": d.class_name,
            "confidence": d.confidence,
            "is_healthy": "healthy" in d.class_name.lower(),
        } for d in detections])

        weather_data = (
            session.query(WeatherRecord)
            .filter(
                WeatherRecord.field_id == field_id,
                WeatherRecord.date >= datetime.utcnow() - timedelta(days=7),
            ).all()
        )

        weather_df = None
        if weather_data:
            weather_df = pd.DataFrame([{
                "temperature_mean": w.temperature,
                "humidity_mean": w.humidity,
                "precipitation_total": w.rainfall,
                "solar_radiation_mean": w.solar_radiation,
            } for w in weather_data]).mean().to_dict()

        features = yield_predictor.extract_features(obs_df, weather_df)
        prediction = yield_predictor.predict(features)

        session.add(YieldPrediction(
            field_id=field_id,
            estimated_yield=prediction["predicted_yield"],
            lower_bound=prediction["lower_bound"],
            upper_bound=prediction["upper_bound"],
            confidence=prediction["confidence"],
            factors={},
        ))
        session.commit()
        return prediction
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Yield prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------
@app.post("/api/weather")
async def add_weather_data(weather: WeatherData):
    session = db.get_session()
    try:
        session.add(WeatherRecord(
            field_id=weather.field_id,
            date=weather.date,
            temperature=weather.temperature,
            humidity=weather.humidity,
            rainfall=weather.rainfall,
            wind_speed=weather.wind_speed,
            solar_radiation=weather.solar_radiation,
        ))
        session.commit()
        return {"message": "Weather data added successfully"}
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        session.close()


@app.get("/api/field/{field_id}/weather")
async def get_field_weather(field_id: int, days: int = 7):
    session = db.get_session()
    try:
        cutoff = datetime.utcnow() - timedelta(days=days)
        records = (
            session.query(WeatherRecord)
            .filter(WeatherRecord.field_id == field_id, WeatherRecord.date >= cutoff)
            .order_by(WeatherRecord.date.desc())
            .all()
        )
        return {
            "field_id": field_id,
            "weather_data": [{
                "date": w.date.isoformat(),
                "temperature": w.temperature,
                "humidity": w.humidity,
                "rainfall": w.rainfall,
                "wind_speed": w.wind_speed,
                "solar_radiation": w.solar_radiation,
            } for w in records]
        }
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------
@app.post("/api/alerts")
async def create_alert(alert: AlertCreate):
    session = db.get_session()
    try:
        try:
            level = AlertLevel(alert.alert_level.upper())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid alert_level '{alert.alert_level}'. "
                       f"Must be one of {[e.value for e in AlertLevel]}",
            )

        new_alert = Alert(
            field_id=alert.field_id,
            alert_level=level,
            title=alert.title,
            description=alert.description,
            action_required=alert.action_required,
        )
        session.add(new_alert)
        session.commit()

        if level in (AlertLevel.HIGH, AlertLevel.CRITICAL):
            if settings.telegram_bot_token:
                telegram.send(f"ALERT: {alert.title}\n{alert.description}")
            if settings.smtp_user:
                email.send(
                    to=settings.smtp_user,
                    subject=f"Farm Alert: {alert.title}",
                    body=alert.description,
                )

        return {"id": new_alert.id, "message": "Alert created successfully"}
    except HTTPException:
        session.rollback()
        raise
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        session.close()


@app.get("/api/field/{field_id}/alerts")
async def get_field_alerts(field_id: int, resolved: bool = False):
    session = db.get_session()
    try:
        alerts = (
            session.query(Alert)
            .filter(Alert.field_id == field_id, Alert.resolved == resolved)
            .order_by(Alert.created_at.desc())
            .all()
        )
        return {
            "alerts": [{
                "id": a.id,
                "alert_level": a.alert_level.value if a.alert_level else None,
                "title": a.title,
                "description": a.description,
                "action_required": a.action_required,
                "created_at": a.created_at.isoformat(),
            } for a in alerts]
        }
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Notifications
# ---------------------------------------------------------------------------
@app.post("/api/notify/telegram")
async def send_telegram(message: str):
    try:
        telegram.send(message)
        return {"status": "sent", "platform": "telegram"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send Telegram message: {str(e)}")


@app.post("/api/notify/email")
async def send_email(to: str, subject: str, body: str):
    try:
        email.send(to, subject, body)
        return {"status": "sent", "platform": "email"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send email: {str(e)}")


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
@app.get("/api/field/{field_id}/report")
async def generate_field_report(field_id: int, format: str = "html"):
    session = db.get_session()
    try:
        field = session.query(Field).filter(Field.id == field_id).first()
        if not field:
            raise HTTPException(404, "Field not found")

        latest_scan = (
            session.query(Scan)
            .filter(Scan.field_id == field_id)
            .order_by(Scan.scan_date.desc())
            .first()
        )

        health_data = await get_field_health(field_id)

        try:
            yield_data = await get_yield_prediction(field_id)
        except HTTPException as he:
            yield_data = {"error": he.detail}

        report_data = {
            "field_info": {
                "field_id": field.id,
                "field_name": field.field_name,
                "crop_type": field.crop_type,
                "area": field.area,
            },
            "health_metrics": health_data,
            "yield_prediction": yield_data,
            "scan_date": latest_scan.scan_date.isoformat() if latest_scan else None,
            "diseases": [],
            "pests": [],
            "alerts": [],
            "recommendations": [],
            "weather_forecast": [],
            "summary": f"Report generated for field {field.field_name} on {datetime.now().strftime('%Y-%m-%d')}."
        }

        # Add disease data if available
        if latest_scan:
            detections = session.query(Detection).filter(Detection.scan_id == latest_scan.id).all()
            disease_counts = {}
            for det in detections:
                diseases = session.query(DiseaseRecord).filter(DiseaseRecord.detection_id == det.id).all()
                for dr in diseases:
                    if dr.disease_name not in disease_counts:
                        disease_counts[dr.disease_name] = {"name": dr.disease_name, "severity": dr.severity, "affected_plants": 0}
                    disease_counts[dr.disease_name]["affected_plants"] += 1
            report_data["diseases"] = list(disease_counts.values())

        # Add pest data if available
        if latest_scan:
            detections = session.query(Detection).filter(Detection.scan_id == latest_scan.id).all()
            pest_counts = {}
            for det in detections:
                pests = session.query(PestRecord).filter(PestRecord.detection_id == det.id).all()
                for pr in pests:
                    if pr.pest_name not in pest_counts:
                        pest_counts[pr.pest_name] = {"name": pr.pest_name, "count": 0, "density": 0}
                    pest_counts[pr.pest_name]["count"] += pr.count
                    pest_counts[pr.pest_name]["density"] = pr.density
            report_data["pests"] = list(pest_counts.values())

        # Add alerts if available
        alerts = session.query(Alert).filter(Alert.field_id == field_id, Alert.resolved.is_(False)).all()
        report_data["alerts"] = [{"level": a.alert_level.value, "title": a.title, "description": a.description} for a in alerts]

        # Add recommendations based on health score
        if health_data.get("health_score", 0) < 80:
            report_data["recommendations"].append({
                "priority": "High",
                "action": "Implement disease treatment and pest control measures",
                "rationale": f"Health score of {health_data.get('health_score', 0)}% indicates immediate attention needed"
            })
        else:
            report_data["recommendations"].append({
                "priority": "Low",
                "action": "Continue current farming practices and regular monitoring",
                "rationale": f"Health score of {health_data.get('health_score', 0)}% indicates good condition"
            })

        if format == "html":
            from src.agent.tools.report_tool import ReportTool   # lazy
            report_tool = ReportTool()
            html_report = report_tool.generate_report(report_data)
            return {"format": "html", "content": html_report}
        return report_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")
    finally:
        session.close()
from sqlalchemy import Column, Integer, Float, String, DateTime, JSON, ForeignKey, Text, Boolean, Enum
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from src.data.base import Base


class AlertLevel(str, enum.Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Farm(Base):
    __tablename__ = 'farms'
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    location = Column(String(255))
    latitude = Column(Float)
    longitude = Column(Float)
    total_area = Column(Float)
    crop_type = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    fields = relationship('Field', back_populates='farm')


class Field(Base):
    __tablename__ = 'fields'
    id = Column(Integer, primary_key=True)
    farm_id = Column(Integer, ForeignKey('farms.id'))
    field_name = Column(String(100))
    area = Column(Float)
    crop_type = Column(String(100))
    planting_date = Column(DateTime)
    expected_harvest_date = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    farm = relationship('Farm', back_populates='fields')
    scans = relationship('Scan', back_populates='field')


class Scan(Base):
    __tablename__ = 'scans'
    id = Column(String(36), primary_key=True)  # uuid
    field_id = Column(Integer, ForeignKey('fields.id'))
    scan_date = Column(DateTime, nullable=False)
    drone_altitude = Column(Float)
    video_duration = Column(Float)
    total_frames = Column(Integer)
    processed_frames = Column(Integer)
    status = Column(String(50), default='processing')
    created_at = Column(DateTime, default=datetime.utcnow)
    field = relationship('Field', back_populates='scans')
    detections = relationship('Detection', back_populates='scan')


class Detection(Base):
    __tablename__ = 'detections'
    id = Column(Integer, primary_key=True)
    scan_id = Column(String(36), ForeignKey('scans.id'))
    frame_id = Column(Integer)
    track_id = Column(Integer)
    object_type = Column(String(50))
    class_name = Column(String(100))
    confidence = Column(Float)
    bbox_x1 = Column(Float)
    bbox_y1 = Column(Float)
    bbox_x2 = Column(Float)
    bbox_y2 = Column(Float)
    zone = Column(String(10))
    created_at = Column(DateTime, default=datetime.utcnow)
    scan = relationship('Scan', back_populates='detections')


class DiseaseRecord(Base):
    __tablename__ = 'disease_records'
    id = Column(Integer, primary_key=True)
    detection_id = Column(Integer, ForeignKey('detections.id'))
    disease_name = Column(String(100))
    confidence = Column(Float)
    severity = Column(Float)
    severity_level = Column(String(20))
    lesion_pixels = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)


class PestRecord(Base):
    __tablename__ = 'pest_records'
    id = Column(Integer, primary_key=True)
    detection_id = Column(Integer, ForeignKey('detections.id'))
    pest_name = Column(String(100))
    confidence = Column(Float)
    count = Column(Integer)
    density = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


class WeatherRecord(Base):
    __tablename__ = 'weather_records'
    id = Column(Integer, primary_key=True)
    field_id = Column(Integer, ForeignKey('fields.id'))
    date = Column(DateTime, nullable=False)
    temperature = Column(Float)
    humidity = Column(Float)
    rainfall = Column(Float)
    wind_speed = Column(Float)
    solar_radiation = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


class YieldPrediction(Base):
    __tablename__ = 'yield_predictions'
    id = Column(Integer, primary_key=True)
    field_id = Column(Integer, ForeignKey('fields.id'))
    prediction_date = Column(DateTime, default=datetime.utcnow)
    estimated_yield = Column(Float)
    lower_bound = Column(Float)
    upper_bound = Column(Float)
    confidence = Column(Float)
    factors = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)


class Alert(Base):
    __tablename__ = 'alerts'
    id = Column(Integer, primary_key=True)
    field_id = Column(Integer, ForeignKey('fields.id'))
    alert_level = Column(Enum(AlertLevel))
    title = Column(String(255))
    description = Column(Text)
    action_required = Column(Boolean, default=True)
    resolved = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime)
from src.data.database import DatabaseManager
from src.data.base import Base
from src.data.models import Farm, Field
from configs.settings import settings
from datetime import datetime, timedelta

if __name__ == "__main__":
    db = DatabaseManager(settings.database_url)
    Base.metadata.create_all(db.engine)
    
    # Create sample data
    session = db.get_session()
    try:
        # Check if farms exist
        if not session.query(Farm).first():
            # Create sample farm
            sample_farm = Farm(
                name="Sample Farm",
                location="Demo Location",
                latitude=40.7128,
                longitude=-74.0060,
                total_area=100.0,
                crop_type="Mixed Vegetables"
            )
            session.add(sample_farm)
            session.flush()
            
            # Create sample field
            sample_field = Field(
                farm_id=sample_farm.id,
                field_name="Field 1",
                area=25.0,
                crop_type="Spinach",
                planting_date=datetime.now(),
                expected_harvest_date=datetime.now() + timedelta(days=90)
            )
            session.add(sample_field)
            session.commit()
            print("Sample farm and field created successfully")
        else:
            print("Farm data already exists")
    except Exception as e:
        session.rollback()
        print(f"Error creating sample data: {e}")
    finally:
        session.close()
    
    print("Database tables created successfully")
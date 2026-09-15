"""
Initialize SQL Server database for Omni-AgriVision
"""
import pyodbc
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_database():
    """Create the database if it doesn't exist"""
    # Use pyodbc directly for CREATE DATABASE (requires autocommit)
    try:
        conn_str = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=localhost;DATABASE=master;Trusted_Connection=yes;"
        conn = pyodbc.connect(conn_str, autocommit=True)
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute("SELECT name FROM sys.databases WHERE name = 'omni_agri'")
        if cursor.fetchone():
            print("Database 'omni_agri' already exists")
        else:
            cursor.execute("CREATE DATABASE omni_agri")
            print("Database 'omni_agri' created successfully")
        
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error creating database: {e}")
        return False
    
    return True

def create_tables():
    """Create all tables in the database"""
    from src.data.base import Base
    from configs.settings import settings
    
    try:
        engine = create_engine(settings.database_url)
        Base.metadata.create_all(engine)
        print("Database tables created successfully")
        return True
    except Exception as e:
        print(f"Error creating tables: {e}")
        return False

def add_sample_data():
    """Add sample farm and field data"""
    from src.data.models import Farm, Field
    from src.data.database import DatabaseManager
    from configs.settings import settings
    from datetime import datetime
    
    try:
        db = DatabaseManager(settings.database_url)
        session = db.get_session()
        
        # Check if sample data already exists
        existing_farm = session.query(Farm).filter(Farm.name == "Sample Farm").first()
        if existing_farm:
            print("Sample data already exists")
            session.close()
            return True
        
        # Create sample farm
        farm = Farm(
            name="Sample Farm",
            location="Sample Location",
            crop_type="Tomato",
            total_area=10.0,
            created_at=datetime.utcnow()
        )
        session.add(farm)
        session.flush()
        
        # Create sample field
        field = Field(
            farm_id=farm.id,
            field_name="Field 1",
            crop_type="Tomato",
            area=5.0,
            planting_date=datetime.utcnow()
        )
        session.add(field)
        session.commit()
        
        print("Sample data added successfully")
        session.close()
        return True
    except Exception as e:
        session.rollback()
        print(f"Error adding sample data: {e}")
        session.close()
        return False

if __name__ == "__main__":
    print("Initializing SQL Server database...")
    
    # Step 1: Create database
    if not create_database():
        print("Failed to create database")
        sys.exit(1)
    
    # Step 2: Create tables
    if not create_tables():
        print("Failed to create tables")
        sys.exit(1)
    
    # Step 3: Add sample data
    if not add_sample_data():
        print("Failed to add sample data")
        sys.exit(1)
    
    print("\nDatabase initialization completed successfully!")
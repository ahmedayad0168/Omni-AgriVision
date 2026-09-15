from src.weather.nasa_power_client import NASAPowerClient
from configs.settings import settings
import datetime
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class WeatherTool:
    def __init__(self):
        self.client = NASAPowerClient()

    def get_forecast(self, field_id: str = None) -> str:
        """Fetch 7-day agricultural forecast with temperature, humidity, rainfall, and wind."""
        lat, lon = 30.0, 31.0
        try:
            df = self.client.get_forecast(lat, lon, days=7)
            if df.empty:
                return "Weather forecast data currently unavailable."
            
            # Format as clean markdown table for agent/user
            lines = ["| Date | Avg Temp (°C) | Max/Min (°C) | Humidity (%) | Rain (mm) | Wind (km/h) |",
                     "|------|---------------|--------------|--------------|-----------|-------------|"]
            for _, row in df.iterrows():
                dt = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])[:10]
                lines.append(
                    f"| {dt} | {row['temperature_mean']:.1f} | {row.get('temperature_max', row['temperature_mean']):.1f}/{row.get('temperature_min', row['temperature_mean']):.1f} | {row['humidity']:.0f}% | {row['precipitation']:.1f} | {row['wind_speed']:.1f} |"
                )
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"Weather forecast error: {e}")
            return f"Weather forecast error: {e}"

    def get_historical(self, start_date: str, end_date: str, lat: float = 30.0, lon: float = 31.0) -> str:
        # Convert date strings from YYYY-MM-DD to YYYYMMDD if needed
        if '-' in start_date:
            start_date = start_date.replace('-', '')
        if '-' in end_date:
            end_date = end_date.replace('-', '')
        df = self.client.get_daily(lat, lon, start_date, end_date)
        if df.empty:
            return "No historical weather data."
        return df.to_string(index=False)
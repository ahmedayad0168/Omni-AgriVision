from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    debug: bool = Field(default=False, alias="DEBUG")
    secret_key: str = Field(..., alias="SECRET_KEY")

    # Database
    database_url: str = Field(
        default="sqlite:///./omni_agri.db",
        alias="APP_DATABASE_URL",
        validation_alias="APP_DATABASE_URL",
        serialization_alias="APP_DATABASE_URL",
    )
    redis_url: Optional[str] = Field(default=None, alias="REDIS_URL")

    # Ollama
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="qwen3:8b", alias="OLLAMA_MODEL")
    ollama_embed_model: str = Field(default="qwen3:8b", alias="OLLAMA_EMBED_MODEL")

    # NASA POWER
    nasa_power_base_url: str = Field(
        default="https://power.larc.nasa.gov/api",
        alias="NASA_POWER_BASE_URL",
    )

    # Sentinel
    sentinel_client_id: Optional[str] = Field(default=None, alias="SENTINEL_CLIENT_ID")
    sentinel_client_secret: Optional[str] = Field(default=None, alias="SENTINEL_CLIENT_SECRET")

    # Notifications
    telegram_bot_token: Optional[str] = Field(default=None, alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: Optional[str] = Field(default=None, alias="TELEGRAM_CHAT_ID")
    smtp_host: Optional[str] = Field(default=None, alias="SMTP_HOST")
    smtp_port: Optional[int] = Field(default=None, alias="SMTP_PORT")
    smtp_user: Optional[str] = Field(default=None, alias="SMTP_USER")
    smtp_password: Optional[str] = Field(default=None, alias="SMTP_PASSWORD")

    # ML
    model_cache_dir: str = Field(default="./models/cache", alias="MODEL_CACHE_DIR")
    mlflow_tracking_uri: Optional[str] = Field(default=None, alias="MLFLOW_TRACKING_URI")

    # Storage
    upload_dir: str = Field(default="./data/uploads", alias="UPLOAD_DIR")
    output_dir: str = Field(default="./data/outputs", alias="OUTPUT_DIR")
    knowledge_base_dir: str = Field(default="./data/knowledge_base", alias="KNOWLEDGE_BASE_DIR")

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: str = Field(default="json", alias="LOG_FORMAT")


settings = Settings()
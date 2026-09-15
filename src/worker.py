from celery import Celery
from configs.settings import settings
import logging

logger = logging.getLogger(__name__)

# Only initialize Celery if Redis is configured
if settings.redis_url:
    app = Celery('omni_agri', broker=settings.redis_url, backend=settings.redis_url)
    app.conf.update(
        task_serializer='json',
        accept_content=['json'],
        result_serializer='json',
        timezone='UTC',
        enable_utc=True,
    )
else:
    app = None
    logger.warning("Redis URL not configured - Celery worker disabled")

# Import tasks (will be registered)
if app:
    from src.tasks import process_video_task  # noqa
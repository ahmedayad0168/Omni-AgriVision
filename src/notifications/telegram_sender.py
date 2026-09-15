import requests
from configs.settings import settings
import logging

logger = logging.getLogger(__name__)


class TelegramSender:
    def __init__(self):
        self.token = settings.telegram_bot_token
        self.chat_id = settings.telegram_chat_id

    def send(self, message: str):
        if not self.token or not self.chat_id:
            logger.warning("Telegram not configured; message not sent.")
            return
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {'chat_id': self.chat_id, 'text': message}
        try:
            r = requests.post(url, json=payload, timeout=10)
            r.raise_for_status()
            logger.info("Telegram message sent.")
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
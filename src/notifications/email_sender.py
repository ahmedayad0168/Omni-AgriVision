import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from configs.settings import settings
import logging

logger = logging.getLogger(__name__)


class EmailSender:
    def __init__(self):
        self.smtp_host = settings.smtp_host
        self.smtp_port = settings.smtp_port
        self.user = settings.smtp_user
        self.password = settings.smtp_password

    def send(self, to: str, subject: str, body: str):
        if not all([self.smtp_host, self.smtp_port, self.user, self.password]):
            logger.warning("SMTP not configured; email not sent.")
            return
        msg = MIMEMultipart()
        msg['From'] = self.user
        msg['To'] = to
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.user, self.password)
                server.send_message(msg)
            logger.info(f"Email sent to {to}")
        except Exception as e:
            logger.error(f"Email failed: {e}")
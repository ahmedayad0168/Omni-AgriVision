import json
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class FarmState:
    """
    Simple in-memory state per farm (can be extended to Redis).
    """
    def __init__(self, field_id: int):
        self.field_id = field_id
        self.state: Dict[str, Any] = {}

    def update(self, key: str, value: Any):
        self.state[key] = value

    def get(self, key: str, default=None):
        return self.state.get(key, default)

    def to_json(self) -> str:
        return json.dumps(self.state)
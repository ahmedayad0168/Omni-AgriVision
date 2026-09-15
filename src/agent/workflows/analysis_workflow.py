from src.agent.agent import FarmIntelligenceAgent
from src.data.database import DatabaseManager
import logging

logger = logging.getLogger(__name__)


class AnalysisWorkflow:
    """
    Orchestrates the complete analysis: data ingestion, model inference, agent reasoning.
    """
    def __init__(self, field_id: int, db_manager: DatabaseManager):
        self.field_id = field_id
        self.db = db_manager
        self.agent = FarmIntelligenceAgent(field_id, db_manager)

    def run(self) -> str:
        # Step 1: Trigger agent analysis
        report = self.agent.analyze()
        return report
from src.agent.tools.field_tool import FieldTool
from src.data.database import DatabaseManager
from configs.settings import settings

def test_field_tool():
    db = DatabaseManager(settings.database_url)
    tool = FieldTool(db, 1)
    data = tool.get_data()
    assert isinstance(data, str)
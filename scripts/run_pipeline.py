from src.data.database import DatabaseManager
from src.agent.workflows.analysis_workflow import AnalysisWorkflow
from configs.settings import settings
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--field', type=int, required=True)
    args = parser.parse_args()
    db = DatabaseManager(settings.database_url)
    workflow = AnalysisWorkflow(args.field, db)
    report = workflow.run()
    print(report)

if __name__ == "__main__":
    main()
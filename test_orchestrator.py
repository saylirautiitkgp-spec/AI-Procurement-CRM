from agents.orchestrator.orchestrator import Orchestrator
from services.database_service import DatabaseService

db = DatabaseService()

company = db.get_company(10)

orchestrator = Orchestrator(
    dry_run=True
)

orchestrator.process_company(company)
"""
=========================================================
Workflow
=========================================================

Runs the Company Intelligence Workflow.

Author : AI Procurement CRM
"""

from datetime import datetime

from services.database_service import DatabaseService
from agents.orchestrator.orchestrator import Orchestrator


class Workflow:

    def __init__(self):

        self.db = DatabaseService()

        self.agent = Orchestrator()

    def run(self):

        print("=" * 70)
        print("AI PROCUREMENT CRM")
        print("=" * 70)

        start = datetime.now()

        companies = self.db.get_pending_companies()[:1]

        if len(companies) == 0:

            print("\nNo pending companies found.")

            return

        print(f"\nFound {len(companies)} pending companies\n")

        success = 0
        failed = 0

        for company in companies:

            company_id = company["company_id"]

            company_name = company["company_name"]

            try:

                print(f"\n{'='*60}")
                print(company_name)
                print(f"{'='*60}")

                self.db.update_status(
                    company_id,
                    "Searching"
                )

                self.agent.process_company(company)

                self.db.update_status(
                    company_id,
                    "Completed"
                )

                success += 1

            except Exception as e:

                failed += 1

                self.db.update_status(

                    company_id,

                    "Failed",

                    str(e)

                )

                print(e)

        finish = datetime.now()

        workflow = {

            "workflow_name": "Company Intelligence",

            "started_at": start.isoformat(),

            "finished_at": finish.isoformat(),

            "companies_processed": len(companies),

            "successful": success,

            "failed": failed,

            "status": "Completed",

            "duration_seconds": (finish-start).total_seconds(),

            "agent_version": "1.0"

        }

        self.db.save_workflow_run(workflow)

        print("\n")
        print("="*70)
        print("WORKFLOW FINISHED")
        print("="*70)
        print(f"Processed : {len(companies)}")
        print(f"Success   : {success}")
        print(f"Failed    : {failed}")
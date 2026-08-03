"""
=========================================================
AI ORCHESTRATOR
=========================================================

Central Brain of the AI Procurement CRM.

Author : AI Procurement CRM
"""

from agents.company.company_agent import CompanyAgent
from services.duplicate_manager import DuplicateManager


class Orchestrator:

    def __init__(self, dry_run=False):

        self.dry_run = dry_run

        # --------------------------------------------------
        # Registered Agents
        # --------------------------------------------------

        self.agents = [

            CompanyAgent()

        ]

        self.duplicate_manager = DuplicateManager()

    # =====================================================
    # Register Agent
    # =====================================================

    def register(self, agent):

        self.agents.append(agent)

    # =====================================================
    # Run Pipeline
    # =====================================================

    def process_company(self, company):

        print("\n" + "=" * 80)
        print("AI PROCUREMENT CRM ORCHESTRATOR")
        print("=" * 80)

        print(f"\nCompany : {company['company_name']}")

        # --------------------------------------------------
        # Execute Registered Agents
        # --------------------------------------------------

        for agent in self.agents:

            print()

            print("-" * 60)

            print(f"Running {agent.__class__.__name__}")

            print("-" * 60)

            if self.dry_run:

                print("Skipped (Dry Run)")

                continue

            agent.process_company(company)

        # --------------------------------------------------
        # Duplicate Resolution
        # --------------------------------------------------

        if self.dry_run:

            print("\nDuplicate Manager Skipped")

        else:

            print("\nRunning Duplicate Manager")

            self.duplicate_manager.merge_all()

        print()

        print("=" * 80)

        print("ORCHESTRATOR COMPLETED")

        print("=" * 80)
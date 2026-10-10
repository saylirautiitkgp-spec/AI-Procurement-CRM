"""
=========================================================
ORCHESTRATOR TEST
=========================================================

Tests the AI Procurement CRM Orchestrator.

Author : AI Procurement CRM
"""

from agents.orchestrator.orchestrator import Orchestrator
from services.database_service import DatabaseService


# =====================================================
# DATABASE
# =====================================================

db = DatabaseService()


# =====================================================
# GET TEST COMPANY
# =====================================================

company = db.get_company(10)


# =====================================================
# VALIDATE COMPANY
# =====================================================

if not company:

    print("Company not found.")

    raise SystemExit


# =====================================================
# CREATE ORCHESTRATOR
# =====================================================

orchestrator = Orchestrator(
    dry_run=True
)


# =====================================================
# RUN ORCHESTRATOR
# =====================================================

context = orchestrator.process_company(
    company
)


# =====================================================
# DISPLAY FINAL CONTEXT
# =====================================================

print("\n" + "=" * 80)
print("FINAL CONTEXT")
print("=" * 80)


if context:

    print(
        "\nCompany ID:",
        context.company_id
    )

    print("\nRegistered Agents:")

    for agent in context.get_metadata(
        "registered_agents",
        []
    ):

        print(f"  - {agent}")

    print(
        "\nSuccessful Agents:",
        context.get_metadata(
            "successful_agents",
            0
        )
    )

    print(
        "Failed Agents:",
        context.get_metadata(
            "failed_agents",
            0
        )
    )

    print(
        "Errors:",
        len(context.errors)
    )


print("\n" + "=" * 80)
print("ORCHESTRATOR TEST COMPLETED")
print("=" * 80)
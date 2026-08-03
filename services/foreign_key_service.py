"""
=========================================================
Foreign Key Service
=========================================================

Moves all foreign keys from duplicate company
to the master company.

Author : AI Procurement CRM
"""

from services.database_service import DatabaseService


class ForeignKeyService:

    def __init__(self):

        self.db = DatabaseService()

        # Add future tables here
        self.tables = [

            "company_profile",

            "ai_insights",

            "contacts",

            "projects",

            "meetings",

            "documents",

            "opportunities"

        ]

    # =====================================================
    # Move Company References
    # =====================================================

    def move_company(

        self,

        master_company_id,

        duplicate_company_id

    ):

        print("\nMoving Foreign Keys...")

        for table in self.tables:

            try:

                (
                    self.db.db
                    .table(table)
                    .update({
                        "company_id": master_company_id
                    })
                    .eq(
                        "company_id",
                        duplicate_company_id
                    )
                    .execute()
                )

                print(f"✓ {table}")

            except Exception:

                print(f"Skipped {table}")

        print("\nForeign Keys Updated")
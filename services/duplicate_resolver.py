"""
=========================================================
Duplicate Resolver
=========================================================

Merges duplicate companies into a single Golden Record.

Author : AI Procurement CRM
"""

from services.company_merge_service import CompanyMergeService
from services.foreign_key_service import ForeignKeyService
from services.database_service import DatabaseService


class DuplicateResolver:

    def __init__(self):

        self.merge = CompanyMergeService()

        self.foreign = ForeignKeyService()

        self.db = DatabaseService()

    # =====================================================
    # Resolve Duplicate
    # =====================================================

    def resolve(

        self,

        master_company_id,

        duplicate_company_id

    ):

        print()

        print("=" * 70)

        print("RESOLVING DUPLICATE")

        print("=" * 70)

        # -----------------------------
        # Merge Company Profile
        # -----------------------------

        duplicate_profile = self.db.get_company_profile(

            duplicate_company_id

        )

        if duplicate_profile:

            self.merge.merge_profile(

                master_company_id,

                duplicate_profile

            )

        # -----------------------------
        # Move Foreign Keys
        # -----------------------------

        self.foreign.move_company(

            master_company_id,

            duplicate_company_id

        )

        # -----------------------------
        # Delete Duplicate Company
        # -----------------------------

        self.db.save_merge_log({

    "master_company_id": master_company_id,

    "duplicate_company_id": duplicate_company_id,

    "merged_by": "DuplicateResolver",

    "confidence": 1.0,

    "notes": "Automatic duplicate merge"

})

        self.db.delete(

            "companies",

            "company_id",

            duplicate_company_id

        )

        print()

        print("Duplicate Removed")
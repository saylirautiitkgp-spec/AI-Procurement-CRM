"""
=========================================================
Duplicate Manager
=========================================================

Controls the complete duplicate resolution workflow.

Author : AI Procurement CRM
"""

from services.duplicate_detection_service import CompanyDuplicateService
from services.duplicate_resolver import DuplicateResolver


class DuplicateManager:

    def __init__(self):

        self.detector = CompanyDuplicateService()

        self.resolver = DuplicateResolver()

    # =====================================================
    # Merge Everything
    # =====================================================

    def merge_all(self):

        duplicates = self.detector.find_duplicates()

        if len(duplicates) == 0:

            print("\nNo duplicates found.")

            return

        print("\nDuplicates Found :", len(duplicates))

        for duplicate in duplicates:

            master_id = duplicate["master_company_id"]

            duplicate_id = duplicate["duplicate_company_id"]

            print()

            print("=" * 70)

            print(

                duplicate["master_company_name"],

                "<--",

                duplicate["duplicate_company_name"]

            )

            self.resolver.resolve(

                master_id,

                duplicate_id

            )

        print()

        print("=" * 70)

        print("Duplicate Resolution Finished")

        print("=" * 70)
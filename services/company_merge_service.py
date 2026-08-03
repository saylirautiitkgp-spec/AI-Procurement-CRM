"""
=========================================================
Company Merge Service
=========================================================

Merges duplicate companies into a single complete record.

Author : AI Procurement CRM
"""

from services.database_service import DatabaseService
from services.merge_engine import MergeEngine

class CompanyMergeService:

    def __init__(self):

        self.db = DatabaseService()

        self.engine = MergeEngine()

    # =====================================================
    # Merge two dictionaries
    # =====================================================

    def merge_dicts(self, old_data, new_data):

        merged = {}

        keys = set(old_data.keys()) | set(new_data.keys())

        for key in keys:

            old = old_data.get(key)

            new = new_data.get(key)

            # Prefer new value if old is empty

            if old in [None, "", [], {}]:

                merged[key] = new

            else:

                merged[key] = old

        return merged

    # =====================================================
    # Merge company profile
    # =====================================================

    def merge_profile(self, company_id, profile):

        existing = self.db.get_company_profile(company_id)

        if existing:

            merged = self.engine.merge(

            existing,

            profile

        )

            

            self.db.update(

                "company_profile",

                merged,

                "company_id",

                company_id

            )

        else:

            profile["company_id"] = company_id

            self.db.insert(

                "company_profile",

                profile

            )
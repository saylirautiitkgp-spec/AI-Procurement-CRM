"""
=========================================================
Database Writer
=========================================================

Maps AI outputs to the correct database tables.

Author: AI Procurement CRM
"""

from datetime import datetime
from services.database_service import DatabaseService
from services.company_merge_service import CompanyMergeService

class DatabaseWriter:

    def __init__(self):

        self.db = DatabaseService()
        self.merge = CompanyMergeService()
    # =====================================================
    # Main Entry
    # =====================================================

    def write(self, company_id, data):

        self._update_company(company_id, data)

        self.merge.merge_profile(

            company_id,

            data

        )

        self._write_ai_insight(company_id, data)

    # =====================================================
    # Companies Table
    # =====================================================

    def _update_company(self, company_id, data):

        update = {}

        company_fields = [

            "website",
            "linkedin_url",
            "sector",
            "sub_sector"

        ]

        for field in company_fields:

            if field in data and data[field] is not None:

                update[field] = data[field]

        if len(update) == 0:

            return

        update["last_enriched"] = datetime.now().isoformat()

        update["enrichment_status"] = "Completed"

        (
            self.db.db
            .table("companies")
            .update(update)
            .eq("company_id", company_id)
            .execute()
        )

    # =====================================================
    # Company Profile
    # =====================================================

    def _update_company_profile(self, company_id, data):

        existing = self.db.get_company_profile(company_id)

        profile = {}

        profile_fields = [

            "page_title",
            "meta_description",
            "company_description",
            "company_summary",
            "industry",
            "headquarters",
            "founded_year",
            "employee_count",
            "company_size",
            "annual_revenue",
            "ownership_type",
            "company_type",
            "parent_company",
            "website_domain",
            "products",
            "services",
            "business_units",
            "technologies",
            "certifications",
            "industries_served",
            "subsidiaries",
            "manufacturing_locations",
            "supplier_portal",
            "procurement_page",
            "contact_page",
            "about_page",
            "sustainability_page",
            "investor_relations_page",
            "confidence_score",
            "extracted_text"

        ]

        for field in profile_fields:

            value = data.get(field)

            if value is None:

                continue

            if isinstance(value, str) and value.strip() == "":

                continue

            profile[field] = value

        if existing:

            (
                self.db.db
                .table("company_profile")
                .update(profile)
                .eq("company_id", company_id)
                .execute()
            )

        else:

            profile["company_id"] = company_id

            (
                self.db.db
                .table("company_profile")
                .insert(profile)
                .execute()
            )

    # =====================================================
    # AI Insights
    # =====================================================

    def _write_ai_insight(self, company_id, data):

        if not data.get("company_summary"):

            return

        (
            self.db.db
            .table("ai_insights")
            .insert({

                "company_id": company_id,

                "insight_type": "Company Summary",

                "insight": data["company_summary"],

                "confidence_score": data.get(
                    "confidence_score",
                    0.95
                ),

                "generated_by": "CompanyAgent"

            })
            .execute()
        )
"""
=========================================================
Enrichment Planner
=========================================================

Determines which fields still need to be enriched.

Author: AI Procurement CRM
"""


class EnrichmentPlanner:

    def __init__(self):
        pass

    # =====================================================
    # BUILD ENRICHMENT PLAN
    # =====================================================

    def build_plan(self, company, profile=None):

        tasks = []

        # ---------------------------------------------------
        # COMPANIES TABLE
        # ---------------------------------------------------

        if not company.get("website"):
            tasks.append("website")

        if not company.get("linkedin_url"):
            tasks.append("linkedin")

        if not company.get("sector"):
            tasks.append("scrape")

        if not company.get("sub_sector"):
            tasks.append("scrape")

        # ---------------------------------------------------
        # NO PROFILE EXISTS
        # ---------------------------------------------------

        if profile is None:

            tasks.extend([

                "scrape",

                "supplier_portal",

                "procurement_page"

            ])

            return list(set(tasks))

        # ---------------------------------------------------
        # PROFILE FIELDS
        # ---------------------------------------------------

        important_fields = [

            "company_summary",

            "company_description",

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

            "manufacturing_locations"

        ]

        for field in important_fields:

            value = profile.get(field)

            if value is None:

                if "scrape" not in tasks:

                    tasks.append("scrape")

                break

            if isinstance(value, str) and value.strip() == "":

                if "scrape" not in tasks:

                    tasks.append("scrape")

                break

            if isinstance(value, list) and len(value) == 0:

                if "scrape" not in tasks:

                    tasks.append("scrape")

                break

        # ---------------------------------------------------
        # PROCUREMENT
        # ---------------------------------------------------

        if not profile.get("supplier_portal"):

            tasks.append("supplier_portal")

        if not profile.get("procurement_page"):

            tasks.append("procurement_page")

        # ---------------------------------------------------
        # REMOVE DUPLICATES
        # ---------------------------------------------------

        return list(set(tasks))

    # =====================================================
    # DISPLAY
    # =====================================================

    def display_plan(self, tasks):

        print("\nEnrichment Plan")

        print("-" * 40)

        for task in sorted(tasks):

            print(f"✓ {task}")

        print("-" * 40)
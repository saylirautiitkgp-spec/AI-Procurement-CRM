"""
=========================================================
Business Search Service
=========================================================

Searches trusted business sources ONLY for fields that
are still missing.

Author : AI Procurement CRM
"""

from services.search_service import SearchService


class BusinessSearchService:

    def __init__(self):

        self.search_service = SearchService()

    # =====================================================
    # BUILD SEARCH QUERIES
    # =====================================================

    def build_queries(self, company_name, profile):

        if profile is None:
            profile = {}

        queries = {}

        # -------------------------------------------------
        # Revenue
        # -------------------------------------------------

        if not profile.get("annual_revenue"):

            queries["annual_revenue"] = [

                f"{company_name} annual revenue",

                f"{company_name} turnover",

                f"{company_name} annual report revenue"

            ]

        # -------------------------------------------------
        # Employee Count
        # -------------------------------------------------

        if not profile.get("employee_count"):

            queries["employee_count"] = [

                f"{company_name} employees",

                f"{company_name} employee count",

                f"{company_name} company size",

                f"{company_name} LinkedIn employees"

            ]

        # -------------------------------------------------
        # Founded Year
        # -------------------------------------------------

        if not profile.get("founded_year"):

            queries["founded_year"] = [

                f"{company_name} founded",

                f"{company_name} established"

            ]

        # -------------------------------------------------
        # Parent Company
        # -------------------------------------------------

        if not profile.get("parent_company"):

            queries["parent_company"] = [

                f"{company_name} parent company",

                f"{company_name} group company"

            ]

        # -------------------------------------------------
        # Ownership Type
        # -------------------------------------------------

        if not profile.get("ownership_type"):

            queries["ownership_type"] = [

                f"{company_name} listed private public"

            ]

        # -------------------------------------------------
        # Manufacturing Locations
        # -------------------------------------------------

        if not profile.get("manufacturing_locations"):

            queries["manufacturing_locations"] = [

                f"{company_name} manufacturing plants",

                f"{company_name} factories",

                f"{company_name} manufacturing locations"

            ]

        # -------------------------------------------------
        # Certifications
        # -------------------------------------------------

        if not profile.get("certifications"):

            queries["certifications"] = [

                f"{company_name} ISO",

                f"{company_name} AS9100",

                f"{company_name} NADCAP",

                f"{company_name} certifications"

            ]

        # -------------------------------------------------
        # Business Units
        # -------------------------------------------------

        if not profile.get("business_units"):

            queries["business_units"] = [

                f"{company_name} business units",

                f"{company_name} divisions",

                f"{company_name} segments"

            ]

        # -------------------------------------------------
        # Subsidiaries
        # -------------------------------------------------

        if not profile.get("subsidiaries"):

            queries["subsidiaries"] = [

                f"{company_name} subsidiaries",

                f"{company_name} group companies"

            ]

        # -------------------------------------------------
        # Supplier Portal
        # -------------------------------------------------

        if not profile.get("supplier_portal"):

            queries["supplier_portal"] = [

                f"{company_name} supplier portal",

                f"{company_name} vendor registration"

            ]

        # -------------------------------------------------
        # Procurement
        # -------------------------------------------------

        if not profile.get("procurement_page"):

            queries["procurement_page"] = [

                f"{company_name} procurement",

                f"{company_name} sourcing",

                f"{company_name} purchasing"

            ]

        return queries

    # =====================================================
    # SEARCH MISSING FIELDS
    # =====================================================

    def search(self, company_name, profile):

        queries = self.build_queries(

            company_name,

            profile

        )

        all_results = {}

        for field, field_queries in queries.items():

            print(f"Searching {field}")

            field_results = []

            for query in field_queries:

                results = self.search_service.search(

                    query=query,

                    company_name=company_name,

                    max_results=3

                )

                field_results.extend(results)

            # ------------------------------------------
            # Remove duplicate URLs
            # ------------------------------------------

            unique = {}

            for result in field_results:

                url = result.get("url")

                if url not in unique:

                    unique[url] = result

            all_results[field] = list(unique.values())

        return all_results
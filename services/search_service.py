"""
============================================================
Search Service V2
============================================================

Enterprise Search Engine

Pipeline

Query
    ↓
Tavily
    ↓
Validation
    ↓
Deduplicate
    ↓
Ranking
    ↓
Return Results

Author : AI Procurement CRM
"""

from utils.tavily_client import client
from services.ranking_service import RankingService
from services.website_validator import WebsiteValidator


class SearchService:

    def __init__(self):

        self.ranker = RankingService()

        self.validator = WebsiteValidator()

    # =====================================================
    # Generic Search
    # =====================================================

    def search(

        self,

        query,

        company_name="",

        max_results=8

    ):

        try:

            response = client.search(

                query=query,

                search_depth="advanced",

                max_results=max_results

            )

            results = response.get(

                "results",

                []

            )

        except Exception as e:

            print(e)

            return []

        # ------------------------------------------
        # Remove unwanted websites
        # ------------------------------------------

        results = self.validator.filter_results(

            results

        )

        # ------------------------------------------
        # Remove duplicate URLs
        # ------------------------------------------

        unique = {}

        for result in results:

            url = result.get("url")

            if url not in unique:

                unique[url] = result

        results = list(unique.values())

        # ------------------------------------------
        # Rank
        # ------------------------------------------

        results = self.ranker.rank(

            results,

            company_name

        )

        return results

    # =====================================================
    # Multi Query Search
    # =====================================================

    def multi_search(

        self,

        queries,

        company_name="",

        max_results=5

    ):

        all_results = []

        for query in queries:

            print(f"Searching: {query}")

            results = self.search(

                query,

                company_name,

                max_results

            )

            all_results.extend(results)

        # Remove duplicates

        unique = {}

        for result in all_results:

            url = result.get("url")

            if url not in unique:

                unique[url] = result

        ranked = self.ranker.rank(

            list(unique.values()),

            company_name

        )

        return ranked

    # =====================================================
    # Official Website
    # =====================================================

    def search_official_website(

        self,

        company_name

    ):

        queries = [

            f"{company_name} official website",

            f"{company_name} corporate website",

            f"{company_name} homepage"

        ]

        return self.multi_search(

            queries,

            company_name

        )

    # =====================================================
    # LinkedIn
    # =====================================================

    def search_linkedin(

        self,

        company_name

    ):

        try:

            response = client.search(

                query=f"{company_name} LinkedIn",

                search_depth="advanced",

                max_results=5

            )

            return response.get(

                "results",

                []

            )

        except Exception:

            return []

    # =====================================================
    # Supplier
    # =====================================================

    def search_supplier_portal(

        self,

        company_name

    ):

        queries = [

            f"{company_name} supplier portal",

            f"{company_name} vendor registration",

            f"{company_name} supplier",

            f"{company_name} procurement"

        ]

        return self.multi_search(

            queries,

            company_name

        )

    # =====================================================
    # Procurement
    # =====================================================

    def search_procurement(

        self,

        company_name

    ):

        queries = [

            f"{company_name} procurement",

            f"{company_name} sourcing",

            f"{company_name} purchasing"

        ]

        return self.multi_search(

            queries,

            company_name

        )

    # =====================================================
    # Business Intelligence
    # =====================================================

    def search_business(

        self,

        company_name

    ):

        queries = [

            f"{company_name} annual report",

            f"{company_name} annual revenue",

            f"{company_name} employees",

            f"{company_name} parent company",

            f"{company_name} subsidiaries",

            f"{company_name} manufacturing plants",

            f"{company_name} ISO certifications",

            f"{company_name} company profile"

        ]

        return self.multi_search(

            queries,

            company_name

        )

    # =====================================================
    # Best Result
    # =====================================================

    def best(self, results):

        if len(results) == 0:

            return None

        return results[0]

    def best_official_website(self, company):

        return self.best(

            self.search_official_website(company)

        )

    def best_linkedin(self, company):

        return self.best(

            self.search_linkedin(company)

        )

    def best_supplier_portal(self, company):

        return self.best(

            self.search_supplier_portal(company)

        )

    def best_procurement_page(self, company):

        return self.best(

            self.search_procurement(company)

        )

    # =====================================================
    # Debug
    # =====================================================

    def display_results(

        self,

        results

    ):

        print("=" * 80)

        for i, result in enumerate(results, start=1):

            print()

            print(i)

            print(result.get("title"))

            print(result.get("url"))

            print(result.get("score"))

            print("-" * 80)
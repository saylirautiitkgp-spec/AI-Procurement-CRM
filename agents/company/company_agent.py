"""
=========================================================
Company Agent
=========================================================

Intelligent Company Enrichment Agent

Author : AI Procurement CRM
"""

from services.search_service import SearchService
from services.scraper_service import ScraperService
from services.website_parser import WebsiteParser
from services.business_search_service import BusinessSearchService
from services.business_parser import BusinessParser

from services.enrichment_planner import EnrichmentPlanner
from services.database_writer import DatabaseWriter
from services.database_service import DatabaseService
from services.page_finder_service import PageFinderService

class CompanyAgent:

    def __init__(self):

        self.search = SearchService()

        self.scraper = ScraperService()

        self.parser = WebsiteParser()

        self.business_search = BusinessSearchService()

        self.business_parser = BusinessParser()

        self.planner = EnrichmentPlanner()

        self.writer = DatabaseWriter()

        self.db = DatabaseService()

        self.page_finder = PageFinderService()

    # ===================================================
    # MAIN
    # ===================================================

    def process_company(self, company):

        company_id = company["company_id"]

        company_name = company["company_name"]

        print(f"\nProcessing : {company_name}")

        # -------------------------------------------------
        # Existing Profile
        # -------------------------------------------------

        profile = self.db.get_company_profile(company_id)

        # -------------------------------------------------
        # Build Enrichment Plan
        # -------------------------------------------------

        tasks = self.planner.build_plan(

            company,

            profile

        )

        self.planner.display_plan(tasks)

        result = {}

        # =================================================
        # WEBSITE
        # =================================================

        if "website" in tasks:

            website = self.search.best_official_website(

                company_name

            )

            if website:

                result["website"] = website["url"]

                print("\nOfficial Website")

                print(result["website"])

        # =================================================
        # LINKEDIN
        # =================================================

        if "linkedin" in tasks:

            linkedin = self.search.best_linkedin(

                company_name

            )

            if linkedin:

                result["linkedin_url"] = linkedin["url"]

        # ===================================================
        # SCRAPE WEBSITE
        # ===================================================

        if "scrape" in tasks:

            website_url = result.get(

            "website",

            company.get("website")

        )

        if website_url:

            print("\nDownloading Website...")

            pages = self.scraper.scrape_company(

            website_url

        )

        # ------------------------------
        # Find Important Pages
        # ------------------------------

        homepage = pages.get("homepage")

        if homepage:

            discovered_pages = self.page_finder.find_pages(

                homepage,

                website_url

            )

            result.update(discovered_pages)

        print("Parsing Website using Gemini...")

        website_profile = self.parser.parse(

            pages,

            website_url

        )

        result.update(

            website_profile

        )
        # =================================================
        # BUSINESS INTELLIGENCE
        # =================================================

        print("\nSearching Business Intelligence...")
        business_results = self.business_search.search(

        company_name,

        profile

        )

        business_profile = self.business_parser.parse(

        business_results

        )

        result.update(

        business_profile

        )


       

        # =================================================
        # SUPPLIER PORTAL
        # =================================================

        if "supplier_portal" in tasks:

            supplier = self.search.best_supplier_portal(

                company_name

            )

            if supplier:

                result["supplier_portal"] = supplier["url"]

        # =================================================
        # PROCUREMENT PAGE
        # =================================================

        if "procurement_page" in tasks:

            procurement = self.search.best_procurement_page(

                company_name

            )

            if procurement:

                result["procurement_page"] = procurement["url"]

        # =================================================
        # SAVE
        # =================================================

        print("\nSaving to database...")

        self.writer.write(

            company_id,

            result

        )

        print("\nCompleted Successfully")
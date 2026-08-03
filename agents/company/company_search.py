from services.search_service import SearchService


class CompanySearch:

    def __init__(self):

        self.search_service = SearchService()

    def search_company(self, company_name):

        company_data = {

            "company_name": company_name,

            "website": self.search_service.best_official_website(company_name),

            "linkedin": self.search_service.best_linkedin(company_name),

            "supplier_portal": self.search_service.best_supplier_portal(company_name),

            "procurement": self.search_service.best_procurement_page(company_name)

        }

        return company_data
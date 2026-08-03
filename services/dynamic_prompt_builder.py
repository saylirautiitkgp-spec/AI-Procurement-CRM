"""
=========================================================
Dynamic Prompt Builder
=========================================================

Builds prompts based on missing fields.

Author : AI Procurement CRM
"""


class DynamicPromptBuilder:

    def __init__(self):

        self.field_descriptions = {

            "annual_revenue":
                "Annual Revenue",

            "employee_count":
                "Employee Count",

            "headquarters":
                "Headquarters",

            "founded_year":
                "Founded Year",

            "industry":
                "Industry",

            "company_size":
                "Company Size",

            "ownership_type":
                "Ownership Type",

            "company_type":
                "Company Type",

            "parent_company":
                "Parent Company",

            "products":
                "Products",

            "services":
                "Services",

            "business_units":
                "Business Units",

            "technologies":
                "Technologies",

            "certifications":
                "Certifications",

            "industries_served":
                "Industries Served",

            "subsidiaries":
                "Subsidiaries",

            "manufacturing_locations":
                "Manufacturing Locations",

            "supplier_portal":
                "Supplier Portal",

            "procurement_page":
                "Procurement Page",

            "about_page":
                "About Page",

            "contact_page":
                "Contact Page",

            "investor_relations_page":
                "Investor Relations Page",

            "sustainability_page":
                "Sustainability Page"
        }

    # ===================================================
    # Missing Fields
    # ===================================================

    def get_missing_fields(self, profile):

        if profile is None:

            return list(self.field_descriptions.keys())

        missing = []

        for field in self.field_descriptions:

            value = profile.get(field)

            if value in [

                None,

                "",

                [],

                {}

            ]:

                missing.append(field)

        return missing

    # ===================================================
    # Build Prompt
    # ===================================================

    def build_prompt(

        self,

        profile

    ):

        missing = self.get_missing_fields(profile)

        prompt = "Extract ONLY the following fields:\n\n"

        for field in missing:

            prompt += f"- {self.field_descriptions[field]}\n"

        return prompt
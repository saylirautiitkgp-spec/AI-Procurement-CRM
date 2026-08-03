"""
=========================================================
Business Parser
=========================================================

Uses Gemini to extract structured business intelligence
from trusted business search results.

Author : AI Procurement CRM
"""

from services.llm_service import LLMService
from agents.company.prompts.business_profile_prompt import (
    BUSINESS_PROFILE_PROMPT,
)


class BusinessParser:

    def __init__(self):

        self.llm = LLMService()

    # =====================================================
    # BUILD CONTEXT
    # =====================================================

    def build_context(self, search_results):

        context = ""

        preferred_order = [

            "annual_revenue",

            "employee_count",

            "founded_year",

            "parent_company",

            "ownership_type",

            "manufacturing_locations",

            "certifications",

            "business_units",

            "subsidiaries",

            "supplier_portal",

            "procurement_page"

        ]

        for field in preferred_order:

            if field not in search_results:

                continue

            results = search_results[field]

            if not results:

                continue

            context += "\n"
            context += "=" * 70
            context += "\n"

            context += field.upper()

            context += "\n"

            context += "=" * 70

            context += "\n\n"

            for result in results:

                title = result.get("title", "")

                url = result.get("url", "")

                content = result.get("content", "")

                context += f"TITLE : {title}\n"

                context += f"URL : {url}\n"

                context += f"CONTENT : {content}\n\n"

        return context

    # =====================================================
    # PARSE
    # =====================================================

    def parse(self, search_results):

        context = self.build_context(search_results)

        prompt = BUSINESS_PROFILE_PROMPT.replace(

            "{{TEXT}}",

            context

        )

        result = self.llm.generate_json(prompt)

        if result is None:

            return {}

        return result
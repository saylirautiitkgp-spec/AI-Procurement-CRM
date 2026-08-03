"""
=========================================================
Website Parser
=========================================================

Extracts clean website text and sends it to the LLM.

Author : AI Procurement CRM
"""

import re

from bs4 import BeautifulSoup

from services.llm_service import LLMService
from agents.company.prompts.company_profile_prompt import (
    COMPANY_PROFILE_PROMPT,
)


class WebsiteParser:

    def __init__(self):

        self.llm = LLMService()

    # =====================================================
    # REMOVE UNWANTED HTML
    # =====================================================

    def remove_noise(self, soup: BeautifulSoup):

        remove_tags = [

            "script",
            "style",
            "noscript",
            "svg",
            "img",
            "iframe",
            "footer",
            "header",
            "form"

        ]

        for tag in remove_tags:

            for element in soup.find_all(tag):

                element.decompose()

        return soup

    # =====================================================
    # CLEAN TEXT
    # =====================================================

    def clean_text(self, text):

        text = re.sub(r"\s+", " ", text)

        text = text.replace("\xa0", " ")

        text = text.strip()

        return text

    # =====================================================
    # EXTRACT WEBSITE TEXT
    # =====================================================

    def extract_text(self, soup: BeautifulSoup):

        soup = self.remove_noise(soup)

        text = soup.get_text(separator=" ")

        text = self.clean_text(text)

        return text

    # =====================================================
    # LIMIT TOKENS
    # =====================================================

    def truncate_text(

        self,

        text,

        max_chars=30000

    ):

        if len(text) <= max_chars:

            return text

        return text[:max_chars]

    # =====================================================
    # BUILD PROMPT
    # =====================================================

    def build_prompt(self, text):

        return COMPANY_PROFILE_PROMPT.replace(

            "{{TEXT}}",

            text

        )

    # =====================================================
    # LLM EXTRACTION
    # =====================================================

    def extract_company_profile(self, text):

        prompt = self.build_prompt(text)

        result = self.llm.generate_json(prompt)

        if result is None:

            return {}

        return result

    def parse_text(self, text, url=None):

        text = self.truncate_text(text)

        profile = self.extract_company_profile(text)

        profile = self.normalize(profile)

        profile["source_url"] = url

        profile["extracted_text"] = text

        return profile
    # =====================================================
    # SAFE GET
    # =====================================================

    def safe_get(

        self,

        data,

        key,

        default=None

    ):

        if key not in data:

            return default

        return data[key]

    # =====================================================
    # NORMALIZE JSON
    # =====================================================
    def normalize(self, data):

        return {

            # ---------------------------
            # Companies Table
            # ---------------------------

            "sector": self.safe_get(data, "sector"),

            "sub_sector": self.safe_get(data, "sub_sector"),

            # ---------------------------
            # Company Profile
            # ---------------------------

            "page_title": self.safe_get(data, "page_title"),

            "meta_description": self.safe_get(data, "meta_description"),

            "company_description": self.safe_get(data, "company_description"),

            "company_summary": self.safe_get(data, "company_summary"),

            "industry": self.safe_get(data, "industry"),

            "headquarters": self.safe_get(data, "headquarters"),

            "founded_year": self.safe_get(data, "founded_year"),

            "employee_count": self.safe_get(data, "employee_count"),

            "company_size": self.safe_get(data, "company_size"),

            "annual_revenue": self.safe_get(data, "annual_revenue"),

            "ownership_type": self.safe_get(data, "ownership_type"),

            "company_type": self.safe_get(data, "company_type"),

            "parent_company": self.safe_get(data, "parent_company"),

            "website_domain": self.safe_get(data, "website_domain"),

            "products": self.safe_get(data, "products", []),

            "services": self.safe_get(data, "services", []),

            "business_units": self.safe_get(data, "business_units", []),

            "technologies": self.safe_get(data, "technologies", []),

            "certifications": self.safe_get(data, "certifications", []),

            "industries_served": self.safe_get(data, "industries_served", []),

            "subsidiaries": self.safe_get(data, "subsidiaries", []),

            "manufacturing_locations": self.safe_get(
                data,
                "manufacturing_locations",
                []
            ),

            "supplier_portal": self.safe_get(data, "supplier_portal"),

            "procurement_page": self.safe_get(data, "procurement_page"),

            "contact_page": self.safe_get(data, "contact_page"),

            "about_page": self.safe_get(data, "about_page"),

            "sustainability_page": self.safe_get(data, "sustainability_page"),

            "investor_relations_page": self.safe_get(
                data,
                "investor_relations_page"
            ),

            "confidence_score": self.safe_get(
                data,
                "confidence_score",
                0.0
            )

        }
        # =====================================================
    # PARSE WEBSITE
    # =====================================================

    def parse(self, pages, url=None):

        combined_text = ""

        metadata = {}

        for page_name, soup in pages.items():

            if soup is None:

                continue

            text = self.extract_text(soup)

            combined_text += f"\n\n===== {page_name.upper()} =====\n"

            combined_text += text

            if page_name == "homepage":

                if soup.title:

                    metadata["page_title"] = soup.title.text.strip()

                meta = soup.find(

                    "meta",

                    attrs={"name": "description"}

                )

                if meta:

                    metadata["meta_description"] = meta.get(

                        "content"

                    )

        combined_text = self.truncate_text(

            combined_text,

            max_chars=100000

        )

        profile = self.extract_company_profile(

            combined_text

        )

        profile = self.normalize(profile)

        profile["page_title"] = metadata.get(

            "page_title"

        )

        profile["meta_description"] = metadata.get(

            "meta_description"

        )

        profile["company_description"] = profile.get(

            "company_summary"

        )

        profile["extracted_text"] = combined_text

        profile["source_url"] = url

        return profile

        # -----------------------------
        # Add Metadata
        # -----------------------------

        profile["page_title"] = soup.title.text.strip() if soup.title else None

        description = None

        meta = soup.find("meta", attrs={"name": "description"})

        if meta:

            description = meta.get("content")

        profile["meta_description"] = description

        profile["company_description"] = profile.get(
            "company_summary"
        )

        profile["extracted_text"] = text

        profile["source_url"] = url

        return profile
"""
=========================================================
Scraper Service
=========================================================

Downloads webpages and automatically discovers important
company pages.

Author : AI Procurement CRM
"""

import requests

from bs4 import BeautifulSoup
from urllib.parse import urljoin


class ScraperService:

    def __init__(self):

        self.headers = {

            "User-Agent":
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

        }

    # =====================================================
    # DOWNLOAD PAGE
    # =====================================================

    def scrape(self, url):

        try:

            response = requests.get(

                url,

                headers=self.headers,

                timeout=20

            )

            response.raise_for_status()

            return BeautifulSoup(

                response.text,

                "html.parser"

            )

        except Exception as e:

            print(f"Failed : {url}")

            print(e)

            return BeautifulSoup("", "html.parser")

    # =====================================================
    # FIND IMPORTANT LINKS
    # =====================================================

    def discover_links(self, soup, base_url):

        keywords = {

            "about": [
                "about",
                "about-us",
                "company",
                "who-we-are"
            ],

            "products": [
                "products",
                "solutions",
                "portfolio"
            ],

            "services": [
                "services"
            ],

            "technology": [
                "technology",
                "innovation",
                "capabilities"
            ],

            "manufacturing": [
                "manufacturing",
                "plants",
                "locations",
                "facilities"
            ],

            "certifications": [
                "certifications",
                "quality",
                "iso"
            ],

            "supplier": [
                "supplier",
                "vendor"
            ],

            "procurement": [
                "procurement",
                "purchase",
                "sourcing"
            ],

            "careers": [
                "career",
                "careers",
                "jobs"
            ],

            "contact": [
                "contact"
            ],

            "sustainability": [
                "sustainability",
                "esg"
            ],

            "investors": [
                "investor",
                "investors"
            ]

        }

        pages = {}

        for link in soup.find_all("a", href=True):

            href = link["href"].lower()

            full = urljoin(base_url, href)

            for page_type, words in keywords.items():

                if page_type in pages:
                    continue

                if any(word in href for word in words):

                    pages[page_type] = full

        return pages

    # =====================================================
    # SCRAPE COMPLETE WEBSITE
    # =====================================================

    def scrape_company(self, website):

        pages = {}

        home = self.scrape(website)

        pages["homepage"] = home

        discovered = self.discover_links(

            home,

            website

        )

        for page, url in discovered.items():

            print(f"Scraping {page}: {url}")

            pages[page] = self.scrape(url)

        return pages
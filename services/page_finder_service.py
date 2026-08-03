"""
=========================================================
Page Finder Service
=========================================================

Discovers important company pages from the homepage
and intelligently probes common URLs.

Author : AI Procurement CRM
"""

import requests

from urllib.parse import urljoin
from urllib.parse import urlparse


class PageFinderService:

    def __init__(self):

        self.page_keywords = {

            "about_page": [

                "about",
                "about-us",
                "who-we-are",
                "company",
                "corporate"

            ],

            "contact_page": [

                "contact",
                "contact-us",
                "reach-us",
                "locations"

            ],

            "supplier_portal": [

                "supplier",
                "supplier-portal",
                "vendor",
                "vendor-registration",
                "partner"

            ],

            "procurement_page": [

                "procurement",
                "purchasing",
                "sourcing",
                "rfq",
                "tender"

            ],

            "investor_relations_page": [

                "investor",
                "investors",
                "investor-relations",
                "annual-report",
                "financials"

            ],

            "sustainability_page": [

                "sustainability",
                "esg",
                "csr",
                "environment"

            ]

        }

    # =====================================================
    # FIND FROM NAVIGATION
    # =====================================================

    def find_pages(self, soup, base_url):

        pages = {}

        links = soup.find_all("a", href=True)

        for link in links:

            href = link.get("href")

            text = link.get_text(" ", strip=True).lower()

            href_lower = href.lower()

            combined = text + " " + href_lower

            full_url = urljoin(base_url, href)

            for page, keywords in self.page_keywords.items():

                if page in pages:

                    continue

                for keyword in keywords:

                    if keyword in combined:

                        pages[page] = full_url

                        break

        # Try standard URLs for anything still missing

        pages = self.try_standard_urls(

            pages,

            base_url

        )

        return pages

    # =====================================================
    # TRY COMMON URLS
    # =====================================================

    def try_standard_urls(self, pages, base_url):

        parsed = urlparse(base_url)

        root = f"{parsed.scheme}://{parsed.netloc}"

        candidates = {

            "about_page": [

                "/about",

                "/about-us",

                "/company"

            ],

            "contact_page": [

                "/contact",

                "/contact-us"

            ],

            "supplier_portal": [

                "/supplier",

                "/suppliers",

                "/vendor",

                "/vendor-registration"

            ],

            "procurement_page": [

                "/procurement",

                "/sourcing",

                "/purchasing"

            ],

            "investor_relations_page": [

                "/investors",

                "/investor-relations",

                "/annual-report"

            ],

            "sustainability_page": [

                "/sustainability",

                "/esg",

                "/csr"

            ]

        }

        headers = {

            "User-Agent":

            "Mozilla/5.0"

        }

        for page, urls in candidates.items():

            if page in pages:

                continue

            for path in urls:

                url = root + path

                try:

                    r = requests.get(

                        url,

                        headers=headers,

                        timeout=5,

                        allow_redirects=True

                    )

                    if r.status_code == 200:

                        pages[page] = r.url

                        break

                except Exception:

                    pass

        return pages
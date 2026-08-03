"""
============================================================
Link Discovery Service
============================================================
"""

from urllib.parse import urljoin


class LinkDiscovery:

    KEYWORDS = {

        "about": [
            "about",
            "about-us",
            "company",
            "who-we-are"
        ],

        "contact": [
            "contact",
            "contact-us",
            "reach-us"
        ],

        "careers": [
            "career",
            "careers",
            "jobs"
        ],

        "supplier": [
            "supplier",
            "vendors",
            "vendor",
            "partners"
        ],

        "procurement": [
            "procurement",
            "purchasing",
            "sourcing",
            "rfq"
        ],

        "sustainability": [
            "sustainability",
            "esg",
            "environment"
        ]
    }

    def discover(self, soup, base_url):

        pages = {}

        for a in soup.find_all("a", href=True):

            href = a["href"].lower()

            full_url = urljoin(base_url, href)

            for page_type, words in self.KEYWORDS.items():

                if page_type in pages:
                    continue

                if any(word in href for word in words):

                    pages[page_type] = full_url

        return pages
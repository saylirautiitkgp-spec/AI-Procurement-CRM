"""
============================================================
Website Validator
============================================================

Filters Tavily search results before they are ranked.

Rejects:
- Social media
- News sites
- PDFs
- Job portals
- Search pages
- File hosts

Author : AI Procurement CRM
"""

from urllib.parse import urlparse


class WebsiteValidator:

    def __init__(self):

        # ----------------------------------------------------
        # Domains to completely reject
        # ----------------------------------------------------

        self.blacklisted_domains = {

            # Social

            "linkedin.com",
            "facebook.com",
            "instagram.com",
            "twitter.com",
            "x.com",
            "youtube.com",

            # Jobs

            "glassdoor.com",
            "indeed.com",
            "ambitionbox.com",
            "naukri.com",
            "foundit.in",

            # Business directories

            "crunchbase.com",
            "zoominfo.com",
            "rocketreach.co",

            # News

            "moneycontrol.com",
            "economictimes.indiatimes.com",
            "business-standard.com",
            "livemint.com",
            "bloomberg.com",
            "reuters.com",

            # Knowledge

            "wikipedia.org",

            # Documents

            "scribd.com",
            "researchgate.net",
            "academia.edu",
            "slideshare.net",
            "issuu.com",
            "pdfcoffee.com",

            # File hosts

            "drive.google.com",
            "dropbox.com"

        }

        # ----------------------------------------------------
        # URL keywords to reject
        # ----------------------------------------------------

        self.blacklisted_keywords = {

            "career",
            "careers",
            "job",
            "jobs",

            "news",

            "press",

            "media",

            "login",

            "signin",

            "register",

            "support",

            "help",

            "search",

            "download",

            "store",

            "shop"

        }

        # ----------------------------------------------------
        # File extensions
        # ----------------------------------------------------

        self.invalid_extensions = {

            ".pdf",

            ".doc",

            ".docx",

            ".ppt",

            ".pptx",

            ".xls",

            ".xlsx",

            ".zip"

        }

    # ========================================================
    # Extract domain
    # ========================================================

    def get_domain(self, url):

        try:

            return urlparse(url).netloc.lower()

        except Exception:

            return ""

    # ========================================================
    # Reject bad domains
    # ========================================================

    def is_valid_domain(self, url):

        domain = self.get_domain(url)

        for bad in self.blacklisted_domains:

            if bad in domain:

                return False

        return True

    # ========================================================
    # Reject PDFs / documents
    # ========================================================

    def is_document(self, url):

        url = url.lower()

        for ext in self.invalid_extensions:

            if url.endswith(ext):

                return True

        return False

    # ========================================================
    # Reject unwanted pages
    # ========================================================

    def is_valid_page(self, url):

        url = url.lower()

        for keyword in self.blacklisted_keywords:

            if keyword in url:

                return False

        return True

    # ========================================================
    # Reject search results
    # ========================================================

    def is_search_page(self, url):

        url = url.lower()

        patterns = [

            "/search",

            "?q=",

            "&q=",

            "/results",

            "/find"

        ]

        for pattern in patterns:

            if pattern in url:

                return True

        return False

    # ========================================================
    # Final Validation
    # ========================================================

    def validate(self, result):

        url = result.get("url", "")

        if not url:

            return False

        if self.is_document(url):

            return False

        if not self.is_valid_domain(url):

            return False

        if not self.is_valid_page(url):

            return False

        if self.is_search_page(url):

            return False

        return True

    # ========================================================
    # Filter Results
    # ========================================================

    def filter_results(self, results):

        filtered = []

        seen = set()

        for result in results:

            url = result.get("url", "")

            if url in seen:

                continue

            if self.validate(result):

                filtered.append(result)

                seen.add(url)

        return filtered
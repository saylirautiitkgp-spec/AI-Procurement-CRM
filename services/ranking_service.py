"""
============================================================
Ranking Service
============================================================

Ranks Tavily search results and returns the most likely
official company website.

Shared by all AI Agents.

Author : AI Procurement CRM
"""

from urllib.parse import urlparse


class RankingService:

    def __init__(self):
        pass

    # --------------------------------------------------------
    # Extract Domain
    # --------------------------------------------------------

    def get_domain(self, url):

        try:
            return urlparse(url).netloc.lower()

        except Exception:
            return ""

    # --------------------------------------------------------
    # Score URL
    # --------------------------------------------------------

    def score(self, result, company_name=""):

        url = result.get("url", "")
        title = result.get("title", "")

        url_lower = url.lower()
        title_lower = title.lower()

        domain = self.get_domain(url)

        company = (
            company_name.lower()
            .replace(" ", "")
            .replace("-", "")
            .replace("&", "")
        )

        score = 0

        # =====================================================
        # Domain Quality
        # =====================================================

        if domain.startswith("www."):
            score += 10

        if domain.endswith(".com"):
            score += 40

        elif domain.endswith(".in"):
            score += 35

        elif domain.endswith(".co"):
            score += 20

        elif domain.endswith(".net"):
            score += 5

        elif domain.endswith(".org"):
            score -= 20

        elif domain.endswith(".gov"):
            score -= 40

        # =====================================================
        # Company Name Match
        # =====================================================

        clean_domain = (
            domain.replace(".", "")
            .replace("-", "")
        )

        if company:

            if company in clean_domain:
                score += 250

            if company in title_lower.replace(" ", ""):
                score += 100

        # =====================================================
        # Homepage Bonus
        # =====================================================

        parsed = urlparse(url)

        if parsed.path in ["", "/"]:
            score += 120

        elif len(parsed.path.split("/")) <= 2:
            score += 40

        # =====================================================
        # Official Website Keywords
        # =====================================================

        official_keywords = [

            "official",
            "corporate",
            "global",
            "homepage",
            "home"

        ]

        for word in official_keywords:

            if word in title_lower:

                score += 35

        # =====================================================
        # About Company Pages
        # =====================================================

        about_keywords = [

            "about",
            "company",
            "who-we-are",
            "our-company"

        ]

        for word in about_keywords:

            if word in url_lower:

                score += 25

        # =====================================================
        # Procurement Pages
        # =====================================================

        procurement_keywords = [

            "supplier",
            "suppliers",
            "vendor",
            "vendors",
            "procurement",
            "purchase",
            "purchasing",
            "sourcing",
            "rfq",
            "partner"

        ]

        for word in procurement_keywords:

            if word in url_lower:

                score += 50

            if word in title_lower:

                score += 20

        # =====================================================
        # Bad Domains
        # =====================================================

        bad_domains = [

            "linkedin.com",
            "facebook.com",
            "instagram.com",
            "twitter.com",
            "youtube.com",

            "glassdoor.com",
            "ambitionbox.com",
            "indeed.com",
            "naukri.com",
            "foundit.in",

            "wikipedia.org",
            "crunchbase.com",
            "zoominfo.com",

            "scribd.com",
            "researchgate.net",
            "academia.edu",
            "issuu.com",
            "slideshare.net",
            "pdfcoffee.com",

            "moneycontrol.com",
            "economictimes.indiatimes.com",
            "business-standard.com",
            "bloomberg.com"

        ]

        for bad in bad_domains:

            if bad in domain:

                score -= 500

        # =====================================================
        # Bad URL Keywords
        # =====================================================

        bad_keywords = [

            "career",
            "careers",
            "jobs",
            "job",
            "press",
            "media",
            "news",
            "store",
            "shop",
            "support",
            "help",
            "login"

        ]

        for word in bad_keywords:

            if word in url_lower:

                score -= 80

        # =====================================================
        # PDF / Documents Penalty
        # =====================================================

        document_keywords = [

            ".pdf",
            "pdf",
            "brochure",
            "catalog",
            "manual",
            "datasheet",
            "document"

        ]

        for word in document_keywords:

            if word in url_lower:

                score -= 400

        # =====================================================
        # Bonus for Short URLs
        # =====================================================

        if len(url) < 40:
            score += 30

        return score

    # --------------------------------------------------------
    # Rank Results
    # --------------------------------------------------------

    def rank(self, results, company_name=""):

        ranked = sorted(

            results,

            key=lambda x: self.score(
                x,
                company_name
            ),

            reverse=True

        )

        return ranked

    # --------------------------------------------------------
    # Best Result
    # --------------------------------------------------------

    def best_result(self, results, company_name=""):

        ranked = self.rank(
            results,
            company_name
        )

        if not ranked:
            return None

        return ranked[0]

    # --------------------------------------------------------
    # Display Rankings
    # --------------------------------------------------------

    def display(self, results, company_name=""):

        ranked = self.rank(
            results,
            company_name
        )

        print("=" * 80)
        print("SEARCH RANKINGS")
        print("=" * 80)

        for result in ranked:

            print()

            print("Score :", self.score(result, company_name))

            print("Title :", result.get("title"))

            print("URL   :", result.get("url"))

            print("-" * 80)

        return ranked
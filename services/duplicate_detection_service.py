"""
=========================================================
Duplicate Detection Service
=========================================================

Detects duplicate companies and automatically
selects the Golden Record.

Author : AI Procurement CRM
"""

from difflib import SequenceMatcher

from services.database_service import DatabaseService
from services.duplicate_resolver import DuplicateResolver


class CompanyDuplicateService:

    def __init__(self):

        self.db = DatabaseService()

        self.resolver = DuplicateResolver()

    # =====================================================
    # Normalize Company Name
    # =====================================================

    def normalize(self, name):

        if not name:

            return ""

        name = name.lower()

        remove = [

            "private",

            "limited",

            "ltd",

            "pvt",

            "inc",

            "corp",

            "corporation",

            "&",

            ",",

            ".",

            "-"

        ]

        for word in remove:

            name = name.replace(word, "")

        return " ".join(name.split())

    # =====================================================
    # Similarity
    # =====================================================

    def similarity(self, a, b):

        return SequenceMatcher(

            None,

            self.normalize(a),

            self.normalize(b)

        ).ratio()

    # =====================================================
    # Score Company
    # =====================================================

    def score_company(self, company):

        score = 0

        if company.get("website"):
            score += 10

        if company.get("linkedin_url"):
            score += 10

        if company.get("sector"):
            score += 5

        if company.get("revenue"):
            score += 5

        profile = self.db.get_company_profile(

            company["company_id"]

        )

        if profile:

            score += 20

            if profile.get("company_summary"):
                score += 20

            if profile.get("employee_count"):
                score += 5

            if profile.get("parent_company"):
                score += 5

            if profile.get("manufacturing_locations"):
                score += 5

        insights = self.db.select(

            "ai_insights",

            "company_id",

            company["company_id"]

        )

        score += len(insights) * 10

        return score

    # =====================================================
    # Find Duplicates
    # =====================================================

    def find_duplicates(self):

        companies = self.db.select_all(

            "companies"

        )

        duplicates = []

        visited = set()

        for i, company1 in enumerate(companies):

            id1 = company1["company_id"]

            if id1 in visited:

                continue

            for company2 in companies[i + 1:]:

                id2 = company2["company_id"]

                if id2 in visited:

                    continue

                similarity = self.similarity(

                    company1["company_name"],

                    company2["company_name"]

                )

                if similarity < 0.92:

                    continue

                score1 = self.score_company(

                    company1

                )

                score2 = self.score_company(

                    company2

                )

                if score1 >= score2:

                    master = company1

                    duplicate = company2

                else:

                    master = company2

                    duplicate = company1

                duplicates.append({

                    "master_company_id": master["company_id"],

                    "duplicate_company_id": duplicate["company_id"],

                    "master_company_name": master["company_name"],

                    "duplicate_company_name": duplicate["company_name"],

                    "master_score": max(score1, score2),

                    "duplicate_score": min(score1, score2),

                    "similarity": similarity

                })

                visited.add(

                    duplicate["company_id"]

                )

        return duplicates

    # =====================================================
    # Merge Two Companies
    # =====================================================

    def merge_duplicates(

        self,

        master_company_id,

        duplicate_company_id

    ):

        self.resolver.resolve(

            master_company_id,

            duplicate_company_id

        )
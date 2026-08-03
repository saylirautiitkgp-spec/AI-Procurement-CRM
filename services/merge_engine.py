"""
=========================================================
Merge Engine
=========================================================

Generic Master Data Merge Engine.

Used by:
- Company Agent
- Contact Agent
- Supplier Agent
- News Agent
- Procurement Agent

Author : AI Procurement CRM
"""

from copy import deepcopy


class MergeEngine:

    # =====================================================
    # Empty Value
    # =====================================================

    def is_empty(self, value):

        if value is None:
            return True

        if value == "":
            return True

        if value == []:
            return True

        if value == {}:
            return True

        return False

    # =====================================================
    # Merge Lists
    # =====================================================

    def merge_lists(self, old, new):

        if old is None:
            old = []

        if new is None:
            new = []

        merged = []

        for item in old + new:

            if item not in merged:

                merged.append(item)

        return merged

    # =====================================================
    # Merge Confidence
    # =====================================================

    def merge_confidence(self, old, new):

        if old is None:
            return new

        if new is None:
            return old

        return max(old, new)

    # =====================================================
    # Merge Text
    # =====================================================

    def merge_text(self, old, new):

        if self.is_empty(old):

            return new

        if self.is_empty(new):

            return old

        if len(new) > len(old):

            return new

        return old

    # =====================================================
    # Merge Single Value
    # =====================================================

    def merge_value(self, key, old, new):

        if self.is_empty(old):

            return new

        if self.is_empty(new):

            return old

        if isinstance(old, list):

            return self.merge_lists(old, new)

        if key in [

            "confidence_score",

            "ai_confidence"

        ]:

            return self.merge_confidence(old, new)

        if isinstance(old, str):

            return self.merge_text(old, new)

        return old

    # =====================================================
    # Merge Dictionaries
    # =====================================================

    def merge(self, old_data, new_data):

        merged = deepcopy(old_data)

        keys = set(old_data.keys()) | set(new_data.keys())

        for key in keys:

            old = old_data.get(key)

            new = new_data.get(key)

            merged[key] = self.merge_value(

                key,

                old,

                new

            )

        return merged
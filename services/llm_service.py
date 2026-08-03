"""
=========================================================
LLM Service
=========================================================

Centralized Gemini Service

All AI agents communicate with Gemini through this file.

Author : AI Procurement CRM
"""

import json
import re

from google import genai

from config.settings import GEMINI_API_KEY


class LLMService:

    def __init__(self):

        self.client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        self.model = "gemini-2.5-flash"

    # =====================================================
    # GENERATE
    # =====================================================

    def generate(
        self,
        prompt,
        temperature=0.2
    ):

        response = self.client.models.generate_content(

            model=self.model,

            contents=prompt,

            config={

                "temperature": temperature,

                "response_mime_type": "application/json"

            }

        )

        return response.text

    # =====================================================
    # GENERATE JSON
    # =====================================================

    def generate_json(
        self,
        prompt,
        temperature=0.2
    ):

        response = self.client.models.generate_content(

            model=self.model,

            contents=prompt,

            config={

                "temperature": temperature,

                "response_mime_type": "application/json"

            }

        )

        text = response.text.strip()

        # Remove markdown if Gemini returns it
        text = text.replace("```json", "")
        text = text.replace("```", "")

        # Extract JSON object only
        match = re.search(r"\{.*\}", text, re.DOTALL)

        if match:
            text = match.group()

        try:

            return json.loads(text)

        except Exception as e:

            print("\n" + "=" * 80)
            print("INVALID JSON RETURNED BY GEMINI")
            print("=" * 80)
            print(text)
            print("=" * 80)

            raise e

    # =====================================================
    # SUMMARIZE
    # =====================================================

    def summarize(self, text):

        prompt = f"""

Summarize the following company in under 200 words.

{text}

"""

        return self.generate(prompt)

    # =====================================================
    # EXTRACT
    # =====================================================

    def extract(
        self,
        prompt_template,
        text
    ):

        prompt = prompt_template.replace(

            "{{TEXT}}",

            text

        )

        return self.generate_json(prompt)
from dotenv import load_dotenv
import os

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
print("URL:", SUPABASE_URL)
print("KEY:", "Loaded" if SUPABASE_KEY else "Not Loaded")
print("TAVILY:", "Loaded" if TAVILY_API_KEY else "Not Loaded")
print("Gemini:", "Loaded" if GEMINI_API_KEY else "Not Loaded")
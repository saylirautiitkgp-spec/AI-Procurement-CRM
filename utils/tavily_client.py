"""
Reusable Tavily Client
"""

from tavily import TavilyClient
from config.settings import TAVILY_API_KEY

client = TavilyClient(api_key=TAVILY_API_KEY)
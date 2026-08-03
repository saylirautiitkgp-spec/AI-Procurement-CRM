from services.search_service import SearchService
from services.scraper_service import ScraperService
from services.website_parser import WebsiteParser

search = SearchService()
scraper = ScraperService()
parser = WebsiteParser()

company_name = "Bosch India"

print("=" * 70)
print(f"Searching {company_name}")
print("=" * 70)

website = search.best_official_website(company_name)

print("\nWebsite Found:")
if website is None:

    print("Website not found")

    exit()

print(website["url"])

print("\nDownloading Website...")

soup = scraper.scrape(website["url"])

print("\nSending to Gemini...\n")

profile = parser.parse(
    soup,
    website["url"]
)

print("=" * 70)

for key, value in profile.items():

    print(f"\n{key}")

    print(value)

print("\n" + "=" * 70)
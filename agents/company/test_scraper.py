from services.search_service import SearchService
from services.scraper_service import ScraperService
from services.website_parser import WebsiteParser


company_name = "Bosch India"

search = SearchService()
scraper = ScraperService()
parser = WebsiteParser()

print("=" * 70)
print("Searching for official website...")
print("=" * 70)

website = search.best_official_website(company_name)

if website is None:
    print("No website found.")
    exit()

print("\nWebsite Found:")
print(website["title"])
print(website["url"])

print("\nDownloading Website...\n")

soup = scraper.scrape(website["url"])

company = parser.parse(soup)

print("=" * 70)

print("Company")
print(company_name)

print("\nWebsite")
print(website["url"])

print("\nTitle")
print(company["title"])

print("\nDescription")
print(company["description"])

print("=" * 70)
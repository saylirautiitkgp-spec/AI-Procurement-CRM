from services.duplicate_detection_service import CompanyDuplicateService

service = CompanyDuplicateService()

duplicates = service.find_duplicates()

print()

print("=" * 80)

print("DUPLICATES")

print("=" * 80)

for item in duplicates:

    print()

    print(item["master"]["company_name"])

    print(item["duplicate"]["company_name"])

    print(item["score"])
from agents.company.company_search import CompanySearch

search = CompanySearch()

company = search.search_company("Bosch India")

print("=" * 60)
print(company["company_name"])
print("=" * 60)

print("\nWebsite")
print(company["website"])

print("\nLinkedIn")
print(company["linkedin"])

print("\nSupplier Portal")
print(company["supplier_portal"])

print("\nProcurement")
print(company["procurement"])
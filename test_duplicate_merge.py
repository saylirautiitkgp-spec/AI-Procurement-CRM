from services.duplicate_detection_service import CompanyDuplicateService

service = CompanyDuplicateService()

service.merge_duplicates(

    master_company_id=4,

    duplicate_company_id=3

)
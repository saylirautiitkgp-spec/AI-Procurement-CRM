"""LEGACY MANUAL SCRIPT - merges company 3 into 4 and DELETES company 3 (ON DELETE CASCADE removes its contacts, products,
components, supplier links, lead status, insights and profile). It used to run on import. It now refuses unless an
operator opts in explicitly, and even then only when executed as a script."""

import os

if __name__ == "__main__":
    if os.environ.get("LEGACY_DELETES_ENABLED") != "1":
        raise SystemExit("blocked: legacy merge deletes data; set LEGACY_DELETES_ENABLED=1 only after a reviewed plan")

    from services.duplicate_detection_service import CompanyDuplicateService

    CompanyDuplicateService().merge_duplicates(master_company_id=4, duplicate_company_id=3)

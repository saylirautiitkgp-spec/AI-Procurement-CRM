"""
=========================================================
Database Service
=========================================================

Centralized database access layer for the AI Procurement CRM.

Author: AI Procurement CRM
"""

from datetime import datetime
from database.supabase_client import supabase


class DatabaseService:

    def __init__(self):
        self.db = supabase

    # =====================================================
    # MERGE LOGS
    # =====================================================

    def save_merge_log(self, values):

        (
        self.db
        .table("merge_logs")
        .insert(values)
        .execute()
        )

    # =====================================================
    # COMPANY METHODS
    # =====================================================

    def get_pending_companies(self):

        response = (
            self.db
            .table("companies")
            .select("*")
            .or_(
                "enrichment_status.eq.Pending,"
                "enrichment_status.eq.Failed,"
                "enrichment_status.eq.Searching"
            )
            .execute()
        )

        return response.data

    def get_company(self, company_id):

        response = (
            self.db
            .table("companies")
            .select("*")
            .eq("company_id", company_id)
            .single()
            .execute()
        )

        return response.data

    def update_company(self, company_id, values):

        values["last_enriched"] = datetime.now().isoformat()

        (
            self.db
            .table("companies")
            .update(values)
            .eq("company_id", company_id)
            .execute()
        )

    # =====================================================
    # INCOMPLETE COMPANY PROFILES
    # =====================================================

    def get_incomplete_profiles(self):

        response = (
            self.db
            .table("company_profile")
            .select("company_id")
            .or_(
                ",".join([
                    "company_summary.is.null",
                    "employee_count.is.null",
                    "annual_revenue.is.null",
                    "company_size.is.null",
                    "industry.is.null",
                    "headquarters.is.null",
                    "products.is.null",
                    "services.is.null",
                    "technologies.is.null",
                    "certifications.is.null",
                    "industries_served.is.null"
                ])
            )
            .execute()
        )

        return response.data

    # =====================================================
    # COMPANY PROFILE
    # =====================================================

    def get_company_profile(self, company_id):

        response = (
            self.db
            .table("company_profile")
            .select("*")
            .eq("company_id", company_id)
            .execute()
        )

        if len(response.data) == 0:
            return None

        return response.data[0]

    def upsert_company_profile(self, values):

        (
            self.db
            .table("company_profile")
            .upsert(values)
            .execute()
        )

    # =====================================================
    # AI INSIGHTS
    # =====================================================

    def insert_ai_insight(self, values):

        (
            self.db
            .table("ai_insights")
            .insert(values)
            .execute()
        )

    # =====================================================
    # WORKFLOW STATUS
    # =====================================================

    def update_status(self, company_id, status, error=None):

        update = {
            "enrichment_status": status
        }

        if error:
            update["last_error"] = error

        (
            self.db
            .table("companies")
            .update(update)
            .eq("company_id", company_id)
            .execute()
        )

    # =====================================================
    # WORKFLOW RUNS
    # =====================================================

    def save_workflow_run(self, values):

        (
            self.db
            .table("workflow_runs")
            .insert(values)
            .execute()
        )

    # =====================================================
    # GENERIC UPSERT
    # =====================================================

    def upsert(self, table_name, values):

        (
            self.db
            .table(table_name)
            .upsert(values)
            .execute()
        )

    # =====================================================
    # GENERIC INSERT
    # =====================================================

    def insert(self, table_name, values):

        (
            self.db
            .table(table_name)
            .insert(values)
            .execute()
        )

    # =====================================================
    # GENERIC UPDATE
    # =====================================================

    def update(self, table_name, values, column, value):

        (
            self.db
            .table(table_name)
            .update(values)
            .eq(column, value)
            .execute()
        )

    # =====================================================
    # GENERIC SELECT
    # =====================================================

    def select(self, table_name, column, value):

        response = (
            self.db
            .table(table_name)
            .select("*")
            .eq(column, value)
            .execute()
        )

        return response.data
    
    # =====================================================
    # ALL COMPANIES
    # =====================================================

    def select_all(self, table):

        response = (

            self.db

            .table(table)

            .select("*")

            .execute()

        )

        return response.data
    
    # =====================================================
    # Merge Logs
    # =====================================================

    def save_merge_log(self, values):

        (

        self.db

        .table("merge_logs")

        .insert(values)

        .execute()

        )
        
    # =====================================================
    # DELETE
    # =====================================================

    def delete(
    self,
    table_name,
    column,
    value
    ):

        (
        self.db
        .table(table_name)
        .delete()
        .eq(column, value)
        .execute()
    )
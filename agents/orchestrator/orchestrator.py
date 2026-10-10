"""
=========================================================
AI ORCHESTRATOR
=========================================================

Central Brain of the AI Procurement CRM.

Responsibilities:
- Register AI agents
- Execute agents sequentially
- Maintain shared agent context
- Track execution status
- Handle agent failures
- Support dry-run execution
- Run duplicate resolution
- Provide execution summary

Author : AI Procurement CRM
"""

import inspect
import time

from agents.company.company_agent import CompanyAgent
from services.duplicate_manager import DuplicateManager
from services.agent_context import AgentContext


class Orchestrator:

    # =====================================================
    # INITIALIZATION
    # =====================================================

    def __init__(self, dry_run=False):

        self.dry_run = dry_run

        # -------------------------------------------------
        # AGENT REGISTRY
        # -------------------------------------------------

        self.agents = []

        # Register existing Company Agent
        self.register(CompanyAgent())

        # -------------------------------------------------
        # DUPLICATE MANAGER
        # -------------------------------------------------

        self.duplicate_manager = DuplicateManager()

    # =====================================================
    # REGISTER AGENT
    # =====================================================

    def register(self, agent):

        if agent is None:

            return

        self.agents.append(agent)

    # =====================================================
    # GET REGISTERED AGENTS
    # =====================================================

    def get_agents(self):

        return self.agents

    # =====================================================
    # GET AGENT NAME
    # =====================================================

    def _get_agent_name(self, agent):

        return agent.__class__.__name__

    # =====================================================
    # CHECK CONTEXT SUPPORT
    # =====================================================

    def _supports_context(self, agent):

        try:

            signature = inspect.signature(
                agent.process_company
            )

            parameters = signature.parameters

            # Explicit context parameter
            if "context" in parameters:

                return True

            # Agent accepts **kwargs
            for parameter in parameters.values():

                if (
                    parameter.kind
                    == inspect.Parameter.VAR_KEYWORD
                ):

                    return True

        except Exception:

            pass

        return False

    # =====================================================
    # EXECUTE SINGLE AGENT
    # =====================================================

    def _execute_agent(
        self,
        agent,
        company,
        context
    ):

        agent_name = self._get_agent_name(agent)

        print()
        print("-" * 60)
        print(f"Running {agent_name}")
        print("-" * 60)

        # -------------------------------------------------
        # DRY RUN
        # -------------------------------------------------

        if self.dry_run:

            print("Skipped (Dry Run)")

            context.add_execution_log(
                agent_name=agent_name,
                status="SKIPPED",
                message="Dry Run"
            )

            return

        # -------------------------------------------------
        # START TIMER
        # -------------------------------------------------

        start_time = time.perf_counter()

        try:

            # -------------------------------------------------
            # CONTEXT-AWARE AGENT
            # -------------------------------------------------

            if self._supports_context(agent):

                result = agent.process_company(
                    company,
                    context=context
                )

            # -------------------------------------------------
            # LEGACY / EXISTING AGENT
            # -------------------------------------------------

            else:

                result = agent.process_company(
                    company
                )

            # -------------------------------------------------
            # EXECUTION TIME
            # -------------------------------------------------

            duration = (
                time.perf_counter()
                - start_time
            )

            # -------------------------------------------------
            # SAVE RESULT
            # -------------------------------------------------

            context.set_result(
                agent_name,
                result
            )

            # -------------------------------------------------
            # EXECUTION LOG
            # -------------------------------------------------

            context.add_execution_log(
                agent_name=agent_name,
                status="SUCCESS",
                duration=duration
            )

            print(
                f"✓ {agent_name} "
                f"completed in {duration:.2f}s"
            )

        # -----------------------------------------------------
        # AGENT FAILURE
        # -----------------------------------------------------

        except Exception as error:

            duration = (
                time.perf_counter()
                - start_time
            )

            # Save error
            context.add_error(
                agent_name,
                error
            )

            # Save execution log
            context.add_execution_log(
                agent_name=agent_name,
                status="FAILED",
                duration=duration,
                message=str(error)
            )

            print(
                f"✗ {agent_name} "
                f"failed after {duration:.2f}s"
            )

            print(f"Error: {error}")

            print(
                f"Continuing orchestrator "
                f"after {agent_name} failure..."
            )

    # =====================================================
    # RUN DUPLICATE MANAGER
    # =====================================================

    def _run_duplicate_manager(self, context):

        print()
        print("-" * 60)
        print("Running Duplicate Manager")
        print("-" * 60)

        # -------------------------------------------------
        # DRY RUN
        # -------------------------------------------------

        if self.dry_run:

            print("Skipped (Dry Run)")

            context.add_execution_log(
                agent_name="DuplicateManager",
                status="SKIPPED",
                message="Dry Run"
            )

            return

        # -------------------------------------------------
        # START TIMER
        # -------------------------------------------------

        start_time = time.perf_counter()

        try:

            result = self.duplicate_manager.merge_all()

            duration = (
                time.perf_counter()
                - start_time
            )

            # Save result
            context.set_result(
                "DuplicateManager",
                result
            )

            # Save execution log
            context.add_execution_log(
                agent_name="DuplicateManager",
                status="SUCCESS",
                duration=duration
            )

            print(
                f"✓ Duplicate Manager "
                f"completed in {duration:.2f}s"
            )

        except Exception as error:

            duration = (
                time.perf_counter()
                - start_time
            )

            context.add_error(
                "DuplicateManager",
                error
            )

            context.add_execution_log(
                agent_name="DuplicateManager",
                status="FAILED",
                duration=duration,
                message=str(error)
            )

            print(
                f"✗ Duplicate Manager "
                f"failed after {duration:.2f}s"
            )

            print(f"Error: {error}")

    # =====================================================
    # PROCESS COMPANY
    # =====================================================

    def process_company(self, company):

        print("\n" + "=" * 80)
        print("AI PROCUREMENT CRM ORCHESTRATOR")
        print("=" * 80)

        # -------------------------------------------------
        # VALIDATE COMPANY
        # -------------------------------------------------

        if not company:

            print("\n✗ No company supplied.")

            return None

        company_name = company.get(
            "company_name",
            "Unknown Company"
        )

        company_id = company.get(
            "company_id"
        )

        print(f"\nCompany : {company_name}")

        if company_id is not None:

            print(
                f"Company ID : {company_id}"
            )

        # -------------------------------------------------
        # CREATE SHARED CONTEXT
        # -------------------------------------------------

        context = AgentContext(
            company=company
        )

        # -------------------------------------------------
        # INITIAL METADATA
        # -------------------------------------------------

        context.set_metadata(
            "dry_run",
            self.dry_run
        )

        context.set_metadata(
            "registered_agents",
            [
                self._get_agent_name(agent)
                for agent in self.agents
            ]
        )

        # -------------------------------------------------
        # EXECUTE AGENTS
        # -------------------------------------------------

        for agent in self.agents:

            self._execute_agent(
                agent=agent,
                company=company,
                context=context
            )

        # -------------------------------------------------
        # DUPLICATE MANAGEMENT
        # -------------------------------------------------

        self._run_duplicate_manager(
            context
        )

        # -------------------------------------------------
        # EXECUTION STATISTICS
        # -------------------------------------------------

        failed_agents = [
            entry
            for entry in context.execution_log
            if entry["status"] == "FAILED"
        ]

        successful_agents = [
            entry
            for entry in context.execution_log
            if entry["status"] == "SUCCESS"
        ]

        context.set_metadata(
            "successful_agents",
            len(successful_agents)
        )

        context.set_metadata(
            "failed_agents",
            len(failed_agents)
        )

        # -------------------------------------------------
        # PRINT SUMMARY
        # -------------------------------------------------

        context.print_summary()

        # -------------------------------------------------
        # FINAL STATUS
        # -------------------------------------------------

        print()

        print("=" * 80)

        if failed_agents:

            print(
                "ORCHESTRATOR COMPLETED "
                "WITH ERRORS"
            )

        else:

            print(
                "ORCHESTRATOR COMPLETED "
                "SUCCESSFULLY"
            )

        print("=" * 80)

        # -------------------------------------------------
        # RETURN CONTEXT
        # -------------------------------------------------

        return context
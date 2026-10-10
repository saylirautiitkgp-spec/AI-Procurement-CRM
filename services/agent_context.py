"""
=========================================================
AGENT CONTEXT
=========================================================

Shared execution context used by the AI Procurement CRM
Orchestrator and its registered agents.

The context allows multiple agents to share:
- Company information
- Agent results
- Sources
- Insights
- Errors
- Metadata
- Execution logs

Author : AI Procurement CRM
"""

from datetime import datetime


class AgentContext:

    def __init__(self, company=None):

        # -------------------------------------------------
        # COMPANY INFORMATION
        # -------------------------------------------------

        self.company = company or {}

        self.company_id = self.company.get("company_id")

        # -------------------------------------------------
        # AGENT OUTPUTS
        # -------------------------------------------------

        self.results = {}

        # -------------------------------------------------
        # SHARED INFORMATION
        # -------------------------------------------------

        self.sources = []

        self.insights = []

        # -------------------------------------------------
        # ERRORS
        # -------------------------------------------------

        self.errors = []

        # -------------------------------------------------
        # METADATA
        # -------------------------------------------------

        self.metadata = {}

        # -------------------------------------------------
        # EXECUTION LOG
        # -------------------------------------------------

        self.execution_log = []

    # =====================================================
    # RESULTS
    # =====================================================

    def set_result(self, agent_name, result):

        self.results[agent_name] = result

    def get_result(self, agent_name, default=None):

        return self.results.get(agent_name, default)

    # =====================================================
    # SOURCES
    # =====================================================

    def add_source(self, source):

        if source and source not in self.sources:

            self.sources.append(source)

    # =====================================================
    # INSIGHTS
    # =====================================================

    def add_insight(self, insight):

        if insight:

            self.insights.append(insight)

    # =====================================================
    # ERRORS
    # =====================================================

    def add_error(self, agent_name, error):

        self.errors.append(
            {
                "agent": agent_name,
                "error": str(error),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

    # =====================================================
    # METADATA
    # =====================================================

    def set_metadata(self, key, value):

        self.metadata[key] = value

    def get_metadata(self, key, default=None):

        return self.metadata.get(key, default)

    # =====================================================
    # EXECUTION LOG
    # =====================================================

    def add_execution_log(
        self,
        agent_name,
        status,
        duration=None,
        message=None
    ):

        entry = {
            "agent": agent_name,
            "status": status,
            "duration": duration,
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        }

        self.execution_log.append(entry)

    # =====================================================
    # CONVERT TO DICTIONARY
    # =====================================================

    def to_dict(self):

        return {
            "company": self.company,
            "company_id": self.company_id,
            "results": self.results,
            "sources": self.sources,
            "insights": self.insights,
            "errors": self.errors,
            "metadata": self.metadata,
            "execution_log": self.execution_log
        }

    # =====================================================
    # PRINT SUMMARY
    # =====================================================

    def print_summary(self):

        print("\n" + "=" * 80)
        print("AGENT EXECUTION SUMMARY")
        print("=" * 80)

        if not self.execution_log:

            print("No agent execution recorded.")

            return

        for entry in self.execution_log:

            agent = entry.get("agent", "Unknown")

            status = entry.get("status", "UNKNOWN")

            duration = entry.get("duration")

            if duration is not None:

                print(
                    f"{agent:<30} "
                    f"{status:<10} "
                    f"{duration:.2f}s"
                )

            else:

                print(
                    f"{agent:<30} "
                    f"{status:<10}"
                )

        print("=" * 80)

        if self.errors:

            print("\nErrors:")

            for error in self.errors:

                print(
                    f"  [{error['agent']}] "
                    f"{error['error']}"
                )
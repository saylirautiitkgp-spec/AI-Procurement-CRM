"""Read-only CRM context for agents (no names, no contact details).

Contacts are reduced to counts and designations so personal data is not sent to
an LLM provider; an agent never needs the stored names to find public ones."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agentic.tools.registry import Tool, ToolContext, ToolError, ToolOutput

SECTION_ACCESS = {
    "contact": {"contacts_summary"},
    "procurement": {"products", "components", "suppliers"},
    "supplier": {"products", "components", "suppliers"},
}


class CrmContextArgs(BaseModel):
    section: Literal["products", "components", "suppliers", "contacts_summary"] = Field(
        description="Which part of the CRM record to read. All values are UNVERIFIED legacy data.")


def get_crm_context(ctx: ToolContext, args: CrmContextArgs) -> ToolOutput:
    if args.section not in SECTION_ACCESS.get(ctx.agent, set()):
        raise ToolError("section_not_permitted", f"{ctx.agent} may not read '{args.section}'.")
    existing = ctx.subject.get("existing")
    if not existing:
        return ToolOutput(data={"in_crm": False, "note": "Company is not in the CRM; nothing stored."})
    key = {"contacts_summary": "contact_summary"}.get(args.section, args.section)
    value = existing.get(key)
    return ToolOutput(data={"in_crm": True, "section": args.section, "provenance": "database_unverified",
                            "data": value if value is not None else []})


def build_crm_tools(agents: frozenset) -> list[Tool]:
    return [Tool(name="get_crm_context",
                 description=("Read legacy CRM data for this company (products, components, suppliers, or a "
                              "contact COUNT summary). Values are unverified and carry no sources."),
                 args_model=CrmContextArgs, handler=get_crm_context, agents=agents, kind="read")]

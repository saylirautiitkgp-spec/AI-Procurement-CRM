"""Agent registry: the five specialised agents."""

from agentic.agents.company import COMPANY_AGENT
from agentic.agents.contact import CONTACT_AGENT
from agentic.agents.lead import LEAD_AGENT
from agentic.agents.procurement import PROCUREMENT_AGENT
from agentic.agents.supplier import SUPPLIER_AGENT

AGENT_SPECS = {
    "company": COMPANY_AGENT,
    "contact": CONTACT_AGENT,
    "procurement": PROCUREMENT_AGENT,
    "supplier": SUPPLIER_AGENT,
    "lead": LEAD_AGENT,
}

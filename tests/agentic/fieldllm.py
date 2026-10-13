"""Scripted multi-agent double used by the field-request smoke run and the tests (kept in tests/agentic/fieldllm.py)."""
import copy
import re

from agentic.llm.fake import tool_call
from tests.agentic.test_hybrid import HybridLLM
from tests.agentic.test_prefetch import packet_json


class FieldLLM(HybridLLM):
    """Company: HybridLLM. Contact / procurement / supplier: honest submissions quoted from their packets."""

    def __init__(self, mode="honest", fail=()):
        super().__init__(mode)
        self.agent_calls = {}
        self.fail = set(fail)

    def generate(self, *, system, messages, tools, max_output_tokens, force_tool=False):
        names = [t.name for t in tools]
        finish = next((n for n in names if n.endswith("_findings")), None)
        if names == ["route"] or finish in (None, "submit_company_findings", "submit_lead_findings"):
            return super().generate(system=system, messages=messages, tools=tools,
                                    max_output_tokens=max_output_tokens, force_tool=force_tool)
        agent = finish.replace("submit_", "").replace("_findings", "")
        self.agent_calls[agent] = self.agent_calls.get(agent, 0) + 1
        self.calls.append({"agent": agent, "tools": names, "messages": copy.deepcopy(messages), "force": force_tool})
        if agent in self.fail:
            from agentic.llm.base import LLMError
            raise LLMError("invalid", "400 scripted failure")
        packet = packet_json(messages[0]["content"]) or {"documents": [], "search_results": []}
        docs = packet.get("documents", [])
        snips = packet.get("search_results", [])
        text_of = [(d["url"], d["text"]) for d in docs] + [(s["url"], s["snippet"]) for s in snips]
        if agent == "procurement":
            items = []
            for url, text in text_of:
                m = re.search(r"[^.\n]*supplier portal at (https?://[^\s]+?)\.?\n", text + "\n")
                if m and not any(i["kind"] == "supplier_portal" for i in items):
                    items.append({"kind": "supplier_portal", "value": m.group(1).rstrip("."),
                                  "evidence": [{"url": url, "snippet": m.group(0).strip()[:300]}]})
                m = re.search(r"Procurement categories: [^\n]+", text)
                if m and not any(i["kind"] == "purchasing_category" for i in items):
                    items.append({"kind": "purchasing_category", "value": m.group(0).split(": ", 1)[1].split(",")[0].strip(),
                                  "evidence": [{"url": url, "snippet": m.group(0).strip()[:300]}]})
            return tool_call(finish, status="satisfied" if items else "exhausted", items=items)
        if agent == "contact":
            contacts = []
            for url, text in text_of:
                m = re.search(r"Head of Procurement: ([A-Z][a-z]+ [A-Z][a-z]+)", text)
                if m:
                    contacts.append({"name": m.group(1), "designation": "Head of Procurement",
                                     "evidence": [{"url": url, "snippet": m.group(0)}]})
                    break
            return tool_call(finish, status="satisfied" if contacts else "exhausted", contacts=contacts)
        rels = []
        for url, text in text_of:
            m = re.search(r"([A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)*) supplies (\w+) to Apollo Micro Systems", text)
            if m:
                rels.append({"supplier_name": m.group(1), "customer_name": "Apollo Micro Systems", "component": m.group(2),
                             "evidence": [{"url": url, "snippet": m.group(0)}]})
                break
        return tool_call(finish, status="satisfied" if rels else "exhausted", relationships=rels)

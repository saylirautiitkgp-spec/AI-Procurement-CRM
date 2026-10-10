"""The Gemini smoke test is prepared but must stay disabled: prove the guards, the request cap, and that the
harness logic itself works against a fake model. No network."""

import json

import pytest

from agentic.eval import gemini_smoke as gs
from agentic.llm.base import LLMResponse, ToolCallRequest, Usage
from agentic.llm.fake import FakeLLM


def call(name, **args):
    return LLMResponse(tool_calls=[ToolCallRequest(id="c", name=name, args=args)], usage=Usage(input_tokens=500, output_tokens=40))


def test_live_run_is_refused_without_every_approval(monkeypatch):
    ok = ["--model", "gemini-3-flash-preview", "--i-have-checked-my-free-quota"]
    with pytest.raises(gs.SmokeGuardError, match="AGENTIC_SMOKE_APPROVED"):
        gs.check_guards({"GEMINI_API_KEY": "k"}, ok)
    with pytest.raises(gs.SmokeGuardError, match="free-quota"):
        gs.check_guards({"AGENTIC_SMOKE_APPROVED": "1", "GEMINI_API_KEY": "k"}, ["--model", "m"])
    with pytest.raises(gs.SmokeGuardError, match="--model"):
        gs.check_guards({"AGENTIC_SMOKE_APPROVED": "1", "GEMINI_API_KEY": "k"}, ["--i-have-checked-my-free-quota"])
    with pytest.raises(gs.SmokeGuardError, match="GEMINI_API_KEY"):
        gs.check_guards({"AGENTIC_SMOKE_APPROVED": "1"}, ok)
    gs.check_guards({"AGENTIC_SMOKE_APPROVED": "1", "GEMINI_API_KEY": "k"}, ok)


def test_dry_run_prints_the_plan_and_makes_no_request(capsys, monkeypatch):
    monkeypatch.delenv("AGENTIC_SMOKE_APPROVED", raising=False)
    gs.main(["--dry-run"])
    assert "NOT RUN" in capsys.readouterr().out


def test_main_without_approval_exits_before_touching_any_client(monkeypatch):
    monkeypatch.delenv("AGENTIC_SMOKE_APPROVED", raising=False)
    import agentic.llm.gemini as g
    monkeypatch.setattr(g, "GeminiClient", lambda *a, **k: (_ for _ in ()).throw(AssertionError("client created")))
    with pytest.raises(gs.SmokeGuardError):
        gs.main(["out.json"])


def test_request_cap_is_hard():
    llm = gs.CountingLLM(FakeLLM(lambda s, m, t: call("route", next="company", reason="x")), cap=2)
    for _ in range(2):
        llm.generate(system="s", messages=[], tools=[], max_output_tokens=10)
    with pytest.raises(gs.SmokeGuardError, match="cap"):
        llm.generate(system="s", messages=[], tools=[], max_output_tokens=10)


def test_harness_logic_passes_with_a_well_behaved_fake_model():
    home = "https://www.acmepreciionworks.example/"
    script = [
        call("route", next="company", reason="only eligible agent"),
        call("web_search", query="Acme Precision Works official website"),
        call("fetch_page", url="https://www.acmeprecisionworks.example/"),
    ]
    from agentic.eval.synthetic import make_world
    f = make_world(gs.COMPANY, "standard")["facts"]
    site = f"https://www.{f['domain']}/"
    line = f"Acme Precision Works is a electronics company headquartered in {f['city']}, India."
    script.append(call("submit_company_findings", status="satisfied", identity_website=site,
                       claims=[{"field": "official_website", "value": site,
                                "evidence": [{"url": site, "snippet": line}]},
                               {"field": "headquarters", "value": f["city"], "evidence": [{"url": site, "snippet": line}]}]))
    llm = gs.CountingLLM(FakeLLM(script))
    out = gs.run_smoke(llm)
    assert out["requests_made"] == 4 == gs.MAX_REQUESTS
    assert out["all_passed"], out["checks"]
    assert out["total_input_tokens"] == 2000


def test_harness_detects_a_model_that_fabricates_evidence():
    from agentic.eval.synthetic import make_world
    f = make_world(gs.COMPANY, "standard")["facts"]
    site = f"https://www.{f['domain']}/"
    script = [call("route", next="company", reason="x"), call("web_search", query="Acme Precision Works"),
              call("fetch_page", url=site),
              call("submit_company_findings", status="satisfied", identity_website=site,
                   claims=[{"field": "headquarters", "value": "Atlantis",
                            "evidence": [{"url": site, "snippet": "headquartered in Atlantis, Oceania"}]}])]
    out = gs.run_smoke(gs.CountingLLM(FakeLLM(script)))
    assert out["checks"]["quoted_snippets_pass_evidence_validation"]["ok"] is False and not out["all_passed"]


def test_smoke_inputs_are_synthetic_and_it_imports_no_database_or_search_client():
    import ast
    tree = ast.parse(open("agentic/eval/gemini_smoke.py", encoding="utf-8").read())
    imported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imported |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            imported.add(n.module or "")
    assert not any(m and (m.startswith(("supabase", "tavily")) or "backends.search" in m or "repo" in m) for m in imported)
    assert gs.COMPANY["name"] == "Acme Precision Works" and gs.MAX_REQUESTS == 4 and gs.MAX_OUTPUT_TOKENS == 1024


def test_env_file_is_never_read_unless_explicitly_requested(monkeypatch):
    import dotenv
    monkeypatch.setenv("AGENTIC_SMOKE_APPROVED", "1")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: (_ for _ in ()).throw(AssertionError(".env was read")))
    with pytest.raises(gs.SmokeGuardError, match="GEMINI_API_KEY"):          # stops at the guard, .env untouched
        gs.main(["--model", "gemini-3-flash-preview", "--i-have-checked-my-free-quota", "out.json"])


def test_a_failed_request_is_not_retried_and_stops_the_run():
    from agentic.llm.base import LLMError
    fake = FakeLLM([LLMError("invalid", "404 model not found"), call("route", next="company", reason="x")])
    llm = gs.CountingLLM(fake)
    with pytest.raises(LLMError):
        gs.run_smoke(llm)
    assert llm.n == 1 and llm.log[0]["error_kind"] == "invalid" and len(fake.script) == 1     # second item never consumed


def test_raw_synthetic_responses_are_logged_without_secrets():
    secret = "AIzaSyDUMMYKEYDUMMYKEYDUMMYKEY12345"
    resp = LLMResponse(text=f"note {secret}", tool_calls=[ToolCallRequest(id="1", name="route", args={"next": "company", "reason": secret})],
                       usage=Usage(input_tokens=5, output_tokens=2))
    llm = gs.CountingLLM(FakeLLM([resp]))
    llm.generate(system="s", messages=[], tools=[], max_output_tokens=10)
    dumped = json.dumps(llm.log)
    assert "AIzaSy" not in dumped and llm.log[0]["raw"]["tool_calls"][0]["name"] == "route"

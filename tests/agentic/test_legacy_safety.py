"""Regression tests for the legacy code paths touched in Phase 0."""

import importlib
import sys
import types


def _import_orchestrator(monkeypatch):
    # The legacy modules create Supabase/Tavily clients at import time; fake them so the test
    # needs no credentials and makes no network calls.
    monkeypatch.setitem(sys.modules, "database.supabase_client", types.SimpleNamespace(supabase=object()))
    monkeypatch.setitem(sys.modules, "utils.tavily_client", types.SimpleNamespace(client=object()))
    for name in [m for m in sys.modules if m.startswith(("agents.orchestrator", "agents.company", "services."))]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module("agents.orchestrator.orchestrator")


class _Spy:
    def __init__(self):
        self.called = 0

    def merge_all(self):
        self.called += 1


def _orchestrator(mod, merge_duplicates):
    o = mod.Orchestrator.__new__(mod.Orchestrator)      # bypass __init__ (builds real clients)
    o.dry_run, o.agents, o.merge_duplicates = False, [], merge_duplicates
    o.duplicate_manager = _Spy()
    return o


def test_orchestrator_does_not_auto_merge_by_default(monkeypatch, capsys):
    mod = _import_orchestrator(monkeypatch)
    o = _orchestrator(mod, merge_duplicates=False)
    ctx = o.process_company({"company_id": 1, "company_name": "Acme"})
    assert o.duplicate_manager.called == 0
    entry = [e for e in ctx.execution_log if e["agent"] == "DuplicateManager"][0]
    assert entry["status"] == "SKIPPED" and "reviewed merge plan" in entry["message"]


def test_auto_merge_only_runs_when_explicitly_enabled(monkeypatch):
    mod = _import_orchestrator(monkeypatch)
    o = _orchestrator(mod, merge_duplicates=True)
    o.process_company({"company_id": 1, "company_name": "Acme"})
    assert o.duplicate_manager.called == 1


def test_default_constructor_has_auto_merge_disabled(monkeypatch):
    mod = _import_orchestrator(monkeypatch)
    import inspect
    assert inspect.signature(mod.Orchestrator.__init__).parameters["merge_duplicates"].default is False


def test_importing_settings_prints_nothing(capsys, monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    sys.modules.pop("config.settings", None)
    importlib.import_module("config.settings")
    out = capsys.readouterr().out
    assert out == "" and "supabase.co" not in out

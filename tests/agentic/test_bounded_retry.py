"""One bounded follow-up attempt: a single retry only for a 503 capacity error, a hard cap on requests SENT (retries
included), every sent request counted by the guard, and a second agent that only starts when the cap has room.
Offline only."""

import time

import pytest

from agentic import company_live
from agentic import preflight as pf
from agentic.agents import AGENT_SPECS
from agentic.backends.fixtures import FixtureFetcher, FixtureSearch
from agentic.config import Efficiency, Limits, Settings, load_settings
from agentic.llm.base import LLMError, LLMResponse, RetryingLLM, Usage, is_capacity_503
from agentic.llm.pacing import PacedLLM, ProviderQuota
from agentic.persistence.store import Store
from agentic.repo import InMemoryRepo
from agentic.runtime import AgenticRunner, make_deps
from tests.agentic.fieldllm import FieldLLM
from tests.agentic.support import COMPANY_A
from tests.agentic.test_company_only import seed
from tests.agentic.test_prefetch import world

SCOPE = ["company", "procurement"]
E503 = LLMError("transient", "503 UNAVAILABLE. This model is currently experiencing high demand.")


class Scripted:
    """Raises the queued errors first (one per request), then delegates. Counts requests that reach it."""
    name, model = "gemini", "gemini-3.8-flash"

    def __init__(self, errors=(), inner=None):
        self.errors, self.inner, self.reached = list(errors), inner or FieldLLM(), 0

    @property
    def calls(self):
        return self.inner.calls

    def generate(self, **kw):
        self.reached += 1
        if self.errors:
            raise self.errors.pop(0)
        return self.inner.generate(**kw)


def retrying(inner, retries=1):
    return RetryingLLM(inner, max_retries=retries, base_delay=0, sleep=lambda s: None, retry_if=is_capacity_503)


# ============================================================================================ which errors may be retried
@pytest.mark.parametrize("err,retried", [
    (E503, True),
    (LLMError("transient", "request timed out: ReadTimeout"), False),
    (LLMError("transient", "500 INTERNAL"), False),
    (LLMError("rate_limit", "429 RESOURCE_EXHAUSTED per minute"), False),
    (LLMError("quota", "429 per day"), False),
    (LLMError("invalid", "404 NOT_FOUND. no longer available"), False),
    (LLMError("invalid", "401 UNAUTHENTICATED"), False),
    (LLMError("invalid", "400 INVALID_ARGUMENT"), False),
    (LLMError("other", "weird"), False),
])
def test_only_a_503_capacity_error_is_retried(err, retried):
    inner = Scripted([err])
    llm = retrying(inner)
    kw = dict(system="s", messages=[{"role": "user", "content": "x"}], tools=[], max_output_tokens=10)
    if retried:
        inner.inner = type("Ok", (), {"calls": [], "generate": lambda self, **k: LLMResponse(text="ok", usage=Usage(requests=1, input_tokens=1, output_tokens=1))})()
        assert llm.generate(**kw).text == "ok" and inner.reached == 2 and llm.retries == 1
    else:
        with pytest.raises(LLMError):
            llm.generate(**kw)
        assert inner.reached == 1 and llm.retries == 0


def test_a_second_503_is_not_retried_again():
    inner = Scripted([E503, E503])
    llm = retrying(inner)
    with pytest.raises(LLMError):
        llm.generate(system="s", messages=[], tools=[], max_output_tokens=10)
    assert inner.reached == 2 and llm.retries == 1


def test_the_environment_switches_build_this_policy():
    s = load_settings(company_live.base_env("rpm=5,tpm=250000,rpd=20", "d", "gemini-3.8-flash", 0, None, 1, True, 2, 20))
    assert (s.limits.llm_max_retries, s.limits.retry_503_only, s.limits.max_requests_total,
            s.limits.llm_backoff_base_s, s.limits.company_correction_requests) == (1, True, 2, 20.0, 0)
    deps = make_deps(llm=Scripted(), repo=InMemoryRepo([COMPANY_A]), search=None, fetcher=None, store=Store(":memory:"),
                     settings=s, specs=AGENT_SPECS)
    assert deps.llm.retry_if is is_capacity_503 and deps.llm.max_retries == 1
    plain = make_deps(llm=Scripted(), repo=InMemoryRepo([COMPANY_A]), search=None, fetcher=None, store=Store(":memory:"),
                      settings=Settings(), specs=AGENT_SPECS)
    assert plain.llm.retry_if is None


# ============================================================================================ hard cap + guard accounting
QUOTA = ProviderQuota(rpm=5, tpm=250000, rpd=20, tpd=10 ** 12)
KW = dict(system="s", messages=[{"role": "user", "content": "hi"}], tools=[], max_output_tokens=10)


def test_the_hard_cap_counts_retries_and_every_sent_request_reaches_the_daily_guard():
    store = Store(":memory:")
    inner = Scripted([E503, E503, E503])
    paced = PacedLLM(inner, QUOTA, store, sleep=lambda s: None)
    paced.max_total = 2
    llm = retrying(paced, retries=5)                                  # even a generous retry setting cannot exceed the cap
    with pytest.raises(LLMError) as e:
        llm.generate(**KW)
    assert inner.reached == 2 and paced.sent == 2 and e.value.kind == "quota"
    assert store.usage_since("gemini:gemini-3.8-flash", 86400, time.time())[0] == 2      # both failed requests are counted
    with pytest.raises(LLMError):
        paced.generate(**KW)
    assert inner.reached == 2                                         # the cap is sticky: nothing more is sent


# ============================================================================================ scoped runs
def run(errors=(), *, retries=1, cap=2, store=None):
    inner = Scripted(errors)
    store = store or Store(":memory:")
    paced = PacedLLM(inner, QUOTA, store, sleep=lambda s: None)
    paced.max_total = cap
    lim = Limits(llm_max_retries=retries, retry_503_only=True, max_requests_total=cap, max_llm_calls_run=cap,
                 company_correction_requests=0)
    repo = InMemoryRepo([COMPANY_A], products={7: ["Avionics"]}, components={7: ["PCBAs"]})
    w = world()
    search, fetch = FixtureSearch(w), FixtureFetcher(w)
    deps = make_deps(llm=paced, repo=repo, search=search, fetcher=fetch, store=store,
                     settings=Settings(limits=lim, efficiency=Efficiency(lazy_crm=True)), specs=AGENT_SPECS,
                     sleep=lambda s: None)
    out = AgenticRunner(deps, AGENT_SPECS).start("Apollo Micro Systems", scope_agents=SCOPE, expected_company_id=7)
    return inner, paced, search, out, store


def test_a_clean_run_uses_two_requests_one_per_agent_and_no_retry():
    inner, paced, search, out, store = run()
    assert inner.reached == 2 and paced.sent == 2 and out["budgets"]["llm_retries"] == 0
    assert out["agent_runs"] == {"company": 1, "procurement": 1} and len(search.calls) == 2


def test_one_503_is_retried_once_and_the_second_agent_waits_for_budget_that_is_gone():
    inner, paced, search, out, store = run([E503])
    assert inner.reached == 2 and paced.sent == 2 and out["budgets"]["llm_retries"] == 1
    assert out["claims"]["company"], "the retried company request produced validated claims"
    assert out["agent_results"]["procurement"]["end_reason"] == "request_budget_exhausted"   # no room under the cap
    assert out["agent_results"]["procurement"]["llm_calls"] == 0 and out["budgets"]["llm_calls"] == 1
    assert len(search.calls) == 1                                                  # and no Tavily search was spent on it
    assert store.usage_since("gemini:gemini-3.8-flash", 86400, time.time())[0] == 2


def test_two_503s_stop_the_run_with_nothing_more_sent():
    inner, paced, search, out, store = run([E503, E503])
    assert inner.reached == 2 and paced.sent == 2 and not out["claims"].get("company")
    assert out["agent_results"]["company"]["end_reason"] in ("llm_transient", "llm_quota")
    assert out["agent_runs"] == {"company": 1} and out["status"] == "partial"


@pytest.mark.parametrize("err", [LLMError("invalid", "404 NOT_FOUND"), LLMError("invalid", "401 UNAUTHENTICATED"),
                                 LLMError("invalid", "400 INVALID_ARGUMENT"), LLMError("rate_limit", "429 per minute")])
def test_non_503_errors_are_not_retried_and_stop_the_run(err):
    inner, paced, search, out, store = run([err])
    assert inner.reached == 1 and out["budgets"]["llm_retries"] == 0 and out["agent_runs"] == {"company": 1}
    assert len(search.calls) == 1 and out["status"] == "partial"


# ============================================================================================ preflight for this attempt
ARGS = ["--company", "AIRBUS India", "--company-id", "11", "--agents", "company", "procurement", "--model", "gemini-3.8-flash",
        "--retries", "1", "--retry-503-only", "--total-requests-cap", "2", "--max-requests", "2", "--max-credits", "2",
        "--apply", "--preflight-only", "--quota", "rpm=5,tpm=250000,rpd=20", "--console-remaining", "20"]


def test_the_preflight_plans_exactly_two_requests_with_the_retry_inside_the_cap(tmp_path, capsys):
    seed(tmp_path, 0)
    assert company_live.main(ARGS + ["--data-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "min 0, max 2" in out and "hard cap: 2 requests SENT in total, retries included" in out
    assert "only for a 503 capacity error" in out and "remaining under the cap: 16" in out
    assert "Tavily credits: max planned 2" in out and "SELECTs (research): min 6, max planned 8" in out
    assert "write-phase verification SELECTs capped at 40" in out and "correction requests permitted: 0" in out


def test_the_preflight_blocks_when_the_guard_has_less_room_than_the_cap(tmp_path, capsys):
    store = Store(str(tmp_path / "agentic.sqlite"))
    for i in range(15):
        store.usage_add("gemini:gemini-3.8-flash", 1000, time.time() - 60 - i)           # 1 left under the cap of 16
    assert company_live.main(ARGS + ["--data-dir", str(tmp_path)]) == 3
    out = capsys.readouterr().out
    assert pf.BLOCK_LINE in out and "remaining daily allowance 1" in out

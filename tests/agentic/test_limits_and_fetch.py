"""Cost / quota limits (zero-cost constraint) and the safe page fetcher."""

import pytest

from agentic.backends.fetch import FetchError, SafeHttpFetcher
from agentic.config import Limits
from agentic.llm.fake import FakeLLM, tool_call
from agentic.persistence.store import Store
from tests.agentic.support import make_runner

PROFILE = "company profile"


def public(host):
    return ["93.184.216.34"]


# ============================================================== credit / quota budgets
def test_per_run_search_credit_cap_blocks_further_searches():
    r = make_runner(limits=Limits(max_search_credits_run=1))
    out = r.start("Apollo Micro Systems", PROFILE)
    assert out["budgets"]["search_credits"] == 1 and len(r.search.calls) == 1
    codes = [e["data"]["error"]["code"] for e in out["trace"] if e["type"] == "tool_result" and not e["data"]["ok"]]
    assert "credit_budget_exhausted" in codes


def test_monthly_budget_disables_search_and_run_ends_partial():
    store = Store(":memory:")
    store.charge_credits(299)                                   # 1 credit left this month
    r = make_runner(limits=Limits(monthly_credit_budget=300), store=store)
    out = r.start("Apollo Micro Systems", PROFILE)
    assert store.credits_used() == 300 and len(r.search.calls) == 1
    assert out["budgets"]["search_disabled"] and out["budgets"]["search_disabled_reason"] == "monthly_credit_budget_exhausted"
    assert out["status"] == "partial"


def test_advanced_depth_costs_two_credits_and_default_is_cheap():
    script = [tool_call("route", next="company", reason="x"),
              tool_call("web_search", query="Apollo Micro Systems official website", depth="advanced"),
              tool_call("web_search", query="Apollo Micro Systems about us"),
              tool_call("submit_company_findings", status="exhausted"), tool_call("route", next="finish", reason="d")]
    r = make_runner(llm=FakeLLM(script))
    out = r.start("Apollo Micro Systems", PROFILE)
    assert [c[1] for c in r.search.calls] == ["advanced", "basic"] and out["budgets"]["search_credits"] == 3


def test_identical_searches_are_served_from_cache_and_cost_nothing():
    store = Store(":memory:")
    r1 = make_runner(store=store)
    first = r1.start("Apollo Micro Systems", PROFILE)
    used = store.credits_used()
    assert used == first["budgets"]["search_credits"] > 0
    r2 = make_runner(store=store)
    second = r2.start("Apollo Micro Systems", PROFILE)
    assert r2.search.calls == [] and second["budgets"]["search_credits"] == 0 and store.credits_used() == used
    assert any(e["data"].get("cached") for e in second["trace"] if e["type"] == "tool_result")
    assert second["report"]["company_profile"].keys() == first["report"]["company_profile"].keys()


def test_cached_results_remain_usable_after_the_credit_budget_is_gone():
    store = Store(":memory:")
    make_runner(store=store).start("Apollo Micro Systems", PROFILE)
    store.charge_credits(10_000)                                # monthly budget now exhausted
    r = make_runner(store=store, limits=Limits(monthly_credit_budget=300))
    out = r.start("Apollo Micro Systems", PROFILE)
    assert out["status"] == "completed" and not out["budgets"]["search_disabled"]


def test_token_and_wall_clock_limits_are_enforced():
    out = make_runner(limits=Limits(max_total_tokens_run=700)).start("Apollo Micro Systems", "comprehensive")
    assert out["status"] == "partial" and out["stop_reason"] in ("max_tokens_reached",) or \
        out["agent_results"]["company"]["end_reason"] == "max_tokens_reached"
    out2 = make_runner(limits=Limits(max_wall_clock_s=-1)).start("Apollo Micro Systems", PROFILE)
    assert out2["stop_reason"] == "wall_clock_exceeded" and out2["status"] == "partial"


def test_rate_limited_search_backs_off_then_degrades():
    sleeps = []
    r = make_runner(variant="search_rate_limit")
    r.deps.sleep = sleeps.append
    out = r.start("Apollo Micro Systems", PROFILE)
    assert len(sleeps) == 2 and out["budgets"]["search_disabled_reason"] == "search_rate_limited_persistent"
    assert out["status"] == "partial" and out["report"]["company_profile"] is None


def test_quota_exhaustion_returns_a_partial_result_not_a_retry_loop():
    r = make_runner(variant="search_quota")
    out = r.start("Apollo Micro Systems", PROFILE)
    assert len(r.search.calls) <= 3 and out["status"] == "partial"
    assert out["budgets"]["search_disabled_reason"] == "search_quota_exhausted"


def test_live_provider_requires_explicit_opt_in():
    from agentic.config import load_settings
    assert load_settings({}).allow_live is False and load_settings({}).provider == "fake"
    assert load_settings({"AGENTIC_ALLOW_LIVE": "1"}).allow_live is True


# ============================================================================== fetcher
class Resp:
    def __init__(self, status=200, headers=None, body=b"<html></html>", encoding="utf-8"):
        self.status_code, self.headers, self._body, self.encoding = status, headers or {"Content-Type": "text/html"}, body, encoding
        self.text = body.decode("utf-8", "replace")

    def iter_content(self, n):
        for i in range(0, len(self._body), n):
            yield self._body[i:i + n]


class Session:
    def __init__(self, routes):
        self.routes, self.requested = routes, []

    def get(self, url, **kw):
        self.requested.append(url)
        r = self.routes.get(url)
        if r is None:
            return Resp(404)
        return r


def fetcher(routes, **kw):
    return SafeHttpFetcher(resolver=kw.pop("resolver", public), session=Session(routes), respect_robots=False, **kw)


def test_fetcher_extracts_text_title_and_same_site_links_only():
    html = (b"<html><head><title>Acme</title><script>alert(1)</script></head><body><h1>Hello</h1>"
            b"<a href='/about'>About</a><a href='https://other.example/x'>Other</a></body></html>")
    page = fetcher({"https://acme.example/": Resp(body=html)}).fetch("https://acme.example/")
    assert page.title == "Acme" and "Hello" in page.text and "alert" not in page.text
    assert [l["url"] for l in page.links] == ["https://acme.example/about"]


def test_redirect_to_internal_address_is_blocked_at_each_hop():
    routes = {"https://acme.example/": Resp(302, {"Location": "http://169.254.169.254/latest/meta-data"})}
    f = fetcher(routes)
    with pytest.raises(FetchError) as exc:
        f.fetch("https://acme.example/")
    assert exc.value.code == "blocked_non_public_address"
    assert f.session.requested == ["https://acme.example/"]            # the internal URL was never requested


def test_dns_pointing_to_private_range_is_blocked_and_redirect_loops_stop():
    with pytest.raises(FetchError) as e1:
        fetcher({}, resolver=lambda h: ["192.168.1.10"]).fetch("https://evil.example/")
    assert e1.value.code == "blocked_non_public_address"
    loop = {"https://a.example/": Resp(302, {"Location": "https://a.example/"})}
    with pytest.raises(FetchError) as e2:
        fetcher(loop, max_redirects=3).fetch("https://a.example/")
    assert e2.value.code == "too_many_redirects"


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://a.example/x", "http://a.example:8080/",
                                 "http://user:pw@a.example/", "http://localhost/", "gopher://a.example"])
def test_dangerous_urls_never_reach_the_network(url):
    f = fetcher({})
    with pytest.raises(FetchError):
        f.fetch(url)
    assert f.session.requested == []


def test_page_size_content_type_and_status_limits():
    big = Resp(body=b"<html><body>" + b"x" * 200_000 + b"</body></html>")
    page = fetcher({"https://a.example/": big}, max_bytes=50_000).fetch("https://a.example/")
    assert page.truncated and len(page.text) <= 50_000
    with pytest.raises(FetchError) as e1:
        fetcher({"https://a.example/p": Resp(headers={"Content-Type": "application/pdf"})}).fetch("https://a.example/p")
    assert e1.value.code == "unsupported_content_type"
    with pytest.raises(FetchError) as e2:
        fetcher({}).fetch("https://a.example/missing")
    assert e2.value.code == "http_404"


def test_robots_txt_disallow_is_respected():
    routes = {"https://a.example/robots.txt": Resp(headers={"Content-Type": "text/plain"},
                                                    body=b"User-agent: *\nDisallow: /private\n"),
              "https://a.example/private/x": Resp(), "https://a.example/ok": Resp()}
    f = SafeHttpFetcher(resolver=public, session=Session(routes), respect_robots=True)
    assert f.fetch("https://a.example/ok").status == 200
    with pytest.raises(FetchError) as e:
        f.fetch("https://a.example/private/x")
    assert e.value.code == "disallowed_by_robots_txt"


# ============================================================ fetch tool domain gate
def test_fetch_tool_only_reaches_domains_discovered_by_search_and_blocks_internal_urls():
    script = [tool_call("route", next="company", reason="x"),
              tool_call("fetch_page", url="https://never-seen.example/"),          # undiscovered domain
              tool_call("fetch_page", url="http://169.254.169.254/latest"),
              tool_call("fetch_page", url="file:///etc/passwd"),
              tool_call("submit_company_findings", status="exhausted"), tool_call("route", next="finish", reason="d")]
    r = make_runner(llm=FakeLLM(script))
    out = r.start("Apollo Micro Systems", PROFILE)
    codes = [e["data"]["error"]["code"] for e in out["trace"] if e["type"] == "tool_result" and not e["data"]["ok"]]
    assert codes == ["url_not_discovered", "url_blocked", "invalid_arguments"] or codes[:2] == \
        ["url_not_discovered", "url_blocked"]
    assert r.fetcher.calls == []

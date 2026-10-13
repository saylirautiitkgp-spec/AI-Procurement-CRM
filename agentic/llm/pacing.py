"""Client-side pacing against the provider's free-tier limits (RPM, TPM, RPD, TPD).

Why: a 429 is a *failure* and retries cost requests. With limits like 8,000 tokens/minute the right move is to wait
BEFORE sending. The limits are supplied by the operator from their own console (`AGENTIC_QUOTA`) - nothing is
assumed. Daily budgets are enforced over a rolling 24 hours from a persistent log (so restarts and parallel
processes cannot silently overspend); the provider's reset time is not assumed.

The pre-send estimate is deliberately conservative: prompt characters / 4 PLUS the full max_output_tokens
reservation, because Groq's documentation does not say whether requested output tokens count toward TPM.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Callable

from agentic.llm.base import LLMError, LLMResponse

CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class ProviderQuota:
    rpm: int
    tpm: int
    rpd: int
    tpd: int
    headroom: float = 0.8          # never plan to use more than 80% of any limit

    @classmethod
    def parse(cls, text: str, headroom: float = 0.8, require_tpd: bool = True) -> "ProviderQuota":
        """'rpm=30,tpm=8000,rpd=1000,tpd=200000' - all four are required (no assumed limits). With
        require_tpd=False an omitted tpd means "the console shows no tokens/day limit" (not enforced)."""
        parts = dict(p.strip().split("=", 1) for p in text.split(",") if "=" in p)
        if not require_tpd and "tpd" not in parts:
            parts["tpd"] = str(10 ** 12)
        missing = [k for k in ("rpm", "tpm", "rpd", "tpd") if k not in parts]
        if missing:
            raise ValueError(f"quota is missing: {', '.join(missing)}")
        vals = {k: int(parts[k].replace("_", "")) for k in ("rpm", "tpm", "rpd", "tpd")}
        if any(v <= 0 for v in vals.values()):
            raise ValueError("quota values must be positive")
        return cls(**vals, headroom=headroom)


def estimate_request_tokens(system: str, messages: list, tools: list, max_output_tokens: int) -> int:
    chars = len(system) + len(json.dumps(messages, default=str)) + \
        sum(len(json.dumps(t.parameters)) + len(t.description) for t in tools)
    return chars // CHARS_PER_TOKEN + max_output_tokens


class PacedLLM:
    """Wraps an LLMClient; blocks (sleeps) to stay under the per-minute limits, refuses locally when a daily budget
    or a single request would exceed a limit. `log` persists usage: any object with
    `usage_since(key, seconds)->(requests, tokens)` and `usage_add(key, tokens, ts)`."""

    def __init__(self, inner, quota: ProviderQuota, log=None, *, clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep, max_wait_s: float = 180.0):
        self.inner, self.quota, self.log = inner, quota, log
        self.name, self.model = inner.name, inner.model
        self._clock, self._sleep, self.max_wait_s = clock, sleep, max_wait_s
        self._window: list[tuple[float, int]] = []          # (timestamp, tokens) in the last 60 s
        self.waited_s = 0.0
        self.local_refusals = 0
        self._key = f"{inner.name}:{inner.model}"

    def _prune(self, now: float) -> None:
        self._window = [(t, k) for t, k in self._window if now - t < 60.0]

    def generate(self, **kw) -> LLMResponse:
        q = self.quota
        est = estimate_request_tokens(kw["system"], kw["messages"], kw["tools"], kw["max_output_tokens"])
        if est > q.tpm:
            self.local_refusals += 1
            raise LLMError("invalid", f"estimated request size {est} tokens exceeds the {q.tpm} tokens/minute limit; "
                                      "shrink the prompt (fewer/shorter tool results) - not sent")
        self._daily_check(est)
        waited = 0.0
        while True:
            now = self._clock()
            self._prune(now)
            used = sum(k for _, k in self._window)
            n = len(self._window)
            if n < int(q.rpm * q.headroom) or n == 0:
                if used + est <= q.tpm * q.headroom or not self._window:
                    break
            oldest = self._window[0][0]
            delay = max(60.0 - (now - oldest), 0.5) + 0.05
            if waited + delay > self.max_wait_s:
                self.local_refusals += 1
                raise LLMError("rate_limit", f"local pacing would need to wait over {self.max_wait_s:.0f}s "
                                             "to stay under the per-minute limits - not sent")
            self._sleep(delay)
            waited += delay
        self.waited_s += waited
        resp = self.inner.generate(**kw)
        used_tokens = (resp.usage.input_tokens + resp.usage.output_tokens) or est
        stamp = self._clock()
        self._window.append((stamp, used_tokens))
        if self.log is not None:
            self.log.usage_add(self._key, used_tokens, stamp)
        return resp

    def _daily_check(self, est: int) -> None:
        if self.log is None:
            return
        q = self.quota
        reqs, toks = self.log.usage_since(self._key, 86400, self._clock())
        if reqs + 1 > q.rpd * q.headroom:
            self.local_refusals += 1
            raise LLMError("quota", f"local daily request budget reached ({reqs} in the last 24 h, limit "
                                    f"{q.rpd}, planning cap {int(q.rpd * q.headroom)}) - not sent")
        if toks + est > q.tpd * q.headroom:
            self.local_refusals += 1
            raise LLMError("quota", f"local daily token budget would be exceeded ({toks} used in the last 24 h + "
                                    f"~{est}, limit {q.tpd}, planning cap {int(q.tpd * q.headroom)}) - not sent")

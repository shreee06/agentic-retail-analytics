"""Pacing and backoff for rate-limited LLM calls.

Wraps any LLM client. Waits are counted separately so the evaluation can leave
them out of latency and never mistake a rate limit for a wrong answer.
"""

from __future__ import annotations

import time
from typing import Callable

from .llm import LLM, LLMError


class PacedLLM:
    def __init__(
        self,
        inner: LLM,
        *,
        min_interval_s: float = 2.0,
        max_retries: int = 5,
        base_backoff_s: float = 5.0,
        max_backoff_s: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.inner = inner
        self.min_interval_s = min_interval_s
        self.max_retries = max_retries
        self.base_backoff_s = base_backoff_s
        self.max_backoff_s = max_backoff_s
        self._sleep = sleep
        self._clock = clock
        self._last_call_start: float | None = None
        # running totals; the harness reads deltas per question via snapshot()
        self.wait_seconds = 0.0
        self.rate_limit_retries = 0
        self.calls = 0

    def snapshot(self) -> tuple[float, int, int]:
        return self.wait_seconds, self.rate_limit_retries, self.calls

    def _wait(self, seconds: float) -> None:
        if seconds > 0:
            self.wait_seconds += seconds
            self._sleep(seconds)

    def complete(self, system: str, user: str, *, step: str = "", max_tokens: int = 1500) -> str:
        for attempt in range(self.max_retries + 1):
            if self._last_call_start is not None:  # pacing: keep calls spread out
                self._wait(self.min_interval_s - (self._clock() - self._last_call_start))
            started = self._last_call_start = self._clock()
            self.calls += 1
            try:
                return self.inner.complete(system, user, step=step, max_tokens=max_tokens)
            except LLMError as error:
                if error.kind != "rate_limit" or attempt == self.max_retries:
                    raise
                # the failed call itself, plus the backoff, is "rate-limit time"
                self.wait_seconds += self._clock() - started
                self.rate_limit_retries += 1
                backoff = min(self.max_backoff_s, self.base_backoff_s * 2**attempt)
                self._wait(max(backoff, error.retry_after or 0.0))
        raise AssertionError("unreachable")

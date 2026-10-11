"""Unit tests: search circuit breaker — burned providers skipped instantly."""
from __future__ import annotations

import asyncio
import time

from app.core.errors import SearchError
from app.tools.web_search import (
    OfflineSearchProvider,
    SearchProvider,
    WebSearchResultItem,
    WebSearchTool,
    WebSearchInput,
)
from app.core.config import get_settings


class FlakyProvider(SearchProvider):
    """Fails N times, then succeeds."""

    name = "flaky"

    def __init__(self, fail_times: int) -> None:
        self.fail_times = fail_times
        self.calls = 0

    async def search(self, query: str, max_results: int) -> list[WebSearchResultItem]:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise SearchError("backend down", retryable=True)
        return [WebSearchResultItem(title="result", url="https://example.com/a")]


class WorkingProvider(SearchProvider):
    name = "working"

    def __init__(self) -> None:
        self.calls = 0

    async def search(self, query: str, max_results: int) -> list[WebSearchResultItem]:
        self.calls += 1
        return [WebSearchResultItem(title="fallback", url="https://example.com/b")]


def _tool(primary: SearchProvider) -> WebSearchTool:
    tool = WebSearchTool(provider=primary)
    # isolate: only the injected primary + one working fallback
    tool._fallbacks = [WorkingProvider()]
    return tool


class TestCircuitBreaker:
    async def test_first_failure_falls_back(self):
        primary = FlakyProvider(fail_times=1)
        tool = _tool(primary)
        out = await tool.run(WebSearchInput(query="test query"))
        assert out.total == 1 and "working" in out.provider

    async def test_burned_provider_skipped_instantly(self):
        """After 2 consecutive failures the breaker opens: the provider is
        NOT called again (no wasted timeout), fallback answers instantly."""
        primary = FlakyProvider(fail_times=99)  # always fails
        tool = _tool(primary)
        # first two searches trip the breaker
        await tool.run(WebSearchInput(query="q1"))
        await tool.run(WebSearchInput(query="q2"))
        assert primary.calls == 2  # tripped after threshold
        # third search: breaker open — primary skipped
        t0 = time.time()
        out = await tool.run(WebSearchInput(query="q3"))
        assert primary.calls == 2  # NOT called again
        assert time.time() - t0 < 1.0  # instant, no burned timeout
        assert out.total == 1

    async def test_success_resets_breaker(self):
        primary = FlakyProvider(fail_times=1)
        tool = _tool(primary)
        await tool.run(WebSearchInput(query="q1"))  # failure → fallback
        await tool.run(WebSearchInput(query="q2"))  # primary succeeds
        assert primary.calls == 2
        # breaker reset: another failure starts the count from zero
        primary.fail_times = 99
        await tool.run(WebSearchInput(query="q3"))
        assert primary.calls == 3  # tried again (was healthy)

    async def test_cooldown_expiry_retries_provider(self):
        primary = FlakyProvider(fail_times=99)
        tool = _tool(primary)
        await tool.run(WebSearchInput(query="q1"))
        await tool.run(WebSearchInput(query="q2"))  # breaker open
        # simulate cooldown expiry
        tool._health[primary.name]["blocked_until"] = time.time() - 1
        await tool.run(WebSearchInput(query="q3"))
        assert primary.calls == 3  # retried after cooldown

    async def test_empty_results_do_not_trip_breaker(self):
        class EmptyProvider(SearchProvider):
            name = "empty"

            def __init__(self):
                self.calls = 0

            async def search(self, query, max_results):
                self.calls += 1
                return []

        primary = EmptyProvider()
        tool = _tool(primary)
        await tool.run(WebSearchInput(query="q1"))
        await tool.run(WebSearchInput(query="q2"))
        await tool.run(WebSearchInput(query="q3"))
        assert primary.calls == 3  # always tried (empty ≠ failure)

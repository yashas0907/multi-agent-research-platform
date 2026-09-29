"""Web search tool with pluggable providers.

Providers:
  * `OfflineSearchProvider` — searches a small *curated local corpus* of real
    public technical sources (bundled as JSON). Deterministic, no network.
    Used for development, CI, and evaluation reproducibility.
  * `DuckDuckGoSearchProvider` — live search (rate-limited, best-effort).

Security: search results are metadata only; page content is fetched by the
`web_fetch` tool which treats everything as untrusted data.
"""
from __future__ import annotations

import json
import pathlib
from abc import ABC, abstractmethod
from typing import ClassVar

from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.errors import SearchError
from app.tools.base import BaseTool

CORPUS_PATH = pathlib.Path(__file__).parent / "offline_corpus.json"


class WebSearchInput(BaseModel):
    query: str = Field(min_length=2, max_length=300)
    max_results: int = Field(default=6, ge=1, le=15)
    recency_months: int | None = Field(default=None, ge=1, le=60)


class WebSearchResultItem(BaseModel):
    title: str
    url: str
    snippet: str = ""
    source_type: str = "other"
    domain: str | None = None
    published_at: str | None = None


class WebSearchOutput(BaseModel):
    query: str
    results: list[WebSearchResultItem] = Field(default_factory=list)
    provider: str = "offline"
    total: int = 0


class SearchProvider(ABC):
    name: str = "abstract"

    @abstractmethod
    async def search(self, query: str, max_results: int) -> list[WebSearchResultItem]:
        ...


class OfflineSearchProvider(SearchProvider):
    """Deterministic corpus search — TF-IDF-lite scoring over bundled sources.

    The corpus contains only *real* public sources (title/URL/snippet/type) —
    no fabricated pages. Full text is fetched at runtime for web sources, or
    bundled for the demo corpus.
    """

    name = "offline"

    def __init__(self, corpus_path: pathlib.Path = CORPUS_PATH) -> None:
        self.corpus_path = corpus_path
        self._corpus: list[dict] | None = None

    def _load(self) -> list[dict]:
        if self._corpus is None:
            try:
                self._corpus = json.loads(self.corpus_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise SearchError(f"offline corpus unreadable: {exc}", retryable=False) from exc
        return self._corpus

    async def search(self, query: str, max_results: int) -> list[WebSearchResultItem]:
        corpus = self._load()
        terms = _tokenize(query)
        scored: list[tuple[float, dict]] = []
        for entry in corpus:
            score = _score(entry, terms)
            if score > 0:
                scored.append((score, entry))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            WebSearchResultItem(
                title=item["title"],
                url=item["url"],
                snippet=item.get("snippet", ""),
                source_type=item.get("source_type", "other"),
                domain=item.get("domain"),
                published_at=item.get("published_at"),
            )
            for _, item in scored[:max_results]
        ]


class DuckDuckGoSearchProvider(SearchProvider):
    """Live search via DuckDuckGo's HTML endpoint (no API key required).

    Deliberately simple and best-effort: live search is a fallback for real
    deployments; the offline corpus is the reproducible default.
    NOTE: DDG tarpits/blocks datacenter IPs (AWS/GCP/etc.) — production
    deployments need the Wikipedia fallback below.
    """

    name = "duckduckgo"

    async def search(self, query: str, max_results: int) -> list[WebSearchResultItem]:
        import httpx

        url = "https://html.duckduckgo.com/html/"
        try:
            async with httpx.AsyncClient(
                timeout=get_settings().WEB_FETCH_TIMEOUT_SECONDS,
                headers={"User-Agent": "research-platform/1.0"},
            ) as client:
                resp = await client.post(url, data={"q": query})
            if resp.status_code != 200:
                raise SearchError(f"search backend returned {resp.status_code}", retryable=True)
            return _parse_ddg_html(resp.text, max_results)
        except httpx.TimeoutException as exc:
            raise SearchError("search timed out", retryable=True) from exc
        except httpx.TransportError as exc:
            raise SearchError(f"search transport error: {exc}", retryable=True) from exc


class WikipediaSearchProvider(SearchProvider):
    """Wikipedia search — free, no key, works from datacenter IPs.

    Reliable where DDG blocks: definitions, encyclopedic topics, general
    knowledge. Returns real article metadata (title/URL/snippet).
    """

    name = "wikipedia"
    ENDPOINT = "https://en.wikipedia.org/w/api.php"

    async def search(self, query: str, max_results: int) -> list[WebSearchResultItem]:
        import httpx

        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": str(max_results),
            "format": "json",
            "utf8": "1",
            "origin": "*",
        }
        try:
            async with httpx.AsyncClient(
                timeout=get_settings().WEB_FETCH_TIMEOUT_SECONDS,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (compatible; ResearchPlatform/1.0; "
                        "+https://github.com/yashas0907/multi-agent-research-platform)"
                    )
                },
            ) as client:
                resp = await client.get(self.ENDPOINT, params=params)
            if resp.status_code != 200:
                raise SearchError(f"wikipedia returned {resp.status_code}", retryable=True)
            data = resp.json()
        except httpx.TimeoutException as exc:
            raise SearchError("wikipedia search timed out", retryable=True) from exc
        except httpx.TransportError as exc:
            raise SearchError(f"wikipedia transport error: {exc}", retryable=True) from exc

        from app.tools.web_fetch import strip_html

        results: list[WebSearchResultItem] = []
        for item in data.get("query", {}).get("search", []):
            title = item.get("title", "").strip()
            snippet = strip_html(item.get("snippet", ""))
            if not title:
                continue
            slug = title.replace(" ", "_")
            results.append(
                WebSearchResultItem(
                    title=title,
                    url=f"https://en.wikipedia.org/wiki/{slug}",
                    snippet=snippet[:400],
                    source_type="documentation",
                    domain="en.wikipedia.org",
                )
            )
        return results


def _parse_ddg_html(html: str, max_results: int) -> list[WebSearchResultItem]:
    import re

    results: list[WebSearchResultItem] = []
    # DDG html results: <a rel="nofollow" class="result__a" href="...">Title</a>
    pattern = re.compile(
        r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.S
    )
    for href, title in pattern.findall(html)[: max_results * 2]:
        title = re.sub(r"<[^>]+>", "", title).strip()
        href = href.strip()
        if not title or "uddg=" in href:
            # decode ddg redirect
            if "uddg=" in href:
                import urllib.parse

                q = urllib.parse.parse_qs(urllib.parse.urlsplit(href).query)
                href = q.get("uddg", [href])[0]
        if title and href.startswith("http"):
            domain = href.split("//", 1)[-1].split("/", 1)[0]
            results.append(
                WebSearchResultItem(title=title, url=href, snippet="", domain=domain)
            )
        if len(results) >= max_results:
            break
    return results


class WebSearchTool(BaseTool[WebSearchInput, WebSearchOutput]):
    name: ClassVar[str] = "web_search"
    description: ClassVar[str] = (
        "Search the web (or the offline corpus in offline mode) for sources "
        "relevant to a research query. Returns title/URL/snippet metadata."
    )

    def __init__(self, provider: SearchProvider | None = None) -> None:
        self._provider = provider or _default_provider()
        # Production resilience chain: search engines tarpit/block datacenter
        # IPs — fall through Wikipedia (free, datacenter-friendly), then the
        # curated offline corpus (real sources). Deployed research stays cited.
        self._fallbacks: list[SearchProvider] = []
        if self._provider.name != "wikipedia":
            self._fallbacks.append(WikipediaSearchProvider())
        if self._provider.name != "offline":
            self._fallbacks.append(OfflineSearchProvider())

    def input_schema(self) -> type[WebSearchInput]:
        return WebSearchInput

    def output_schema(self) -> type[WebSearchOutput]:
        return WebSearchOutput

    async def run(self, params: WebSearchInput) -> WebSearchOutput:
        try:
            results = await self._provider.search(params.query, params.max_results)
        except SearchError:
            results = []
        except Exception as exc:  # noqa: BLE001
            raise SearchError(f"search failed: {exc}", retryable=True) from exc

        used = self._provider.name
        if not results:
            for fb in self._fallbacks:
                try:
                    results = await fb.search(params.query, params.max_results)
                except SearchError:
                    continue
                if results:
                    used = f"{self._provider.name}+{fb.name}"
                    break
        return WebSearchOutput(
            query=params.query, results=results, provider=used, total=len(results)
        )


def _default_provider() -> SearchProvider:
    settings = get_settings()
    if settings.WEB_SEARCH_PROVIDER == "duckduckgo":
        return DuckDuckGoSearchProvider()
    return OfflineSearchProvider()


# --- offline corpus scoring helpers ---
def _tokenize(text: str) -> list[str]:
    return [t for t in re_split_tokens(text.lower()) if len(t) > 2]


def re_split_tokens(text: str) -> list[str]:
    import re

    return re.split(r"[^a-z0-9+#.-]+", text)


def _score(entry: dict, terms: list[str]) -> float:
    title = entry.get("title", "").lower()
    snippet = entry.get("snippet", "").lower()
    keywords = " ".join(entry.get("keywords", [])).lower()
    score = 0.0
    for term in terms:
        if term in title:
            score += 3.0
        if term in snippet:
            score += 1.0
        if term in keywords:
            score += 2.0
    return score

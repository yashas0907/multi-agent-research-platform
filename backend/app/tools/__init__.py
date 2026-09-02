"""Tool registry wiring — the single place tools are registered."""
from __future__ import annotations

from app.core.config import get_settings
from app.tools.base import ToolRegistry
from app.tools.calculator import CalculatorTool
from app.tools.vector_search import VectorSearchTool
from app.tools.web_fetch import WebFetchTool
from app.tools.web_search import (
    DuckDuckGoSearchProvider,
    OfflineSearchProvider,
    WebSearchTool,
)


def build_tool_registry() -> ToolRegistry:
    """Create a registry with all platform tools wired to configured providers."""
    settings = get_settings()
    registry = ToolRegistry()

    if settings.WEB_SEARCH_PROVIDER == "duckduckgo":
        search_provider = DuckDuckGoSearchProvider()
    else:
        search_provider = OfflineSearchProvider()

    registry.register(WebSearchTool(provider=search_provider))
    registry.register(WebFetchTool())
    registry.register(CalculatorTool())
    registry.register(VectorSearchTool())
    return registry

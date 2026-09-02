"""Domain error taxonomy.

Every failure mode the platform anticipates maps to one of these classes so
the orchestrator can decide: retry, skip, degrade, or abort.
"""
from __future__ import annotations


class PlatformError(Exception):
    """Base class for all platform errors."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.message = message
        self.retryable = retryable

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{type(self).__name__}: {self.message}"


class ToolError(PlatformError):
    """A tool call failed (network, parsing, upstream error)."""


class ToolValidationError(ToolError):
    """Tool arguments failed schema validation."""

    def __init__(self, message: str, *, details: str = "") -> None:
        super().__init__(message, retryable=False)
        self.details = details


class LLMError(PlatformError):
    """The LLM provider call failed (timeout, rate limit, server error)."""


class StructuredOutputError(LLMError):
    """The model returned output that could not be parsed/validated.

    Retryable once: a repair pass may salvage a slightly malformed JSON.
    """


class SearchError(ToolError):
    """The web search backend failed."""


class DocumentIngestError(PlatformError):
    """Uploaded document could not be parsed or ingested."""


class BudgetExceededError(PlatformError):
    """Research budget (iterations/searches/tokens/runtime) exhausted."""

    def __init__(self, message: str, *, budget_type: str, limit: int) -> None:
        super().__init__(message, retryable=False)
        self.budget_type = budget_type
        self.limit = limit


class ResearchCancelledError(PlatformError):
    """The user cancelled the research job."""


class NotFoundError(PlatformError):
    """Requested entity does not exist."""


class VectorStoreError(PlatformError):
    """Vector store operation failed."""

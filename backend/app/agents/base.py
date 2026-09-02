"""Agent base classes.

An agent = role-specific prompt + typed LLM I/O + narrow responsibility.
Agents receive a `ResearchState` (or a slice of it), never raw conversation
history. All LLM calls go through `complete_structured` (validated output).
"""
from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, ClassVar, TypeVar

from pydantic import BaseModel

from app.core.errors import PlatformError
from app.core.llm import LLMClient, get_llm_client
from app.core.logging import get_logger
from app.prompts.library import Prompt, get_prompt_library

logger = get_logger(__name__)

# Safe operational event callback (orchestrator persists these)
EventCallback = Callable[[dict[str, Any]], Awaitable[None]]

T = TypeVar("T", bound=BaseModel)


class AgentContext:
    """Per-run context handed to agents by the orchestrator."""

    def __init__(
        self,
        llm: LLMClient,
        event_emitter: EventCallback,
        profile: dict[str, Any],
    ) -> None:
        self.llm = llm
        self.emit = event_emitter
        self.profile = profile  # depth profile (limits)


class BaseAgent:
    """Base class for all specialized agents."""

    name: ClassVar[str] = "base"
    prompt_name: ClassVar[str] = ""

    def __init__(self, ctx: AgentContext) -> None:
        self.ctx = ctx
        self._prompt: Prompt = get_prompt_library().get(self.prompt_name)

    async def _structured(self, user_payload: str, schema: type[T]) -> T:
        messages = [
            {"role": "system", "content": self._prompt.system},
            {"role": "user", "content": user_payload},
        ]
        return await self.ctx.llm.complete_structured(messages, schema)

    async def emit(self, event_type: str, message: str, **data: Any) -> None:
        await self.ctx.emit(
            {
                "agent": self.name,
                "event_type": event_type,
                "message": message[:300],
                "stage": None,
                "data": _safe(data),
            }
        )


def _safe(data: dict[str, Any]) -> dict[str, Any]:
    """Strip values that could leak prompts/completions into traces."""
    banned = {"system", "prompt", "messages", "completion", "chain_of_thought"}
    return {k: v for k, v in data.items() if k not in banned}


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"



def _safe(data: dict[str, Any]) -> dict[str, Any]:
    """Strip values that could leak prompts/completions into traces."""
    banned = {"system", "prompt", "messages", "completion", "chain_of_thought"}
    return {k: v for k, v in data.items() if k not in banned}


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"

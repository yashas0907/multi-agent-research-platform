"""Tool abstraction layer.

Every tool is a class with:
  * a unique `name`
  * Pydantic input schema (`InputT`) and output schema (`OutputT`)
  * explicit error handling (`ToolError` subclasses; never raw exceptions)
  * structured logging of calls/results (args summary, duration, outcome —
    never full untrusted content in logs)

Agents never construct tool arguments by hand from prose — they go through
`ToolRunner`, which validates inputs against the tool's schema, enforces the
session's tool-call budget, and records observability events.
"""
from __future__ import annotations

import abc
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from app.core.errors import ToolError, ToolValidationError
from app.core.logging import get_logger

logger = get_logger(__name__)

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class ToolResult(BaseModel):
    """Uniform envelope for tool execution."""

    tool: str
    ok: bool
    output: dict[str, Any] | None = None
    error: str | None = None
    error_kind: str | None = None  # timeout | upstream | validation | empty | other
    duration_ms: int = 0
    call_id: str = ""


class BaseTool(abc.ABC, Generic[InputT, OutputT]):
    """Base class for all tools."""

    name: ClassVar[str] = "tool"
    description: ClassVar[str] = ""

    @abc.abstractmethod
    def input_schema(self) -> type[BaseModel]:
        """Return the Pydantic input model class."""

    @abc.abstractmethod
    def output_schema(self) -> type[BaseModel]:
        """Return the Pydantic output model class."""

    @abc.abstractmethod
    async def run(self, params: InputT) -> OutputT:
        """Execute with *validated* params. Raise ToolError on failure."""

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema().model_json_schema(),
        }


class ToolRegistry:
    """Registry of available tools — the tool catalog exposed to agents."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            raise ToolError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        if name not in self._tools:
            raise ToolError(f"unknown tool: {name}")
        return self._tools[name]

    def all(self) -> list[BaseTool]:
        return list(self._tools.values())

    def catalog(self) -> list[dict[str, Any]]:
        return [t.describe() for t in self._tools.values()]


class ToolRunner:
    """Validated, budgeted, logged tool execution.

    Usage:
        runner = ToolRunner(registry, budget=BudgetTracker(...), on_event=cb)
        result = await runner.execute("web_search", {"query": "..."})
    """

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        on_event: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        max_tool_calls: int | None = None,
    ) -> None:
        self.registry = registry
        self.on_event = on_event
        self.max_tool_calls = max_tool_calls
        self.calls_made = 0

    def budget_exhausted(self) -> bool:
        return self.max_tool_calls is not None and self.calls_made >= self.max_tool_calls

    async def execute(self, tool_name: str, args: dict[str, Any]) -> ToolResult:
        call_id = uuid.uuid4().hex[:12]
        started = time.perf_counter()

        if self.budget_exhausted():
            return ToolResult(
                tool=tool_name,
                ok=False,
                error="tool call budget exhausted for this session",
                error_kind="budget",
                call_id=call_id,
            )

        try:
            tool = self.registry.get(tool_name)
        except ToolError as exc:
            return self._fail(tool_name, exc, "unknown_tool", started, call_id)

        try:
            params = tool.input_schema().model_validate(args)
        except ValidationError as exc:
            logger.warning("tool_args_invalid", tool=tool_name, error=str(exc)[:300])
            return self._fail(
                tool_name,
                ToolValidationError("invalid tool arguments", details=str(exc)[:400]),
                "validation",
                started,
                call_id,
            )

        self.calls_made += 1
        await self._emit(
            {
                "agent": "tool_runner",
                "event_type": "tool_call",
                "message": f"Calling tool {tool_name}",
                "stage": None,
                "data": {"tool": tool_name, "call_id": call_id},
            }
        )
        try:
            result = await tool.run(params)
            duration_ms = int((time.perf_counter() - started) * 1000)
            # Validate output against the declared schema too.
            tool.output_schema().model_validate(result.model_dump())
            logger.info(
                "tool_ok", tool=tool_name, duration_ms=duration_ms, call_id=call_id
            )
            await self._emit(
                {
                    "agent": "tool_runner",
                    "event_type": "tool_result",
                    "message": f"{tool_name} completed",
                    "stage": None,
                    "data": {
                        "tool": tool_name,
                        "ok": True,
                        "duration_ms": duration_ms,
                        "call_id": call_id,
                    },
                }
            )
            return ToolResult(
                tool=tool_name,
                ok=True,
                output=result.model_dump(),
                duration_ms=duration_ms,
                call_id=call_id,
            )
        except ToolError as exc:
            duration_ms = int((time.perf_counter() - started) * 1000)
            logger.warning("tool_failed", tool=tool_name, error=str(exc)[:300], duration_ms=duration_ms)
            await self._emit(
                {
                    "agent": "tool_runner",
                    "event_type": "tool_result",
                    "message": f"{tool_name} failed: {str(exc)[:120]}",
                    "stage": None,
                    "data": {
                        "tool": tool_name,
                        "ok": False,
                        "duration_ms": duration_ms,
                        "call_id": call_id,
                    },
                }
            )
            return self._fail(tool_name, exc, "tool_error", started, call_id)
        except Exception as exc:  # noqa: BLE001 — tools must never crash the pipeline
            logger.error("tool_crashed", tool=tool_name, error=str(exc)[:300])
            return self._fail(tool_name, exc, "crash", started, call_id)

    def _fail(
        self, tool_name: str, exc: Exception, kind: str, started: float, call_id: str
    ) -> ToolResult:
        return ToolResult(
            tool=tool_name,
            ok=False,
            error=str(exc)[:400],
            error_kind=kind,
            duration_ms=int((time.perf_counter() - started) * 1000),
            call_id=call_id,
        )

    async def _emit(self, event: dict[str, Any]) -> None:
        if self.on_event is not None:
            try:
                await self.on_event(event)
            except Exception:  # noqa: BLE001 — observability must not break research
                pass

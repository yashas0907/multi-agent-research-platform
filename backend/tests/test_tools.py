"""Unit tests: tool system — validation, budgets, error handling, calculator."""
from __future__ import annotations

import pytest

from app.tools.base import ToolRegistry, ToolRunner
from app.tools.calculator import CalculatorTool, CalculatorInput
from app.core.errors import ToolError, ToolValidationError


class TestCalculator:
    async def test_basic_ops(self):
        tool = CalculatorTool()
        out = await tool.run(CalculatorInput(expression="2 + 3 * 4"))
        assert out.result == 14

    async def test_power(self):
        tool = CalculatorTool()
        out = await tool.run(CalculatorInput(expression="2 ** 10"))
        assert out.result == 1024

    async def test_division_by_zero_rejected(self):
        tool = CalculatorTool()
        with pytest.raises(ToolValidationError):
            await tool.run(CalculatorInput(expression="1 / 0"))

    async def test_rejects_eval_injection(self):
        tool = CalculatorTool()
        with pytest.raises(ToolValidationError):
            await tool.run(CalculatorInput(expression="__import__('os').system('rm -rf /')"))

    async def test_rejects_names(self):
        tool = CalculatorTool()
        with pytest.raises(ToolValidationError):
            await tool.run(CalculatorInput(expression="open('x')"))


class TestToolRunner:
    async def test_validates_input_schema(self):
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        runner = ToolRunner(registry)
        result = await runner.execute("calculator", {"expression": "  "})  # fails min_length
        assert result.ok is False
        assert result.error_kind == "validation"

    async def test_unknown_tool(self):
        runner = ToolRunner(ToolRegistry())
        result = await runner.execute("nonexistent", {})
        assert result.ok is False
        assert result.error_kind == "unknown_tool"

    async def test_budget_exhaustion(self):
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        runner = ToolRunner(registry, max_tool_calls=1)
        ok = await runner.execute("calculator", {"expression": "1+1"})
        assert ok.ok is True
        blocked = await runner.execute("calculator", {"expression": "1+1"})
        assert blocked.ok is False
        assert blocked.error_kind == "budget"

    async def test_event_emission(self):
        events: list[dict] = []

        async def sink(ev: dict) -> None:
            events.append(ev)

        registry = ToolRegistry()
        registry.register(CalculatorTool())
        runner = ToolRunner(registry, on_event=sink)
        await runner.execute("calculator", {"expression": "1+1"})
        kinds = [e["event_type"] for e in events]
        assert "tool_call" in kinds and "tool_result" in kinds

    async def test_crashing_tool_never_raises(self):
        from app.tools.base import BaseTool
        from pydantic import BaseModel

        class In(BaseModel):
            x: int

        class Out(BaseModel):
            y: int

        class CrashTool(BaseTool[In, Out]):
            name = "crash"
            description = "always crashes"

            def input_schema(self): return In
            def output_schema(self): return Out

            async def run(self, params): raise RuntimeError("boom")

        registry = ToolRegistry()
        registry.register(CrashTool())
        runner = ToolRunner(registry)
        result = await runner.execute("crash", {"x": 1})
        assert result.ok is False
        assert result.error_kind == "crash"

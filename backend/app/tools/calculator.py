"""Safe calculator tool.

Agents must not do arithmetic in their heads — numeric claims go through
this tool. Implementation uses Python's AST evaluator (NO eval()) with a
whitelist of safe operations only.
"""
from __future__ import annotations

import ast
import operator
from typing import ClassVar

from pydantic import BaseModel, Field, field_validator

from app.core.errors import ToolValidationError
from app.tools.base import BaseTool

_OPS: dict[type[ast.AST], object] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.FloorDiv: operator.floordiv,
}


class CalculatorInput(BaseModel):
    expression: str = Field(min_length=1, max_length=500)

    @field_validator("expression")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("expression must not be empty/whitespace")
        return v


class CalculatorOutput(BaseModel):
    expression: str
    result: float
    result_text: str


class CalculatorTool(BaseTool[CalculatorInput, CalculatorOutput]):
    name: ClassVar[str] = "calculator"
    description: ClassVar[str] = (
        "Evaluate a mathematical expression safely (add, sub, mul, div, pow, mod). "
        "Use for any numeric claim derivation."
    )

    def input_schema(self) -> type[CalculatorInput]:
        return CalculatorInput

    def output_schema(self) -> type[CalculatorOutput]:
        return CalculatorOutput

    async def run(self, params: CalculatorInput) -> CalculatorOutput:
        try:
            tree = ast.parse(params.expression, mode="eval")
            value = _eval_node(tree.body)  # type: ignore[arg-type]
        except SyntaxError as exc:
            raise ToolValidationError(f"invalid expression: {exc}") from exc
        except ZeroDivisionError as exc:
            raise ToolValidationError("division by zero") from exc
        result = float(value)
        if abs(result) > 1e100:
            raise ToolValidationError("result magnitude too large")
        return CalculatorOutput(
            expression=params.expression,
            result=result,
            result_text=_format_number(result),
        )


def _eval_node(node: ast.AST) -> float | int:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ToolValidationError(f"unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp):
        op = _OPS.get(type(node.op))
        if op is None:
            raise ToolValidationError(f"unsupported operator: {type(node.op).__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(float(right)) > 100:
            raise ToolValidationError("exponent too large")
        return op(left, right)  # type: ignore[operator]
    if isinstance(node, ast.UnaryOp):
        op = _OPS.get(type(node.op))
        if op is None:
            raise ToolValidationError(f"unsupported operator: {type(node.op).__name__}")
        return op(_eval_node(node.operand))  # type: ignore[operator]
    raise ToolValidationError(f"unsupported syntax: {type(node).__name__}")


def _format_number(value: float) -> str:
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.6g}"

"""Plan schema for plan-then-execute orchestration (Step 8).

The planner LLM must emit JSON matching this schema. Pydantic validation
means a malformed or hallucinated plan (unknown tool name, missing
required field) is rejected before any step executes — the same
"validate before you trust it" philosophy as the SQL Node's sqlglot check.

Value references let one step's real output feed into a later step
without the LLM ever re-typing the number: instead of the LLM writing
"1500" as an arg, it writes {"from_step": 0, "field": "revenue"}, and the
executor resolves that from the actual stored result of step 0.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

ToolName = Literal["rag", "sql", "calc", "custom_calc"]


class ValueRef(BaseModel):
    """A reference to a field in an earlier step's real output."""
    from_step: int
    field: str
    row: int = 0  # which row, for steps that return multiple rows (e.g. SQL)


class PlanStep(BaseModel):
    tool: ToolName
    # rag / sql: natural-language query text for that tool
    query: str | None = None
    # calc: which standard metric to compute (must match calc.deterministic registry)
    intent: str | None = None
    # custom_calc: an arithmetic expression over named variables
    expression: str | None = None
    # calc / custom_calc: arg name -> literal value OR ValueRef (as a dict)
    args: dict[str, Any] = Field(default_factory=dict)

    @field_validator("tool")
    @classmethod
    def _require_fields_for_tool(cls, v):
        return v  # cross-field validation happens in Plan.validate_steps


class Plan(BaseModel):
    steps: list[PlanStep]

    @field_validator("steps")
    @classmethod
    def _validate_steps(cls, steps: list[PlanStep]) -> list[PlanStep]:
        if not steps:
            raise ValueError("Plan must contain at least one step.")
        for i, step in enumerate(steps):
            if step.tool in ("rag", "sql") and not step.query:
                raise ValueError(f"Step {i} ({step.tool}) requires a 'query'.")
            if step.tool == "calc" and not step.intent:
                raise ValueError(f"Step {i} (calc) requires an 'intent'.")
            if step.tool == "custom_calc" and not step.expression:
                raise ValueError(f"Step {i} (custom_calc) requires an 'expression'.")
            if step.tool in ("calc", "custom_calc"):
                for arg_name, arg_value in step.args.items():
                    if isinstance(arg_value, dict) and "from_step" in arg_value:
                        ref_idx = arg_value["from_step"]
                        if not (0 <= ref_idx < i):
                            raise ValueError(
                                f"Step {i} arg {arg_name!r} references step {ref_idx}, "
                                "which must be an earlier step in the plan."
                            )
        return steps

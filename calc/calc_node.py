"""Calc Node: hybrid dispatch between the deterministic and edge-case paths.

Standard metrics (gross_margin, growth_rate, etc.) never touch the LLM's
text output for the numbers — compute_standard_metric() calls the real
Python function directly on structured values.

For a calculation outside the predefined set, the LLM is allowed to write
the *formula* (an expression string, e.g. "(a - b) / a * 100"), but the
`values` dict is still supplied by the calling code from real structured
data — the LLM never re-types the numbers themselves, only the shape of
the calculation.
"""
from __future__ import annotations

from dataclasses import dataclass

from calc.deterministic import METRIC_REGISTRY, CalcResult, compute_standard_metric
from calc.metrics import CalcError
from calc.safe_eval import safe_eval


@dataclass
class CalcNodeResult:
    path: str  # "deterministic" | "custom_expression"
    value: float | None
    detail: dict
    error: str | None

    @property
    def ok(self) -> bool:
        return self.error is None


def run_calc_node(intent: str, values: dict) -> CalcNodeResult:
    """Standard-metric path. `intent` must be one of METRIC_REGISTRY's keys."""
    try:
        result: CalcResult = compute_standard_metric(intent, values)
    except CalcError as e:
        return CalcNodeResult(path="deterministic", value=None, detail={"intent": intent}, error=str(e))

    return CalcNodeResult(
        path="deterministic",
        value=result.value,
        detail={"intent": result.intent, "inputs_used": result.inputs_used},
        error=None,
    )


def run_custom_calc(expression: str, values: dict) -> CalcNodeResult:
    """Edge-case path for calculations outside METRIC_REGISTRY.

    `expression` is LLM-generated and untrusted; `values` must come from the
    calling agent code (SQL/RAG output), never from the LLM's text.
    """
    try:
        value = safe_eval(expression, values)
    except CalcError as e:
        return CalcNodeResult(
            path="custom_expression", value=None,
            detail={"expression": expression}, error=str(e),
        )

    return CalcNodeResult(
        path="custom_expression",
        value=value,
        detail={"expression": expression, "values_used": values},
        error=None,
    )


def known_metric_intents() -> list[str]:
    return sorted(METRIC_REGISTRY.keys())

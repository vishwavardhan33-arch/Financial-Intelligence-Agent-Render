"""Deterministic metric dispatch — the "safe path" from Step 7.

Given a known intent (set by the router/planner) and a dict of structured
values (typically straight from the SQL Node's typed result rows), this
calls the matching pure function directly. The LLM never sees or re-types
the numbers here — it only ever chooses *which* intent applies.
"""
from __future__ import annotations

from dataclasses import dataclass

from calc import metrics
from calc.metrics import CalcError

METRIC_REGISTRY: dict[str, dict] = {
    "gross_margin": {"fn": metrics.gross_margin, "args": ["revenue", "cogs"]},
    "operating_margin": {"fn": metrics.operating_margin, "args": ["operating_income", "revenue"]},
    "net_margin": {"fn": metrics.net_margin, "args": ["net_income", "revenue"]},
    "growth_rate": {"fn": metrics.growth_rate, "args": ["current", "previous"]},
    "cagr": {"fn": metrics.cagr, "args": ["begin_value", "end_value", "periods"]},
    "ratio": {"fn": metrics.ratio, "args": ["numerator", "denominator"]},
}


@dataclass
class CalcResult:
    intent: str
    inputs_used: dict
    value: float


def compute_standard_metric(intent: str, values: dict) -> CalcResult:
    """Look up `intent` in the registry and call its function with the
    matching fields pulled out of `values`, in the function's own
    positional order. Raises CalcError on an unknown intent or missing
    required fields — never silently guesses."""
    spec = METRIC_REGISTRY.get(intent)
    if spec is None:
        raise CalcError(
            f"Unknown standard metric intent: {intent!r}. "
            f"Known intents: {sorted(METRIC_REGISTRY.keys())}"
        )

    missing = [arg for arg in spec["args"] if arg not in values]
    if missing:
        raise CalcError(f"Missing required value(s) for '{intent}': {missing}")

    args = [values[arg] for arg in spec["args"]]
    result_value = spec["fn"](*args)

    inputs_used = {arg: values[arg] for arg in spec["args"]}
    return CalcResult(intent=intent, inputs_used=inputs_used, value=result_value)

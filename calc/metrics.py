"""Core financial metric functions.

These are plain, pure functions — no LLM involved anywhere in here. The
"never let the LLM re-type numbers" rule from Step 7 depends on these being
called directly with real values (from SQL/RAG output), never with numbers
an LLM has copied out of a text response.

All functions raise CalcError instead of letting a ZeroDivisionError or
similar leak out, so the agent layer always gets one consistent error type
to handle for "this metric isn't computable from the given inputs."
"""
from __future__ import annotations


class CalcError(Exception):
    pass


def _require_nonzero(value: float, name: str) -> None:
    if value == 0:
        raise CalcError(f"Cannot compute: {name} is zero (division by zero).")


def gross_margin(revenue: float, cogs: float) -> float:
    """(revenue - cogs) / revenue"""
    _require_nonzero(revenue, "revenue")
    return (revenue - cogs) / revenue


def operating_margin(operating_income: float, revenue: float) -> float:
    """operating_income / revenue"""
    _require_nonzero(revenue, "revenue")
    return operating_income / revenue


def net_margin(net_income: float, revenue: float) -> float:
    """net_income / revenue"""
    _require_nonzero(revenue, "revenue")
    return net_income / revenue


def growth_rate(current: float, previous: float) -> float:
    """(current - previous) / previous — e.g. YoY, QoQ growth."""
    _require_nonzero(previous, "previous")
    return (current - previous) / previous


def cagr(begin_value: float, end_value: float, periods: float) -> float:
    """Compound annual growth rate: (end/begin)^(1/periods) - 1."""
    _require_nonzero(begin_value, "begin_value")
    _require_nonzero(periods, "periods")
    if begin_value < 0 or end_value < 0:
        raise CalcError("CAGR is undefined for negative begin/end values.")
    return (end_value / begin_value) ** (1 / periods) - 1


def ratio(numerator: float, denominator: float) -> float:
    """Generic numerator / denominator, guarded against division by zero."""
    _require_nonzero(denominator, "denominator")
    return numerator / denominator

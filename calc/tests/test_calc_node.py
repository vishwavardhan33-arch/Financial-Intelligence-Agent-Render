import pytest

from calc.calc_node import run_calc_node, run_custom_calc
from calc.deterministic import compute_standard_metric
from calc.metrics import CalcError


def test_compute_standard_metric_gross_margin():
    result = compute_standard_metric("gross_margin", {"revenue": 4235102, "cogs": 2100000})
    assert result.value == pytest.approx(0.504144, rel=1e-5)
    assert result.inputs_used == {"revenue": 4235102, "cogs": 2100000}


def test_compute_standard_metric_unknown_intent_raises():
    with pytest.raises(CalcError):
        compute_standard_metric("ebitda_margin_but_this_isnt_defined", {"revenue": 100})


def test_compute_standard_metric_missing_field_raises():
    with pytest.raises(CalcError):
        compute_standard_metric("gross_margin", {"revenue": 100})  # missing cogs


def test_run_calc_node_deterministic_path_success():
    result = run_calc_node("growth_rate", {"current": 1500, "previous": 1200})
    assert result.ok
    assert result.path == "deterministic"
    assert result.value == pytest.approx(0.25)


def test_run_calc_node_deterministic_path_error_is_captured_not_raised():
    result = run_calc_node("growth_rate", {"current": 100, "previous": 0})
    assert not result.ok
    assert "zero" in result.error.lower()


def test_run_custom_calc_edge_case_success():
    result = run_custom_calc("(a - b) / a * 100", {"a": 4235102, "b": 2100000})
    assert result.ok
    assert result.path == "custom_expression"
    assert result.value == pytest.approx(50.4144, rel=1e-5)


def test_run_custom_calc_rejects_unsafe_expression():
    result = run_custom_calc("__import__('os').system('echo pwned')", {})
    assert not result.ok
    assert result.value is None

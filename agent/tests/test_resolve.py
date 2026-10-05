import pytest

from agent.resolve import ResolutionError, resolve_args


def test_resolves_literal_values_unchanged():
    resolved = resolve_args({"a": 5, "b": 10}, [])
    assert resolved == {"a": 5, "b": 10}


def test_resolves_sql_row_reference():
    step_results = [{"rows": [{"total_amount": "1200.00"}, {"total_amount": "1500.00"}]}]
    resolved = resolve_args(
        {"current": {"from_step": 0, "field": "total_amount", "row": 1},
         "previous": {"from_step": 0, "field": "total_amount", "row": 0}},
        step_results,
    )
    assert resolved == {"current": 1500.0, "previous": 1200.0}


def test_resolves_calc_step_value_reference():
    step_results = [{"value": 0.5, "intent": "gross_margin"}]
    resolved = resolve_args({"x": {"from_step": 0, "field": "value"}}, step_results)
    assert resolved == {"x": 0.5}


def test_missing_row_raises():
    step_results = [{"rows": [{"total_amount": "1200.00"}]}]
    with pytest.raises(ResolutionError):
        resolve_args({"x": {"from_step": 0, "field": "total_amount", "row": 5}}, step_results)


def test_missing_field_raises():
    step_results = [{"rows": [{"total_amount": "1200.00"}]}]
    with pytest.raises(ResolutionError):
        resolve_args({"x": {"from_step": 0, "field": "nonexistent"}}, step_results)


def test_reference_to_step_that_hasnt_run_raises():
    with pytest.raises(ResolutionError):
        resolve_args({"x": {"from_step": 3, "field": "value"}}, [])

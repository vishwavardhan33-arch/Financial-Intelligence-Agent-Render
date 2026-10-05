import pytest
from pydantic import ValidationError

from agent.plan import Plan


def test_valid_plan_parses():
    plan = Plan.model_validate({
        "steps": [
            {"tool": "sql", "query": "get amounts", "args": {}},
            {"tool": "calc", "intent": "growth_rate",
             "args": {"current": {"from_step": 0, "field": "x"}, "previous": 100}},
        ]
    })
    assert len(plan.steps) == 2


def test_empty_plan_rejected():
    with pytest.raises(ValidationError):
        Plan.model_validate({"steps": []})


def test_sql_step_without_query_rejected():
    with pytest.raises(ValidationError):
        Plan.model_validate({"steps": [{"tool": "sql", "args": {}}]})


def test_calc_step_without_intent_rejected():
    with pytest.raises(ValidationError):
        Plan.model_validate({"steps": [{"tool": "calc", "args": {}}]})


def test_forward_reference_rejected():
    """A step can't reference a step that hasn't run yet (or itself)."""
    with pytest.raises(ValidationError):
        Plan.model_validate({
            "steps": [
                {"tool": "calc", "intent": "growth_rate",
                 "args": {"current": {"from_step": 1, "field": "x"}, "previous": 100}},
                {"tool": "sql", "query": "get amounts", "args": {}},
            ]
        })

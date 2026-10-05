"""Router/planner (Step 8): turns a natural-language question into a
validated, ordered multi-step Plan.

Like the SQL Node, the LLM call is injected (`llm_generate_fn`) so this
isn't tied to a specific model provider.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from pydantic import ValidationError

from agent.plan import Plan
from calc.deterministic import METRIC_REGISTRY

PLANNER_SYSTEM_PROMPT = """You are the planner for a financial intelligence agent. \
Given a user question, output a JSON plan describing which tool(s) to call, in order.

Available tools:
- "rag": search financial filings (10-K/10-Q/earnings reports) for narrative/text answers.
  Requires "query" (natural language).
- "sql": query the invoice database (vendors, invoices, line_items).
  Requires "query" (natural language question about invoices/vendors).
- "calc": compute a standard financial metric. Requires "intent" (one of: {intents}) \
and "args" (a dict of the metric's required inputs).
- "custom_calc": compute something NOT in the standard metric list. Requires "expression" \
(an arithmetic formula using named variables) and "args" (the variable values).

For "calc"/"custom_calc" args, a value can either be a literal number, OR a reference to an \
earlier step's real output: {{"from_step": <step_index>, "field": "<field_name>", "row": 0}}. \
ALWAYS use a reference instead of typing out a number you saw in an earlier step's result — \
never re-type a number yourself.

Output ONLY a JSON object of the form {{"steps": [...]}}. No markdown fences, no explanation.
"""


@dataclass
class PlannerResult:
    plan: Plan | None
    raw_response: str
    error: str | None

    @property
    def ok(self) -> bool:
        return self.error is None


def build_planner_prompt(nl_query: str) -> str:
    system = PLANNER_SYSTEM_PROMPT.format(intents=", ".join(sorted(METRIC_REGISTRY.keys())))
    return f"{system}\n\nUser question: {nl_query}\n\nJSON plan:"


def generate_plan(nl_query: str, llm_generate_fn: Callable[[str], str]) -> PlannerResult:
    prompt = build_planner_prompt(nl_query)
    raw = llm_generate_fn(prompt)
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        parsed_json = json.loads(cleaned)
    except json.JSONDecodeError as e:
        return PlannerResult(plan=None, raw_response=raw, error=f"planner output was not valid JSON: {e}")

    try:
        plan = Plan.model_validate(parsed_json)
    except ValidationError as e:
        return PlannerResult(plan=None, raw_response=raw, error=f"planner output failed schema validation: {e}")

    return PlannerResult(plan=plan, raw_response=raw, error=None)

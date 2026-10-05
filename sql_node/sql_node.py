"""SQL Node: natural language -> validated, executed, structured result.

The `llm_generate_fn` is injected rather than hardcoded to a specific
provider, so this can be wired to Claude, GPT, or any other model at the
agent layer. It must be a callable: (prompt: str) -> str, returning the
raw SQL text the model produced.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sql_node.executor import SQLExecutionError, run_validated_sql
from sql_node.prompts import build_prompt
from sql_node.validator import SQLValidationError, validate_sql


@dataclass
class SQLNodeResult:
    nl_query: str
    generated_sql: str
    executed_sql: str | None
    rows: list[dict] | None
    error: str | None

    @property
    def ok(self) -> bool:
        return self.error is None


def run_sql_node(nl_query: str, llm_generate_fn: Callable[[str], str]) -> SQLNodeResult:
    prompt = build_prompt(nl_query)
    raw_sql = llm_generate_fn(prompt)
    # Models sometimes wrap SQL in markdown fences despite instructions not to.
    cleaned = raw_sql.strip().removeprefix("```sql").removeprefix("```").removesuffix("```").strip()

    try:
        validated = validate_sql(cleaned)
    except SQLValidationError as e:
        return SQLNodeResult(
            nl_query=nl_query, generated_sql=cleaned,
            executed_sql=None, rows=None, error=f"validation failed: {e}",
        )

    try:
        rows = run_validated_sql(validated)
    except SQLExecutionError as e:
        return SQLNodeResult(
            nl_query=nl_query, generated_sql=cleaned,
            executed_sql=validated.sql, rows=None, error=f"execution failed: {e}",
        )

    return SQLNodeResult(
        nl_query=nl_query, generated_sql=cleaned,
        executed_sql=validated.sql, rows=rows, error=None,
    )

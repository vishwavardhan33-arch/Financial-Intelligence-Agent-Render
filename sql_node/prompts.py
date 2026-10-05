"""Prompt construction for the text-to-SQL LLM call."""
from __future__ import annotations

from sql_node.schema_context import FEW_SHOT_EXAMPLES, build_schema_description

SYSTEM_INSTRUCTIONS = """You are a SQL generator for a financial invoice database.

Rules:
- Output exactly one PostgreSQL SELECT statement. Nothing else — no
  explanation, no markdown fences, no semicolon-separated multiple statements.
- Only use the tables and columns listed in the schema below.
- Never write INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, or any DDL/DML.
- Use explicit JOINs (not comma joins) and qualify ambiguous columns with
  their table alias.
"""


def build_prompt(nl_query: str) -> str:
    schema_description = build_schema_description()
    examples_text = "\n\n".join(
        f"Q: {ex['question']}\nSQL: {ex['sql']}" for ex in FEW_SHOT_EXAMPLES
    )
    return (
        f"{SYSTEM_INSTRUCTIONS}\n"
        f"Schema:\n{schema_description}\n\n"
        f"Examples:\n{examples_text}\n\n"
        f"Q: {nl_query}\nSQL:"
    )

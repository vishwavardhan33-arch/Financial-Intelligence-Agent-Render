"""Execute validated, read-only SQL and return structured results.

Deliberately returns plain Python data (list of dicts), never a
re-serialized/LLM-restated version of the numbers — this is what keeps the
"never let the LLM re-type numbers" guarantee from Step 7 intact: whatever
comes out of here goes straight into the Calc Node as real values.
"""
from __future__ import annotations

from sqlalchemy import text

from db.session import ReadonlySessionLocal
from sql_node.validator import ValidatedSQL


class SQLExecutionError(Exception):
    pass


def run_validated_sql(validated: ValidatedSQL) -> list[dict]:
    try:
        with ReadonlySessionLocal() as session:
            result = session.execute(text(validated.sql))
            rows = [dict(row._mapping) for row in result]
    except Exception as e:
        raise SQLExecutionError(f"Query failed to execute: {e}") from e

    return rows

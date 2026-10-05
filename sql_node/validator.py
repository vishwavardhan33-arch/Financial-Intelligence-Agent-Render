"""Validate LLM-generated SQL before it is ever executed.

This is the safety net from Step 6: even though the DB connection itself
uses a read-only role (belt), this validator is the suspenders — it
rejects anything that isn't a single, well-formed SELECT statement
referencing only allow-listed tables/columns, before the query ever
reaches the database.
"""
from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import expressions as exp

from sql_node.schema_context import ALLOWED_COLUMNS, ALLOWED_TABLES

DISALLOWED_NODE_TYPES = (
    exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Alter,
    exp.Create, exp.Grant, exp.Merge, exp.TruncateTable,
)


class SQLValidationError(Exception):
    pass


@dataclass
class ValidatedSQL:
    sql: str
    tables_used: set[str]


def validate_sql(raw_sql: str, dialect: str = "postgres") -> ValidatedSQL:
    """Raise SQLValidationError if raw_sql is anything other than a single,
    safe, allow-listed SELECT statement. Returns the parsed/normalized SQL
    on success."""
    raw_sql = raw_sql.strip().rstrip(";")

    try:
        statements = sqlglot.parse(raw_sql, read=dialect)
    except Exception as e:  # sqlglot raises its own ParseError subclasses
        raise SQLValidationError(f"SQL failed to parse: {e}") from e

    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        raise SQLValidationError(
            f"Expected exactly one SQL statement, got {len(statements)} "
            "(multi-statement input is rejected outright)."
        )

    stmt = statements[0]

    # Reject if the statement (or anything nested in it) is a write/DDL node.
    for node in stmt.walk():
        if isinstance(node, DISALLOWED_NODE_TYPES):
            raise SQLValidationError(f"Disallowed statement type: {type(node).__name__}")

    # The top-level statement itself must be a SELECT (optionally wrapped in
    # a CTE / WITH clause).
    root = stmt
    if isinstance(root, exp.With):
        root = root.this
    if not isinstance(root, (exp.Select, exp.Union, exp.Subquery)):
        raise SQLValidationError(f"Only SELECT statements are allowed, got: {type(root).__name__}")

    # Table allow-list. CTE aliases (e.g. "recent" in "WITH recent AS (...)")
    # are syntactically indistinguishable from real table references at this
    # level, so they must be excluded before checking against the allow-list.
    cte_names = {cte.alias_or_name.lower() for cte in stmt.find_all(exp.CTE)}
    tables_used = {t.name.lower() for t in stmt.find_all(exp.Table)} - cte_names
    disallowed_tables = tables_used - ALLOWED_TABLES
    if disallowed_tables:
        raise SQLValidationError(f"Query references non-allow-listed table(s): {disallowed_tables}")
    if not tables_used:
        raise SQLValidationError("Query does not reference any known table.")

    # Column allow-list (heuristic: every bare column identifier used must be
    # a known column somewhere in the schema; this doesn't fully resolve
    # per-table qualification, but it does catch made-up or unrelated
    # columns / any attempt to reference system columns/tables).
    # Aliases defined in the SELECT list (e.g. "SUM(x) AS total") are not
    # real columns and must be excluded, since they legitimately get
    # referenced again in ORDER BY / GROUP BY.
    defined_aliases = {a.alias_or_name.lower() for a in stmt.find_all(exp.Alias)}
    columns_used = {c.name.lower() for c in stmt.find_all(exp.Column)} - defined_aliases
    disallowed_columns = columns_used - ALLOWED_COLUMNS
    if disallowed_columns:
        raise SQLValidationError(f"Query references non-allow-listed column(s): {disallowed_columns}")

    return ValidatedSQL(sql=stmt.sql(dialect=dialect), tables_used=tables_used)

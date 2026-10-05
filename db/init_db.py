"""Idempotent database bootstrap for deployment.

    python -m db.init_db

1. Applies the table definitions from schema.sql (all CREATE ... IF NOT EXISTS).
2. Tries to create the least-privilege `agent_readonly` role. Managed hosts
   (e.g. Render) may not allow role creation; that is fine, because the
   read-only engine in db/session.py also enforces read-only transactions.
3. If the invoices table is empty, loads the bundled sample invoices so a
   fresh deployment has something to query.
"""
from __future__ import annotations

import sys
from pathlib import Path

from sqlalchemy import text

from db.session import engine

SCHEMA_PATH = Path(__file__).parent / "schema.sql"
ROLE_MARKER = "-- Read-only role"
SAMPLE_CSV = Path(__file__).parent.parent / "ingestion" / "sample_data" / "invoices.csv"


def apply_schema() -> None:
    sql = SCHEMA_PATH.read_text()
    tables_sql, _, role_sql = sql.partition(ROLE_MARKER)

    with engine.begin() as conn:
        conn.exec_driver_sql(tables_sql)
    print("schema: tables ready")

    if not role_sql.strip():
        return
    try:
        with engine.begin() as conn:
            # literal % (from format('%I')) must be escaped for the driver
            conn.exec_driver_sql((ROLE_MARKER + role_sql).replace("%", "%%"))
        print("schema: agent_readonly role ready")
    except Exception as e:  # noqa: BLE001 - any failure here is non-fatal by design
        first_line = str(e).strip().splitlines()[0]
        print(f"schema: skipped agent_readonly role ({first_line}); "
              "read-only transactions still enforced by the engine")


def seed_if_empty() -> None:
    with engine.connect() as conn:
        count = conn.execute(text("SELECT count(*) FROM invoices")).scalar_one()
    if count:
        print(f"seed: invoices table already has {count} rows, skipping")
        return
    from ingestion.load_invoices import load_invoices

    stats = load_invoices(str(SAMPLE_CSV))
    print(f"seed: loaded {stats}")


if __name__ == "__main__":
    try:
        apply_schema()
        seed_if_empty()
    except Exception as e:  # noqa: BLE001
        print(f"init_db failed: {e}", file=sys.stderr)
        raise

"""Integration test: SQL Node output chained directly into the Calc Node.

This is the concrete version of the Step 8 multi-step pattern (e.g. "what
was the growth in spend with vendor X between these two invoices") — SQL
pulls the raw numbers, Calc computes the derived metric, and at no point
does an LLM re-type any number in between. Skips gracefully if no DB is
reachable (same convention as sql_node/tests/test_sql_node_integration.py).
"""
import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from calc.calc_node import run_calc_node
from db.session import ReadonlySessionLocal
from sql_node.sql_node import run_sql_node


def _db_reachable() -> bool:
    try:
        with ReadonlySessionLocal() as session:
            session.execute(text("SELECT 1"))
        return True
    except OperationalError:
        return False


pytestmark = pytest.mark.skipif(not _db_reachable(), reason="No reachable Postgres DB with agent_readonly role.")


def test_sql_to_calc_growth_chain_uses_real_db_numbers():
    """'How much did Acme Cloud Services' invoiced amount grow between
    INV-1001 and INV-1050?' — SQL fetches both real amounts, Calc computes
    the growth rate on them directly."""
    sql_result = run_sql_node(
        "Acme Cloud Services invoice amounts for INV-1001 and INV-1050",
        lambda prompt: (
            "SELECT invoice_number, total_amount FROM invoices "
            "WHERE invoice_number IN ('INV-1001', 'INV-1050') ORDER BY invoice_number;"
        ),
    )
    assert sql_result.ok
    amounts = {row["invoice_number"]: float(row["total_amount"]) for row in sql_result.rows}
    assert amounts == {"INV-1001": 1200.0, "INV-1050": 1500.0}

    calc_result = run_calc_node("growth_rate", {
        "current": amounts["INV-1050"],
        "previous": amounts["INV-1001"],
    })
    assert calc_result.ok
    assert calc_result.value == pytest.approx(0.25)  # 1200 -> 1500 is +25%

    # The exact real numbers pulled from the DB must be the ones recorded as
    # having been used — nothing substituted, nothing re-typed.
    assert calc_result.detail["inputs_used"] == {"current": 1500.0, "previous": 1200.0}


def test_sql_to_calc_ratio_of_vendor_spend_to_total():
    sql_result = run_sql_node(
        "total spend and Acme's spend",
        lambda prompt: (
            "SELECT "
            "  (SELECT SUM(total_amount) FROM invoices) AS total_spend, "
            "  (SELECT SUM(total_amount) FROM invoices i JOIN vendors v ON v.vendor_id = i.vendor_id "
            "     WHERE v.name = 'Acme Cloud Services') AS acme_spend;"
        ),
    )
    assert sql_result.ok
    row = sql_result.rows[0]

    calc_result = run_calc_node("ratio", {
        "numerator": float(row["acme_spend"]),
        "denominator": float(row["total_spend"]),
    })
    assert calc_result.ok
    # 2700 (Acme) / 5900.50 (all four invoices) from the seeded sample data
    assert calc_result.value == pytest.approx(2700 / 5900.50, rel=1e-4)

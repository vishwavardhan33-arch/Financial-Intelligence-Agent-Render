import pytest

from sql_node.validator import SQLValidationError, validate_sql


@pytest.mark.parametrize("sql", [
    "SELECT v.name, SUM(i.total_amount) AS total_invoiced "
    "FROM invoices i JOIN vendors v ON v.vendor_id = i.vendor_id "
    "GROUP BY v.name ORDER BY total_invoiced DESC;",

    "SELECT invoice_number, total_amount FROM invoices WHERE status = 'overdue';",

    "WITH recent AS (SELECT * FROM invoices WHERE invoice_date > '2024-01-01') "
    "SELECT * FROM recent;",

    "WITH recent AS (SELECT * FROM invoices), v AS (SELECT * FROM vendors) "
    "SELECT recent.invoice_number, v.name FROM recent "
    "JOIN v ON v.vendor_id = recent.vendor_id;",
])
def test_valid_queries_pass(sql):
    result = validate_sql(sql)
    assert result.sql  # normalized SQL returned


@pytest.mark.parametrize("sql,expected_substring", [
    ("DROP TABLE invoices;", "Disallowed statement type"),
    ("DELETE FROM invoices;", "Disallowed statement type"),
    ("SELECT 1; DROP TABLE invoices;", "exactly one SQL statement"),
    ("SELECT * FROM pg_shadow;", "non-allow-listed table"),
    ("SELECT vendor_id, ssn FROM vendors;", "non-allow-listed column"),
    ("WITH x AS (SELECT 1) UPDATE invoices SET total_amount = 0;", "Disallowed statement type"),
    ("not even valid sql at all (((", None),  # just must raise, message is a parse error
])
def test_attacks_and_invalid_sql_are_rejected(sql, expected_substring):
    with pytest.raises(SQLValidationError) as exc_info:
        validate_sql(sql)
    if expected_substring:
        assert expected_substring in str(exc_info.value)

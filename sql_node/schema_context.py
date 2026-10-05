"""Schema description + allow-list, kept in sync with db/schema.sql by hand.

Used for two things:
  1. Building the schema context injected into the text-to-SQL prompt.
  2. The allow-list the validator checks generated SQL against.
"""
from __future__ import annotations

SCHEMA: dict[str, dict] = {
    "vendors": {
        "columns": ["vendor_id", "name", "category", "created_at"],
        "description": "One row per vendor/supplier that has sent an invoice.",
    },
    "invoices": {
        "columns": [
            "invoice_id", "vendor_id", "invoice_number", "invoice_date",
            "due_date", "total_amount", "currency", "status", "created_at",
        ],
        "description": "One row per invoice. vendor_id references vendors.vendor_id. "
                        "status is one of: pending, paid, rejected, overdue.",
    },
    "line_items": {
        "columns": [
            "line_item_id", "invoice_id", "description", "quantity",
            "unit_price", "amount", "category",
        ],
        "description": "One row per line item on an invoice. invoice_id references "
                        "invoices.invoice_id.",
    },
}

ALLOWED_TABLES = set(SCHEMA.keys())
ALLOWED_COLUMNS = {col for t in SCHEMA.values() for col in t["columns"]}

FEW_SHOT_EXAMPLES = [
    {
        "question": "What is the total amount invoiced by each vendor?",
        "sql": (
            "SELECT v.name, SUM(i.total_amount) AS total_invoiced "
            "FROM invoices i JOIN vendors v ON v.vendor_id = i.vendor_id "
            "GROUP BY v.name ORDER BY total_invoiced DESC;"
        ),
    },
    {
        "question": "List all overdue invoices with the vendor name and amount.",
        "sql": (
            "SELECT i.invoice_number, v.name AS vendor, i.total_amount, i.due_date "
            "FROM invoices i JOIN vendors v ON v.vendor_id = i.vendor_id "
            "WHERE i.status = 'overdue';"
        ),
    },
    {
        "question": "What was total revenue-relevant spend by category last quarter?",
        "sql": (
            "SELECT li.category, SUM(li.amount) AS total_amount "
            "FROM line_items li JOIN invoices i ON i.invoice_id = li.invoice_id "
            "WHERE i.invoice_date >= date_trunc('quarter', CURRENT_DATE) - INTERVAL '3 months' "
            "AND i.invoice_date < date_trunc('quarter', CURRENT_DATE) "
            "GROUP BY li.category;"
        ),
    },
]


def build_schema_description() -> str:
    lines = []
    for table, info in SCHEMA.items():
        cols = ", ".join(info["columns"])
        lines.append(f"- {table}({cols})\n  {info['description']}")
    return "\n".join(lines)

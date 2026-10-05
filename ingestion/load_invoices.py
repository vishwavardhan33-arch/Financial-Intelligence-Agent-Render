"""Load invoice data from CSV/Excel into the normalized Postgres schema.

Expected input format: one row per line item, invoice-level fields repeated
across the line items that belong to the same invoice. Grouped by
(vendor_name, invoice_number) to build Invoice rows, with one LineItem per
input row.

Usage:
    python -m ingestion.load_invoices path/to/invoices.csv
    python -m ingestion.load_invoices path/to/invoices.xlsx
"""
from __future__ import annotations

import sys
import datetime as dt

import pandas as pd
from sqlalchemy.orm import Session

from db.models import Vendor, Invoice, LineItem
from db.session import SessionLocal

REQUIRED_COLUMNS = [
    "vendor_name",
    "invoice_number",
    "invoice_date",
    "total_amount",
    "item_description",
    "item_quantity",
    "item_unit_price",
    "item_amount",
]


def load_dataframe(path: str) -> pd.DataFrame:
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")

    # Normalize dtypes.
    df["invoice_date"] = pd.to_datetime(df["invoice_date"]).dt.date
    if "due_date" in df.columns:
        df["due_date"] = pd.to_datetime(df["due_date"], errors="coerce").dt.date
    df["currency"] = df.get("currency", "USD")
    df["status"] = df.get("status", "pending")
    df["vendor_category"] = df.get("vendor_category")
    df["item_category"] = df.get("item_category")

    return df


def get_or_create_vendor(session: Session, name: str, category: str | None) -> Vendor:
    vendor = session.query(Vendor).filter_by(name=name).one_or_none()
    if vendor is None:
        vendor = Vendor(name=name, category=category)
        session.add(vendor)
        session.flush()  # populate vendor_id without committing
    return vendor


def get_or_create_invoice(
    session: Session,
    vendor: Vendor,
    invoice_number: str,
    invoice_date: dt.date,
    due_date: dt.date | None,
    total_amount: float,
    currency: str,
    status: str,
) -> Invoice:
    invoice = (
        session.query(Invoice)
        .filter_by(vendor_id=vendor.vendor_id, invoice_number=invoice_number)
        .one_or_none()
    )
    if invoice is None:
        invoice = Invoice(
            vendor_id=vendor.vendor_id,
            invoice_number=invoice_number,
            invoice_date=invoice_date,
            due_date=due_date,
            total_amount=total_amount,
            currency=currency,
            status=status,
        )
        session.add(invoice)
        session.flush()
    return invoice


def load_invoices(path: str) -> dict:
    df = load_dataframe(path)
    stats = {"vendors": 0, "invoices": 0, "line_items": 0}

    with SessionLocal() as session:
        seen_vendors: set[str] = set()
        seen_invoices: set[tuple[int, str]] = set()

        for _, row in df.iterrows():
            vendor = get_or_create_vendor(session, row["vendor_name"], row.get("vendor_category"))
            if vendor.name not in seen_vendors:
                seen_vendors.add(vendor.name)
                stats["vendors"] += 1

            invoice = get_or_create_invoice(
                session,
                vendor=vendor,
                invoice_number=str(row["invoice_number"]),
                invoice_date=row["invoice_date"],
                due_date=row.get("due_date"),
                total_amount=float(row["total_amount"]),
                currency=row["currency"],
                status=row["status"],
            )
            key = (vendor.vendor_id, invoice.invoice_number)
            if key not in seen_invoices:
                seen_invoices.add(key)
                stats["invoices"] += 1

            line_item = LineItem(
                invoice_id=invoice.invoice_id,
                description=row["item_description"],
                quantity=float(row["item_quantity"]),
                unit_price=float(row["item_unit_price"]),
                amount=float(row["item_amount"]),
                category=row.get("item_category"),
            )
            session.add(line_item)
            stats["line_items"] += 1

        session.commit()

    return stats


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m ingestion.load_invoices <path_to_csv_or_xlsx>")
        sys.exit(1)

    result = load_invoices(sys.argv[1])
    print(f"Loaded: {result['vendors']} vendors, {result['invoices']} invoices, "
          f"{result['line_items']} line items.")

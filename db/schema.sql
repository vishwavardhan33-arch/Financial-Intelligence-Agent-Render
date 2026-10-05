-- Financial Intelligence Agent: invoice DB schema
-- Normalized: vendors -> invoices -> line_items

CREATE TABLE IF NOT EXISTS vendors (
    vendor_id       SERIAL PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,
    category        TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS invoices (
    invoice_id      SERIAL PRIMARY KEY,
    vendor_id       INTEGER NOT NULL REFERENCES vendors(vendor_id),
    invoice_number  TEXT,
    invoice_date    DATE NOT NULL,
    due_date        DATE,
    total_amount    NUMERIC(14, 2) NOT NULL,
    currency        TEXT NOT NULL DEFAULT 'USD',
    status          TEXT NOT NULL DEFAULT 'pending', -- pending / paid / rejected / overdue
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS line_items (
    line_item_id    SERIAL PRIMARY KEY,
    invoice_id      INTEGER NOT NULL REFERENCES invoices(invoice_id) ON DELETE CASCADE,
    description     TEXT,
    quantity        NUMERIC(12, 2) NOT NULL DEFAULT 1,
    unit_price      NUMERIC(14, 2) NOT NULL,
    amount          NUMERIC(14, 2) NOT NULL,
    category        TEXT
);

-- Indexes for the query patterns the SQL Node will generate most often:
-- filtering/joining by vendor, filtering by date range, aggregating by status.
CREATE INDEX IF NOT EXISTS idx_invoices_vendor_id   ON invoices(vendor_id);
CREATE INDEX IF NOT EXISTS idx_invoices_invoice_date ON invoices(invoice_date);
CREATE INDEX IF NOT EXISTS idx_invoices_status       ON invoices(status);
CREATE INDEX IF NOT EXISTS idx_line_items_invoice_id ON line_items(invoice_id);

-- Read-only role for the text-to-SQL execution layer (Step 6 safety net).
-- The agent's DB connection should use this role, never the owner role,
-- so a validation-layer gap can't turn into a write/delete.
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'agent_readonly') THEN
        CREATE ROLE agent_readonly LOGIN PASSWORD 'change_me_in_env';
    END IF;
END
$$;

DO $$
BEGIN
    EXECUTE format('GRANT CONNECT ON DATABASE %I TO agent_readonly', current_database());
END
$$;

GRANT USAGE ON SCHEMA public TO agent_readonly;
GRANT SELECT ON vendors, invoices, line_items TO agent_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO agent_readonly;

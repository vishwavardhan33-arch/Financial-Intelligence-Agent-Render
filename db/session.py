"""Engine/session factory.

Two engines are exposed:
- `engine`           : owner connection, used by ingestion scripts (read/write)
- `readonly_engine`  : used by the SQL Node at query time

Connection config comes from either:
- DATABASE_URL (what Render and most hosts provide), or
- the individual DB_HOST / DB_PORT / DB_NAME / DB_*_USER / DB_*_PASSWORD vars
  used by docker-compose.

The read-only engine is defence in depth beneath the sqlglot validator. It
always opens connections with `default_transaction_read_only=on` and a
statement timeout, so even if the validator had a gap the session itself
refuses writes. With the individual-vars setup it additionally connects as
the `agent_readonly` role, which only has SELECT grants. Managed Postgres
hosts that don't let you create roles can set only DATABASE_URL and still
get the read-only transaction guarantee.
"""
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

STATEMENT_TIMEOUT_MS = int(os.getenv("SQL_STATEMENT_TIMEOUT_MS", "10000"))


def _normalize(url: str) -> str:
    """Hosts hand out postgres:// or postgresql://; SQLAlchemy needs the driver named."""
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


def _build_urls() -> tuple[str, str]:
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        owner = _normalize(database_url)
        readonly = _normalize(os.getenv("DATABASE_READONLY_URL", database_url))
        return owner, readonly

    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME", "financial_agent")
    owner_user = os.getenv("DB_OWNER_USER", "postgres")
    owner_password = os.getenv("DB_OWNER_PASSWORD", "postgres")
    ro_user = os.getenv("DB_READONLY_USER", "agent_readonly")
    ro_password = os.getenv("DB_READONLY_PASSWORD", "change_me_in_env")
    return (
        f"postgresql+psycopg://{owner_user}:{owner_password}@{host}:{port}/{name}",
        f"postgresql+psycopg://{ro_user}:{ro_password}@{host}:{port}/{name}",
    )


_OWNER_URL, _READONLY_URL = _build_urls()

_READONLY_CONNECT_ARGS = {
    "options": f"-c default_transaction_read_only=on -c statement_timeout={STATEMENT_TIMEOUT_MS}"
}

engine = create_engine(_OWNER_URL, future=True, pool_pre_ping=True)
readonly_engine = create_engine(
    _READONLY_URL, future=True, pool_pre_ping=True, connect_args=_READONLY_CONNECT_ARGS
)

SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)
ReadonlySessionLocal = sessionmaker(bind=readonly_engine, expire_on_commit=False, future=True)

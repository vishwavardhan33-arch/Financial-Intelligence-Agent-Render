"""FastAPI app: the agent's HTTP API plus the web UI.

Uses an app-factory pattern (`create_app`) so tests can inject fake LLM
functions and a test index, while real deployment uses the module-level
`app`, wired to an OpenAI-compatible open-model API (Groq by default) and the
prebuilt filings corpus.
"""
from __future__ import annotations

import datetime as dt
import os
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from decimal import Decimal
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent.graph import build_agent_graph, run_agent
from api.llm_provider import LLMError

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


class QueryRequest(BaseModel):
    query: str = Field(min_length=3, max_length=500)


class QueryResponse(BaseModel):
    answer: str | None
    citations: list[int]
    plan: list[dict]
    step_results: list[dict]
    error: str | None


class RateLimiter:
    """Sliding-window limiter, per client IP. In-memory is enough for a single
    free-tier instance; it exists so a public URL can't burn through the
    free LLM quota."""

    def __init__(self, max_requests: int, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] > self.window:
                hits.popleft()
            if len(hits) >= self.max_requests:
                return False
            hits.append(now)
            return True


def _plain(value):
    """Make step results JSON-friendly. Postgres NUMERIC arrives as Decimal,
    which pydantic would serialize as a string ("1200.00"); the UI needs real
    numbers so it can align and format them."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def create_app(
    planner_llm_fn: Callable[[str], str] | None = None,
    sql_llm_fn: Callable[[str], str] | None = None,
    synthesizer_llm_fn: Callable[[str], str] | None = None,
    rag_index=None,
    rag_reranker=None,
    rate_limit_per_minute: int | None = None,
) -> FastAPI:
    using_default_llm = planner_llm_fn is None or sql_llm_fn is None or synthesizer_llm_fn is None
    if using_default_llm:
        from api.llm_provider import build_llm_fn

        default_llm_fn = build_llm_fn()
        planner_llm_fn = planner_llm_fn or default_llm_fn
        sql_llm_fn = sql_llm_fn or default_llm_fn
        synthesizer_llm_fn = synthesizer_llm_fn or default_llm_fn

        if rag_index is None:
            from api.rag_setup import build_default_rag

            rag_index, rag_reranker = build_default_rag()

    graph = build_agent_graph(planner_llm_fn, sql_llm_fn, synthesizer_llm_fn, rag_index, rag_reranker)

    limit = rate_limit_per_minute
    if limit is None:
        limit = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "8"))
    limiter = RateLimiter(limit) if limit > 0 else None

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Load the embedding model in the background so the first question
        # isn't the one that pays for it.
        if hasattr(rag_index, "warm_up"):
            threading.Thread(target=rag_index.warm_up, daemon=True).start()
        yield

    app = FastAPI(title="Financial Intelligence Agent", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/status")
    def status():
        return {
            "llm_configured": (not using_default_llm) or bool(os.environ.get("LLM_API_KEY")),
            "model": os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile"),
            "filings_indexed": rag_index is not None,
        }

    @app.post("/query", response_model=QueryResponse)
    def query(request: QueryRequest, http_request: Request) -> QueryResponse:
        if limiter is not None and not limiter.allow(_client_ip(http_request)):
            raise HTTPException(status_code=429, detail="Too many questions in a short time. Wait a minute and try again.")

        try:
            state = run_agent(request.query, graph)
        except LLMError as e:
            return QueryResponse(answer=None, citations=[], plan=[], step_results=[], error=str(e))

        plan = state.get("plan")
        return QueryResponse(
            answer=state.get("final_answer"),
            citations=state.get("citations", []),
            plan=[step.model_dump(exclude_none=True) for step in plan.steps] if plan else [],
            step_results=_plain(state.get("step_results", [])),
            error=state.get("error"),
        )

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @app.get("/", include_in_schema=False)
        def index():
            return FileResponse(STATIC_DIR / "index.html")

    return app


# Real deployment entrypoint: `uvicorn api.main:app`. Nothing here connects to
# the LLM provider or loads a model at import time; both happen lazily.
app = create_app()

import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from api.main import create_app
from db.session import ReadonlySessionLocal


def _db_reachable() -> bool:
    try:
        with ReadonlySessionLocal() as session:
            session.execute(text("SELECT 1"))
        return True
    except OperationalError:
        return False


needs_db = pytest.mark.skipif(not _db_reachable(), reason="No reachable Postgres DB.")


def test_health_endpoint():
    app = create_app(planner_llm_fn=lambda p: "", sql_llm_fn=lambda p: "", synthesizer_llm_fn=lambda p: "")
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@needs_db
def test_query_endpoint_full_flow_against_real_db():
    def planner(prompt):
        return json.dumps({"steps": [
            {"tool": "sql", "query": "Acme invoice amounts", "args": {}},
            {"tool": "calc", "intent": "growth_rate", "args": {
                "current": {"from_step": 0, "field": "total_amount", "row": 1},
                "previous": {"from_step": 0, "field": "total_amount", "row": 0},
            }},
        ]})

    def sql_llm(prompt):
        return ("SELECT invoice_number, total_amount FROM invoices "
                "WHERE invoice_number IN ('INV-1001', 'INV-1050') ORDER BY invoice_number;")

    def synthesizer(prompt):
        return json.dumps({"answer": "Grew 25%.", "citations": []})

    app = create_app(planner_llm_fn=planner, sql_llm_fn=sql_llm, synthesizer_llm_fn=synthesizer)
    client = TestClient(app)

    response = client.post("/query", json={"query": "How much did Acme spend grow?"})
    assert response.status_code == 200

    body = response.json()
    assert body["answer"] == "Grew 25%."
    assert body["error"] is None
    assert body["step_results"][1]["value"] == pytest.approx(0.25)


def test_query_endpoint_returns_error_field_on_malformed_plan():
    app = create_app(
        planner_llm_fn=lambda p: "not valid json {{{",
        sql_llm_fn=lambda p: "",
        synthesizer_llm_fn=lambda p: "",
    )
    client = TestClient(app)

    response = client.post("/query", json={"query": "anything"})
    assert response.status_code == 200  # the API itself succeeded; the AGENT failed gracefully
    body = response.json()
    assert body["error"] is not None
    assert body["answer"] is None


def test_importing_default_app_needs_no_api_key_or_network(monkeypatch):
    """Import-time app creation must not call the LLM provider or load an
    embedding model, so importing works with no key set (test collection,
    docs builds, first deploy before the key is added)."""
    import importlib

    monkeypatch.delenv("LLM_API_KEY", raising=False)
    import api.main as main_module
    importlib.reload(main_module)  # re-run module-level `app = create_app()`
    assert main_module.app is not None


def test_response_includes_the_plan_for_the_ui():
    planner = lambda p: json.dumps({"steps": [
        {"tool": "calc", "intent": "growth_rate", "args": {"current": 150, "previous": 100}},
    ]})
    synthesizer = lambda p: json.dumps({"answer": "Up 50%.", "citations": []})
    client = TestClient(create_app(planner, lambda p: "", synthesizer))

    body = client.post("/query", json={"query": "growth from 100 to 150?"}).json()
    assert body["plan"] == [{"tool": "calc", "intent": "growth_rate", "args": {"current": 150, "previous": 100}}]
    assert body["step_results"][0]["value"] == pytest.approx(0.5)


def test_llm_failure_returns_error_field_not_500():
    from api.llm_provider import LLMError

    def planner(prompt):
        raise LLMError("Provider unavailable after 4 attempts (429)")

    client = TestClient(create_app(planner, lambda p: "", lambda p: ""))
    response = client.post("/query", json={"query": "anything at all"})
    assert response.status_code == 200
    assert "Provider unavailable" in response.json()["error"]


def test_rate_limit_blocks_after_limit_per_ip():
    client = TestClient(create_app(
        planner_llm_fn=lambda p: "not json",
        sql_llm_fn=lambda p: "",
        synthesizer_llm_fn=lambda p: "",
        rate_limit_per_minute=2,
    ))
    codes = [client.post("/query", json={"query": "some question"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_rejects_empty_and_oversized_queries():
    client = TestClient(create_app(lambda p: "", lambda p: "", lambda p: ""))
    assert client.post("/query", json={"query": ""}).status_code == 422
    assert client.post("/query", json={"query": "x" * 501}).status_code == 422


def test_rag_step_uses_injected_index_and_citations_survive():
    from retrieval.chunker import Chunk
    from retrieval.embeddings import DeterministicFakeEmbedder
    from retrieval.vector_store import FakeReranker, HybridIndex

    index = HybridIndex(embedder=DeterministicFakeEmbedder())
    index.build([Chunk(text="Gross margin improved on favorable product mix.", chunk_type="text", section="MD&A", doc_id="d1")])

    planner = lambda p: json.dumps({"steps": [{"tool": "rag", "query": "gross margin drivers"}]})
    synthesizer = lambda p: json.dumps({"answer": "Product mix.", "citations": [0, 99]})
    client = TestClient(create_app(planner, lambda p: "", synthesizer, rag_index=index, rag_reranker=FakeReranker()))

    body = client.post("/query", json={"query": "why did gross margin improve?"}).json()
    assert body["citations"] == [0]  # fabricated id 99 dropped
    assert body["step_results"][0]["chunks"][0]["doc_id"] == "d1"


def test_plain_converts_decimal_and_dates_to_json_friendly_values():
    import datetime as dt
    from decimal import Decimal

    from api.main import _plain

    out = _plain([{"total": Decimal("1200.50"), "due": dt.date(2024, 2, 14), "name": "Acme"}])
    assert out == [{"total": 1200.5, "due": "2024-02-14", "name": "Acme"}]
    assert isinstance(out[0]["total"], float)


@needs_db
def test_sql_amounts_reach_the_client_as_numbers():
    planner = lambda p: json.dumps({"steps": [{"tool": "sql", "query": "totals"}]})
    sql_llm = lambda p: "SELECT invoice_number, total_amount FROM invoices WHERE invoice_number = 'INV-1001'"
    synth = lambda p: json.dumps({"answer": "ok", "citations": []})
    client = TestClient(create_app(planner, sql_llm, synth))

    row = client.post("/query", json={"query": "show INV-1001"}).json()["step_results"][0]["rows"][0]
    assert row["total_amount"] == 1200.0 and isinstance(row["total_amount"], float)

import json

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from agent.graph import build_agent_graph, run_agent
from db.session import ReadonlySessionLocal
from retrieval.chunker import chunk_filing
from retrieval.embeddings import DeterministicFakeEmbedder
from retrieval.vector_store import FakeReranker, HybridIndex


def _db_reachable() -> bool:
    try:
        with ReadonlySessionLocal() as session:
            session.execute(text("SELECT 1"))
        return True
    except OperationalError:
        return False


needs_db = pytest.mark.skipif(not _db_reachable(), reason="No reachable Postgres DB.")


@needs_db
def test_full_sql_to_calc_plan_executes_against_real_db():
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

    graph = build_agent_graph(planner, sql_llm, synthesizer)
    state = run_agent("growth question", graph)

    assert state["error"] is None
    assert state["final_answer"] == "Grew 25%."
    calc_step = state["step_results"][1]
    assert calc_step["value"] == pytest.approx(0.25)


def test_malformed_plan_json_halts_cleanly_without_crashing():
    graph = build_agent_graph(lambda p: "not json {{{", lambda p: "", lambda p: "")
    state = run_agent("anything", graph)
    assert state["error"] is not None
    assert state.get("final_answer") is None


@needs_db
def test_failing_step_halts_before_dependent_step_runs():
    def planner(prompt):
        return json.dumps({"steps": [
            {"tool": "sql", "query": "malicious", "args": {}},
            {"tool": "calc", "intent": "growth_rate", "args": {
                "current": {"from_step": 0, "field": "x", "row": 0}, "previous": 100,
            }},
        ]})

    def sql_llm_malicious(prompt):
        return "DROP TABLE invoices;"

    def synthesizer(prompt):
        return json.dumps({"answer": "Could not complete: the query was rejected.", "citations": []})

    graph = build_agent_graph(planner, sql_llm_malicious, synthesizer)
    state = run_agent("bad flow", graph)

    # Only the failed SQL step ran — the dependent calc step must NOT have executed.
    assert len(state["step_results"]) == 1
    assert "step_error" in state["step_results"][0]
    assert state["final_answer"] == "Could not complete: the query was rejected."


def test_rag_citations_validated_against_real_chunk_ids():
    with open("retrieval/tests/sample_filing.md") as f:
        md = f.read()
    chunks = chunk_filing(md, doc_id="ACME_10K_2024", company="Acme Corp", fiscal_period="FY2024")
    index = HybridIndex(embedder=DeterministicFakeEmbedder())
    index.build(chunks)

    def planner(prompt):
        return json.dumps({"steps": [{"tool": "rag", "query": "gross margin", "args": {}}]})

    def synthesizer(prompt):
        # chunk_id 0 is real; 999 is fabricated and must be dropped.
        return json.dumps({"answer": "Margin improved.", "citations": [0, 999]})

    graph = build_agent_graph(planner, lambda p: "", synthesizer, rag_index=index, rag_reranker=FakeReranker())
    state = run_agent("What was the gross margin?", graph)

    assert state["citations"] == [0]


def test_rag_backend_failure_becomes_a_failed_step_not_a_crash():
    from agent.dispatch import execute_step
    from agent.plan import PlanStep

    class BrokenIndex:
        def hybrid_search(self, *a, **k):
            raise RuntimeError("model download failed")

    result = execute_step(PlanStep(tool="rag", query="anything"), [], lambda p: "", BrokenIndex(), None)
    assert not result.ok
    assert "model download failed" in result.error

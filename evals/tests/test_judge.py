import json

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from agent.graph import build_agent_graph
from db.session import ReadonlySessionLocal
from evals.judge import JudgeParseError, judge_end_to_end, judge_retrieval, judge_sql
from evals.run_evals import run_eval_suite, summarize


def _db_reachable() -> bool:
    try:
        with ReadonlySessionLocal() as session:
            session.execute(text("SELECT 1"))
        return True
    except OperationalError:
        return False


needs_db = pytest.mark.skipif(not _db_reachable(), reason="No reachable Postgres DB.")


def test_judge_retrieval_parses_structured_score():
    def fake_judge(prompt):
        assert "revenue" in prompt.lower() or "chunk" in prompt.lower()
        return json.dumps({"score": 4, "justification": "Mostly relevant."})

    result = judge_retrieval("What was revenue?", [{"chunk_id": 0, "text": "Revenue grew"}], fake_judge)
    assert result.score == 4


def test_judge_sql_parses_structured_score():
    def fake_judge(prompt):
        return json.dumps({"score": 5, "justification": "Correct join and filter."})

    result = judge_sql("total by vendor", "SELECT ...", [{"total": 100}], fake_judge)
    assert result.score == 5


def test_judge_end_to_end_parses_three_scores():
    def fake_judge(prompt):
        return json.dumps({"correctness": 5, "groundedness": 4, "routing_appropriateness": 5,
                            "justification": "Good."})

    result = judge_end_to_end("q", "a", [], fake_judge)
    assert result.average == pytest.approx((5 + 4 + 5) / 3)


def test_judge_raises_on_unparseable_output():
    with pytest.raises(JudgeParseError):
        judge_end_to_end("q", "a", [], lambda p: "not json")


@needs_db
def test_eval_suite_runs_agent_then_judges_with_real_data():
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

    def judge(prompt):
        # Prove the judge actually receives the real answer, not a stub.
        assert "Grew 25%" in prompt
        return json.dumps({"correctness": 5, "groundedness": 5, "routing_appropriateness": 5,
                            "justification": "Matches SQL result."})

    graph = build_agent_graph(planner, sql_llm, synthesizer)
    results = run_eval_suite(["growth question"], graph, judge)

    assert len(results) == 1
    assert results[0].judgment.correctness == 5

    summary = summarize(results)
    assert summary["n_failed_to_run"] == 0
    assert summary["avg_correctness"] == 5.0


def test_summarize_counts_failed_cases_separately():
    from evals.run_evals import EvalCaseResult
    results = [
        EvalCaseResult(query="q1", final_answer="a1", error=None, judgment=None),
    ]
    summary = summarize(results)
    assert summary["n_failed_to_run"] == 1
    assert summary["avg_correctness"] is None

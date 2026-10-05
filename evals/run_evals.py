"""Run a batch of eval queries through the agent graph and score each with
the end-to-end LLM-as-judge (Step 10).

This is deliberately not tied to a hand-labeled golden set (per the chosen
eval strategy: pure LLM-as-judge) — but see the calibration note in
evals/judge.py before trusting scores at scale.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from agent.graph import run_agent
from evals.judge import EndToEndJudgment, judge_end_to_end


@dataclass
class EvalCaseResult:
    query: str
    final_answer: str | None
    error: str | None
    judgment: EndToEndJudgment | None


def run_eval_suite(
    queries: list[str],
    graph,
    judge_llm_fn: Callable[[str], str],
) -> list[EvalCaseResult]:
    results = []
    for query in queries:
        state = run_agent(query, graph)

        if state.get("error") or state.get("final_answer") is None:
            results.append(EvalCaseResult(
                query=query, final_answer=None, error=state.get("error"), judgment=None,
            ))
            continue

        judgment = judge_end_to_end(
            query=query,
            answer=state["final_answer"],
            step_results=state["step_results"],
            judge_llm_fn=judge_llm_fn,
        )
        results.append(EvalCaseResult(
            query=query, final_answer=state["final_answer"], error=None, judgment=judgment,
        ))

    return results


def summarize(results: list[EvalCaseResult]) -> dict:
    judged = [r.judgment for r in results if r.judgment is not None]
    n_failed = sum(1 for r in results if r.judgment is None)

    if not judged:
        return {"n_cases": len(results), "n_failed_to_run": n_failed, "avg_correctness": None,
                "avg_groundedness": None, "avg_routing": None}

    return {
        "n_cases": len(results),
        "n_failed_to_run": n_failed,
        "avg_correctness": sum(j.correctness for j in judged) / len(judged),
        "avg_groundedness": sum(j.groundedness for j in judged) / len(judged),
        "avg_routing": sum(j.routing_appropriateness for j in judged) / len(judged),
    }


def print_report(results: list[EvalCaseResult]) -> None:
    for r in results:
        print(f"\nQ: {r.query}")
        if r.judgment is None:
            print(f"  FAILED TO RUN: {r.error}")
            continue
        print(f"  A: {r.final_answer}")
        j = r.judgment
        print(f"  correctness={j.correctness} groundedness={j.groundedness} "
              f"routing={j.routing_appropriateness} -- {j.justification}")

    summary = summarize(results)
    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

"""Dispatch a single plan step to its underlying tool implementation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from agent.plan import PlanStep
from agent.resolve import ResolutionError, resolve_args
from calc.calc_node import run_calc_node, run_custom_calc
from retrieval.vector_store import HybridIndex
from sql_node.sql_node import run_sql_node


@dataclass
class StepExecutionResult:
    tool: str
    output: dict          # stored verbatim in step_results for later reference resolution / synthesis
    error: str | None

    @property
    def ok(self) -> bool:
        return self.error is None


def execute_step(
    step: PlanStep,
    step_results: list[dict],
    sql_llm_fn: Callable[[str], str],
    rag_index: HybridIndex | None = None,
    rag_reranker=None,
) -> StepExecutionResult:
    if step.tool == "sql":
        result = run_sql_node(step.query, sql_llm_fn)
        if not result.ok:
            return StepExecutionResult(tool="sql", output={}, error=result.error)
        return StepExecutionResult(tool="sql", output={"rows": result.rows, "sql": result.executed_sql}, error=None)

    if step.tool == "rag":
        if rag_index is None:
            return StepExecutionResult(tool="rag", output={}, error="No RAG index configured for this agent run.")
        try:
            chunks = rag_index.hybrid_search(step.query, top_k=5, reranker=rag_reranker)
        except Exception as e:  # noqa: BLE001 - e.g. embedding model failed to load
            return StepExecutionResult(tool="rag", output={}, error=f"Filing search is unavailable: {e}")
        return StepExecutionResult(
            tool="rag",
            output={"chunks": [
                {"chunk_id": i, "text": c.chunk.text, "section": c.chunk.section,
                 "doc_id": c.chunk.doc_id, "chunk_type": c.chunk.chunk_type}
                for i, c in enumerate(chunks)
            ]},
            error=None,
        )

    if step.tool == "calc":
        try:
            resolved_args = resolve_args(step.args, step_results)
        except ResolutionError as e:
            return StepExecutionResult(tool="calc", output={}, error=str(e))
        calc_result = run_calc_node(step.intent, resolved_args)
        if not calc_result.ok:
            return StepExecutionResult(tool="calc", output={}, error=calc_result.error)
        return StepExecutionResult(tool="calc", output={"value": calc_result.value, **calc_result.detail}, error=None)

    if step.tool == "custom_calc":
        try:
            resolved_args = resolve_args(step.args, step_results)
        except ResolutionError as e:
            return StepExecutionResult(tool="custom_calc", output={}, error=str(e))
        calc_result = run_custom_calc(step.expression, resolved_args)
        if not calc_result.ok:
            return StepExecutionResult(tool="custom_calc", output={}, error=calc_result.error)
        return StepExecutionResult(tool="custom_calc", output={"value": calc_result.value, **calc_result.detail}, error=None)

    return StepExecutionResult(tool=step.tool, output={}, error=f"Unknown tool: {step.tool}")

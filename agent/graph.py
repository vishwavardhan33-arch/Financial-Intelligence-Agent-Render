"""LangGraph agent graph (Step 8): plan-then-execute.

Graph shape:

    plan --(plan parsed OK)--> execute_step --(more steps, no halt)--> execute_step (loop)
      \\--(plan failed)--> END                        \\--(steps done, or a step failed)--> synthesize --> END

A step failure halts further execution (we don't try to run step 2 with
garbage inputs if step 1 failed) but still routes to synthesize, so the
final answer can honestly explain what could and couldn't be determined,
rather than the whole run just dying silently.
"""
from __future__ import annotations

from typing import Callable, TypedDict

from langgraph.graph import END, StateGraph

from agent.dispatch import execute_step
from agent.plan import Plan
from agent.planner import generate_plan
from agent.synthesizer import synthesize_answer
from retrieval.vector_store import HybridIndex


class AgentState(TypedDict, total=False):
    query: str
    plan: Plan | None
    step_index: int
    step_results: list[dict]
    halted: bool
    final_answer: str | None
    citations: list[int]
    error: str | None  # only set for a hard failure (e.g. unparseable plan)


def build_agent_graph(
    planner_llm_fn: Callable[[str], str],
    sql_llm_fn: Callable[[str], str],
    synthesizer_llm_fn: Callable[[str], str],
    rag_index: HybridIndex | None = None,
    rag_reranker=None,
):
    def plan_node(state: AgentState) -> dict:
        result = generate_plan(state["query"], planner_llm_fn)
        if not result.ok:
            return {"error": result.error, "plan": None}
        return {"plan": result.plan, "step_index": 0, "step_results": [], "halted": False}

    def execute_step_node(state: AgentState) -> dict:
        plan = state["plan"]
        idx = state["step_index"]
        step = plan.steps[idx]

        exec_result = execute_step(step, state["step_results"], sql_llm_fn, rag_index, rag_reranker)
        output = exec_result.output if exec_result.ok else {"step_error": exec_result.error}

        return {
            "step_results": state["step_results"] + [output],
            "step_index": idx + 1,
            "halted": not exec_result.ok,
        }

    def route_after_plan(state: AgentState) -> str:
        return "execute_step" if state.get("plan") is not None else END

    def route_after_step(state: AgentState) -> str:
        plan = state["plan"]
        if state.get("halted"):
            return "synthesize"
        if state["step_index"] < len(plan.steps):
            return "execute_step"
        return "synthesize"

    def synthesize_node(state: AgentState) -> dict:
        result = synthesize_answer(state["query"], state["step_results"], synthesizer_llm_fn)
        return {"final_answer": result.answer, "citations": result.citations, "error": result.error}

    graph = StateGraph(AgentState)
    graph.add_node("plan", plan_node)
    graph.add_node("execute_step", execute_step_node)
    graph.add_node("synthesize", synthesize_node)

    graph.set_entry_point("plan")
    graph.add_conditional_edges("plan", route_after_plan, {"execute_step": "execute_step", END: END})
    graph.add_conditional_edges("execute_step", route_after_step, {"execute_step": "execute_step", "synthesize": "synthesize"})
    graph.add_edge("synthesize", END)

    return graph.compile()


def run_agent(query: str, graph) -> AgentState:
    return graph.invoke({"query": query})

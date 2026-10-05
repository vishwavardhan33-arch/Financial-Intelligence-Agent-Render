"""Synthesizer (Step 9): combines all step outputs into a final answer.

Grounding strategy: structured citations. The LLM must output
{"answer": ..., "citations": [chunk_id, ...]}, and every citation is
checked against the chunk_ids that were actually retrieved in this run —
any citation pointing at a chunk_id that doesn't exist is stripped out
rather than trusted, since an invented citation is worse than none.

Note: the final answer text is naturally allowed to *state* numbers (e.g.
"gross margin was 50.4%") — the "never let the LLM re-type numbers" rule
governs step-to-step data flow (Step 7/8), not the prose of the final
answer, which is inherently text the LLM writes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

SYNTHESIZER_SYSTEM_PROMPT = """You are answering a financial question using the results of \
several tool calls (SQL query results, computed metrics, and/or retrieved filing excerpts).

Rules:
- Base your answer ONLY on the tool results provided below. Do not invent numbers or facts.
- If retrieved filing excerpts are provided, every factual claim drawn from them must be \
supported by a citation to that chunk's id.
- Output ONLY a JSON object: {"answer": "<your answer text>", "citations": [<chunk_id>, ...]}. \
No markdown fences, no extra text. If no filing excerpts were used, citations should be [].
"""


@dataclass
class SynthesisResult:
    answer: str | None
    citations: list[int]
    dropped_citations: list[int]  # citations the LLM gave that didn't match a real chunk_id
    error: str | None

    @property
    def ok(self) -> bool:
        return self.error is None


def build_synthesis_prompt(nl_query: str, step_results: list[dict]) -> str:
    parts = [SYNTHESIZER_SYSTEM_PROMPT, f"\nUser question: {nl_query}\n", "Tool results:"]
    for i, result in enumerate(step_results):
        if "rows" in result:
            parts.append(f"[Step {i} - SQL] {result['rows']}")
        elif "chunks" in result:
            for c in result["chunks"]:
                parts.append(f"[Step {i} - RAG chunk_id={c['chunk_id']}] {c['text']}")
        elif "value" in result:
            parts.append(f"[Step {i} - Calc] value={result['value']}, detail={ {k: v for k, v in result.items() if k != 'value'} }")
        elif "step_error" in result:
            parts.append(f"[Step {i} - FAILED] {result['step_error']}")
    parts.append("\nJSON:")
    return "\n".join(parts)


def synthesize_answer(
    nl_query: str, step_results: list[dict], llm_generate_fn: Callable[[str], str]
) -> SynthesisResult:
    prompt = build_synthesis_prompt(nl_query, step_results)
    raw = llm_generate_fn(prompt)
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        return SynthesisResult(answer=None, citations=[], dropped_citations=[], error=f"synthesizer output was not valid JSON: {e}")

    answer = parsed.get("answer")
    raw_citations = parsed.get("citations", [])
    if answer is None:
        return SynthesisResult(answer=None, citations=[], dropped_citations=[], error="synthesizer output missing 'answer'.")

    valid_chunk_ids = {
        c["chunk_id"]
        for result in step_results if "chunks" in result
        for c in result["chunks"]
    }
    valid_citations = [c for c in raw_citations if c in valid_chunk_ids]
    dropped = [c for c in raw_citations if c not in valid_chunk_ids]

    return SynthesisResult(answer=answer, citations=valid_citations, dropped_citations=dropped, error=None)

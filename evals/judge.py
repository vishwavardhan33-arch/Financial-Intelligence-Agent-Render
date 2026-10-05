"""LLM-as-judge harness (Step 10).

Like the other LLM-calling components, `judge_llm_fn` is injected so this
isn't tied to a specific provider. Each judge function returns a
structured, parsed result — never raw text — so scores can be aggregated
across a batch of eval cases.

Calibration note: an uncalibrated judge can drift or rate its own model's
outputs generously. Before trusting judge scores at scale, spot-check a
handful (10-15) against your own read of the same cases — see
evals/run_evals.py for where that would plug in.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from evals.judge_prompts import END_TO_END_JUDGE_PROMPT, RETRIEVAL_JUDGE_PROMPT, SQL_JUDGE_PROMPT


class JudgeParseError(Exception):
    pass


def _call_judge(prompt: str, judge_llm_fn: Callable[[str], str]) -> dict:
    raw = judge_llm_fn(prompt)
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise JudgeParseError(f"Judge output was not valid JSON: {e}. Raw: {raw!r}") from e


@dataclass
class RetrievalJudgment:
    score: int
    justification: str


def judge_retrieval(query: str, chunks: list[dict], judge_llm_fn: Callable[[str], str]) -> RetrievalJudgment:
    chunks_text = "\n".join(f"- [{c.get('chunk_id')}] {c.get('text', '')[:300]}" for c in chunks)
    prompt = RETRIEVAL_JUDGE_PROMPT.format(query=query, chunks=chunks_text or "(none retrieved)")
    parsed = _call_judge(prompt, judge_llm_fn)
    return RetrievalJudgment(score=int(parsed["score"]), justification=parsed.get("justification", ""))


@dataclass
class SQLJudgment:
    score: int
    justification: str


def judge_sql(query: str, sql: str, result_rows: list[dict], judge_llm_fn: Callable[[str], str]) -> SQLJudgment:
    prompt = SQL_JUDGE_PROMPT.format(query=query, sql=sql, result_preview=result_rows[:5])
    parsed = _call_judge(prompt, judge_llm_fn)
    return SQLJudgment(score=int(parsed["score"]), justification=parsed.get("justification", ""))


@dataclass
class EndToEndJudgment:
    correctness: int
    groundedness: int
    routing_appropriateness: int
    justification: str

    @property
    def average(self) -> float:
        return (self.correctness + self.groundedness + self.routing_appropriateness) / 3


def judge_end_to_end(
    query: str, answer: str, step_results: list[dict], judge_llm_fn: Callable[[str], str]
) -> EndToEndJudgment:
    prompt = END_TO_END_JUDGE_PROMPT.format(query=query, answer=answer, step_results=step_results)
    parsed = _call_judge(prompt, judge_llm_fn)
    return EndToEndJudgment(
        correctness=int(parsed["correctness"]),
        groundedness=int(parsed["groundedness"]),
        routing_appropriateness=int(parsed["routing_appropriateness"]),
        justification=parsed.get("justification", ""),
    )

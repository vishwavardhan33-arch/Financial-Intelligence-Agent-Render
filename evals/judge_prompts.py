"""Judge prompt templates (Step 10). Each judge outputs a structured
{"score": 1-5, "justification": "..."} so results can be aggregated into
a dashboard rather than eyeballed pass/fail.
"""
from __future__ import annotations

RETRIEVAL_JUDGE_PROMPT = """You are evaluating a RAG system's retrieval quality.

Question: {query}

Retrieved chunks:
{chunks}

Score (1-5) whether the retrieved chunks contain SUFFICIENT information to \
answer the question. 5 = everything needed is present, 1 = completely irrelevant.

Output ONLY JSON: {{"score": <1-5>, "justification": "<one sentence>"}}
"""

SQL_JUDGE_PROMPT = """You are evaluating whether a generated SQL query correctly \
answers a natural-language question about an invoice database.

Question: {query}
Generated SQL: {sql}
Execution result (first few rows): {result_preview}

Score (1-5) whether the SQL's LOGIC matches the question's intent (right tables, \
right filters, right aggregation) and the result looks plausible. \
5 = clearly correct, 1 = clearly wrong or nonsensical.

Output ONLY JSON: {{"score": <1-5>, "justification": "<one sentence>"}}
"""

END_TO_END_JUDGE_PROMPT = """You are evaluating a financial intelligence agent's final answer.

Question: {query}
Final answer: {answer}
Tool results used to produce it: {step_results}

Score THREE things, each 1-5:
- correctness: does the answer correctly address the question given the tool results?
- groundedness: is every claim in the answer traceable to the tool results \
(no invented numbers or facts)?
- routing_appropriateness: were sensible tools chosen for this question?

Output ONLY JSON: {{"correctness": <1-5>, "groundedness": <1-5>, \
"routing_appropriateness": <1-5>, "justification": "<one or two sentences>"}}
"""

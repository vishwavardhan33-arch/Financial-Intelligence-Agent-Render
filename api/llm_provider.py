"""LLM provider: any OpenAI-compatible chat-completions API serving an open model.

Default is Groq (free tier, no credit card) running Llama 3.3 70B. Because
Groq, OpenRouter, Together, Cerebras and others all speak the same
/chat/completions protocol, switching provider is three env vars:

    LLM_BASE_URL   e.g. https://api.groq.com/openai/v1
    LLM_API_KEY    key from the provider's console
    LLM_MODEL      e.g. llama-3.3-70b-versatile

Every node (planner, SQL, synthesizer, judge) takes a plain
(prompt: str) -> str callable, so this module only has to produce one.

Free tiers rate-limit aggressively, so 429 and 5xx responses are retried
with exponential backoff (honouring Retry-After when the provider sends it).
The HTTP client is created lazily, so importing this module never touches
the network or requires a key.
"""
from __future__ import annotations

import os
import time
from typing import Callable

import httpx

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.3-70b-versatile"

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    """Raised when the provider cannot produce a completion."""


def build_llm_fn(
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float = 60.0,
    max_retries: int = 3,
    temperature: float = 0.0,
) -> Callable[[str], str]:
    """Return a (prompt: str) -> str function backed by an OpenAI-compatible API.

    temperature defaults to 0 because the planner, SQL generator and judge
    all need deterministic, parseable output.
    """
    resolved_model = model or os.environ.get("LLM_MODEL", DEFAULT_MODEL)
    resolved_base = (base_url or os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
    resolved_key = api_key if api_key is not None else os.environ.get("LLM_API_KEY", "")
    state: dict = {"client": None}

    def _client() -> httpx.Client:
        if state["client"] is None:
            state["client"] = httpx.Client(base_url=resolved_base, timeout=timeout)
        return state["client"]

    def llm_fn(prompt: str) -> str:
        if not resolved_key:
            raise LLMError("LLM_API_KEY is not set. Create a free key and add it to the environment.")

        payload = {
            "model": resolved_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        headers = {"Authorization": f"Bearer {resolved_key}"}

        last_error = "unknown error"
        for attempt in range(max_retries + 1):
            try:
                response = _client().post("/chat/completions", json=payload, headers=headers)
            except httpx.HTTPError as e:
                last_error = f"network error: {e}"
            else:
                if response.status_code == 200:
                    try:
                        return response.json()["choices"][0]["message"]["content"] or ""
                    except (KeyError, IndexError, ValueError) as e:
                        raise LLMError(f"Unexpected response shape from provider: {e}") from e
                if response.status_code not in RETRYABLE_STATUS:
                    raise LLMError(f"Provider returned {response.status_code}: {response.text[:300]}")
                last_error = f"{response.status_code}: {response.text[:200]}"
                retry_after = response.headers.get("retry-after")
                if retry_after and retry_after.replace(".", "", 1).isdigit() and attempt < max_retries:
                    time.sleep(min(float(retry_after), 20.0))
                    continue

            if attempt < max_retries:
                time.sleep(min(2 ** attempt, 8))

        raise LLMError(f"Provider unavailable after {max_retries + 1} attempts ({last_error})")

    return llm_fn

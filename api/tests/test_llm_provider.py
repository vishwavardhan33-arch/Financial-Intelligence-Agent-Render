"""Tests for api/llm_provider.py against a fake local HTTP server that
mimics the OpenAI-compatible /chat/completions response shape, so the real
request/response/retry code is exercised without any network or API key."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from api.llm_provider import LLMError, build_llm_fn


def _make_server(handler_cls):
    server = HTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, f"http://127.0.0.1:{server.server_address[1]}"


class _EchoHandler(BaseHTTPRequestHandler):
    seen: list[dict] = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _EchoHandler.seen.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
        reply = {"choices": [{"message": {"role": "assistant", "content": f"echo: {body['messages'][0]['content']}"}}]}
        payload = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class _FlakyHandler(BaseHTTPRequestHandler):
    calls = 0

    def log_message(self, *args):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        _FlakyHandler.calls += 1
        if _FlakyHandler.calls == 1:
            self.send_response(429)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        payload = json.dumps({"choices": [{"message": {"content": "ok after retry"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class _BadRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        payload = b'{"error": "invalid api key"}'
        self.send_response(401)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def echo_server():
    _EchoHandler.seen = []
    server, thread, url = _make_server(_EchoHandler)
    yield url
    server.shutdown()
    thread.join()


def test_sends_real_prompt_with_auth_and_parses_response(echo_server):
    llm_fn = build_llm_fn(model="test-model", base_url=echo_server, api_key="secret")
    assert llm_fn("What is the gross margin?") == "echo: What is the gross margin?"

    request = _EchoHandler.seen[0]
    assert request["path"] == "/chat/completions"
    assert request["auth"] == "Bearer secret"
    assert request["body"]["model"] == "test-model"
    assert request["body"]["temperature"] == 0.0


def test_missing_api_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    llm_fn = build_llm_fn(base_url="http://127.0.0.1:1")
    with pytest.raises(LLMError, match="LLM_API_KEY"):
        llm_fn("hi")


def test_retries_on_rate_limit_then_succeeds(monkeypatch):
    monkeypatch.setattr("api.llm_provider.time.sleep", lambda s: None)
    _FlakyHandler.calls = 0
    server, thread, url = _make_server(_FlakyHandler)
    try:
        llm_fn = build_llm_fn(base_url=url, api_key="k", max_retries=2)
        assert llm_fn("hi") == "ok after retry"
        assert _FlakyHandler.calls == 2
    finally:
        server.shutdown()
        thread.join()


def test_non_retryable_error_fails_immediately():
    server, thread, url = _make_server(_BadRequestHandler)
    try:
        llm_fn = build_llm_fn(base_url=url, api_key="bad", max_retries=3)
        with pytest.raises(LLMError, match="401"):
            llm_fn("hi")
    finally:
        server.shutdown()
        thread.join()


def test_importing_and_building_makes_no_network_call():
    """Constructing the llm_fn must not connect anywhere; only a call does."""
    llm_fn = build_llm_fn(base_url="http://127.0.0.1:1", api_key="k")
    assert callable(llm_fn)

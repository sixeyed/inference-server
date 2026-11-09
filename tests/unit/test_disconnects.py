"""A client hanging up is not the model failing.

Two things made a client disconnect look like a server fault. The turn line
said "-> EMPTY", which sends you looking at the model when it was still
prefilling and the client had simply stopped waiting. And every idle keep-alive
socket a client reset printed a full socketserver traceback, which reads as the
engine crashing.
"""
import http.server
import io
import sys

import pytest

from inference import server, tracing


def _record(**overrides) -> dict:
    record = {
        "api": "openai", "path": "/v1/chat/completions",
        "timing": {"ttfb_ms": 408.8, "ttft_ms": None, "ttfc_ms": None,
                   "total_ms": 121502.1},
        "response": {"choices": [{"message": {"content": ""}}]},
        "summary": {"n_messages": 389, "n_tools": 17, "context_tokens": None,
                    "prefilled_tokens": None, "cache_n": None,
                    "completion_tokens": None, "called_tools": [],
                    "has_content": False, "has_reasoning": False},
    }
    record.update(overrides)
    return record


def _turn_line(record, capsys) -> str:
    tracing.TraceWriter("unused.jsonl", enabled=False).print_turn(194, record)
    return capsys.readouterr().out


def test_a_client_that_gave_up_before_the_first_token_is_not_empty(capsys):
    line = _turn_line(_record(client_disconnected=True), capsys)
    assert "CLIENT DISCONNECTED before first token" in line
    assert "EMPTY" not in line


def test_a_client_that_gave_up_mid_answer_says_so(capsys):
    record = _record(client_disconnected=True)
    record["timing"]["ttft_ms"] = 7000.0
    record["summary"]["called_tools"] = ["run_in_terminal"]
    line = _turn_line(record, capsys)
    assert "CLIENT DISCONNECTED" in line
    assert "before first token" not in line
    # The tool call never reached the client, so it must not read as delivered.
    assert "-> run_in_terminal" not in line


def test_a_genuinely_empty_answer_is_still_empty(capsys):
    assert "-> EMPTY" in _turn_line(_record(), capsys)


# ------------------------------------------------------------
# socketserver's default handle_error
# ------------------------------------------------------------
@pytest.fixture
def engine_server():
    srv = server.EngineServer(("127.0.0.1", 0), http.server.BaseHTTPRequestHandler,
                              bind_and_activate=False)
    yield srv
    srv.server_close()


def _handle(srv, exc, monkeypatch) -> str:
    stderr = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stderr)
    try:
        raise exc
    except Exception:
        srv.handle_error(None, ("192.168.1.120", 59262))
    return stderr.getvalue()


@pytest.mark.parametrize("exc", [
    ConnectionResetError(54, "Connection reset by peer"),
    BrokenPipeError(32, "Broken pipe"),
])
def test_a_peer_hanging_up_prints_no_traceback(engine_server, exc, monkeypatch):
    assert _handle(engine_server, exc, monkeypatch) == ""


def test_a_real_handler_bug_still_prints_its_traceback(engine_server, monkeypatch):
    out = _handle(engine_server, KeyError("model"), monkeypatch)
    assert "Traceback" in out and "KeyError" in out

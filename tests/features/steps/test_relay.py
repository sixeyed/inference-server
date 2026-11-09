"""Steps for relay.feature."""

import json
import socket
import struct

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

scenarios("relay.feature")


@pytest.fixture
def context():
    return {}


@given(parsers.parse('"{alias}" is loaded'), target_fixture="live")
def loaded(make_harness, alias):
    harness = make_harness()
    assert harness.chat(alias)[0] == 200
    return harness


@given(parsers.parse('"{alias}" is loaded with deprecated models served'),
       target_fixture="live")
def loaded_deprecated(make_harness, alias):
    harness = make_harness(engine_flags=["--serve-deprecated"])
    assert harness.chat(alias)[0] == 200
    return harness


@given(parsers.parse('"{alias}" is loaded and thinks before answering'),
       target_fixture="live")
def loaded_thinking(make_harness, alias):
    harness = make_harness(reasoning=["let ", "me ", "think "],
                           chunks=["the ", "answer"], chunk_delay=0.15)
    assert harness.chat(alias)[0] == 200
    return harness


@given(parsers.parse('"{alias}" is loaded and streams slowly'),
       target_fixture="live")
def loaded_slow(make_harness, alias):
    harness = make_harness(chunks=[f"word{i} " for i in range(20)],
                           chunk_delay=0.1)
    assert harness.chat(alias)[0] == 200
    return harness


@when("a client hangs up after the first chunk")
def hang_up(live, context):
    """A raw socket, because urllib cannot be made to abandon a stream halfway.

    SO_LINGER 0 closes with a RST rather than a FIN - what a client that has
    given up and torn its connection down actually sends - so the engine's next
    write fails promptly instead of filling a buffer nobody reads.
    """
    body = json.dumps({"model": "qwen3.8", "stream": True,
                       "messages": [{"role": "user", "content": "hi"}]}).encode()
    port = int(live.base.rsplit(":", 1)[1])
    sock = socket.create_connection(("127.0.0.1", port), timeout=10)
    sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\n"
                 b"Host: 127.0.0.1\r\nContent-Type: application/json\r\n"
                 + f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
    received = b""
    while b"data:" not in received:
        received += sock.recv(4096)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    sock.close()
    live.settle(2)


@when("a client sends a conversation containing a tool call with empty arguments")
def send_empty_args(live, context):
    payload = {
        "model": "qwen3.8",
        "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "list_dir", "arguments": ""}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        ],
    }
    context["status"], context["body"] = live.request(
        "/v1/chat/completions", payload)


@when(parsers.parse('a client sends a chat request for "{alias}"'))
def send_chat(live, context, alias):
    context["status"], context["body"] = live.chat(alias)


@when("a client sends a streaming chat request")
def send_stream(live, context):
    context["status"], context["body"] = live.chat("qwen3.8", stream=True)


def _last_trace(live) -> dict:
    return live.last_trace()


@then(parsers.parse('the upstream received "{value}" as the arguments'))
def upstream_args(live, value):
    sent = live.launcher.server.requests[-1]
    call = sent["messages"][1]["tool_calls"][0]["function"]
    assert call["arguments"] == value, call


@then("the trace records the repair")
def trace_repair(live):
    assert _last_trace(live)["repaired_tool_args"] == ["list_dir"]


@then("the trace keeps the untouched original")
def trace_original(live):
    original = _last_trace(live)["request_original"]
    call = original["messages"][1]["tool_calls"][0]["function"]
    assert call["arguments"] == "", call


@then(parsers.parse('the upstream received the model "{value}"'))
def upstream_model(live, value):
    assert live.launcher.server.requests[-1]["model"] == value


@then("the trace records the rewrite")
def trace_rewrite(live):
    record = _last_trace(live)
    assert record["rewrote_model"][0] == "gemma-4-31b"
    assert record["summary"]["model"] == "gemma-4-31b", "client's name must survive"


@then("the trace records no rewrite")
def trace_no_rewrite(live):
    assert "rewrote_model" not in _last_trace(live)


@then("the trace records that the client disconnected")
def trace_disconnect(live):
    record = _last_trace(live)
    assert record.get("client_disconnected") is True, record.keys()
    # Stopped at the hang-up rather than streaming all 20 chunks to nobody.
    assert record["timing"]["total_ms"] < 20 * 100, record["timing"]


@then("time-to-first-token is earlier than time-to-first-content")
def deferral_visible(live):
    timing = _last_trace(live)["timing"]
    assert timing["ttft_ms"] < timing["ttfc_ms"], timing


@then("the recorded deferral is greater than zero")
def deferral_recorded(live):
    row = live.rows("SELECT deferral_ms FROM requests ORDER BY id DESC LIMIT 1")[0]
    assert row["deferral_ms"] > 0, row

"""Steps for preemption.feature."""

import threading
import time
import urllib.request

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

scenarios("preemption.feature")


@pytest.fixture
def context():
    return {}


@given('"qwen3.8" is loaded and answering slowly', target_fixture="live")
def slow_model(make_harness):
    # Enough chunks, slowly enough, that a swap lands squarely mid-stream.
    harness = make_harness(chunks=[f"word{i} " for i in range(40)],
                           chunk_delay=0.25)
    assert harness.chat("qwen3.8")[0] == 200
    return harness


@when("a streaming request is in flight")
def start_stream(live, context):
    def run():
        started = time.perf_counter()
        request = urllib.request.Request(
            live.base + "/v1/chat/completions",
            data=b'{"model":"qwen3.8","messages":[{"role":"user","content":"hi"}],'
                 b'"stream":true}',
            headers={"Content-Type": "application/json"}, method="POST")
        body = b""
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                while True:
                    chunk = response.read1(4096)
                    if not chunk:
                        break
                    body += chunk
        except Exception as exc:            # the server going away is the point
            context["stream_error"] = exc
        context["stream_body"] = body
        context["stream_ms"] = (time.perf_counter() - started) * 1000

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    context["thread"] = thread
    # Let it actually get going, so the swap is genuinely mid-stream.
    time.sleep(1.0)


@when(parsers.parse('a client sends a chat request for "{alias}"'))
def swap_to(live, context, alias):
    started = time.perf_counter()
    context["status"], context["body"] = live.chat(alias)
    context["swap_ms"] = (time.perf_counter() - started) * 1000
    context["thread"].join(timeout=30)
    # Three turns: the one that loaded the model, the one that got cut, and the
    # one that caused the swap.
    live.settle(3)


@then("the in-flight response is cut short")
def cut_short(context):
    # 40 chunks at 0.25s is ten seconds of stream; it cannot have completed.
    assert b"[DONE]" not in context["stream_body"]
    assert context["stream_ms"] < 9_000, context["stream_ms"]


@then("the swap did not wait for it")
def did_not_wait(context):
    assert context["swap_ms"] < 9_000, context["swap_ms"]


@then(parsers.parse('the engine reports "{alias}" as loaded'))
def reports_loaded(live, alias):
    assert live.get("/health")["model"] == alias


@then("the engine is not in a failed state")
def not_failed(live):
    assert live.get("/health")["state"] == "ready"


@then("one request is recorded as preempted")
def one_preempted(live):
    rows = live.rows("SELECT alias, preempted FROM requests WHERE preempted = 1")
    assert len(rows) == 1, live.rows("SELECT alias, ok, preempted FROM requests")


@then("the preempted request is absent from the per-model statistics")
def absent_from_stats(live):
    rows = live.rows("SELECT id, alias, ok, preempted FROM requests ORDER BY id")
    counted = sum(row["requests"] for row in live.engine.usage.per_model())
    eligible = sum(1 for row in rows if row["ok"] and not row["preempted"])
    assert counted == eligible, (
        f"stats counted {counted} of {eligible} eligible rows: {rows}")
    assert eligible == len(rows) - 1, f"expected exactly one cut turn: {rows}"


@then("the preempted request still appears in the recent requests")
def present_in_recent(live):
    assert any(row["preempted"] for row in live.engine.usage.recent(20))

"""Steps for session_header.feature."""

import json
import threading
import time

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

scenarios("session_header.feature")

HEADER = "x-mtplx-session-id"
VSCODE = "GitHubCopilotChat/0.64.1"


@pytest.fixture
def context():
    return {}


@given(parsers.parse('"{alias}" is loaded'), target_fixture="live")
def loaded(make_harness, alias):
    harness = make_harness()
    assert harness.chat(alias)[0] == 200
    return harness


@given(parsers.parse('"{alias}" is loaded with the API key "{key}"'),
       target_fixture="live")
def loaded_with_key(make_harness, alias, key):
    harness = make_harness(api_key=key)
    assert harness.chat(alias)[0] == 200
    return harness


@given(parsers.parse('"{alias}" is loaded and streams slowly'),
       target_fixture="live")
def loaded_slow(make_harness, alias):
    harness = make_harness(chunks=[f"word{i} " for i in range(20)],
                           chunk_delay=0.1)
    assert harness.chat(alias)[0] == 200
    return harness


@given(parsers.parse("VS Code collapses loops after {count:d} identical calls"))
def collapse_after(live, count):
    from inference import clients
    live.engine.settings.set_loop_threshold(clients.identify({}, VSCODE), count)


# ------------------------------------------------------------
def _conversation(turns: int, opening="you are a test fixture") -> list:
    messages = [{"role": "system", "content": opening}]
    for i in range(turns):
        messages.append({"role": "user", "content": f"turn {i}"})
        messages.append({"role": "assistant", "content": f"answer {i}"})
    messages.append({"role": "user", "content": "next"})
    return messages


def _loop(calls: int) -> list:
    messages = [{"role": "system", "content": "you are a test fixture"},
                {"role": "user", "content": "check the build"}]
    for i in range(calls):
        messages.append({"role": "assistant", "content": "", "tool_calls": [{
            "id": f"call_{i}", "type": "function",
            "function": {"name": "run_in_terminal",
                         "arguments": json.dumps({"command": "az build show"})}}]})
        messages.append({"role": "tool", "tool_call_id": f"call_{i}",
                         "content": "ERR: 404 Not Found"})
    return messages


def _send(live, messages, headers=None, stream=False, model=None):
    model = model or live.engine.supervisor.status()["model"]
    return live.request("/v1/chat/completions",
                        {"model": model, "messages": messages, "stream": stream},
                        headers={"User-Agent": VSCODE, **(headers or {})})


def _upstream_headers(live, back: int) -> dict:
    """Headers of the request `back` places from the end, lower-cased -
    urllib capitalises names on the way out, MTPLX lower-cases them on the
    way in, so only the lower-case form means anything."""
    seen = live.launcher.server.headers_seen[-back]
    return {k.lower(): v for k, v in seen.items()}


def _last_trace(live) -> dict:
    return live.last_trace()


# ------------------------------------------------------------
@when("a client sends 2 turns of one conversation")
def two_turns(live):
    assert _send(live, _conversation(1))[0] == 200
    assert _send(live, _conversation(2))[0] == 200


@when("a client sends a long conversation and then starts a new one")
def long_then_new(live):
    assert _send(live, _conversation(5))[0] == 200
    assert _send(live, _conversation(0))[0] == 200


@when(parsers.parse('a client sends a turn with its own "{name}" of "{value}"'))
def own_header(live, name, value):
    assert _send(live, _conversation(1), headers={name: value})[0] == 200


@when(parsers.parse("VS Code sends a turn with {first:d} identical calls and "
                    "then one with {second:d}"))
def loop_turns(live, first, second):
    assert _send(live, _loop(first))[0] == 200
    assert _send(live, _loop(second))[0] == 200
    # The second really was collapsed - otherwise this proves nothing.
    assert _last_trace(live).get("collapsed_loops"), "loop was not collapsed"


@when("a second turn of a conversation arrives while its first is streaming")
def overlapping(live, context):
    server = live.launcher.server
    before = len(server.requests)
    first = threading.Thread(
        target=_send, args=(live, _conversation(1)), kwargs={"stream": True})
    first.start()
    deadline = time.monotonic() + 5
    while len(server.requests) == before and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(server.requests) > before, "first turn never reached upstream"
    assert _send(live, _conversation(2))[0] == 200
    first.join(10)
    # Upstream order: the streaming first turn, then the overlapping second.
    context["first"] = {k.lower(): v for k, v in server.headers_seen[before].items()}
    context["second"] = {k.lower(): v
                         for k, v in server.headers_seen[before + 1].items()}
    assert _send(live, _conversation(3))[0] == 200
    context["later"] = _upstream_headers(live, 1)


@when("a client sends a turn with that API key and an interaction id")
def key_and_interaction(live):
    assert _send(live, _conversation(1), headers={
        "Authorization": "Bearer sk-do-not-record-me",
        "X-Interaction-Id": "b1946ac9-2c2f-4e8e-9e5f-7c1e0c9f3a11"})[0] == 200


# ------------------------------------------------------------
@then("both turns reached the upstream with the same session id")
def same_id(live):
    first, second = _upstream_headers(live, 2), _upstream_headers(live, 1)
    assert first.get(HEADER), first
    assert first.get(HEADER) == second.get(HEADER), (first, second)


@then("the two reached the upstream with different session ids")
def different_ids(live):
    first, second = _upstream_headers(live, 2), _upstream_headers(live, 1)
    assert first.get(HEADER) and second.get(HEADER), (first, second)
    assert first[HEADER] != second[HEADER]


@then(parsers.parse('the upstream received "{name}" as "{value}"'))
def upstream_header(live, name, value):
    assert _upstream_headers(live, 1).get(name.lower()) == value


@then("the upstream received no engine session id")
def no_engine_id(live):
    assert HEADER not in _upstream_headers(live, 1)


@then("no request reached the upstream with an engine session id")
def never_an_engine_id(live):
    for seen in live.launcher.server.headers_seen:
        assert HEADER not in {k.lower() for k in seen}, seen


@then("the first turn carried a session id and the second did not")
def only_first(context):
    assert context["first"].get(HEADER), context["first"]
    # MTPLX would 409 the second: one name, two requests in flight.
    assert HEADER not in context["second"], context["second"]


@then("a later turn carries the session id again")
def released(context):
    assert context["later"].get(HEADER) == context["first"][HEADER]


@then("the trace lists the client's header names")
def header_names(live):
    names = _last_trace(live)["routing"]["client_headers"]
    assert "authorization" in names and "x-interaction-id" in names, names


@then("the trace does not contain the API key")
def no_key(live):
    # Wait for the record first: reading before it lands would pass without
    # having looked at anything.
    live.last_trace()
    with open(live.engine.tracer.path) as handle:
        assert "sk-do-not-record-me" not in handle.read()


@then("the trace records the session id sent upstream")
def session_recorded(live):
    routing = _last_trace(live)["routing"]
    assert routing["upstream_session"] == _upstream_headers(live, 1)[HEADER]

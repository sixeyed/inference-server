"""Steps for loops.feature."""

import json
import time

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from inference.settings import Settings

scenarios("loops.feature")

AGENTS = {"VS Code": "GitHubCopilotChat/0.32.0", "Cline": "Cline/3.2.0"}

COMMAND = "az pipelines runs show --id 24131618 --expand jobs"
RESULT = "ERR: 404 Not Found"


@pytest.fixture
def context():
    return {}


def _label(client):
    from inference import clients
    return clients.identify({}, AGENTS[client])


@given(parsers.parse('"{alias}" is loaded'), target_fixture="live")
def loaded(make_harness, alias):
    harness = make_harness(ui=True)
    assert harness.chat(alias)[0] == 200
    return harness


@given(parsers.parse("{client} collapses loops after {count:d} identical calls"))
@when(parsers.parse("{client} collapses loops after {count:d} identical calls"))
def configure(live, client, count):
    live.engine.settings.set_loop_threshold(_label(client), count)


def _history(n, moving=False):
    msgs = [{"role": "system", "content": "be helpful"},
            {"role": "user", "content": "check the build"}]
    for i in range(n):
        msgs.append({"role": "assistant", "content": "",
                     "tool_calls": [{"id": f"call_{i}", "type": "function",
                                     "function": {"name": "run_in_terminal",
                                                  "arguments": json.dumps(
                                                      {"command": COMMAND})}}]})
        msgs.append({"role": "tool", "tool_call_id": f"call_{i}",
                     "content": f"{i}% complete" if moving else RESULT})
    return msgs


@when(parsers.parse("{client} sends a history with {n:d} identical tool calls"))
def send_loop(live, context, client, n):
    context["sent"] = _history(n)
    context["status"], _ = live.request(
        "/v1/chat/completions",
        {"model": "qwen3.8", "messages": context["sent"]},
        headers={"User-Agent": AGENTS[client]})


@when(parsers.parse("{client} sends a history with {n:d} polls that each "
                    "report progress"))
def send_poll(live, context, client, n):
    context["sent"] = _history(n, moving=True)
    context["status"], _ = live.request(
        "/v1/chat/completions",
        {"model": "qwen3.8", "messages": context["sent"]},
        headers={"User-Agent": AGENTS[client]})


@when("the engine is restarted")
def restart(live, context):
    context["reloaded"] = Settings(live.engine.settings.path)


def _upstream(live):
    return live.launcher.server.requests[-1]["messages"]


@then("the upstream received the whole history")
def whole_history(live, context):
    assert _upstream(live) == context["sent"]


@then("the upstream history is shorter than the client sent")
def shorter(live, context):
    assert len(_upstream(live)) < len(context["sent"])


@then("the upstream no longer contains the repeated command")
def no_command(live):
    """The measured difference between breaking the loop and not: an example of
    the command in front of the model is an example it copies."""
    assert COMMAND not in json.dumps(_upstream(live))


@then("the upstream still contains what the command returned")
def keeps_result(live):
    assert RESULT in json.dumps(_upstream(live))


@then(parsers.parse("the trace records that {n:d} calls were collapsed"))
def trace_records(live, n):
    record = live.last_trace()
    runs = record.get("collapsed_loops")
    assert runs and runs[0]["repeats"] == n, runs
    assert runs[0]["tool"] == "run_in_terminal"


@then("the trace keeps the whole history the client sent")
def trace_keeps_original(live, context):
    record = live.last_trace()
    assert record["request_original"]["messages"] == context["sent"]
    assert len(record["request"]["messages"]) < len(context["sent"])


@then(parsers.parse("{client} still collapses loops after {count:d}"))
def still_configured(live, context, client, count):
    settings = context.get("reloaded") or live.engine.settings
    assert settings.loop_threshold_for(_label(client)) == count


@then(parsers.parse("{client} has loop collapsing off"))
def is_off(live, client):
    assert live.engine.settings.loop_threshold_for(_label(client)) == 0


@then(parsers.parse('the log has a "{kind}" entry'))
def log_has(live, kind):
    deadline = time.monotonic() + 5
    while True:
        rows = [r for r in live.get("/api/events")["events"] if r["kind"] == kind]
        if rows or time.monotonic() > deadline:
            assert rows, "no event recorded"
            return
        time.sleep(0.05)


@when(parsers.parse('the config page sets "{client}" to collapse after {n:d}'))
def api_set(live, client, n):
    status, _ = live.request("/api/settings",
                             {"loops": {"client": client, "count": n}})
    assert status == 200

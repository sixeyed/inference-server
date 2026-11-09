"""Steps for events.feature."""

import json
import os
import time

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

scenarios("events.feature")

AGENTS = {"VS Code": "GitHubCopilotChat/0.32.0"}

# Two of these are on vscode-core.txt, one is not. The set also names fifteen
# tools this client never sends at all - which is exactly the noise `drift` has
# to stay quiet about.
BASELINE = ["read_file", "grep_search", "run_notebook_cell"]


@pytest.fixture
def context():
    return {}


@given(parsers.parse('"{alias}" is loaded'), target_fixture="live")
def loaded(make_harness, alias):
    harness = make_harness(ui=True)
    assert harness.chat(alias)[0] == 200
    return harness


def _label(client):
    from inference import clients
    return clients.identify({}, AGENTS[client])


@given(parsers.parse('{client} is configured to use the "{file}" tool set'))
@when(parsers.parse('{client} is configured to use the "{file}" tool set'))
def configure(live, client, file):
    live.engine.settings.set_toolset(_label(client), file)


@when(parsers.parse('a client release renames "{name}"'))
def rename_tool(context, name):
    """What this actually looks like from the engine: the client stops sending
    one name and starts sending another. Nothing announces it."""
    offered = context.setdefault("offered", list(BASELINE))
    offered[offered.index(name)] = f"{name}_v2"


@when("the release is rolled back")
def rollback(context):
    context["offered"] = list(BASELINE)


@when(parsers.parse("{client} sends a request offering its tools"))
def send_tools(live, context, client):
    tools = [{"type": "function",
              "function": {"name": name, "description": f"does {name}",
                           "parameters": {"type": "object"}}}
             for name in context.setdefault("offered", list(BASELINE))]
    context["status"], context["body"] = live.request(
        "/v1/chat/completions",
        {"model": "qwen3.8", "tools": tools,
         "messages": [{"role": "user", "content": "hi"}]},
        headers={"User-Agent": AGENTS[client]})


def _events(live, kind=None, wants=0):
    """The row is written on the request path, so a read the instant the
    response returns can beat the INSERT. Poll rather than sleep - it is a
    race, not a delay."""
    deadline = time.monotonic() + 5
    while True:
        rows = live.get("/api/events")["events"]
        if kind:
            rows = [r for r in rows if r["kind"] == kind]
        if len(rows) >= wants or time.monotonic() > deadline:
            return rows
        time.sleep(0.05)


@then("the log is empty")
def log_empty(live):
    assert _events(live) == []


@then(parsers.parse('the log has one "{kind}" entry'))
def log_has_one(live, kind):
    assert len(_events(live, kind, wants=1)) == 1


@then(parsers.parse('the log has two "{kind}" entries'))
def log_has_two(live, kind):
    assert len(_events(live, kind, wants=2)) == 2


@then("the config page reports no drift")
def config_no_drift(live):
    assert live.get("/api/settings")["drift"] == {}


@then(parsers.parse('the log reports "{name}" as no longer offered'))
def log_reports_drift(live, name):
    rows = _events(live, "tools", wants=2)
    assert rows and name in rows[0]["detail"]["unknown"], rows
    assert name in rows[0]["text"]


@then(parsers.parse('the config page reports drift for "{client}" naming "{name}"'))
def config_reports_drift(live, client, name):
    _events(live, "tools", wants=2)
    drift = live.get("/api/settings")["drift"]
    assert drift.get(client, {}).get("unknown") == [name], drift

"""Steps for toolsets.feature."""

import json

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from inference.settings import Settings

scenarios("toolsets.feature")

# Same trick as prompts.feature: the engine picks the client off the
# User-Agent before it decides whether to filter, so the header is what makes
# these scenarios per-client.
AGENTS = {"VS Code": "GitHubCopilotChat/0.32.0",
          "Cline": "Cline/3.2.0"}

# A slice of the real VS Code array - two tools the traces show it calling
# constantly, and one it has never called once.
OFFERED = ["read_file", "grep_search", "run_notebook_cell"]


def _tools():
    return [{"type": "function",
             "function": {"name": name, "description": f"does {name}",
                          "parameters": {"type": "object"}}}
            for name in OFFERED]


def _label(client):
    from inference import clients
    return clients.identify({}, AGENTS[client])


@pytest.fixture
def context():
    return {}


@given(parsers.parse('"{alias}" is loaded'), target_fixture="live")
def loaded(make_harness, alias):
    # UI on: the last scenario drives the config page through /api/settings.
    harness = make_harness(ui=True)
    assert harness.chat(alias)[0] == 200
    return harness


@given(parsers.parse('a tool set "{file}" whose every name is a typo'))
def typo_set(file):
    """Written into the shipped directory and removed again: the engine under
    test reads the real one, and a permanent fixture there would show up in the
    dashboard's picker as if it were a set worth choosing."""
    import os
    from inference import toolsets
    path = os.path.join(toolsets.TOOLSET_DIR, file)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# matches nothing the client offers\nraed_file\ngrpe_search\n")
    yield
    os.unlink(path)


@given(parsers.parse('{client} is configured to use the "{file}" tool set'))
@when(parsers.parse('{client} is configured to use the "{file}" tool set'))
def configure(live, client, file):
    live.engine.settings.set_toolset(_label(client), file)


@given(parsers.parse('{client} is configured to replace "{block}" with "{file}"'))
def configure_prompt(live, client, block, file):
    live.engine.settings.set_prompt(_label(client), file, [block])


@when(parsers.parse('the prompt for "{label}" is cleared'))
def clear_prompt(live, label):
    live.engine.settings.set_prompt(label, None)


@when(parsers.parse("{client} sends a request offering its tools"))
def send_tools(live, context, client):
    context["status"], context["body"] = live.request(
        "/v1/chat/completions",
        {"model": "qwen3.8", "tools": _tools(),
         "messages": [{"role": "user", "content": "hi"}]},
        headers={"User-Agent": AGENTS[client]})


@when("the engine is restarted")
def restart(live, context):
    context["reloaded"] = Settings(live.engine.settings.path)


def _upstream_tools(live):
    sent = live.launcher.server.requests[-1]
    return [t["function"]["name"] for t in sent.get("tools") or []]


@then("the upstream received all of the tools")
def all_tools(live):
    assert _upstream_tools(live) == OFFERED


@then(parsers.parse('the upstream received the tool "{name}"'))
def received_tool(live, name):
    assert name in _upstream_tools(live)


@then(parsers.parse('the upstream did not receive the tool "{name}"'))
def did_not_receive_tool(live, name):
    assert name not in _upstream_tools(live)


def _last_record(live):
    return live.last_trace()


@then(parsers.parse('the trace records that "{name}" was dropped'))
def trace_records(live, name):
    filtered = _last_record(live).get("filtered_tools")
    assert filtered and filtered["dropped"] == [name], filtered
    assert filtered["file"] == "vscode-core.txt"


@then("the trace keeps all of the tools the client sent")
def trace_keeps_original(live):
    """The trace must never lie about what the client actually sent - the tool
    array is now one of the things the engine changes on the way past."""
    record = _last_record(live)
    original = [t["function"]["name"] for t in record["request_original"]["tools"]]
    assert original == OFFERED
    assert len(record["request"]["tools"]) < len(OFFERED)


@then(parsers.parse('{client}\'s tool set is still "{file}"'))
def still_configured(live, context, client, file):
    settings = context.get("reloaded") or live.engine.settings
    assert settings.toolset_for(_label(client)) == file


@then(parsers.parse("{client} has no configured tool set"))
def no_toolset(live, client):
    assert live.engine.settings.toolset_for(_label(client)) is None


@when(parsers.parse('the config page sets "{client}" to use "{file}"'))
def api_set(live, client, file):
    status, _ = live.request("/api/settings",
                             {"toolset": {"client": client, "file": file}})
    assert status == 200


@when(parsers.parse('the config page clears the tool set for "{client}"'))
def api_clear(live, client):
    status, _ = live.request("/api/settings",
                             {"toolset": {"client": client, "file": None}})
    assert status == 200

"""Steps for prompts.feature."""

import json
import time

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from inference.settings import Settings
from tests.unit.test_prompts import VSCODE

scenarios("prompts.feature")

# The engine identifies the client from the User-Agent before it decides
# whether to rewrite, so the header is what makes these scenarios per-client.
AGENTS = {"VS Code": "GitHubCopilotChat/0.32.0",
          "Cline": "Cline/3.2.0"}


@pytest.fixture
def context():
    return {}


@given(parsers.parse('"{alias}" is loaded'), target_fixture="live")
def loaded(make_harness, alias):
    # UI on: the last two scenarios drive the config page through /api/settings.
    harness = make_harness(ui=True)
    assert harness.chat(alias)[0] == 200
    return harness


@given(parsers.parse('{client} is configured to replace "{block}" with "{file}"'))
def configure(live, client, block, file):
    from inference import clients
    label = clients.identify({}, AGENTS[client])
    live.engine.settings.set_prompt(label, file, [block])


@when(parsers.parse("{client} sends the same system prompt"))
@when(parsers.parse("{client} sends its system prompt"))
def send_prompt(live, context, client):
    context["sent"] = VSCODE
    context["status"], context["body"] = live.request(
        "/v1/chat/completions",
        {"model": "qwen3.8",
         "messages": [{"role": "system", "content": VSCODE},
                      {"role": "user", "content": "hi"}]},
        headers={"User-Agent": AGENTS[client]})


@when("the engine is restarted")
def restart(live, context):
    context["reloaded"] = Settings(live.engine.settings.path)


def _upstream_system(live):
    sent = live.launcher.server.requests[-1]
    return sent["messages"][0]["content"]


@then("the upstream received the prompt unchanged")
def unchanged(live, context):
    assert _upstream_system(live) == context["sent"]


@then(parsers.parse('the upstream prompt no longer contains "{text}"'))
def no_longer_contains(live, text):
    assert text not in _upstream_system(live)


@then(parsers.parse('the upstream prompt still contains "{text}"'))
def still_contains(live, text):
    assert text in _upstream_system(live)


@then("the upstream prompt contains the replacement text")
def contains_replacement(live):
    """Whether the prompt gets shorter depends on the file - against the real
    15k VS Code prompt terse-coder.md takes ~2k off it, but the fixture here
    has a one-line <instructions> block, so length proves nothing. What the
    substitution has to prove is that the shipped text actually landed."""
    from inference import prompts
    section = prompts.load("terse-coder.md")["instructions"]
    assert section.splitlines()[0] in _upstream_system(live)


@then(parsers.parse('the trace records that "{block}" was rewritten'))
def trace_records(live, block):
    record = live.last_trace()
    rewrote = record.get("rewrote_prompt")
    assert rewrote and rewrote["blocks"] == [block], rewrote
    assert rewrote["file"] == "terse-coder.md"
    assert rewrote["chars_after"] != rewrote["chars_before"]


@then("the trace keeps the prompt the client sent")
def trace_keeps_original(live, context):
    record = live.last_trace()
    assert record["request_original"]["messages"][0]["content"] == context["sent"]
    assert record["request"]["messages"][0]["content"] != context["sent"]


@then(parsers.parse('{client} is still configured to replace "{block}"'))
def still_configured(context, client, block):
    from inference import clients
    label = clients.identify({}, AGENTS[client])
    file, replace = context["reloaded"].prompt_for(label)
    assert file and replace == [block]


@when(parsers.parse('the config page sets "{client}" to replace "{block}" '
                    'with "{file}"'))
def api_set(live, client, block, file):
    status, _ = live.request("/api/settings",
                             {"prompt": {"client": client, "file": file,
                                         "replace": [block]}})
    assert status == 200


@when(parsers.parse('the config page clears the prompt for "{client}"'))
def api_clear(live, client):
    status, _ = live.request("/api/settings",
                             {"prompt": {"client": client, "file": None}})
    assert status == 200


def _settings_api(live, wants=None):
    """The usage row is written after the client has its response, so a check
    that runs the instant the request returns can beat the INSERT. Poll rather
    than sleep a fixed amount - it is a race, not a delay."""
    deadline = time.monotonic() + 5
    while True:
        data = live.get("/api/settings")
        if wants is None or wants in data["clients"] or time.monotonic() > deadline:
            return data
        time.sleep(0.05)


@then(parsers.parse('the config page offers "{client}"'))
def api_offers_client(live, client):
    assert client in _settings_api(live, wants=client)["clients"]


@then(parsers.parse('the config page lists "{file}" with its blocks'))
def api_lists_file(live, file):
    entry = next(f for f in _settings_api(live)["available_prompts"]
                 if f["file"] == file)
    assert "preamble" in entry["blocks"]


@then("the config page flags the contract blocks")
def api_flags_contract(live):
    """The UI marks these because replacing one is how you break tool calling."""
    assert "toolUseInstructions" in _settings_api(live)["contract_blocks"]


@then(parsers.parse("{client}'s configured prompt file is \"{file}\""))
def api_configured(live, client, file):
    assert _settings_api(live)["prompts"]["vscode"]["file"] == file


@then(parsers.parse("{client} has no configured prompt"))
def api_cleared(live, client):
    assert "vscode" not in _settings_api(live)["prompts"]

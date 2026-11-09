"""Steps for settings.feature."""

import json

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from inference.settings import Settings

scenarios("settings.feature")


@pytest.fixture
def context():
    return {}


@given(parsers.parse('"{alias}" is loaded'), target_fixture="live")
def loaded(make_harness, alias):
    harness = make_harness()
    assert harness.chat(alias)[0] == 200
    return harness


@given(parsers.parse('"{key}" is stripped'))
@when(parsers.parse('"{key}" is stripped'))
def strip_key(live, key):
    live.engine.settings.set_stripped([key])


@when(parsers.parse("a client sends a chat request with temperature {value:f}"))
def send_with_temperature(live, context, value):
    context["status"], context["body"] = live.request(
        "/v1/chat/completions",
        {"model": "qwen3.8", "temperature": value,
         "messages": [{"role": "user", "content": "hi"}]})


@when("the engine is restarted")
def restart(live, context):
    # A new Settings over the same file is what a restart amounts to.
    context["reloaded"] = Settings(live.engine.settings.path)


@then(parsers.parse("the upstream received a temperature of {value:f}"))
def upstream_temperature(live, value):
    assert live.launcher.server.requests[-1].get("temperature") == value


@then("the upstream received no temperature")
def upstream_no_temperature(live):
    sent = live.launcher.server.requests[-1]
    assert "temperature" not in sent, sent


@then(parsers.parse('the trace records that "{key}" was stripped'))
def trace_records_strip(live, key):
    record = live.last_trace()
    assert record.get("stripped_params") == [key], record.get("stripped_params")


@then("the trace keeps the temperature the client sent")
def trace_keeps_original(live):
    record = live.last_trace()
    assert record["request_original"]["temperature"] == 0.1
    assert "temperature" not in record["request"], "the forwarded body kept it"


@then(parsers.parse('"{key}" is still stripped'))
def still_stripped(context, key):
    assert key in context["reloaded"].stripped


@then(parsers.parse('"{key}" is not in the strip list'))
def not_strippable(live, key):
    assert key not in live.engine.settings.stripped

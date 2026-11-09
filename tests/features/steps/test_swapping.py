"""Steps for swapping.feature."""

import json

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

scenarios("swapping.feature")


@pytest.fixture
def context():
    return {}


@given("nothing is loaded", target_fixture="live")
def nothing_loaded(harness):
    assert harness.get("/health")["state"] == "idle"
    return harness


@given(parsers.parse('"{alias}" is loaded'), target_fixture="live")
def already_loaded(harness, alias):
    status, _ = harness.chat(alias)
    assert status == 200
    return harness


@given(parsers.parse('the model server will fail to start with "{message}"'),
       target_fixture="live")
def failing_server(make_harness, message):
    return make_harness(fail=message)


@when(parsers.parse('a client sends a chat request for "{alias}"'))
def send_chat(live, context, alias):
    context["status"], context["body"] = live.chat(alias)


@then(parsers.parse("the response is {code:d}"))
def response_code(context, code):
    assert context["status"] == code, context["body"][:400]


@then(parsers.parse('"{alias}" was started'))
def was_started(live, alias):
    assert alias in live.launcher.started


@then(parsers.parse('the engine reports "{alias}" as loaded'))
def reports_loaded(live, alias):
    assert live.get("/health")["model"] == alias


@then(parsers.parse("the model was started {count:d} time"))
def started_count(live, count):
    assert len(live.launcher.started) == count, live.launcher.started


@then("the previous server was stopped")
def previous_stopped(live):
    assert live.launcher.stopped >= 1


@then(parsers.parse('the error message contains "{fragment}"'))
def error_contains(context, fragment):
    message = json.loads(context["body"])["error"]["message"]
    assert fragment in message, message


@then("the error message names the log file")
def error_names_log(context):
    message = json.loads(context["body"])["error"]["message"]
    assert "full log:" in message and ".log" in message, message

"""Steps for api_key.feature."""

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

scenarios("api_key.feature")


@pytest.fixture
def context():
    return {}


@given("a model server that checks the API key", target_fixture="live")
def checking_server(make_harness):
    # The mtplx spec is the one that 401s an unauthenticated request, which is
    # what makes any of this observable.
    harness = make_harness()
    assert harness.chat("qwen3.8")[0] == 200
    return harness


@when(parsers.parse('a client sends a chat request with the header "{value}"'))
def with_header(live, context, value):
    context["status"], context["body"] = live.chat(
        "qwen3.8", headers={"Authorization": value})


@when("a client sends a chat request with no authorization header")
def without_header(live, context):
    context["status"], context["body"] = live.chat("qwen3.8")


@when("the model is loaded from cold")
def load_cold(live, context):
    context["status"], context["body"] = live.chat("qwen3.8")


@then(parsers.parse("the response is {code:d}"))
def response_code(context, code):
    assert context["status"] == code, context["body"][:300]


@then(parsers.parse('the upstream saw the header "{value}"'))
def upstream_saw(live, value):
    seen = live.launcher.server.headers_seen[-1]
    assert seen.get("Authorization") == value, seen.get("Authorization")


@then("the engine reported it ready")
def reported_ready(live):
    assert live.get("/health")["state"] == "ready"

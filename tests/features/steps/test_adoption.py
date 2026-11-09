"""Steps for adoption.feature."""

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from inference import models
from tests.fakes import FakeModelServer

scenarios("adoption.feature")


@pytest.fixture
def context():
    return {}


@given(parsers.parse('a model server is already serving "{model_id}"'),
       target_fixture="live")
def already_serving(make_harness, model_id):
    harness = make_harness()
    # Occupy the upstream port before the engine looks at it, which is what
    # `inference run <alias>` in another terminal amounts to.
    server = FakeModelServer(model_id=model_id, engine="llama-server")
    server.start(harness.engine.upstream_port)
    harness.preexisting = server
    return harness


@given("the engine has adopted it")
def engine_adopted(live):
    live.engine.supervisor.adopt(live.engine.index)


@when("the engine starts")
def engine_starts(live):
    live.engine.supervisor.adopt(live.engine.index)


@when(parsers.parse('a client sends a chat request for "{alias}"'))
def send_chat(live, context, alias):
    context["status"], context["body"] = live.chat(alias)


@when("the engine shuts down")
def engine_shuts_down(live):
    live.engine.supervisor.shutdown()


@then(parsers.parse("the response is {code:d}"))
def response_code(context, code):
    assert context["status"] == code, context["body"][:300]


@then(parsers.parse('the engine reports "{alias}" as loaded'))
def reports_loaded(live, alias):
    assert live.get("/health")["model"] == alias


@then("it is marked as adopted")
def marked_adopted(live):
    assert live.get("/status")["adopted"] is True


@then("nothing was started")
def nothing_started(live):
    assert live.launcher.started == []


@then("the model server is still running")
def still_running(live):
    assert live.preexisting._server is not None
    assert live.launcher.stopped == 0

"""Steps for resolution.feature."""

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from inference import models

scenarios("resolution.feature")


@pytest.fixture
def context():
    return {}


@given("the deprecated models are included", target_fixture="index")
def index_with_deprecated(infos):
    return models.build_index(infos, deprecated=True)


@pytest.fixture
def index(infos):
    return models.build_index(infos)


@given("the engine is running", target_fixture="running")
def running(harness):
    return harness


@when(parsers.parse('a client asks for "{sent}"'))
def ask(context, index, sent):
    context["resolved"] = models.resolve(sent, index)


@when("a client asks for \"\"")
def ask_empty(context, index):
    context["resolved"] = models.resolve("", index)


@then(parsers.parse('it resolves to "{alias}"'))
def resolves_to(context, alias):
    assert context["resolved"] == alias


@then("it does not resolve")
def does_not_resolve(context):
    assert context["resolved"] is None


@then(parsers.parse('the model list contains "{alias}"'))
def list_contains(running, alias):
    ids = [m["id"] for m in running.get("/v1/models")["data"]]
    assert alias in ids


@then(parsers.parse('the model list does not contain "{alias}"'))
def list_excludes(running, alias):
    ids = [m["id"] for m in running.get("/v1/models")["data"]]
    assert alias not in ids


@then("no model is loaded")
def nothing_loaded(running):
    assert running.get("/health")["state"] == "idle"

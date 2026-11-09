"""Steps for sandbox.feature."""

import json
import sys

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from inference import models, process

scenarios("sandbox.feature")

# sandbox-exec is macOS. The image's test stage has none - and the engine in
# the image never launches a model itself; the Mac it drives does.
pytestmark = pytest.mark.skipif(sys.platform != "darwin",
                                reason="sandbox-exec is macOS only")


@pytest.fixture
def context():
    return {}


def _on_disk(monkeypatch, missing: set[str], pulled: list[str]) -> None:
    # The real check reads the caches; a scenario must not depend on what this
    # machine happens to have downloaded. A missing repo is on disk once the
    # (fake) puller has fetched it.
    monkeypatch.setattr(process, "weights_are_local",
                        lambda spec: spec.hf_repo not in missing
                        or spec.hf_repo in pulled)


def _load(make_harness, monkeypatch, context, alias, flags=()):
    harness = make_harness(sandbox=True, engine_flags=list(flags))
    context["missing"] = set()
    _on_disk(monkeypatch, context["missing"], harness.launcher.pulled)
    assert harness.chat(alias)[0] == 200
    return harness


@given(parsers.parse('"{alias}" is loaded'), target_fixture="live")
def loaded(make_harness, monkeypatch, context, alias):
    return _load(make_harness, monkeypatch, context, alias)


@given(parsers.parse('"{alias}" is loaded with pulling off'), target_fixture="live")
def loaded_no_pull(make_harness, monkeypatch, context, alias):
    return _load(make_harness, monkeypatch, context, alias, ["--no-pull"])


@given("the sandbox is disabled", target_fixture="live")
def sandbox_off(make_harness, monkeypatch, context):
    harness = make_harness(engine_flags=["--no-sandbox"])
    context["missing"] = set()
    _on_disk(monkeypatch, context["missing"], harness.launcher.pulled)
    return harness


def _repo(alias: str) -> str:
    return models.load_all()[alias].HF_REPO


@given(parsers.parse('"{alias}" has no weights on disk'))
def no_weights(live, context, alias):
    context["missing"].add(_repo(alias))


@given(parsers.parse('pulling fails with "{message}"'))
def pull_fails(live, message):
    live.launcher.pull_error = message


@when(parsers.parse('a client sends a chat request for "{alias}"'))
def send(live, context, alias):
    context["before"] = live.launcher.stopped
    context["status"], context["body"] = live.chat(alias)


@then(parsers.parse("the response is {code:d}"))
def response_code(context, code):
    assert context["status"] == code, context["body"][:300]


@then("the error explains that the sandbox blocks the download")
def explains(context):
    message = json.loads(context["body"])["error"]["message"]
    assert "sandbox" in message and "not on disk" in message, message


@then("the error says how to fetch the weights")
def says_how(context):
    message = json.loads(context["body"])["error"]["message"]
    assert "inference pull" in message, message


@then(parsers.parse('"{alias}" is still serving'))
def still_serving(live, alias):
    status = live.get("/health")
    assert status["state"] == "ready", status
    assert status["model"] == alias, status


@then("nothing was stopped")
def nothing_stopped(live, context):
    assert live.launcher.stopped == context["before"], (
        "the running model was killed for a launch that could not have worked")


@then(parsers.parse('a failed load is recorded for "{alias}"'))
def failure_recorded(live, alias):
    rows = live.rows("SELECT alias, ok, error, log_path FROM loads WHERE ok = 0")
    assert len(rows) == 1, live.rows("SELECT alias, ok, error FROM loads")
    assert rows[0]["alias"] == alias
    assert "sandbox" in (rows[0]["error"] or "")


@then(parsers.parse('the weights for "{alias}" were pulled'))
def was_pulled(live, alias):
    assert live.launcher.pulled == [_repo(alias)]


@then("nothing was pulled")
def nothing_pulled(live):
    assert live.launcher.pulled == []


@then("the error says the weights could not be fetched")
def fetch_failed(context):
    message = json.loads(context["body"])["error"]["message"]
    assert "could not fetch the weights" in message, message

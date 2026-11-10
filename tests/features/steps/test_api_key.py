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


# ------------------------------------------------------------
# A real key: the engine enforces it
# ------------------------------------------------------------
@given(parsers.parse('an engine with the API key "{key}"'), target_fixture="live")
def engine_with_key(make_harness, key):
    return make_harness(api_key=key)


@given(parsers.parse('an engine with the API key "{key}" and its dashboard'),
       target_fixture="live")
def engine_with_key_and_ui(make_harness, key):
    harness = make_harness(api_key=key, ui=True)
    harness.key = key
    return harness


@given("an engine with its dashboard", target_fixture="live")
def engine_with_ui(make_harness):
    return make_harness(ui=True)


@when(parsers.parse('a client sends a chat request with x-api-key "{key}"'))
def with_x_api_key(live, context, key):
    context["status"], context["body"] = live.chat(
        "qwen3.8", headers={"x-api-key": key})


@then("no model was started")
def nothing_started(live):
    assert live.launcher.started == []


def _call(live, method, path, headers=None):
    import urllib.error
    import urllib.request
    request = urllib.request.Request(
        live.base + path, method=method, headers=headers or {},
        data=b"{}" if method == "POST" else None)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code


@then(parsers.parse('GET "{path}" without the key is {code:d}'))
def get_without_key(live, path, code):
    assert _call(live, "GET", path) == code


@then(parsers.parse('GET "{path}" with the key is {code:d}'))
def get_with_key(live, path, code):
    assert _call(live, "GET", path,
                 {"Authorization": f"Bearer {live.key}"}) == code


@then(parsers.parse('POST "{path}" without the key is {code:d}'))
def post_without_key(live, path, code):
    assert _call(live, "POST", path) == code


@then(parsers.parse('POST "{path}" from origin "{origin}" is {code:d}'))
def post_from_origin(live, path, origin, code):
    assert _call(live, "POST", path, {"Origin": origin}) == code


@then(parsers.parse('POST "{path}" from its own origin is {code:d}'))
def post_same_origin(live, path, code):
    assert _call(live, "POST", path, {"Origin": live.base}) == code

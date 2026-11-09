"""Identifying the calling client.

Nothing in an OpenAI request says who sent it, and the failure modes here are
client-shaped - so a session labelled with the wrong client is worse than one
labelled unknown.
"""

from inference.clients import identify


def tools(*names):
    return {"tools": [{"function": {"name": n}} for n in names]}


def test_the_copilot_marker_identifies_vscode():
    # Present in every VS Code request captured here, across tool counts from
    # 51 to 80 - matching the whole set would break on the next tool they add.
    assert identify(tools("read_file", "copilot_getNotebookSummary")) == "vscode"


def test_one_marker_is_enough():
    assert identify(tools("configure_notebook")) == "vscode"
    assert identify(tools("attempt_completion")) == "cline"


def test_the_user_agent_wins_over_the_tool_fingerprint():
    # A header the client set itself beats an inference from its tools.
    assert identify(tools("attempt_completion"), "VSCode/1.99") == "vscode"


def test_a_provider_prefix_identifies_opencode():
    assert identify({"model": "local/qwen3.8"}) == "opencode"
    assert identify({"model": "qwen3.8"}) != "opencode"


def test_no_tools_is_a_bare_api_call_not_an_unknown_client():
    assert identify({"messages": [{"role": "user", "content": "hi"}]}) == "api"


def test_an_unrecognised_tool_set_is_not_guessed_at():
    assert identify(tools("some_tool", "another_tool")) == "unknown"


def test_a_missing_request_does_not_raise():
    assert identify(None) == "api"
    assert identify({}, None) == "api"

"""Repairing empty tool-call arguments."""

from inference import tracing


def conversation(arguments):
    return {"messages": [
        {"role": "user", "content": "go"},
        {"role": "assistant", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "list_dir", "arguments": arguments}}]},
    ]}


def test_empty_arguments_become_an_empty_object():
    # llama.cpp refuses the whole conversation on `"arguments": ""` and 500s
    # every subsequent request, which ends the session.
    request = conversation("")
    assert tracing.repair_tool_args(request) == ["list_dir"]
    call = request["messages"][1]["tool_calls"][0]["function"]
    assert call["arguments"] == "{}"


def test_whitespace_counts_as_empty():
    request = conversation("   ")
    assert tracing.repair_tool_args(request) == ["list_dir"]


def test_real_arguments_are_left_alone():
    request = conversation('{"path":"src"}')
    assert tracing.repair_tool_args(request) == []
    call = request["messages"][1]["tool_calls"][0]["function"]
    assert call["arguments"] == '{"path":"src"}'


def test_a_conversation_with_no_tool_calls_is_untouched():
    assert tracing.repair_tool_args({"messages": [{"role": "user", "content": "hi"}]}) == []
    assert tracing.repair_tool_args({}) == []

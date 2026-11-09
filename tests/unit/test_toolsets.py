"""Filtering the tool list a client offers.

The numbers behind this: across 15 captured trace files VS Code offered 103
distinct tools and the model called 18 of them. The 85 it never called are ~76%
of a ~23,000-token schema resent on every single turn, so the filter is worth
more than the prompt rewrite next to it.

The rule that matters is the one in `apply`: an allowlist that matches nothing
is a typo, not an instruction. Handing a coding agent zero tools turns agent
mode into chat without erroring, and that is a far worse failure than not
filtering at all.
"""
import json
import os
import textwrap

import pytest

from inference import toolsets


def _tools(*names):
    """The shape a client sends - only `function.name` matters here."""
    return [{"type": "function",
             "function": {"name": n, "description": f"does {n}",
                          "parameters": {"type": "object"}}}
            for n in names]


def _names(tools):
    return [t["function"]["name"] for t in tools]


# ------------------------------------------------------------------ parsing
def test_parse_reads_one_name_per_line():
    assert toolsets.parse("read_file\ngrep_search\n") == ["read_file",
                                                          "grep_search"]


def test_parse_ignores_comments_and_blank_lines():
    text = textwrap.dedent("""\
        # what VS Code actually calls
        read_file

        grep_search   # trailing note
        """)
    assert toolsets.parse(text) == ["read_file", "grep_search"]


def test_parse_keeps_the_file_order_and_drops_repeats():
    assert toolsets.parse("b\na\nb\n") == ["b", "a"]


def test_parse_of_nothing_is_empty():
    assert toolsets.parse("# just a comment\n\n") == []


# ------------------------------------------------------------------ loading
def test_load_reads_a_shipped_file():
    assert "read_file" in toolsets.load("vscode-core.txt")


def test_load_refuses_a_path(tmp_path):
    """Reachable from an unauthenticated dashboard, so a separator in the name
    is either a caller bug or someone trying it on. Both want to be loud."""
    for bad in ("../secrets.txt", "sub/dir.txt", "", ".", ".."):
        with pytest.raises(ValueError):
            toolsets.load(bad)


def test_load_of_a_missing_file_is_empty():
    assert toolsets.load("nope.txt") == []


def test_available_lists_the_shipped_files():
    listed = {entry["file"]: entry for entry in toolsets.available()}
    assert "vscode-core.txt" in listed
    core = listed["vscode-core.txt"]
    assert core["count"] == len(toolsets.load("vscode-core.txt"))
    assert core["summary"]


def test_available_skips_a_file_with_no_names(tmp_path, monkeypatch):
    monkeypatch.setattr(toolsets, "TOOLSET_DIR", str(tmp_path))
    (tmp_path / "notes.txt").write_text("# documentation, not a tool set\n")
    assert toolsets.available() == []


# ----------------------------------------------------------------- applying
def test_apply_keeps_only_the_allowed_tools():
    kept, dropped, unknown = toolsets.apply(
        _tools("read_file", "run_notebook_cell", "grep_search"),
        ["read_file", "grep_search"])
    assert _names(kept) == ["read_file", "grep_search"]
    assert dropped == ["run_notebook_cell"]
    assert unknown == []


def test_apply_keeps_the_order_the_client_sent():
    """Not the allowlist's order - the client's. Reordering a tool array is a
    change to the request that buys nothing."""
    kept, _, _ = toolsets.apply(_tools("c", "a", "b"), ["a", "b", "c"])
    assert _names(kept) == ["c", "a", "b"]


def test_apply_reports_names_the_client_did_not_offer():
    """How a stale list shows up: VS Code renamed or removed the tool."""
    _, _, unknown = toolsets.apply(_tools("read_file"),
                                   ["read_file", "gone_tool"])
    assert unknown == ["gone_tool"]


def test_apply_without_an_allowlist_is_a_no_op():
    tools = _tools("read_file", "run_notebook_cell")
    kept, dropped, unknown = toolsets.apply(tools, [])
    assert kept is tools and dropped == [] and unknown == []


def test_apply_to_a_request_with_no_tools_is_a_no_op():
    kept, dropped, unknown = toolsets.apply([], ["read_file"])
    assert kept == [] and dropped == [] and unknown == []


def test_apply_that_would_match_nothing_passes_everything_through():
    """The guard. A typo'd file would otherwise hand a coding agent zero tools
    and turn agent mode into chat, with nothing in the response to say so."""
    tools = _tools("read_file", "grep_search")
    kept, dropped, unknown = toolsets.apply(tools, ["raed_file", "grpe_search"])
    assert kept is tools
    assert dropped == []
    assert unknown == ["raed_file", "grpe_search"]


def test_apply_does_not_mutate_the_list_it_was_given():
    tools = _tools("read_file", "run_notebook_cell")
    toolsets.apply(tools, ["read_file"])
    assert _names(tools) == ["read_file", "run_notebook_cell"]


def test_apply_survives_a_malformed_tool_entry():
    """A tool with no function block cannot be matched, so it goes. Raising
    here would 500 a whole session over one odd entry."""
    kept, dropped, _ = toolsets.apply(
        [{"type": "function"}] + _tools("read_file"), ["read_file"])
    assert _names(kept) == ["read_file"]
    assert dropped == []


# ------------------------------------------------------- the shipped set
def test_the_shipped_vscode_set_is_what_the_traces_showed():
    """core+github: every tool the model actually called across the captured
    sessions, minus the two single-call VS Code ones, plus the companions a
    client falls back to (get_terminal_output, insert_edit_into_file,
    resolve_memory_file_uri) which the traces would not show until needed."""
    core = toolsets.load("vscode-core.txt")
    assert set(core) == {
        "read_file", "grep_search", "file_search", "list_dir",
        "run_in_terminal", "get_terminal_output", "replace_string_in_file",
        "insert_edit_into_file", "create_file", "get_errors",
        "manage_todo_list", "memory", "resolve_memory_file_uri",
        "fetch_webpage", "task_complete", "github_repo", "github_text_search",
    }


# --------------------------------------------------------------- persistence
def _settings(tmp_path):
    from inference.settings import Settings
    return Settings(str(tmp_path / "settings.json"))


def test_no_toolset_is_configured_by_default(tmp_path):
    assert _settings(tmp_path).toolset_for("vscode") is None


def test_a_toolset_is_configured_per_client(tmp_path):
    s = _settings(tmp_path)
    s.set_toolset("vscode", "vscode-core.txt")
    assert s.toolset_for("vscode") == "vscode-core.txt"
    assert s.toolset_for("cline") is None


def test_a_toolset_survives_a_restart(tmp_path):
    """A setting you had to remember to pass on the command line is one you
    will not have set when it matters - same reason the strip list persists."""
    _settings(tmp_path).set_toolset("vscode", "vscode-core.txt")
    assert _settings(tmp_path).toolset_for("vscode") == "vscode-core.txt"


def test_a_toolset_is_cleared_with_none(tmp_path):
    s = _settings(tmp_path)
    s.set_toolset("vscode", "vscode-core.txt")
    s.set_toolset("vscode", None)
    assert s.toolset_for("vscode") is None
    assert _settings(tmp_path).toolset_for("vscode") is None


def test_a_toolset_setting_stores_a_bare_name(tmp_path):
    """set_prompt takes the basename rather than trusting the caller; a hand
    edited settings.json still has to fail loudly at load, not silently read
    some other file."""
    s = _settings(tmp_path)
    s.set_toolset("vscode", "/etc/passwd")
    assert s.toolset_for("vscode") == "passwd"


def test_prompts_and_toolsets_are_independent(tmp_path):
    """They compose: any prompt with any tool set. Clearing one must not take
    the other with it."""
    s = _settings(tmp_path)
    s.set_prompt("vscode", "old-timer.md", ["preamble"])
    s.set_toolset("vscode", "vscode-core.txt")
    s.set_prompt("vscode", None)
    assert s.toolset_for("vscode") == "vscode-core.txt"
    assert s.prompt_for("vscode") == (None, [])


# --------------------------------------------------- through prepare_upstream
def _current():
    """The bit of the model record prepare_upstream reads."""
    import types
    return types.SimpleNamespace(upstream_model_id="qwen3.8")


def _request(*names):
    return {"model": "qwen3.8",
            "messages": [{"role": "user", "content": "go"}],
            "tools": _tools(*names)}


def test_prepare_filters_the_tools_for_a_configured_client(tmp_path):
    from inference import server
    s = _settings(tmp_path)
    s.set_toolset("vscode", "vscode-core.txt")
    request = _request("read_file", "run_notebook_cell", "grep_search")

    body, snapshot, changed = server.prepare_upstream(
        request, _current(), repair=False, settings=s, client="vscode")

    import json as _json
    assert _names(_json.loads(body)["tools"]) == ["read_file", "grep_search"]
    assert changed["filtered_tools"]["file"] == "vscode-core.txt"
    assert changed["filtered_tools"]["kept"] == 2
    assert changed["filtered_tools"]["dropped"] == ["run_notebook_cell"]


def test_prepare_keeps_the_full_tool_list_in_the_snapshot(tmp_path):
    """The trace must never lie about what the client actually sent, and the
    tool array is now one of the things the engine changes on the way past."""
    from inference import server
    s = _settings(tmp_path)
    s.set_toolset("vscode", "vscode-core.txt")

    _, snapshot, _ = server.prepare_upstream(
        _request("read_file", "run_notebook_cell"), _current(),
        repair=False, settings=s, client="vscode")

    assert snapshot is not None
    assert _names(snapshot["tools"]) == ["read_file", "run_notebook_cell"]


def test_prepare_leaves_an_unconfigured_client_alone(tmp_path):
    from inference import server
    s = _settings(tmp_path)
    s.set_toolset("vscode", "vscode-core.txt")

    body, _, changed = server.prepare_upstream(
        _request("read_file", "run_notebook_cell"), _current(),
        repair=False, settings=s, client="cline")

    import json as _json
    assert len(_json.loads(body)["tools"]) == 2
    assert "filtered_tools" not in changed


def test_prepare_records_a_stale_name(tmp_path):
    """How the set drifts: VS Code renames or drops a tool and the engine
    keeps asking for it. Ignored rather than rejected, but recorded."""
    from inference import server
    s = _settings(tmp_path)
    s.set_toolset("vscode", "vscode-core.txt")

    _, _, changed = server.prepare_upstream(
        _request("read_file", "run_notebook_cell"), _current(),
        repair=False, settings=s, client="vscode")

    assert "grep_search" in changed["filtered_tools"]["unknown"]


def test_prepare_does_not_record_a_filter_that_changed_nothing(tmp_path):
    from inference import server
    s = _settings(tmp_path)
    s.set_toolset("vscode", "vscode-core.txt")

    _, _, changed = server.prepare_upstream(
        _request("read_file", "grep_search"), _current(),
        repair=False, settings=s, client="vscode")

    assert "filtered_tools" not in changed


def test_prepare_ignores_a_toolset_file_that_is_gone(tmp_path):
    from inference import server
    s = _settings(tmp_path)
    s.set_toolset("vscode", "deleted.txt")

    body, _, changed = server.prepare_upstream(
        _request("read_file", "run_notebook_cell"), _current(),
        repair=False, settings=s, client="vscode")

    import json as _json
    assert len(_json.loads(body)["tools"]) == 2
    assert "filtered_tools" not in changed


# ------------------------------------------- the trace stays self-describing
def test_the_trace_records_which_client_sent_the_request(tmp_path):
    """Filtering removes the tools `clients.identify` recognises VS Code by -
    `copilot_getNotebookSummary` and the rest are exactly the sort of thing a
    set drops, so re-identifying a stored request no longer finds VS Code.
    The engine knows the answer before it filters, so the record carries it
    rather than leaving it to be re-derived.
    """
    from inference import clients, tracing
    writer = tracing.TraceWriter(str(tmp_path / "t.jsonl"), quiet=True)
    writer.record(path="/v1/chat/completions",
                  request={"model": "qwen3.8", "messages": [],
                           "tools": _tools("read_file", "list_dir")},
                  response={}, streaming=False, timing={}, client="vscode")

    record = json.loads(open(writer.path).readline())
    assert record["client"] == "vscode"
    # The point of recording it: the request alone no longer says so.
    assert clients.identify(record["request"]) != "vscode"

"""Collapsing a repeated tool call out of the conversation.

The case this exists for, from traces/trace-20260827-121604.jsonl: the model
built a malformed Azure DevOps URL, got a 404, and re-issued the byte-identical
command 37 times. Each turn's context differed only by another copy of the
wrong command and the same error, so the wrong continuation stayed the most
likely one. Copilot never stopped it; the engine is the only component that
sees the whole history every turn.

What the shape of the fix had to be was measured, not guessed, by replaying the
stuck turn against the real model:

    keep one copy of the pair + a note          0/3 escaped
    keep one copy + a *directive* note          2/3
    remove every copy, note only                3/3
    remove the calls, keep the result in a note 4/4

So the command text is the anchor, not the error. The note carries what came
back - the model needs that - and never the arguments that produced it, because
an example of the wrong command in front of the model is an example it copies.
"""
import json
import os

import pytest

from inference import loops


def _call(name="run_in_terminal", args='{"command":"az build show"}', cid="c1"):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": args}}


def _pair(result="ERR: 404", args='{"command":"az build show"}', cid="c1",
          text=""):
    """One (assistant asks, tool answers) exchange - the unit that collapses."""
    return [{"role": "assistant", "content": text,
             "tool_calls": [_call(args=args, cid=cid)]},
            {"role": "tool", "tool_call_id": cid, "content": result}]


def _convo(*pairs, head=True):
    out = ([{"role": "system", "content": "be helpful"},
            {"role": "user", "content": "check the build"}] if head else [])
    for p in pairs:
        out += p
    return out


def _roles(msgs):
    return [m["role"] for m in msgs]


# ------------------------------------------------------------------ scanning
def test_spans_finds_each_assistant_and_its_results():
    msgs = _convo(_pair(cid="a"), _pair(cid="b"))
    assert loops.spans(msgs) == [(2, 4), (4, 6)]


def test_spans_ignores_plain_assistant_turns():
    msgs = _convo(_pair(cid="a")) + [{"role": "assistant", "content": "done"}]
    assert loops.spans(msgs) == [(2, 4)]


def test_spans_keeps_several_results_with_their_call():
    """One assistant message can call two tools, and both results belong to
    it. Splitting them would orphan a tool message."""
    msgs = _convo(head=False)
    msgs += [{"role": "assistant", "tool_calls": [_call(cid="a"), _call(cid="b")]},
             {"role": "tool", "tool_call_id": "a", "content": "1"},
             {"role": "tool", "tool_call_id": "b", "content": "2"}]
    assert loops.spans(msgs) == [(0, 3)]


def test_identical_pairs_share_a_fingerprint():
    msgs = _convo(_pair(cid="a"), _pair(cid="b"))
    a, b = loops.spans(msgs)
    assert loops.fingerprint(msgs, *a) == loops.fingerprint(msgs, *b)


def test_the_tool_call_id_is_not_part_of_the_fingerprint():
    """Every retry gets a fresh id from the client, so including it would mean
    nothing ever matched."""
    msgs = _convo(_pair(cid="wildly-different"), _pair(cid="also-different"))
    a, b = loops.spans(msgs)
    assert loops.fingerprint(msgs, *a) == loops.fingerprint(msgs, *b)


def test_a_different_result_is_a_different_fingerprint():
    """The whole point of a poll: same command, output that moves on."""
    msgs = _convo(_pair(result="queued"), _pair(result="running"))
    a, b = loops.spans(msgs)
    assert loops.fingerprint(msgs, *a) != loops.fingerprint(msgs, *b)


# --------------------------------------------------------------- collapsing
def test_a_run_at_the_threshold_collapses():
    msgs = _convo(*[_pair(cid=str(i)) for i in range(3)])
    out, info = loops.collapse(msgs, threshold=3)
    assert len(info) == 1
    assert info[0]["repeats"] == 3
    assert info[0]["tool"] == "run_in_terminal"
    assert _roles(out) == ["system", "user", "user"]


def test_a_run_below_the_threshold_is_left_alone():
    msgs = _convo(*[_pair(cid=str(i)) for i in range(2)])
    out, info = loops.collapse(msgs, threshold=3)
    assert info == [] and out == msgs


def test_a_threshold_of_zero_disables_it():
    """How the Config page turns it off for a client running a real poll."""
    msgs = _convo(*[_pair(cid=str(i)) for i in range(20)])
    out, info = loops.collapse(msgs, threshold=0)
    assert info == [] and out == msgs


def test_a_poll_whose_output_moves_is_never_collapsed():
    msgs = _convo(*[_pair(result=f"{i}% complete", cid=str(i))
                    for i in range(10)])
    out, info = loops.collapse(msgs, threshold=3)
    assert info == [] and out == msgs


def test_different_arguments_are_not_a_loop():
    msgs = _convo(*[_pair(args=f'{{"command":"step {i}"}}', cid=str(i))
                    for i in range(6)])
    out, info = loops.collapse(msgs, threshold=3)
    assert info == []


def test_identical_pairs_that_are_not_consecutive_do_not_count():
    msgs = _convo(_pair(cid="a"), _pair(result="something else", cid="b"),
                  _pair(cid="c"))
    out, info = loops.collapse(msgs, threshold=3)
    assert info == []


# ----------------------------------------------- what the model is left with
def test_the_note_carries_the_result():
    """Deleting the run must not delete what came back - a repeated call that
    returned data still leaves the model needing that data."""
    msgs = _convo(*[_pair(result="ERR: 404 Not Found", cid=str(i))
                    for i in range(5)])
    out, _ = loops.collapse(msgs, threshold=3)
    assert "ERR: 404 Not Found" in out[-1]["content"]


def test_the_note_never_carries_the_arguments():
    """The measured difference between 0/3 and 4/4. An example of the wrong
    command in front of the model is an example it copies."""
    msgs = _convo(*[_pair(args='{"command":"THE-BAD-COMMAND"}', cid=str(i))
                    for i in range(5)])
    out, _ = loops.collapse(msgs, threshold=3)
    assert "THE-BAD-COMMAND" not in json.dumps(out)


def test_the_note_says_how_many_times_and_which_tool():
    msgs = _convo(*[_pair(cid=str(i)) for i in range(7)])
    out, _ = loops.collapse(msgs, threshold=3)
    assert "7" in out[-1]["content"] and "run_in_terminal" in out[-1]["content"]


def test_a_very_large_result_is_capped_and_says_so():
    msgs = _convo(*[_pair(result="x" * 50_000, cid=str(i)) for i in range(4)])
    out, _ = loops.collapse(msgs, threshold=3)
    note = out[-1]["content"]
    assert len(note) < 5_000
    assert "truncated" in note.lower()


# ------------------------------------------------------------- structure
def test_no_tool_message_is_left_without_its_call():
    """An orphaned `tool` message breaks the chat template upstream, which
    fails the whole conversation rather than one turn."""
    msgs = _convo(*[_pair(cid=str(i)) for i in range(6)])
    out, _ = loops.collapse(msgs, threshold=3)
    calls = {c["id"] for m in out for c in m.get("tool_calls") or []}
    for m in out:
        if m["role"] == "tool":
            assert m["tool_call_id"] in calls


def test_work_after_the_loop_survives():
    """A run in the middle still collapses - and it must, or the moment the
    model escapes, the run stops being at the tail and the poison comes back
    on the very next turn."""
    msgs = _convo(*[_pair(cid=str(i)) for i in range(5)])
    msgs += _pair(args='{"command":"the good one"}', result="OK", cid="good")
    out, info = loops.collapse(msgs, threshold=3)
    assert info and info[0]["repeats"] == 5
    assert "the good one" in json.dumps(out)
    assert out[-1]["role"] == "tool" and out[-1]["content"] == "OK"


def test_two_separate_runs_both_collapse():
    msgs = _convo(*[_pair(cid=str(i)) for i in range(4)])
    msgs += _pair(args='{"command":"other"}', result="X", cid="x") * 4
    out, info = loops.collapse(msgs, threshold=3)
    assert len(info) == 2


def test_the_input_is_not_mutated():
    msgs = _convo(*[_pair(cid=str(i)) for i in range(5)])
    before = json.dumps(msgs)
    loops.collapse(msgs, threshold=3)
    assert json.dumps(msgs) == before


def test_a_conversation_with_no_tool_calls_is_untouched():
    msgs = [{"role": "user", "content": "hi"}]
    out, info = loops.collapse(msgs, threshold=3)
    assert out == msgs and info == []


# ---------------------------------------------------------------- regression
REAL = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "traces", "trace-20260827-121604.jsonl")


@pytest.mark.skipif(not os.path.exists(REAL), reason="trace not on this box")
def test_the_session_this_was_built_for():
    """102 messages, of which 74 are the same 404 thirty-seven times over."""
    record = None
    for line in open(REAL, encoding="utf-8"):
        row = json.loads(line)
        if row["seq"] == 118:
            record = row
            break
    msgs = record["request_original"]["messages"]
    out, info = loops.collapse(msgs, threshold=5)
    assert len(msgs) == 102
    assert info[0]["repeats"] == 37
    assert info[0]["tool"] == "run_in_terminal"
    assert len(out) == 29
    assert info[0]["messages_removed"] == 73

    # Three earlier, *non-identical* attempts at the same idea survive - they
    # are not part of the run, and collapsing them would be collapsing work
    # the model actually varied. That is fine: the replay that escaped 4/4 had
    # them present too. What goes is the wall of 37 copies.
    assert json.dumps(msgs).count("[char]38") == 40
    assert json.dumps(out).count("[char]38") == 3

    # And the note itself carries the reason without the command.
    assert "404" in out[-1]["content"]
    assert "[char]38" not in out[-1]["content"]


# --------------------------------------------------------------- the setting
def _settings(tmp_path):
    from inference.settings import Settings
    return Settings(str(tmp_path / "settings.json"))


def test_loop_breaking_is_off_by_default(tmp_path):
    assert _settings(tmp_path).loop_threshold_for("vscode") == 0


def test_a_threshold_is_set_per_client(tmp_path):
    s = _settings(tmp_path)
    s.set_loop_threshold("vscode", 5)
    assert s.loop_threshold_for("vscode") == 5
    assert s.loop_threshold_for("cline") == 0


def test_a_threshold_survives_a_restart(tmp_path):
    _settings(tmp_path).set_loop_threshold("vscode", 5)
    assert _settings(tmp_path).loop_threshold_for("vscode") == 5


def test_a_threshold_of_zero_is_stored_as_off(tmp_path):
    """The Config page's off switch, for a client running a real poll."""
    s = _settings(tmp_path)
    s.set_loop_threshold("vscode", 5)
    s.set_loop_threshold("vscode", 0)
    assert s.loop_threshold_for("vscode") == 0
    assert _settings(tmp_path).loop_threshold_for("vscode") == 0


def test_a_threshold_below_two_is_refused(tmp_path):
    """A threshold of 1 would collapse every single tool call ever made."""
    s = _settings(tmp_path)
    s.set_loop_threshold("vscode", 1)
    assert s.loop_threshold_for("vscode") == 0


def test_a_nonsense_threshold_does_not_wedge_the_engine(tmp_path):
    s = _settings(tmp_path)
    s.set_loop_threshold("vscode", "banana")
    assert s.loop_threshold_for("vscode") == 0


# -------------------------------------------------- through prepare_upstream
def test_prepare_collapses_a_loop_for_a_configured_client(tmp_path):
    import types
    from inference import server
    s = _settings(tmp_path)
    s.set_loop_threshold("vscode", 3)
    request = {"model": "qwen3.8",
               "messages": _convo(*[_pair(cid=str(i)) for i in range(6)])}

    body, snapshot, changed = server.prepare_upstream(
        request, types.SimpleNamespace(upstream_model_id="qwen3.8"),
        repair=False, settings=s, client="vscode")

    sent = json.loads(body)
    assert len(sent["messages"]) == 3
    assert changed["collapsed_loops"][0]["repeats"] == 6
    # The trace must never lie about what the client actually sent.
    assert len(snapshot["messages"]) == 14


def test_prepare_leaves_an_unconfigured_client_alone(tmp_path):
    import types
    from inference import server
    s = _settings(tmp_path)
    s.set_loop_threshold("vscode", 3)
    request = {"model": "qwen3.8",
               "messages": _convo(*[_pair(cid=str(i)) for i in range(6)])}

    body, _, changed = server.prepare_upstream(
        request, types.SimpleNamespace(upstream_model_id="qwen3.8"),
        repair=False, settings=s, client="cline")

    assert len(json.loads(body)["messages"]) == 14
    assert "collapsed_loops" not in changed

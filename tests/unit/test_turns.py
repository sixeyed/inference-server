"""Reading one turn back out of a trace.

The trace is gitignored, rotates per engine run, and can be switched off - so
"not there" is an ordinary answer, and the view has to say which ordinary
answer it is rather than breaking.
"""

import json

from inference import turns


def write_trace(tmp_path, records):
    path = tmp_path / "trace-test.jsonl"
    with open(path, "w") as handle:
        for r in records:
            handle.write(json.dumps(r) + "\n")
    return str(tmp_path)


def record(seq, n_messages, reply="done"):
    return {
        "seq": seq,
        "request": {
            "messages": [{"role": "system", "content": "sys"}]
                        + [{"role": "user", "content": f"m{i}"}
                           for i in range(n_messages - 1)],
            "tools": [{"function": {"name": "read_file"}}],
            "temperature": 0.1,
        },
        "response": {"choices": [{"message": {"role": "assistant",
                                              "content": reply},
                                  "finish_reason": "stop"}]},
    }


def test_only_the_new_messages_come_back(tmp_path):
    # The whole conversation is resent every turn; what is new is the tail.
    d = write_trace(tmp_path, [record(1, 10)])
    out = turns.load(d, "trace-test.jsonl", 1, since=7)
    assert out["found"]
    assert len(out["new_messages"]) == 3
    assert out["skipped"] == 7
    assert out["total_messages"] == 10


def test_a_missing_file_says_so(tmp_path):
    out = turns.load(str(tmp_path), "gone.jsonl", 1)
    assert out["found"] is False
    assert "no longer on disk" in out["reason"]


def test_a_missing_turn_in_a_present_file_says_so(tmp_path):
    d = write_trace(tmp_path, [record(1, 3)])
    out = turns.load(d, "trace-test.jsonl", 99)
    assert out["found"] is False
    assert "not in" in out["reason"]


def test_tracing_off_is_not_an_error(tmp_path):
    out = turns.load(str(tmp_path), None, None)
    assert out["found"] is False
    assert "tracing off" in out["reason"]


def test_a_nonsense_since_falls_back_to_the_tail(tmp_path):
    # A client that compacted its own history can report fewer messages than
    # the previous turn. Show the last few rather than everything or nothing.
    d = write_trace(tmp_path, [record(1, 6)])
    out = turns.load(d, "trace-test.jsonl", 1, since=999)
    assert out["found"] and 0 < len(out["new_messages"]) <= 4


def test_long_content_is_clipped(tmp_path):
    big = dict(record(1, 2))
    big["request"]["messages"][-1]["content"] = "x" * (turns.MAX_CHARS + 500)
    d = write_trace(tmp_path, [big])
    out = turns.load(d, "trace-test.jsonl", 1, since=1)
    message = out["new_messages"][-1]
    assert message["clipped"] is True
    assert len(message["content"]) == turns.MAX_CHARS


def test_a_corrupt_line_does_not_stop_the_search(tmp_path):
    path = tmp_path / "trace-test.jsonl"
    with open(path, "w") as handle:
        handle.write("{not json\n")
        handle.write(json.dumps(record(2, 3)) + "\n")
    out = turns.load(str(tmp_path), "trace-test.jsonl", 2, since=1)
    assert out["found"]

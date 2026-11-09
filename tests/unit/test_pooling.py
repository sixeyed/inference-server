"""Aggregation: pooled rates, and what gets left out."""

import sqlite3

import pytest

from inference import usage


@pytest.fixture
def db(tmp_path):
    database = usage.UsageDB(str(tmp_path / "u.db"))
    yield database
    database.close()


def add(db, *, alias="m", tokens, decode_ms, preempted=0, ok=1, ttft=100.0):
    db._write(
        """INSERT INTO requests (ts, alias, engine, path, stream, ok, preempted,
                                 output_tokens, decode_ms, decode_tps, ttft_ms)
           VALUES ('t', ?, 'e', '/v1/chat/completions', 1, ?, ?, ?, ?, ?, ?)""",
        (alias, ok, preempted, tokens, decode_ms,
         round(tokens / (decode_ms / 1000), 1), ttft))


def test_average_is_pooled_not_the_mean_of_rates(db):
    # One long slow turn and one short fast one. The mean of the two rates is
    # 55.0; pooled - which is what a user actually experiences - is 20.0.
    add(db, tokens=100, decode_ms=10_000)   # 10 tok/s
    add(db, tokens=100, decode_ms=1_000)    # 100 tok/s
    row = db.per_model()[0]
    assert row["avg_tps"] == 18.2           # 200 tokens over 11.0s
    assert row["min_tps"] == 10.0
    assert row["max_tps"] == 100.0
    mean_of_rates = (10.0 + 100.0) / 2
    assert row["avg_tps"] != mean_of_rates


def test_preempted_turns_are_excluded(db):
    add(db, tokens=100, decode_ms=1_000)              # 100 tok/s
    add(db, tokens=3, decode_ms=5_000, preempted=1)   # cut mid-generation
    row = db.per_model()[0]
    assert row["requests"] == 1
    assert row["avg_tps"] == 100.0, "a cut turn must not drag the average down"
    assert db.preempted_counts() == {"m": 1}


def test_failed_turns_are_excluded(db):
    add(db, tokens=100, decode_ms=1_000)
    add(db, tokens=0, decode_ms=1.0, ok=0)
    assert db.per_model()[0]["requests"] == 1


def test_ttft_reports_the_worst_case_alongside_the_mean(db):
    add(db, tokens=10, decode_ms=100, ttft=100.0)
    add(db, tokens=10, decode_ms=100, ttft=60_000.0)
    row = db.per_model()[0]
    assert row["max_ttft_ms"] == 60_000
    assert row["avg_ttft_ms"] == 30_050


def test_preempted_rows_are_kept_for_the_feed(db):
    add(db, tokens=3, decode_ms=5_000, preempted=1)
    assert db.per_model() == []
    assert len(db.recent()) == 1, "excluded from stats, not deleted"


def test_a_recorded_request_actually_lands(db):
    """Guards the INSERT's column/placeholder arity.

    Stats failures are swallowed on purpose - losing a row must never cost a
    turn - so a mismatch here fails completely silently except for one console
    line. That is worth a test rather than a code review.
    """
    import types

    current = types.SimpleNamespace(engine="mtplx serve", load_ms=1234.0,
                                    upstream_model_id="qwen3.8")
    db.record_request(
        seq=1, alias="qwen3.8", requested="qwen3.8", current=current,
        path="/v1/chat/completions", streaming=True,
        timing={"ttfb_ms": 10.0, "ttft_ms": 100.0, "ttfc_ms": 150.0,
                "total_ms": 900.0},
        request={"messages": [{"role": "user", "content": "hi"}], "tools": []},
        response={
            "id": "chatcmpl-1",
            "choices": [{"index": 0, "message": {"role": "assistant",
                                                 "content": "hello"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 5},
            "timings": {"prompt_n": 20, "predicted_n": 5, "prompt_ms": 100.0,
                        "predicted_ms": 200.0, "prompt_per_second": 200.0,
                        "predicted_per_second": 25.0},
            "mtplx_stats": {"mtp_depth": 3, "drafted_by_depth": [3],
                            "accepted_by_depth": [2], "ttft_s": 0.08,
                            "peak_memory_bytes": 1024 ** 3},
        },
        preempted=False, trace_file="traces/t.jsonl", client_ip="192.168.1.9")

    rows = db._query("SELECT * FROM requests")
    assert len(rows) == 1, "the insert silently failed"
    row = rows[0]
    assert row["client_ip"] == "192.168.1.9"
    assert row["alias"] == "qwen3.8"
    assert row["rate_source"] == "server"
    assert row["decode_tps"] == 25.0
    assert row["deferral_ms"] == 50.0
    assert row["mtp_accept"] == 0.667
    assert row["peak_rss_mb"] == 1024.0
    assert row["trace_seq"] == 1


def add_mtp(db, *, drafted, accepted, tokens, decode_ms=1000.0):
    db._write(
        """INSERT INTO requests (ts, alias, engine, path, stream, ok, preempted,
                                 output_tokens, decode_ms, mtp_drafted,
                                 mtp_accepted, mtp_accept)
           VALUES ('t', 'm', 'e', '/v1/chat/completions', 1, 1, 0,
                   ?, ?, ?, ?, ?)""",
        (tokens, decode_ms, drafted, accepted, round(accepted / drafted, 3)))


def test_mtp_acceptance_is_pooled_like_token_rates(db):
    # A two-token reply is a single verify round. Averaging its ratio against a
    # long turn's measures how short your test prompts were, not the model.
    add_mtp(db, drafted=3, accepted=1, tokens=2)        # ratio 0.333
    add_mtp(db, drafted=300, accepted=270, tokens=800)  # ratio 0.900
    row = db.per_model()[0]

    assert row["mtp_drafted"] == 303
    assert row["mtp_accept"] == 0.894          # 271 accepted of 303 drafted
    mean_of_ratios = (0.333 + 0.900) / 2
    assert abs(row["mtp_accept"] - mean_of_ratios) > 0.2, (
        "pooling must not collapse to the mean of per-request ratios")


def test_rows_without_mtp_counts_drop_out_rather_than_skew(db):
    # Rows recorded before the counts existed keep their ratio but cannot be
    # pooled; they must not be silently counted as zero drafts.
    add_mtp(db, drafted=100, accepted=90, tokens=200)
    add(db, tokens=50, decode_ms=1000)   # a llama.cpp turn: no MTP at all
    row = db.per_model()[0]
    assert row["mtp_drafted"] == 100
    assert row["mtp_accept"] == 0.9


def test_no_mtp_anywhere_reports_nothing_rather_than_zero(db):
    add(db, tokens=50, decode_ms=1000)
    assert db.per_model()[0]["mtp_accept"] is None


def test_migration_adds_columns_declared_several_to_a_line(tmp_path):
    """An older usage.db must gain every new column, however SCHEMA lays it out.

    Parsing SCHEMA line-wise silently produced `ADD COLUMN a INTEGER, b REAL`,
    which SQLite rejects - so the migration threw and the engine would not
    start against an existing database.
    """
    import sqlite3

    path = str(tmp_path / "old.db")
    old = sqlite3.connect(path)
    old.execute("""CREATE TABLE requests (
                     id INTEGER PRIMARY KEY, ts TEXT, alias TEXT NOT NULL,
                     engine TEXT NOT NULL, path TEXT NOT NULL,
                     stream INTEGER NOT NULL, ok INTEGER NOT NULL)""")
    old.execute("INSERT INTO requests (ts, alias, engine, path, stream, ok) "
                "VALUES ('t','m','e','/v1/chat/completions',1,1)")
    old.commit()
    old.close()

    db = usage.UsageDB(path)
    try:
        columns = {row[1] for row in db._db.execute("PRAGMA table_info(requests)")}
        for expected in ("status", "client_ip", "session_id", "preempted",
                         "mtp_depth", "mtp_accept", "mtp_bonus_tokens",
                         "mtp_drafted", "mtp_accepted", "decode_ms",
                         "rate_source"):
            assert expected in columns, f"{expected} was not migrated in"
        # And the existing row survives.
        assert db._query("SELECT COUNT(*) AS n FROM requests")[0]["n"] == 1
    finally:
        db.close()


def test_migration_survives_a_comma_inside_a_schema_comment(tmp_path):
    """`-- what was called, in order` must not become a column named "in".

    The parser split the table body on commas before stripping comments, so a
    comma inside one produced `ADD COLUMN in order` and every insert failed.
    """
    import sqlite3

    path = str(tmp_path / "old.db")
    old = sqlite3.connect(path)
    old.execute("CREATE TABLE requests (id INTEGER PRIMARY KEY, ts TEXT, "
                "alias TEXT NOT NULL, engine TEXT NOT NULL, path TEXT NOT NULL, "
                "stream INTEGER NOT NULL, ok INTEGER NOT NULL)")
    old.commit()
    old.close()

    db = usage.UsageDB(path)
    try:
        columns = {row[1] for row in db._db.execute("PRAGMA table_info(requests)")}
        assert "tool_names" in columns and "tool_args" in columns
        assert not any(c in ("in", "order") for c in columns), columns
    finally:
        db.close()

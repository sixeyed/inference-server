"""The engine's event log - what the Logs page reads.

The console already prints most of this, once per engine run, to a terminal
nobody is watching. Two things make the log worth having over that.

The first is dedup by content rather than by a boolean. `warned_toolset` and
friends were set on the first occurrence and never cleared, so drift that
appeared mid-session - a VS Code update renaming a tool between one turn and
the next - was swallowed entirely. Fingerprinting the event means the first
occurrence is recorded, a repeat is not, and a *change* is.

The second is that it survives a restart, which is exactly when you go looking
after an update.
"""
import json

import pytest

from inference import usage as usage_mod
from inference.events import Events


@pytest.fixture
def store(tmp_path):
    return usage_mod.UsageDB(str(tmp_path / "usage.db"))


@pytest.fixture
def events(store):
    return Events(store, quiet=True)


# ------------------------------------------------------------------- storage
def test_an_event_is_recorded(store):
    store.record_event("tools", "vscode", {"dropped": 50})
    rows = store.events()
    assert len(rows) == 1
    assert rows[0]["kind"] == "tools"
    assert rows[0]["client"] == "vscode"
    assert rows[0]["detail"] == {"dropped": 50}


def test_events_come_back_newest_first(store):
    store.record_event("tools", "vscode", {"n": 1})
    store.record_event("prompt", "vscode", {"n": 2})
    assert [r["detail"]["n"] for r in store.events()] == [2, 1]


def test_events_survive_reopening_the_database(tmp_path):
    path = str(tmp_path / "usage.db")
    usage_mod.UsageDB(path).record_event("tools", "vscode", {"dropped": 50})
    assert len(usage_mod.UsageDB(path).events()) == 1


def test_a_database_made_before_events_existed_gains_the_table(tmp_path):
    """The migration path that matters: usage.db here is 12MB of history and
    nobody is going to delete it to get a new table."""
    import sqlite3
    path = str(tmp_path / "old.db")
    old = sqlite3.connect(path)
    old.executescript("CREATE TABLE requests (id INTEGER PRIMARY KEY, ts TEXT);")
    old.commit()
    old.close()

    store = usage_mod.UsageDB(path)
    store.record_event("tools", "vscode", {"dropped": 1})
    assert len(store.events()) == 1


# --------------------------------------------------------------------- dedup
def test_the_same_event_twice_is_recorded_once(events, store):
    for _ in range(2):
        events.note("tools", "vscode", "16 kept, 50 dropped", {"dropped": 50})
    assert len(store.events()) == 1


def test_a_changed_event_is_recorded_again(events, store):
    """The bug this replaces: a boolean set on the first occurrence meant a
    tool renamed mid-session never showed up at all."""
    events.note("tools", "vscode", "no drift", {"unknown": []})
    events.note("tools", "vscode", "drift", {"unknown": ["create_file"]})
    assert [r["detail"]["unknown"] for r in store.events()] == \
        [["create_file"], []]


def test_the_same_event_for_a_different_client_is_its_own(events, store):
    events.note("tools", "vscode", "16 kept", {"dropped": 50})
    events.note("tools", "cline", "16 kept", {"dropped": 50})
    assert len(store.events()) == 2


def test_events_of_different_kinds_do_not_dedup_each_other(events, store):
    events.note("tools", "vscode", "same text", {"a": 1})
    events.note("prompt", "vscode", "same text", {"a": 1})
    assert len(store.events()) == 2


def test_returning_to_a_previous_state_is_recorded(events, store):
    """Drift appearing and then being fixed are both worth a line - the second
    is how you confirm an edit to the set actually landed."""
    events.note("tools", "vscode", "drift", {"unknown": ["create_file"]})
    events.note("tools", "vscode", "clean", {"unknown": []})
    events.note("tools", "vscode", "drift", {"unknown": ["create_file"]})
    assert len(store.events()) == 3


def test_an_event_works_without_a_usage_database(capsys):
    """--no-usage still has to warn. The console line and the row are driven
    by the same call, so losing the database must not lose the warning."""
    events = Events(None, quiet=False)
    events.note("tools", "vscode", "16 kept, 50 dropped", {"dropped": 50})
    assert "16 kept, 50 dropped" in capsys.readouterr().out


def test_the_console_line_is_printed_once_per_change(capsys):
    events = Events(None, quiet=False)
    for _ in range(3):
        events.note("tools", "vscode", "16 kept, 50 dropped", {"dropped": 50})
    assert capsys.readouterr().out.count("16 kept") == 1


# ------------------------------------------------------- drift, specifically
#
# `unknown` alone is the wrong signal. A set may name a tool this client has
# simply never sent - `task_coplete` for a typo, or `task_complete` which only
# some VS Code configurations offer - and that is not drift, it is an entry
# that has never applied. Lighting the indicator for it means it is lit
# permanently, which is the same as not having one.
#
# Drift is narrower and actionable: a tool the client *did* offer and has
# stopped offering. That is what a rename in a client release looks like, and
# the model has silently lost the tool.
def _tools_event(store, matched, unknown):
    store.record_event("tools", "vscode", {"file": "vscode-core.txt",
                                           "matched": matched,
                                           "unknown": unknown})


def test_a_tool_that_stopped_being_offered_is_drift(store):
    _tools_event(store, ["read_file", "create_file"], [])
    _tools_event(store, ["read_file"], ["create_file"])
    drift = store.drift()
    assert set(drift) == {"vscode"}
    assert drift["vscode"]["unknown"] == ["create_file"]
    assert drift["vscode"]["file"] == "vscode-core.txt"


def test_a_name_the_client_never_offered_is_not_drift(store):
    """The permanently-lit case: a set entry that has never applied."""
    _tools_event(store, ["read_file"], ["task_complete"])
    assert store.drift() == {}


def test_drift_reports_nothing_when_the_set_matches(store):
    _tools_event(store, ["read_file", "create_file"], [])
    assert store.drift() == {}


def test_fixing_the_set_clears_the_drift(store):
    """Or it reads as still broken after you have already dealt with it."""
    _tools_event(store, ["read_file", "create_file"], [])
    _tools_event(store, ["read_file"], ["create_file"])
    _tools_event(store, ["read_file", "create_file"], [])
    assert store.drift() == {}


def test_drift_only_reports_names_the_set_still_asks_for(store):
    """Two tools go missing, one has since been dropped from the set, so only
    the other is still a problem worth showing."""
    _tools_event(store, ["read_file", "create_file", "get_errors"], [])
    _tools_event(store, ["read_file"], ["create_file"])
    assert store.drift()["vscode"]["unknown"] == ["create_file"]


def test_drift_is_per_client(store):
    _tools_event(store, ["read_file", "create_file"], [])
    _tools_event(store, ["read_file"], ["create_file"])
    store.record_event("tools", "cline",
                       {"file": "cline.txt", "matched": ["read_file"],
                        "unknown": []})
    assert set(store.drift()) == {"vscode"}

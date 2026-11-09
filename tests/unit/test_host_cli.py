"""The commands an engine elsewhere runs on the Mac over SSH.

Each prints one JSON line and nothing the caller has to scrape, and none of
them may block: the SSH call that asked has to come back.
"""

import json
import os
import types

import pytest

from inference import cli, process
from inference.machine import Unsupported, machine_from_profile


@pytest.fixture
def mac(monkeypatch):
    """A Mac with llama-server and weights, without touching either."""
    monkeypatch.setattr(cli, "weights_root_missing", lambda: None)
    monkeypatch.setattr(cli, "detect_machine", lambda: machine_from_profile("m4"))
    monkeypatch.setattr(process, "resolve_spec_decode", lambda requested: (requested, None))
    monkeypatch.setattr(process, "resolve_engine",
                        lambda spec: ("/opt/homebrew/bin/llama-server", "llama-server", None))
    monkeypatch.setattr(process, "sandbox_wrap",
                        lambda argv: (["/usr/bin/sandbox-exec"] + argv, None))
    monkeypatch.setattr(process, "weights_are_local", lambda spec: True)
    monkeypatch.setattr(process, "checkout_sha", lambda: "abc1234", raising=False)

    def refuse_to_launch(*args, **kwargs):
        raise AssertionError("a plan must never launch anything")
    monkeypatch.setattr(process.subprocess, "Popen", refuse_to_launch)
    return monkeypatch


def test_plan_describes_a_launch_without_starting_it(mac, capsys):
    assert cli.main(["run", "qwen3-tiny", "--plan"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "alias": "qwen3-tiny", "engine": "llama-server",
        "upstream_id": "qwen3-tiny", "ok": True, "reason": None, "needs_pull": False,
        "warnings": [], "sha": "abc1234"}


def test_plan_refuses_a_sandboxed_launch_with_no_weights(mac, capsys):
    mac.setattr(process, "weights_are_local", lambda spec: False)
    assert cli.main(["run", "qwen3-tiny", "--plan"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["ok"] is False
    assert "not on disk" in plan["reason"]


def test_plan_reports_a_model_that_does_not_fit(mac, capsys):
    def too_big(machine, opts):
        raise Unsupported("too large for M4 Max")
    module = types.SimpleNamespace(ALIAS="big", spec=too_big)
    mac.setattr(cli.models, "load_all", lambda: {"big": module})
    assert cli.main(["run", "big", "--plan"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert (plan["ok"], plan["reason"]) == (False, "too large for M4 Max")


def test_detach_starts_in_its_own_session_with_stdin_closed(mac, capsys, tmp_path):
    seen = {}

    class FakePopen:
        pid = 4321

        def __init__(self, argv, **kwargs):
            seen.update(kwargs)

        def poll(self):
            return None

    mac.setattr(process.subprocess, "Popen", FakePopen)
    assert cli.main(["run", "qwen3-tiny", "--detach", "--log-dir", str(tmp_path)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["pid"] == 4321
    assert printed["engine"] == "llama-server"
    assert printed["log_path"].startswith(os.path.abspath(str(tmp_path)))
    assert seen["start_new_session"] is True
    assert seen["stdin"] is process.subprocess.DEVNULL


def test_status_reports_who_holds_the_port(monkeypatch, capsys):
    monkeypatch.setattr(process, "find_server_pid", lambda port: 777 if port == 9123 else None)
    monkeypatch.setattr(process, "rss_mb_of", lambda pid: 512.0, raising=False)
    assert cli.main(["status", "--port", "9123"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "port": 9123, "pid": 777, "alive": True, "rss_mb": 512.0}


def test_status_counts_a_server_that_has_not_bound_yet(monkeypatch, capsys):
    monkeypatch.setattr(process, "find_server_pid", lambda port: None)
    monkeypatch.setattr(process, "pid_alive", lambda pid: pid == 55, raising=False)
    assert cli.main(["status", "--port", "9123", "--pid", "55"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["alive"] is True and printed["pid"] is None

"""Driving a Mac's model servers over SSH, with no SSH.

The runner is the seam: every remote call goes through it, so a fake that
records commands and answers from a script is the whole Mac as far as the
engine can tell.
"""

import json

import pytest

from inference import models, remote
from inference.machine import Options, machine_from_profile
from inference.supervisor import StartupFailed, Supervisor

CONFIG = remote.ModelHostConfig(
    ssh="user@mac.example.com", address="mac.example.com",
    checkout="/Users/user/inference-server",
    python="/opt/homebrew/bin/python3.14",
    identity="/home/inference/.ssh/id_ed25519",
    known_hosts="/home/inference/.ssh/known_hosts", machine="m4")


class FakeMac:
    """Records each remote command; answers with the first scripted match."""

    def __init__(self, *answers):
        self.answers = answers
        self.commands = []

    def __call__(self, argv, timeout):
        command = argv[-1]
        self.commands.append(command)
        for needle, answer in self.answers:
            if needle in command:
                return answer
        raise AssertionError(f"unexpected remote command: {command}")


def ok(payload):
    return 0, json.dumps(payload) + "\n", ""


def process_on(mac, **overrides):
    fields = dict(alias="x", engine="llama-server", port=8083, pid=4321,
                  log_path="/Users/user/inference-server/logs/x.log",
                  started_at=0.0)
    fields.update(overrides)
    return remote.RemoteProcess(host=remote.RemoteHost(CONFIG, runner=mac), **fields)


def test_config_rejects_keys_it_does_not_know():
    with pytest.raises(ValueError, match="identiy"):
        remote.ModelHostConfig.from_dict({"ssh": "a", "address": "b", "checkout": "c",
                                          "identiy": "/typo"})


def test_the_ssh_command_pins_the_key_and_host_keys_and_runs_the_checkout_cli():
    host = remote.RemoteHost(CONFIG)
    argv = host.ssh_argv(host.inference_command(["status", "--port", "8083"]))
    assert argv[:3] == ["ssh", "-p", "22"]
    for option in ("BatchMode=yes", "IdentitiesOnly=yes", "StrictHostKeyChecking=yes",
                   "UserKnownHostsFile=/home/inference/.ssh/known_hosts"):
        assert option in argv
    assert argv[-2] == "user@mac.example.com"
    assert argv[-1] == ("cd /Users/user/inference-server && "
                        "PATH=/opt/homebrew/bin:/usr/local/bin:$PATH "
                        "/opt/homebrew/bin/python3.14 -m inference status --port 8083")


def test_preflight_carries_the_launch_options_and_accepts_ok():
    mac = FakeMac(("--plan", ok({"ok": True, "sha": None})))
    host = remote.RemoteHost(CONFIG, runner=mac)
    assert host.preflight("qwen3-tiny", None, Options(port=8083, ctx=4096)) is None
    assert mac.commands[0].endswith("-m inference run qwen3-tiny --plan --port 8083 --bind 127.0.0.1 --ctx 4096")


def test_preflight_returns_the_macs_refusal():
    mac = FakeMac(("--plan", ok({"ok": False, "reason": "o/r is not on disk"})))
    assert remote.RemoteHost(CONFIG, runner=mac).preflight("x", None, Options()) == "o/r is not on disk"


def test_an_unreachable_mac_is_a_refusal_not_a_crash():
    mac = FakeMac(("--plan", (255, "", "ssh: connect to host mac.example.com port 22: Operation timed out\n")))
    reason = remote.RemoteHost(CONFIG, runner=mac).preflight("x", None, Options())
    assert "Operation timed out" in reason


def test_drift_is_reported_when_the_checkout_is_on_another_commit():
    drift = []
    mac = FakeMac(("--plan", ok({"ok": True, "sha": "def5678"})))
    host = remote.RemoteHost(CONFIG, runner=mac, local_sha="abc1234",
                             on_drift=lambda engine, mac_sha: drift.append((engine, mac_sha)))
    host.preflight("x", None, Options())
    assert drift == [("abc1234", "def5678")]


def test_a_longer_sha_for_the_same_commit_is_not_drift():
    drift = []
    mac = FakeMac(("--plan", ok({"ok": True, "sha": "abc12345"})))
    host = remote.RemoteHost(CONFIG, runner=mac, local_sha="abc1234",
                             on_drift=lambda *shas: drift.append(shas))
    host.preflight("x", None, Options())
    assert drift == []


def test_launch_detaches_and_hands_back_a_process():
    mac = FakeMac(("--detach", ok({"pid": 4321, "engine": "llama-server",
                                   "log_path": "/Users/user/inference-server/logs/x.log"})))
    launched = remote.RemoteHost(CONFIG, runner=mac).launch("x", None, None, Options(port=8083), "/data/logs")
    assert (launched.pid, launched.engine, launched.port) == (4321, "llama-server", 8083)
    assert mac.commands[0].endswith("-m inference run x --detach --no-pull --log-dir logs --port 8083 --bind 127.0.0.1")


def test_a_launch_the_mac_rejects_is_a_system_exit():
    # The supervisor turns SystemExit from a launcher into StartupFailed.
    mac = FakeMac(("--detach", (2, "", "Unknown model: x\n")))
    with pytest.raises(SystemExit, match="Unknown model"):
        remote.RemoteHost(CONFIG, runner=mac).launch("x", None, None, Options(), "/data/logs")


def test_liveness_is_cached_so_each_request_is_not_an_ssh_call():
    mac = FakeMac(("status", ok({"port": 8083, "pid": 4321, "alive": True, "rss_mb": 900.0})))
    launched = process_on(mac)
    assert all(launched.is_alive() for _ in range(5))
    assert len(mac.commands) == 1
    assert mac.commands[0].endswith("status --port 8083 --pid 4321")


def test_an_ssh_blip_is_not_an_exit():
    mac = FakeMac(("status", (255, "", "Connection reset by peer\n")))
    assert process_on(mac).exit_code() is None


def test_a_server_that_died_reports_an_exit():
    mac = FakeMac(("status", ok({"port": 8083, "pid": None, "alive": False, "rss_mb": None})))
    assert process_on(mac).exit_code() == -1


def test_stop_is_scoped_to_the_model_port():
    mac = FakeMac(("stop", (0, "", "")))
    remote.RemoteHost(CONFIG, runner=mac).stop_all(quiet=True, only_port=8083)
    assert mac.commands[0].endswith("-m inference stop --quiet --port 8083")


def test_the_log_tail_is_read_on_the_mac():
    mac = FakeMac(("tail -n 30", (0, "loading\nready\n", "")))
    assert process_on(mac).log_tail(30) == "loading\nready\n"
    assert mac.commands[0] == "tail -n 30 /Users/user/inference-server/logs/x.log"


def test_a_refused_remote_launch_stops_nothing(tmp_path):
    mac = FakeMac(("--plan", ok({"ok": False, "reason": "o/r is not on disk"})))
    machine = machine_from_profile("m4")
    registry = models.load_all()
    sup = Supervisor(machine, registry, models.describe(machine, registry),
                     Options(port=8083), port=8083, log_dir=str(tmp_path),
                     quiet=True, host=remote.RemoteHost(CONFIG, runner=mac))
    with pytest.raises(StartupFailed, match="not on disk"):
        sup.acquire("qwen3-tiny")
    assert not any(" stop " in command or "--detach" in command for command in mac.commands)


def test_a_remote_pull_asks_for_the_weights_the_launch_will_need():
    """kat-coder2.5 pulls a different repo with and without --spec, so the pull
    has to carry the same options the launch will."""
    mac = FakeMac(("pull", (0, "", "")))
    remote.RemoteHost(CONFIG, runner=mac).pull(
        "kat-coder2.5", None, Options(spec_decode=False, thinking=True), say=lambda _: None)
    assert mac.commands[0].endswith("-m inference pull kat-coder2.5 --thinking --no-spec")


def test_a_real_api_key_goes_to_the_mac_with_the_launch():
    # MTPLX on the Mac must be started with the engine's key, or every request
    # the engine relays to it is a 401.
    mac = FakeMac(("--plan", ok({"ok": True, "sha": None})))
    remote.RemoteHost(CONFIG, runner=mac).preflight(
        "qwen3-tiny", None, Options(port=8083, api_key="s3cret"))
    assert "--api-key s3cret" in mac.commands[0]


def test_the_default_key_is_not_sent_so_an_older_mac_checkout_still_works():
    mac = FakeMac(("--plan", ok({"ok": True, "sha": None})))
    remote.RemoteHost(CONFIG, runner=mac).preflight(
        "qwen3-tiny", None, Options(port=8083, api_key="none"))
    assert "--api-key" not in mac.commands[0]

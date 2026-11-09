"""Stopping the right processes, and nothing else.

This exists because of a real near-miss: naming the package `inference` and then
sweeping for it with `pgrep -f inference` also matches macOS's own
siriinferenced. A stop that reaches outside this project is much worse than a
stop that misses something.
"""

from inference import process


def test_server_patterns_are_engine_binaries_only():
    assert process.SERVER_PATTERNS == ["llama-server", "mlx_lm.server", "mtplx"]


def test_no_pattern_matches_the_package_name():
    # `pgrep -f inference` matches siriinferenced, Xcode indexers, and anything
    # else with the substring. The engine is found by its port instead.
    assert not any("inference" in pattern for pattern in process.SERVER_PATTERNS)
    assert not hasattr(process, "ENGINE_PATTERN"), (
        "the engine must be located by port, not by process name")


def test_the_engine_is_killed_by_port(monkeypatch):
    killed = []
    monkeypatch.setattr(process, "find_server_pid",
                        lambda port: 4242 if port == 9999 else None)
    monkeypatch.setattr(process, "_pids", lambda pattern: [])
    monkeypatch.setattr(process.subprocess, "run",
                        lambda argv, **kw: killed.append(argv))

    process.stop_all(quiet=True, engine=True, engine_port=9999)
    assert killed == [["kill", "4242"]]


def test_the_engine_is_left_alone_by_default(monkeypatch):
    killed = []
    monkeypatch.setattr(process, "find_server_pid", lambda port: 4242)
    monkeypatch.setattr(process, "_pids", lambda pattern: [])
    monkeypatch.setattr(process.subprocess, "run",
                        lambda argv, **kw: killed.append(argv))

    process.stop_all(quiet=True)
    assert killed == [], "stop without --engine must not touch the router"


def test_never_signals_this_process(monkeypatch):
    import os
    killed = []
    monkeypatch.setattr(process, "find_server_pid", lambda port: os.getpid())
    monkeypatch.setattr(process, "_pids", lambda pattern: [os.getpid()])
    monkeypatch.setattr(process.subprocess, "run",
                        lambda argv, **kw: killed.append(argv))

    process.stop_all(quiet=True, engine=True)
    assert killed == []


def test_a_server_on_another_port_is_left_alone(monkeypatch):
    """The case that cost a 26GB download.

    `inference stop` swept llama-server by name and killed a fetch running on a
    different port. A sweep scoped to one port must not reach it.
    """
    killed = []
    monkeypatch.setattr(process, "_pids",
                        lambda pattern: [111] if pattern == "llama-server" else [])
    monkeypatch.setattr(process, "_port_of",
                        lambda pid: 8099)          # the download's port
    monkeypatch.setattr(process, "find_server_pid", lambda port: None)
    monkeypatch.setattr(process.subprocess, "run",
                        lambda argv, **kw: killed.append(argv))

    process.stop_all(quiet=True, only_port=8083)
    assert killed == [], "a server on another port must survive"


def test_the_target_port_is_still_stopped(monkeypatch):
    killed = []
    monkeypatch.setattr(process, "_pids",
                        lambda pattern: [111] if pattern == "llama-server" else [])
    monkeypatch.setattr(process, "_port_of", lambda pid: 8083)
    monkeypatch.setattr(process, "find_server_pid", lambda port: None)
    monkeypatch.setattr(process.subprocess, "run",
                        lambda argv, **kw: killed.append(argv))

    process.stop_all(quiet=True, only_port=8083)
    assert ["kill", "111"] in killed


def test_a_process_with_no_discoverable_port_is_still_swept(monkeypatch):
    # An orphan whose command line we cannot read is the case name-matching
    # exists for; narrowing must not make it unkillable.
    killed = []
    monkeypatch.setattr(process, "_pids",
                        lambda pattern: [111] if pattern == "mtplx" else [])
    monkeypatch.setattr(process, "_port_of", lambda pid: None)
    monkeypatch.setattr(process, "find_server_pid", lambda port: None)
    monkeypatch.setattr(process.subprocess, "run",
                        lambda argv, **kw: killed.append(argv))

    process.stop_all(quiet=True, only_port=8083)
    assert ["kill", "111"] in killed


def test_an_unscoped_sweep_still_takes_everything(monkeypatch):
    killed = []
    monkeypatch.setattr(process, "_pids",
                        lambda pattern: [111] if pattern == "llama-server" else [])
    monkeypatch.setattr(process, "_port_of", lambda pid: 8099)
    monkeypatch.setattr(process, "find_server_pid", lambda port: None)
    monkeypatch.setattr(process.subprocess, "run",
                        lambda argv, **kw: killed.append(argv))

    process.stop_all(quiet=True)          # no only_port
    assert ["kill", "111"] in killed

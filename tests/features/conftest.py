"""Fixtures for the behaviour suite.

Two rules here are not stylistic, they are the difference between a test run and
an outage on this machine:

  * never bind the real ports. 8083 and 8084 belong to whatever model is loaded
    for actual work; every fixture takes an ephemeral port.
  * never call the real stop_all(). It matches on process name, so it would
    reach straight past the fakes and SIGTERM a live llama-server or mtplx.
    The supervisor takes its stopper as an argument for exactly this reason.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest

# The repo root, so `inference` and `tests` both import from a bare checkout.
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from inference import cli, models, server  # noqa: E402
from inference.machine import detect_machine  # noqa: E402
from inference.hosts import LocalHost  # noqa: E402
from tests.fakes import FakeLauncher  # noqa: E402

REAL_PORTS = {8083, 8084}


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    assert port not in REAL_PORTS, "refusing to test against the real ports"
    return port


class Harness:
    """An engine, in-process, talking to a fake model server."""

    def __init__(self, args, launcher):
        self.engine = server.Engine(args)
        self.launcher = launcher
        self.engine.supervisor.host = LocalHost(launcher=launcher.launch,
                                                stopper=launcher.stop_all,
                                                puller=launcher.pull,
                                                engine_check=lambda spec: None)

        from inference import usage
        self.engine.attach_usage(usage.UsageDB(args.usage_db))

        # Same wiring as cli.run(), so a scenario that drives the config page
        # is going through the handlers the browser goes through.
        if not args.no_ui:
            from inference import dashboard
            self.engine.dashboard = dashboard.Dashboard(self.engine)

        server.EngineHandler.engine = self.engine
        self.server = server.EngineServer(("127.0.0.1", args.port),
                                          server.EngineHandler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{args.port}"

    # ----------------------------------------------------
    def get(self, path: str):
        with urllib.request.urlopen(self.base + path, timeout=30) as response:
            return json.load(response)

    def request(self, path: str, payload: dict, headers: dict | None = None,
                timeout: float = 60):
        body = json.dumps(payload).encode()
        request = urllib.request.Request(
            self.base + path, data=body,
            headers={"Content-Type": "application/json", **(headers or {})},
            method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()

    def chat(self, model: str, stream: bool = False, headers=None, **extra):
        payload = {"model": model,
                   "messages": [{"role": "user", "content": "hi"}],
                   "stream": stream, **extra}
        return self.request("/v1/chat/completions", payload, headers)

    def rows(self, sql: str, params=()):
        return self.engine.usage._query(sql, params)

    def settle(self, expected: int, timeout: float = 10.0) -> None:
        """Wait for `expected` rows to land.

        The engine writes the trace and the stats row *after* the response has
        gone back to the client - a client should never wait on a database - so
        a test that queries the moment a request returns can race the insert.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.rows("SELECT COUNT(*) AS n FROM requests")[0]["n"] >= expected:
                return
            time.sleep(0.05)
        raise AssertionError(
            f"only {self.rows('SELECT COUNT(*) AS n FROM requests')[0]['n']} "
            f"rows after {timeout}s, wanted {expected}")

    def last_trace(self, timeout: float = 10.0) -> dict:
        """The newest trace record, once the engine has finished writing it.

        The trace is written after the response has gone back to the client,
        so reading it the moment a request returns races the write - rarely on
        a fast Mac, often in a slower container. A request stops counting as
        in flight only after its record is written, so wait for that.
        """
        supervisor = self.engine.supervisor
        deadline = time.monotonic() + timeout
        while True:
            with supervisor._lock:
                inflight = supervisor._inflight
            if inflight == 0:
                break
            if time.monotonic() > deadline:
                raise AssertionError(f"{inflight} request(s) still in flight "
                                     f"after {timeout}s")
            time.sleep(0.02)
        with open(self.engine.tracer.path) as handle:
            return json.loads(handle.readlines()[-1])

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.engine.usage.close()


@pytest.fixture
def make_harness(tmp_path):
    made = []

    def build(*, engine_flags=(), fail=None, ready_after=0.0, api_key="none",
              sandbox=False, ui=False, **server_kwargs):
        # The dashboard is off unless a scenario is about it: with it on, every
        # unmatched GET is answered by the static handler instead of falling
        # through to the relay, which is a different thing to be testing.
        # Sandbox off unless a scenario is about the sandbox. There is no real
        # process to confine - the launcher is a fake - and leaving it on made
        # the suite depend on which models this particular Mac had downloaded,
        # so a fresh checkout would fail on the weights guard rather than on
        # anything it was testing.
        upstream = free_port()
        launcher = FakeLauncher(upstream, api_key=api_key, fail=fail,
                                ready_after=ready_after, **server_kwargs)
        # Through the real parser, so a scenario cannot drift from the flags
        # the command line actually accepts.
        args = cli.build_parser().parse_args([
            "engine",
            "--port", str(free_port()),
            "--upstream-port", str(upstream),
            "--trace-dir", str(tmp_path / "traces"),
            "--log-dir", str(tmp_path / "logs"),
            "--usage-db", str(tmp_path / "usage.db"),
            # Point at a throwaway file. Without this the suite reads and
            # writes the real settings.json in the working directory, so a test
            # run silently rewrites whatever you had configured.
            "--settings", str(tmp_path / "settings.json"),
            "--api-key", api_key,
            *([] if ui else ["--no-ui"]), "--quiet",
            "--start-timeout", "20",
            *([] if sandbox else ["--no-sandbox"]),
            *engine_flags,
        ])
        harness = Harness(args, launcher)
        made.append(harness)
        return harness

    yield build
    for harness in made:
        harness.close()


@pytest.fixture
def harness(make_harness):
    return make_harness()


@pytest.fixture
def machine():
    return detect_machine()


@pytest.fixture
def registry():
    return models.load_all()


@pytest.fixture
def infos(machine, registry):
    return models.describe(machine, registry)

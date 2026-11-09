"""Model servers on another machine, driven over SSH.

For an engine that runs where the models cannot - a Linux container, a pod.
Every decision that depends on the Mac (which llama.cpp build, whether the
weights are on disk, which process owns the port) is made on the Mac by the
checkout's own `inference` CLI; this only carries the question and the answer.

Standard library only, like the rest: it shells out to the ssh binary rather
than taking a dependency on an SSH library.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass, field, fields


class RemoteError(Exception):
    """The model host could not be asked, or answered with something unusable."""


def ssh_runner(argv: list[str], timeout: float) -> tuple[int, str, str]:
    result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    return result.returncode, result.stdout, result.stderr


@dataclass(frozen=True)
class ModelHostConfig:
    """`model_host` from config.json."""

    ssh: str                  # user@host for ssh
    address: str              # where the engine reaches the model port
    checkout: str             # this repo, on the Mac
    python: str = "python3"
    ssh_port: int = 22
    identity: str | None = None
    known_hosts: str | None = None
    # A non-interactive SSH login on a Mac has no Homebrew on its PATH, and the
    # `inference` console script that is on it belongs to another Python.
    path_prefix: str = "/opt/homebrew/bin:/usr/local/bin"
    machine: str = "unknown"  # profile to fall back on while the Mac is unreachable

    @classmethod
    def from_dict(cls, raw: dict) -> "ModelHostConfig":
        # Refuse typos rather than ignore them: a misspelt `identity` means
        # ssh silently offers no key, and every launch fails as "permission
        # denied" with nothing pointing at the config.
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"model_host: unknown keys {unknown}")
        return cls(**raw)


def _run_flags(opts) -> list[str]:
    """Options as the `inference run` flags that reproduce them on the Mac."""
    flags = ["--port", str(opts.port), "--bind", opts.bind]
    if not opts.sandbox:
        flags.append("--no-sandbox")
    if opts.ctx:
        flags += ["--ctx", str(opts.ctx)]
    if opts.thinking is True:
        flags.append("--thinking")
    elif opts.thinking is False:
        flags.append("--no-thinking")
    if opts.spec_decode is True:
        flags.append("--spec")
    elif opts.spec_decode is False:
        flags.append("--no-spec")
    if opts.reasoning_effort:
        flags += ["--reasoning-effort", opts.reasoning_effort]
    return flags


class RemoteHost:
    """The LocalHost interface, answered by a Mac over SSH."""

    remote = True

    def __init__(self, config: ModelHostConfig, runner=ssh_runner, on_drift=None,
                 local_sha: str | None = None, timeout: float = 30.0,
                 pull_timeout: float = 6 * 3600.0):
        self.config = config
        self.address = config.address
        self._run = runner
        self.on_drift = on_drift
        self.local_sha = local_sha
        self.timeout = timeout
        self.pull_timeout = pull_timeout

    # --------------------------------------------------------
    # Commands
    # --------------------------------------------------------
    def ssh_argv(self, remote_command: str) -> list[str]:
        c = self.config
        argv = ["ssh", "-p", str(c.ssh_port),
                "-o", "BatchMode=yes",
                "-o", "ConnectTimeout=10",
                # One connection for everything. Liveness is asked every couple
                # of seconds while a model serves; a handshake each time is
                # latency on every request.
                "-o", "ControlMaster=auto",
                "-o", "ControlPath=/tmp/inference-ssh-%C",
                "-o", "ControlPersist=120"]
        if c.identity:
            argv += ["-i", c.identity, "-o", "IdentitiesOnly=yes"]
        if c.known_hosts:
            argv += ["-o", f"UserKnownHostsFile={c.known_hosts}",
                     "-o", "StrictHostKeyChecking=yes"]
        return argv + [c.ssh, remote_command]

    def inference_command(self, args: list[str]) -> str:
        c = self.config
        return (f"cd {shlex.quote(c.checkout)} && "
                f"PATH={shlex.quote(c.path_prefix)}:$PATH "
                f"{shlex.quote(c.python)} -m inference "
                + " ".join(shlex.quote(a) for a in args))

    def call(self, args: list[str], timeout: float | None = None) -> str:
        """`inference <args>` on the Mac; its stdout, or RemoteError."""
        argv = self.ssh_argv(self.inference_command(args))
        try:
            code, out, err = self._run(argv, timeout or self.timeout)
        except (OSError, subprocess.SubprocessError) as exc:
            raise RemoteError(f"ssh to {self.config.ssh} failed: {exc}") from exc
        if code != 0:
            detail = " | ".join((err or out).strip().splitlines()[-5:])
            raise RemoteError(f"`inference {' '.join(args)}` on {self.config.ssh} "
                              f"exited {code}: {detail}")
        return out

    def call_json(self, args: list[str], timeout: float | None = None) -> dict:
        out = self.call(args, timeout)
        # The last line. Anything before it is a warning the CLI printed.
        lines = [line for line in out.splitlines() if line.strip()]
        try:
            return json.loads(lines[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            raise RemoteError(f"`inference {' '.join(args)}` did not answer "
                              f"with JSON: {out.strip()[:200]!r}") from exc

    # --------------------------------------------------------
    # The host interface
    # --------------------------------------------------------
    def prepare(self, module, machine, opts):
        # No llama.cpp capability gate here - it interrogates the llama-server
        # binary, which is on the Mac. `run --plan` and `--detach` apply it there.
        return module.spec(machine, opts)

    def preflight(self, alias, spec, opts) -> str | None:
        try:
            plan = self.call_json(["run", alias, "--plan", *_run_flags(opts)])
        except RemoteError as exc:
            return str(exc)
        self._check_drift(plan.get("sha"))
        if plan.get("ok"):
            return None
        return plan.get("reason") or f"{self.config.ssh} refused to launch {alias}"

    def needs_pull(self, alias, spec, opts) -> bool:
        try:
            plan = self.call_json(["run", alias, "--plan", *_run_flags(opts)])
        except RemoteError:
            return False  # preflight asks again, and reports why
        return bool(plan.get("needs_pull"))

    def pull(self, alias, spec, opts, say=print) -> None:
        """`inference pull` on the Mac - where the weights have to be.

        A long call: a first pull is tens of GB. It is the Mac's own CLI doing
        the work, so the progress lands in its output, not here.
        """
        say(f"     pulling {alias} on {self.config.ssh} (this can take a while)")
        self.call(["pull", alias], timeout=self.pull_timeout)

    def launch(self, alias, spec, machine, opts, log_dir):
        # log_dir is the engine's; the server's log is written on the Mac, in
        # the checkout, where `inference run` would put it.
        started = time.monotonic()
        try:
            info = self.call_json(["run", alias, "--detach", "--log-dir", "logs",
                                   *_run_flags(opts)])
            pid = int(info["pid"])
        except (RemoteError, KeyError, TypeError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        return RemoteProcess(host=self, alias=alias,
                             engine=info.get("engine") or "unknown",
                             port=opts.port, pid=pid,
                             log_path=info.get("log_path") or "",
                             started_at=started)

    def stop_all(self, quiet: bool = False, only_port: int | None = None):
        args = ["stop", "--quiet"]
        if only_port is not None:
            args += ["--port", str(only_port)]
        try:
            self.call(args, timeout=60)
        except RemoteError as exc:
            print(f"  model host: {exc}", flush=True)
        return 0

    def status(self, port: int, pid: int | None = None) -> dict | None:
        args = ["status", "--port", str(port)]
        if pid is not None:
            args += ["--pid", str(pid)]
        try:
            return self.call_json(args, timeout=15)
        except RemoteError:
            return None

    def server_pid(self, port: int) -> int | None:
        status = self.status(port)
        return status.get("pid") if status else None

    def rss_mb(self, port: int) -> float | None:
        status = self.status(port)
        return status.get("rss_mb") if status else None

    def machine(self) -> dict:
        return self.call_json(["machine"], timeout=15)

    def tail(self, path: str, lines: int) -> str:
        argv = self.ssh_argv(f"tail -n {int(lines)} {shlex.quote(path)}")
        try:
            code, out, _ = self._run(argv, 15)
        except (OSError, subprocess.SubprocessError):
            return ""
        return out if code == 0 else ""

    # --------------------------------------------------------
    def _check_drift(self, mac_sha: str | None) -> None:
        engine_sha = self.local_sha
        if not (self.on_drift and mac_sha and engine_sha):
            return
        # Short shas differ in length for the same commit; a prefix match is
        # the same commit.
        if mac_sha.startswith(engine_sha) or engine_sha.startswith(mac_sha):
            return
        self.on_drift(engine_sha, mac_sha)


@dataclass
class RemoteProcess:
    """A model server on the model host - ServerProcess's remote twin."""

    host: RemoteHost
    alias: str
    engine: str
    port: int
    pid: int
    log_path: str
    started_at: float
    argv: list = field(default_factory=lambda: ["remote"])
    _checked: tuple = (0.0, True)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    CACHE_SECONDS = 2.0

    def is_alive(self) -> bool:
        with self._lock:
            when, value = self._checked
            if time.monotonic() - when < self.CACHE_SECONDS:
                return value
        status = self.host.status(self.port, pid=self.pid)
        # Unreachable is not dead. A Mac that drops one SSH call is still
        # serving; calling that an exit would tear down a healthy session.
        value = True if status is None else bool(status.get("alive"))
        with self._lock:
            self._checked = (time.monotonic(), value)
        return value

    def exit_code(self) -> int | None:
        # The real code went with the detached server's parent. -1 means it
        # has exited; the log tail says why.
        return None if self.is_alive() else -1

    def log_tail(self, lines: int = 30) -> str:
        return self.host.tail(self.log_path, lines) if self.log_path else ""

    def terminate(self, grace: float = 10.0) -> None:
        if self.is_alive():
            self.host.stop_all(quiet=True, only_port=self.port)

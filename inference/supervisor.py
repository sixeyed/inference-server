"""One model server, supervised.

The engine's job is to make "which model" a property of the request rather than
of what you last typed. That means something has to own the model server's
lifecycle: start it on demand, know when it is actually ready, swap it out for a
different one, and notice when it dies. That is this.

Only one model runs at a time - the constraint the whole harness is built on,
since each model is tuned to fill the machine for a single session.

Swapping kills. A request for a different model stops the current server
immediately and whatever was streaming dies with it, because a model switch
means the old model's answer is no longer wanted. The alternative - draining
in-flight work first - lets one wedged socket hold the box hostage, and the
in-flight turn belongs to a model the user just walked away from.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum


from . import models
from .config import DEFAULT_API_KEY, DEFAULT_PORT
from .console import grey, yellow
from .machine import Options, Unsupported
from .hosts import LocalHost
from .process import ServerProcess, upstream_id
from .specs import LlamaServerSpec, MlxServerSpec, MtplxServerSpec


class State(str, Enum):
    IDLE = "idle"
    STARTING = "starting"
    READY = "ready"
    STOPPING = "stopping"
    FAILED = "failed"


class StartupFailed(Exception):
    """A model server did not come up. Carries what the client should be told."""

    def __init__(self, alias: str, reason: str, log_tail: str = "",
                 log_path: str | None = None):
        self.alias = alias
        self.reason = reason
        self.log_tail = log_tail
        self.log_path = log_path
        super().__init__(f"{alias} failed to start: {reason}")

    def message(self) -> str:
        """The body of the 503. Cline and Copilot both render error.message, so
        the server's own last words are the only place a user will ever see
        `unknown flag --spec-type`."""
        parts = [f"{self.alias} failed to start: {self.reason}"]
        if self.log_tail:
            parts.append(self.log_tail)
        if self.log_path:
            parts.append(f"full log: {self.log_path}")
        return "\n\n".join(parts)


@dataclass
class Current:
    """The model that is serving, however it got there."""

    alias: str | None            # None => adopted, serving an id we don't know
    engine: str
    upstream_model_id: str       # what the upstream expects in `model`
    spec: object | None = None
    process: ServerProcess | None = None   # None => we didn't start it
    ready_at: float = 0.0
    load_ms: float = 0.0
    epoch: int = 0

    @property
    def adopted(self) -> bool:
        return self.process is None

    @property
    def log_path(self) -> str | None:
        return self.process.log_path if self.process else None


# ============================================================
# Readiness
# ============================================================
def _get(url: str, api_key: str | None, timeout: float):
    """(status, body) for a probe. status None means nothing is listening."""
    request = urllib.request.Request(url)
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, b""
    except (urllib.error.URLError, TimeoutError, OSError):
        return None, b""


def probe(spec, port: int, api_key: str, timeout: float = 2.0,
          host: str = "127.0.0.1") -> tuple[str, str | None]:
    """Is the server on `port` ready? -> ("ready"|"loading"|"down", warning).

    Each engine answers differently and the differences are load-bearing:

    llama-server  /health 200 when up, 503 while loading. Exact and cheap.

    mtplx         /health, but it *checks the key* - verified against a live
                  server, the same request without a bearer is a 401. So the
                  probe carries the key or it would never see a healthy server.
                  Its body is {"ok":true,...} where llama-server's is
                  {"status":"ok"}, so this looks at the status code only.

    mlx_lm.server /health may not exist; fall back to /v1/models and require
                  the served id to match. Any HTTP answer means the socket is
                  bound, but only the id confirms the weights finished loading.
    """
    base = f"http://{host}:{port}"

    if isinstance(spec, MtplxServerSpec):
        status, _ = _get(f"{base}/health", api_key, timeout)
        if status == 200:
            return "ready", None
        if status in (401, 403):
            # The server is up; our key is wrong. Saying "ready" here is
            # deliberate - the alternative is waiting out the whole start
            # timeout while a perfectly healthy server sits there, and
            # "loading forever" and "wrong key" look identical from outside.
            return "ready", (
                f"{status} from MTPLX /health - the server is up but rejected "
                f"our API key. Requests will fail the same way."
            )
        if status is None:
            return "down", None
        return "loading", None

    if isinstance(spec, MlxServerSpec):
        status, body = _get(f"{base}/health", api_key, timeout)
        if status == 200:
            return "ready", None
        if status is None:
            return "down", None
        # 404 => this build has no /health. Ask what it is serving instead.
        status, body = _get(f"{base}/v1/models", api_key, timeout)
        if status == 200:
            try:
                served = (json.loads(body).get("data") or [{}])[0].get("id")
            except (json.JSONDecodeError, IndexError, AttributeError):
                return "loading", None
            if served and served == getattr(spec, "hf_repo", None):
                return "ready", None
        return "loading", None

    # llama-server, and anything new that behaves like it.
    status, _ = _get(f"{base}/health", api_key, timeout)
    if status == 200:
        return "ready", None
    if status is None:
        return "down", None
    return "loading", None


def served_model_id(port: int, api_key: str, timeout: float = 2.0,
                    host: str = "127.0.0.1") -> str | None:
    """What the server on `port` says it is serving, for adoption."""
    status, body = _get(f"http://{host}:{port}/v1/models", api_key, timeout)
    if status != 200:
        return None
    try:
        return (json.loads(body).get("data") or [{}])[0].get("id")
    except (json.JSONDecodeError, IndexError, AttributeError):
        return None


# ============================================================
# The supervisor
# ============================================================
class Supervisor:
    def __init__(self, machine, registry, infos, opts: Options,
                 port: int = DEFAULT_PORT, log_dir: str = "logs",
                 api_key: str = DEFAULT_API_KEY,
                 start_timeout: float = 1800.0,
                 fail_cooldown: float = 30.0,
                 leave_running: bool = False,
                 quiet: bool = False,
                 host=None, pull: bool = True,
                 on_load=None, on_unload=None, on_failure=None):
        self.machine = machine
        self.registry = registry
        self.infos = {info.alias: info for info in infos}
        self.opts = opts
        self.port = port
        self.log_dir = log_dir
        self.api_key = api_key
        self.start_timeout = start_timeout
        self.fail_cooldown = fail_cooldown
        self.leave_running = leave_running
        self.quiet = quiet

        # Where models run. LocalHost is this machine; RemoteHost drives a Mac
        # over SSH. Tests hand in a LocalHost with a fake launcher and stopper.
        self.host = host or LocalHost()
        # Fetch missing weights before a launch rather than refusing it. The
        # alias being pulled, while one is, for /health and the dashboard.
        self.pull = pull
        self.pulling: str | None = None
        # Set by shutdown() and unload() to stop a pull in progress. The pull
        # runs under the swap lock, so without this Ctrl-C would wait out a
        # 30GB download before the engine could stop.
        self._cancel_pull = threading.Event()

        self.on_load = on_load
        self.on_unload = on_unload
        self.on_failure = on_failure

        self.state = State.IDLE
        self.current: Current | None = None
        self._swap = threading.Lock()
        self._lock = threading.RLock()
        self._inflight = 0
        self._epoch = 0
        self._failed: dict[str, tuple[float, StartupFailed]] = {}
        self._starting_since = 0.0
        self._last_liveness = (0.0, True)

    # --------------------------------------------------------
    # What the HTTP layer calls
    # --------------------------------------------------------
    def acquire(self, alias: str) -> Current:
        """Guarantee `alias` is serving, then register one in-flight request.

        Blocks through a cold start - 30s to 2min warm, longer on a first
        download. That is the deal: the client waits rather than being told to
        come back, because no coding agent retries a 503 gracefully.
        """
        with self._lock:
            if (self.state is State.READY and self.current
                    and self.current.alias == alias and self._alive()):
                self._inflight += 1
                return self.current

        with self._swap:
            # Re-check under the swap lock. Without this, two requests arriving
            # together at session start - Copilot fires a title-generation call
            # alongside the first chat turn - each launch a 30GB server.
            with self._lock:
                if (self.state is State.READY and self.current
                        and self.current.alias == alias and self._alive()):
                    self._inflight += 1
                    return self.current

            failure = self._cooled_down(alias)
            if failure is not None:
                raise failure

            try:
                current = self._start(alias)
            except StartupFailed as exc:
                with self._lock:
                    # Only FAILED when nothing is serving. A launch refused
                    # before the swap leaves the previous model running, and
                    # calling that state "failed" would be a lie - the engine
                    # is fine and so is the model it is still answering with.
                    if self.current is None:
                        self.state = State.FAILED
                self._failed[alias] = (time.monotonic(), exc)
                if self.on_failure:
                    # A load that failed is the one you most want a record of,
                    # and the log file alone means grepping a directory to find
                    # out what happened.
                    self.on_failure(alias, exc)
                raise

            with self._lock:
                self._inflight += 1
                return current

    def acquire_current(self) -> Current | None:
        """Register one in-flight request on whatever is serving, or None.

        For relays that must never start a model. Checking status() and then
        calling acquire() with the alias it named is a race: a swap that lands
        in between makes acquire() swap *back*, killing the new model and its
        stream for the sake of a /props poll.
        """
        with self._lock:
            if (self.state is State.READY and self.current
                    and self._alive()):
                self._inflight += 1
                return self.current
        return None

    def release(self) -> None:
        with self._lock:
            self._inflight = max(0, self._inflight - 1)

    @property
    def epoch(self) -> int:
        with self._lock:
            return self._epoch

    def preempted_since(self, epoch: int) -> bool:
        """Did a swap happen while a request was in flight?

        This is what separates "the user switched model" from "the server
        crashed". Both look like the upstream vanishing mid-stream, and a log
        that conflates them makes the next real crash invisible.
        """
        return self.epoch != epoch

    # --------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------
    def _start(self, alias: str) -> Current:
        module = self.registry.get(alias)
        if module is None:
            raise StartupFailed(alias, "no such model")

        # Build the spec before stopping anything: an Unsupported model or a
        # missing binary should not cost you the server you already had.
        try:
            spec = self.host.prepare(module, self.machine, self.opts)
        except Unsupported as exc:
            raise StartupFailed(alias, str(exc)) from exc
        except SystemExit as exc:
            # find_llama_server() and friends raise SystemExit, which is right
            # for a CLI and fatal for a server.
            raise StartupFailed(alias, str(exc) or "engine executable not found") from exc

        # Missing weights are fetched now, while the current model is still
        # serving: a 30GB download is no reason to stop answering. The server
        # is then launched sandboxed as usual - it never downloads for itself.
        if self.pull and self.host.needs_pull(alias, spec, self.opts):
            self._fetch(alias, spec)

        # A launch known to fail - weights not on disk behind the sandbox, a
        # model host that cannot be reached - is refused here, while the
        # current model is still serving. Discovering it after the kill costs a
        # working session to learn something we could have known beforehand.
        refusal = self.host.preflight(alias, spec, self.opts)
        if refusal:
            hint = ""
            if "not on disk" in refusal:
                hint = (f"Fetch it first:\n"
                        f"  inference pull {alias}\n"
                        f"or set \"auto_pull\": true in config.json so the "
                        f"engine fetches missing weights itself.")
            raise StartupFailed(alias, refusal, log_tail=hint)

        previous = self.current.alias if self.current else None
        self._stop()

        if not self.quiet:
            if previous:
                print(f"  -> switching to {alias} (was {previous})", flush=True)
            else:
                print(f"  -> starting {alias}", flush=True)

        with self._lock:
            self.state = State.STARTING
            self._starting_since = time.monotonic()

        started = time.monotonic()
        try:
            process = self.host.launch(alias, spec, self.machine, self.opts, self.log_dir)
        except SystemExit as exc:
            with self._lock:
                self.state = State.IDLE
            raise StartupFailed(alias, str(exc) or "could not launch") from exc

        if not self.quiet:
            print(grey(f"     {process.engine} -> {process.log_path}"), flush=True)

        tailer = threading.Thread(target=self._tail, args=(process,), daemon=True)
        tailer.start()

        self._wait_ready(alias, spec, process, started)

        load_ms = (time.monotonic() - started) * 1000
        with self._lock:
            self.current = Current(
                alias=alias,
                engine=process.engine,
                upstream_model_id=self._upstream_id(spec, alias),
                spec=spec,
                process=process,
                ready_at=time.time(),
                load_ms=load_ms,
                epoch=self._epoch,
            )
            self.state = State.READY
            self._failed.pop(alias, None)

        if not self.quiet:
            print(f"     ready in {load_ms / 1000:.1f}s", flush=True)
        if self.on_load:
            self.on_load(self.current)
        return self.current

    def _fetch(self, alias: str, spec) -> None:
        if not self.quiet:
            print(f"  -> pulling {alias} - weights not on disk", flush=True)
        say = (lambda line: None) if self.quiet else (lambda line: print(grey(line), flush=True))
        with self._lock:
            self.pulling = alias
        self._cancel_pull.clear()
        try:
            self.host.pull(alias, spec, self.opts, say=say, cancel=self._cancel_pull)
        except Exception as exc:  # PullError, RemoteError, disk full, ...
            raise StartupFailed(alias, f"could not fetch the weights: {exc}") from exc
        finally:
            with self._lock:
                self.pulling = None

    def _wait_ready(self, alias, spec, process: ServerProcess, started: float) -> None:
        deadline = started + self.start_timeout
        warned = False
        heartbeat = started

        while time.monotonic() < deadline:
            # Check the process before the socket. A bad flag kills
            # llama-server in under a second, and without this you sit out the
            # entire timeout staring at something that is already dead.
            code = process.exit_code()
            if code is not None:
                elapsed = time.monotonic() - started
                raise StartupFailed(
                    alias,
                    f"exited with code {code} after {elapsed:.1f}s",
                    log_tail=process.log_tail(30),
                    log_path=process.log_path,
                )

            status, warning = probe(spec, self.port, self.api_key, host=self.host.address)
            if warning and not warned:
                print(yellow(f"     WARNING: {warning}"), flush=True)
                warned = True
            if status == "ready":
                return

            now = time.monotonic()
            if not self.quiet and now - heartbeat >= 15:
                heartbeat = now
                print(grey(f"     starting {alias} ... {now - started:.0f}s"), flush=True)
            time.sleep(0.5)

        process.terminate()
        raise StartupFailed(
            alias,
            f"not ready after {self.start_timeout:.0f}s",
            log_tail=process.log_tail(30),
            log_path=process.log_path,
        )

    def _stop(self) -> None:
        """Stop whatever is serving, and wait for the port to come free."""
        with self._lock:
            current, self.current = self.current, None
            if self.state is not State.IDLE:
                self.state = State.STOPPING
            # Bump the generation here, at the moment the old server goes away
            # - not when the replacement is ready. A request streaming from the
            # server we are about to kill sees its connection drop within
            # milliseconds, and it has to be able to tell right then that this
            # was a deliberate swap rather than a crash. The new model may not
            # be ready for another two minutes.
            if current is not None:
                self._epoch += 1

        adopted = current is not None and current.adopted
        if current is None and self.host.server_pid(self.port) is None:
            with self._lock:
                self.state = State.IDLE
            return

        if adopted and self.leave_running:
            # We did not start it, and we were told not to end things. But the
            # caller is swapping, so it has to go regardless - the alternative
            # is two models in memory at once, which the machine cannot do.
            pass

        started = time.monotonic()
        if not self.quiet and current is not None:
            print(grey(f"     stopping {current.alias or 'adopted server'} ..."),
                  flush=True)

        self.host.stop_all(quiet=True, only_port=self.port)
        if current is not None and current.process is not None:
            current.process.terminate()

        # Binding a new 30GB model before the old one has released its unified
        # memory either OOMs or fails the bind, and the error blames the wrong
        # model. Wait for the socket to actually go.
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if self.host.server_pid(self.port) is None:
                break
            time.sleep(0.25)

        with self._lock:
            self.state = State.IDLE
        if not self.quiet and current is not None:
            print(grey(f"     stopped in {time.monotonic() - started:.1f}s"), flush=True)
        if self.on_unload and current is not None:
            self.on_unload(current)

    def unload(self) -> None:
        self._cancel_pull.set()
        with self._swap:
            self._stop()

    def shutdown(self) -> None:
        """Stop the model on the way out - unless we adopted it, or were asked
        not to. 30GB left resident after the engine quits is the surprising
        outcome; killing something we didn't start is the rude one."""
        self._cancel_pull.set()
        with self._lock:
            current = self.current
        if current is not None and current.adopted:
            return
        if self.leave_running:
            return
        with self._swap:
            self._stop()

    # --------------------------------------------------------
    # Adoption
    # --------------------------------------------------------
    def adopt(self, index: dict[str, str]) -> Current | None:
        """Pick up a server someone started by hand with `inference run`.

        Anything you started deliberately shouldn't be executed on sight by the
        first health probe a client happens to send.
        """
        with self._lock:
            if self.state is not State.IDLE or self.current is not None:
                return None

        if self.host.server_pid(self.port) is None:
            return None

        served = served_model_id(self.port, self.api_key, host=self.host.address)
        if served is None:
            return None

        alias = models.resolve(served, index)
        info = self.infos.get(alias) if alias else None

        with self._lock:
            self._epoch += 1
            self.current = Current(
                alias=alias,
                engine=info.engine if info else "unknown",
                upstream_model_id=served,
                spec=None,
                process=None,
                ready_at=time.time(),
                epoch=self._epoch,
            )
            self.state = State.READY

        if not self.quiet:
            label = alias or f"{served} (unrecognised)"
            print(grey(f"  adopted {label} already serving on {self.port}"), flush=True)
        # Adopted counts as loaded for anything watching - otherwise a server
        # you started by hand records no memory series and no load row, which
        # is exactly the session you are most likely to be measuring.
        if self.on_load:
            self.on_load(self.current)
        return self.current

    # --------------------------------------------------------
    # Introspection
    # --------------------------------------------------------
    def status(self) -> dict:
        # Check liveness here too, not only on the request path. A server that
        # exits on its own - llama.cpp does, on an unparseable generation -
        # otherwise leaves the dashboard reporting "ready" against a pid that
        # is gone, until someone happens to send a request.
        died = None
        with self._lock:
            if self.state is State.READY and not self._alive():
                self.state = State.IDLE
                died, self.current = self.current, None
        # Outside the lock: on_unload writes to the database. Without it the
        # load row never gets its unloaded_at, and the memory sampler goes on
        # polling a server that has gone.
        if died is not None and self.on_unload:
            self.on_unload(died)
        with self._lock:
            current = self.current
            state = self.state
            inflight = self._inflight
            starting = self._starting_since
        payload = {
            "state": state.value,
            "port": self.port,
            "inflight": inflight,
            "model": current.alias if current else None,
            "engine": current.engine if current else None,
            "adopted": current.adopted if current else None,
            "upstream_model": current.upstream_model_id if current else None,
            "load_ms": round(current.load_ms, 1) if current else None,
            "uptime_s": round(time.time() - current.ready_at, 1) if current else None,
            "log": current.log_path if current else None,
            "pid": self.host.server_pid(self.port),
            "epoch": current.epoch if current else self.epoch,
            "pulling": self.pulling,
        }
        if state is State.STARTING:
            payload["starting_s"] = round(time.monotonic() - starting, 1)
            payload["log_tail"] = (
                current.process.log_tail(12) if current and current.process else ""
            )
        return payload

    def rss_mb(self) -> float | None:
        """Resident size of whatever is serving.

        On Apple Silicon the weights sit in unified memory owned by the
        process, so RSS is the real footprint - weights plus KV cache plus
        compute buffers - and that is what decides whether a model fits.

        Sampled from the pid that owns the socket, not from the process we
        spawned: `mtplx serve` re-execs, so those are different processes.
        """
        return self.host.rss_mb(self.port)

    # --------------------------------------------------------
    # Internals
    # --------------------------------------------------------
    def _upstream_id(self, spec, alias: str) -> str:
        """The id the *server* answers to, which is not always the alias."""
        return upstream_id(spec, alias)

    def _alive(self) -> bool:
        """Cheap liveness, because somebody will run `inference stop` while this is up."""
        current = self.current
        if current is None:
            return False
        if current.process is not None:
            return current.process.exit_code() is None
        # Adopted: no handle, so ask the OS - but that forks lsof, far too
        # expensive per request. Cache it briefly.
        now = time.monotonic()
        when, value = self._last_liveness
        if now - when < 5.0:
            return value
        value = self.host.server_pid(self.port) is not None
        self._last_liveness = (now, value)
        return value

    def _cooled_down(self, alias: str) -> StartupFailed | None:
        """Re-raise a recent failure instead of relaunching a doomed model.

        Without this a retrying client restarts a 35GB download six times a
        minute. Sticky per alias only - asking for a different model clears it.
        """
        entry = self._failed.get(alias)
        if entry is None:
            return None
        when, failure = entry
        if time.monotonic() - when < self.fail_cooldown:
            return failure
        del self._failed[alias]
        return None

    def _tail(self, process: ServerProcess) -> None:
        """Echo the server's log while it loads.

        Two minutes of silence on a held-open request is indistinguishable from
        a hang, and the whole cold-start design rests on trusting the block.
        """
        if self.quiet:
            return
        try:
            handle = open(process.log_path, "r", errors="replace")
        except OSError:
            return
        with handle:
            while True:
                with self._lock:
                    if self.state is not State.STARTING:
                        return
                line = handle.readline()
                if not line:
                    time.sleep(0.2)
                    continue
                print(grey(f"     | {line.rstrip()}"), flush=True)

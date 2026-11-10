"""Starting a model server, watching it, and stopping it.

Two ways in, deliberately. `serve()` runs one in the foreground with the full
banner, which is what the `run` command does. `launch()` starts one in the
background and hands back a handle, which is what the engine's supervisor needs.
Both build the command the same way, because a model started through the router
has to be the same server you get from the command line - otherwise nothing
measured through one applies to the other.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime

from . import weights
from .config import DEFAULT_LISTEN_PORT, DEFAULT_PORT, NO_AUTH_KEY, SANDBOX_PROFILE
from .console import blue, cyan, green, grey, quote, red, yellow
from .engines import resolve_engine, resolve_spec_decode
from .machine import Machine, Options, Unsupported
from .specs import (
    LlamaServerSpec,
    MlxServerSpec,
    MtplxServerSpec,
    find_in_hf_cache,
)

# Process names to sweep when stopping. Matching on name rather than on a pid is
# what catches a server started by hand, and a survivor of a re-exec.
SERVER_PATTERNS = ["llama-server", "mlx_lm.server", "mtplx"]

# The engine is found by the port it is listening on, not by name. Name-matching
# "inference" is actively dangerous - it also matches macOS's own siriinferenced,
# among others - and there is no name that is both specific enough to be safe and
# general enough to catch both `python -m inference` and the installed console
# script. The socket is unambiguous.
#
# It is not stopped by default either way: the everyday move is "free the RAM,
# keep the front door" - the engine notices its child died, drops to idle, and
# reloads on the next request.


def find_server_pid(port: int) -> int | None:
    """PID listening on `port`, if it's on this machine.

    Lives here rather than in tests/ because the engine needs it three times
    over: to notice a server someone started by hand, to wait for a port to
    come free before binding the next model to it, and to sample memory.

    That last one is why it matters which pid this is. `mtplx serve` re-execs
    into `python -m mtplx.server.openai`, so the process holding the weights is
    not the process we spawned - asking the OS who owns the socket is the only
    way to get the right one.
    """
    try:
        result = subprocess.run(["lsof", "-ti", f":{port}", "-sTCP:LISTEN"],
                                capture_output=True, text=True, timeout=10)
        pids = [int(p) for p in result.stdout.split() if p.strip().isdigit()]
        return pids[0] if pids else None
    except (subprocess.SubprocessError, ValueError):
        return None


def rss_mb_of(pid: int) -> float | None:
    """Resident size of `pid` in MB. On Apple Silicon that is weights plus KV
    cache plus compute buffers - the number that decides whether a model fits."""
    try:
        result = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)],
                                capture_output=True, text=True, timeout=5)
        return int(result.stdout.strip()) / 1024
    except (subprocess.SubprocessError, ValueError):
        return None


def pid_alive(pid: int) -> bool:
    """Is `pid` still running? For a server that has not bound its port yet -
    one still loading or downloading - which the port alone would call dead."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def checkout_sha() -> str | None:
    """The short git sha of this checkout, or None outside one. An engine in a
    container compares it with its own build, because model files can differ."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        result = subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def upstream_id(spec, alias: str) -> str:
    """The id the *server* answers to, which is not always the alias."""
    for attr in ("alias", "model_id"):
        value = getattr(spec, attr, None)
        if value:
            return value
    # mlx_lm.server has no --alias: it only knows the full repo id.
    return getattr(spec, "hf_repo", None) or alias


def prepare(module, machine: Machine, opts: Options, warn=None):
    """Apply the llama.cpp capability gate, then build the model's spec.

    The gate interrogates `llama-server --help`, so it only means anything for
    llama.cpp models - running it against MTPLX would silently switch off the
    one thing that engine is for.

    Lives here rather than in the `run` command because the engine launches the same
    models and the two must not drift: a model started through the router has
    to be the same server you get from the command line, or nothing measured
    through one applies to the other.

    `warn` is called with the gate's warning at the point `inference run` used to print
    it - before the spec is built - so an Unsupported model still reports the
    build problem first.

    Returns (spec, warning). Raises Unsupported.
    """
    if getattr(module, "ENGINE", "llama-server") == "llama-server":
        spec_decode, warning = resolve_spec_decode(opts.spec_decode)
    else:
        spec_decode, warning = opts.spec_decode, None

    if warning and warn:
        warn(warning)
    if spec_decode is not opts.spec_decode:
        opts = replace(opts, spec_decode=spec_decode)

    spec = module.spec(machine, opts)
    # Model files don't set the key; the run does. Only MTPLX checks one.
    if getattr(spec, "api_key", opts.api_key) != opts.api_key:
        spec = replace(spec, api_key=opts.api_key)
    return spec, warning


def weights_are_local(spec) -> bool:
    """Are this model's weights already, and completely, on disk?

    Every engine here falls back to downloading from HuggingFace at launch
    when the weights are missing - llama-server via `-hf`, mlx_lm.server and
    MTPLX via the repo id - and the sandbox blocks exactly that. So a missing
    model is pulled first (weights.pull), outside the sandbox, and this is the
    question that decides whether to.

    A pull that was interrupted leaves its marker behind, and reads as not on
    disk: the next launch resumes it rather than loading half a model.
    """
    if weights.is_pulling(spec):
        return False
    if isinstance(spec, LlamaServerSpec):
        return bool(spec.local_path or find_in_hf_cache(spec.hf_repo, spec.hf_file))

    repo_dir = spec.hf_repo.replace("/", "--")

    if isinstance(spec, MtplxServerSpec):
        # MTPLX keeps its own cache and does not use the HuggingFace one. The
        # HF cache can hold a metadata-only stub for the same repo - `mtplx
        # inspect` leaves one - so checking there would report weights that
        # are not present.
        #
        # Read the location off the spec rather than assuming ~/.mtplx: that is
        # the directory the server will actually be launched with, so the check
        # and the launch cannot disagree. They did once - the cache moved to an
        # external volume and this went on looking at the old path, so a model
        # that was present read as missing and the sandbox guard refused it.
        return os.path.isdir(os.path.join(
            os.path.expanduser(spec.cache_dir), repo_dir))

    # An MLX repo: a complete snapshot, not just the folder. A partial
    # huggingface_hub download, or a metadata-only fetch, leaves the folder
    # behind - and a sandboxed, offline launch would then fail on it.
    root = weights.hf_repo_dir(spec.hf_repo)
    try:
        with open(os.path.join(root, "refs", "main"), encoding="utf-8") as handle:
            revision = handle.read().strip()
        blobs = os.listdir(os.path.join(root, "blobs"))
    except OSError:
        return False
    snapshot = os.path.join(root, "snapshots", revision)
    return (os.path.isdir(snapshot) and bool(os.listdir(snapshot))
            and not any(b.endswith(".incomplete") for b in blobs))


def sandbox_wrap(argv: list[str]) -> tuple[list[str], str | None]:
    """Prefix argv with sandbox-exec. Returns (argv, warning)."""
    exe = shutil.which("sandbox-exec") or "/usr/bin/sandbox-exec"
    if not os.path.exists(exe):
        return argv, ("sandbox-exec not found - the server is running with "
                      "network access. Expected on macOS; pass --no-sandbox to "
                      "silence this.")
    return [exe, "-p", SANDBOX_PROFILE] + argv, None


def server_env(opts: Options) -> dict | None:
    """The model server's environment - offline, when it is sandboxed.

    A sandboxed launch only goes ahead with the weights on disk, so say so to
    the HuggingFace client. Otherwise mlx_lm.server asks the hub for the latest
    revision, the sandbox denies the socket, and a load that would have worked
    waits out a network timeout first.
    """
    return dict(os.environ, HF_HUB_OFFLINE="1") if opts.sandbox else None


def build_argv(spec, machine: Machine, opts: Options,
               ) -> tuple[list[str], str, list[str]]:
    """(argv, engine name, warnings) - the one place a launch command is built.

    Both front doors call this for the same reason prepare() exists: a model
    started through the router has to be the same server you get from the
    command line. Sandboxing only one of them would mean the same alias is
    confined or not depending on how you started it, which is worse than not
    sandboxing at all - you would think you were protected half the time.
    """
    exe, engine, gate_warning = resolve_engine(spec)
    if opts.sandbox and isinstance(spec, MtplxServerSpec):
        # The weights were pulled before launch; a sandboxed server is never
        # asked to fetch its own. Without the sandbox it still may.
        spec.download = False
    argv = [exe] + spec.argv(machine, opts.port, opts.bind)
    warnings = [gate_warning] if gate_warning else []

    if opts.sandbox:
        argv, sandbox_warning = sandbox_wrap(argv)
        if sandbox_warning:
            warnings.append(sandbox_warning)
        elif not weights_are_local(spec):
            warnings.append(
                f"{spec.hf_repo} is not on disk yet and the egress sandbox "
                "blocks the download. Fetch it first with `inference pull`.")
    return argv, engine, warnings


def plan_launch(alias: str, module, machine: Machine, opts: Options) -> dict:
    """Everything a launch would decide, with nothing launched.

    An engine driving this Mac over SSH asks this before it stops the model it
    already has, so a launch that cannot work is refused while the old model is
    still serving - the guarantee the local supervisor gives, kept across the
    network. The checks are the launcher's own, run where the binaries and
    weights are.
    """
    plan = {"alias": alias, "engine": None, "upstream_id": alias, "ok": False,
            "reason": None, "warnings": [], "sha": checkout_sha(),
            "needs_pull": False}
    try:
        spec, _ = prepare(module, machine, opts)
        _, engine, warnings = build_argv(spec, machine, opts)
    except Unsupported as exc:
        plan["reason"] = str(exc)
        return plan
    except SystemExit as exc:
        # find_llama_server() and friends exit, which is right for a CLI.
        plan["reason"] = str(exc) or "engine executable not found"
        return plan
    plan.update(engine=engine, upstream_id=upstream_id(spec, alias), warnings=warnings)
    if not weights_are_local(spec):
        # Asked whether or not the sandbox is on: an engine that pulls first
        # does so either way, so the server never downloads for itself.
        plan["needs_pull"] = True
        if opts.sandbox:
            plan["reason"] = (f"{spec.hf_repo} is not on disk and the egress "
                              f"sandbox blocks the download")
            return plan
    plan["ok"] = True
    return plan


@dataclass
class ServerProcess:
    """A model server started in the background, and the handle to end it."""

    alias: str
    engine: str
    port: int
    argv: list[str]
    popen: subprocess.Popen
    log_path: str
    started_at: float
    _tail: deque = field(default_factory=lambda: deque(maxlen=200))

    def exit_code(self) -> int | None:
        return self.popen.poll()

    def is_alive(self) -> bool:
        return self.popen.poll() is None

    def log_tail(self, lines: int = 30) -> str:
        """The last few lines of the server's own output.

        Read from the file rather than kept in memory, because the interesting
        case is a process that died during load and the reason is the last
        thing it wrote.
        """
        try:
            with open(self.log_path, "rb") as handle:
                # Enough to cover `lines` of anything llama.cpp or MLX emits.
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - 64 * 1024))
                text = handle.read().decode("utf-8", "replace")
        except OSError:
            return ""
        return "\n".join(text.splitlines()[-lines:])

    def terminate(self, grace: float = 10.0) -> None:
        """SIGTERM the whole process group, then SIGKILL what's left.

        The group, not the process: `mtplx serve` re-execs and mlx_lm.server
        forks workers, so signalling only what we spawned can leave a
        grandchild holding tens of GB of unified memory and the port. The next
        model then fails to bind with a message that blames the wrong thing.
        """
        if self.popen.poll() is not None:
            return
        pid = self.popen.pid
        try:
            group = os.getpgid(pid)
        except ProcessLookupError:
            return

        # Never signal our own group. os.getpgid(0) means "the calling process",
        # so a handle with a bogus pid would have this process kill itself and
        # everything it started - which is how a stray test double took down the
        # whole test run before this guard existed.
        if not pid or group == os.getpgrp():
            return

        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(group, sig)
            except ProcessLookupError:
                return
            deadline = time.monotonic() + (grace if sig == signal.SIGTERM else 5.0)
            while time.monotonic() < deadline:
                if self.popen.poll() is not None:
                    return
                time.sleep(0.2)


def launch(alias: str, spec, machine: Machine, opts: Options,
           log_dir: str = "logs") -> ServerProcess:
    """Start a model server in the background and return a handle.

    No banner and no waiting - the supervisor owns readiness and console
    output. serve() remains the foreground path for `inference run`.
    """
    argv, engine, warnings = build_argv(spec, machine, opts)

    os.makedirs(log_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(log_dir, f"{alias}-{stamp}.log")

    # A file, not a PIPE. A pipe needs a permanent drain thread or the child
    # blocks the moment the 64KB buffer fills, and llama-server logs every
    # request. A file also outlives the crash you want to read about.
    handle = open(log_path, "ab", buffering=0)
    handle.write(f"$ {' '.join(quote(a) for a in argv)}\n\n".encode())
    # No console here - the supervisor owns output - so warnings go where the
    # rest of this server's story is.
    for message in warnings:
        handle.write(f"WARNING: {message}\n\n".encode())

    popen = subprocess.Popen(
        argv,
        env=server_env(opts),
        stdout=handle,
        stderr=subprocess.STDOUT,
        # No stdin. A server started over SSH would otherwise hold the
        # connection's stdin open, and the call that started it never returns.
        stdin=subprocess.DEVNULL,
        # Its own session, so terminate() can signal the whole group.
        start_new_session=True,
    )
    return ServerProcess(
        alias=alias,
        engine=engine,
        port=opts.port,
        argv=argv,
        popen=popen,
        log_path=log_path,
        started_at=time.monotonic(),
    )


def _banner(model_name: str, engine: str, spec, machine: Machine, port: int) -> None:
    bar = "=" * 56
    print(blue(bar))
    print(blue(f"   {engine} - {machine.chip} optimised"))
    print(blue(bar))
    print(cyan(f"   Model:   {model_name}"))
    print(cyan(f"   HF repo: {spec.hf_repo}"))
    if isinstance(spec, LlamaServerSpec) and spec.hf_file:
        print(cyan(f"   Quant:   {spec.hf_file}"))
        local = spec.local_path or find_in_hf_cache(spec.hf_repo, spec.hf_file)
        if local:
            print(cyan(f"   Source:  local file (no download) - {local}"))
    print(cyan(f"   Host:    {machine.host}:{port}"))
    print(blue(bar))
    print()
    print(grey(f"System: {machine.chip} ({machine.cores} cores), {machine.ram_gb}GB RAM"))
    print()
    print(green("Configuration:"))

    if isinstance(spec, LlamaServerSpec):
        rows = [
            ("Context", f"{spec.ctx:,} tokens"),
            ("Batch", f"{machine.batch} (ubatch: {machine.ubatch})"),
            ("Threads", str(machine.threads)),
            ("Max predict", f"{spec.predict:,}"),
            ("Parallel", str(spec.parallel)),
            ("Sampling", f"temp={spec.temp} top_p={spec.top_p} top_k={spec.top_k} "
                         f"min_p={spec.min_p} presence={spec.presence_penalty}"),
            ("GPU layers", "all (999)"),
            ("Flash attn", "on"),
            ("Vision", "on (mmproj)" if spec.vision else "off"),
        ]
    elif isinstance(spec, MtplxServerSpec):
        rows = [
            ("Context", f"{spec.context_window:,} tokens"),
            ("Max tokens", f"{spec.max_tokens:,}"),
            ("Sampling", f"temp={spec.temp} top_p={spec.top_p} top_k={spec.top_k} "
                         f"presence={spec.presence_penalty} "
                         f"frequency={spec.frequency_penalty}"),
            ("Engine", "MTPLX (MLX, native MTP speculative decode)"),
            ("API key", f"{spec.api_key} (not a secret - see config.py)"
                        if spec.api_key == NO_AUTH_KEY else "set (from api_key)"),
        ]
    else:
        rows = [
            ("Max tokens", f"{spec.max_tokens:,}"),
            ("Sampling", f"temp={spec.temp} top_p={spec.top_p}"),
            ("Prefill step", str(spec.prefill_step_size)),
            ("Prompt cache", f"{spec.prompt_cache_bytes // 1024**3}GB"),
            ("Engine", "mlx-lm (Apple Silicon native)"),
        ]
    rows += list(spec.notes)

    for label, value in rows:
        print(grey(f"  {label + ':':<18}{value}"))

    print()
    print(green("Endpoints:"))
    print(yellow(f"  OpenAI-compatible: http://{machine.host}:{port}/v1/chat/completions"))
    print(yellow(f"  Health check:      http://localhost:{port}/health"))
    if isinstance(spec, MtplxServerSpec):
        print(yellow(f"  Anthropic-compat:  http://{machine.host}:{port}/v1/messages"))
        print()
        if spec.api_key == NO_AUTH_KEY:
            print(grey(f"  Clients send `Authorization: Bearer {spec.api_key}` - a"))
            print(grey("  0.0.0.0 bind requires a key, so this is the one they already send."))
        else:
            print(grey("  Clients send `Authorization: Bearer <api_key>`."))
    print()
    print(yellow("Press Ctrl+C to stop"))
    print(blue(bar))
    print()


def serve(model_name: str, spec, machine: Machine, opts: Options) -> int:
    """Print the banner and exec the engine. Returns the engine's exit code."""
    argv, engine, warnings = build_argv(spec, machine, opts)

    if not opts.dry_run:
        print("\033[H\033[2J", end="")  # clear
    _banner(model_name, engine, spec, machine, opts.port)

    for message in warnings:
        print(yellow(f"WARNING: {message}"))
    if warnings:
        print()

    if opts.dry_run:
        print(green("Command (dry run, not launched):"))
        print("  " + " ".join(quote(a) for a in argv))
        print()
        return 0

    try:
        return subprocess.call(argv, env=server_env(opts))
    except KeyboardInterrupt:
        return 130


# ============================================================
# Stopping
# ============================================================
def _pids(pattern: str) -> list[int]:
    result = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    return [int(line) for line in result.stdout.split() if line.strip().isdigit()]


# `mtplx stop` is deliberately not used here. It only manages the daemon that
# `mtplx start` (and the Mac app) registers; against a server this harness
# launched - which execs `mtplx serve` - it exits 1 with "Port 8083 is in use,
# but not by an MTPLX server. Not touching it." The SIGTERM below is what
# actually stops it, and MTPLX handles SIGTERM cleanly.



def _port_of(pid: int) -> int | None:
    """The --port a process was launched with, from its own command line.

    Read from the arguments rather than from the socket, because a server that
    is still downloading weights has not bound anything yet - and that is
    exactly the process a name-based sweep must not kill.
    """
    try:
        result = subprocess.run(["ps", "-o", "command=", "-p", str(pid)],
                                capture_output=True, text=True, timeout=5)
    except subprocess.SubprocessError:
        return None
    argv = result.stdout.split()
    for flag in ("--port", "-p"):
        if flag in argv:
            index = argv.index(flag) + 1
            if index < len(argv) and argv[index].isdigit():
                return int(argv[index])
    return None


def stop_all(quiet: bool = False, engine: bool = False,
             engine_port: int = DEFAULT_LISTEN_PORT,
             only_port: int | None = None) -> int:
    """SIGTERM, then SIGKILL, every model server. Returns how many were stopped.

    Importable because the engine's supervisor uses this same path rather than
    only killing the process it spawned. `mtplx serve` re-execs into
    `python -m mtplx.server.openai`, so the process holding the weights is not
    the one we started - matching on name is what reliably catches it, and what
    catches a server somebody started by hand.

    `only_port` narrows the sweep to servers launched with that `--port`, which
    the supervisor uses so that stopping its own model cannot reach a second
    server you started deliberately on another port - a weights download, or an
    A/B run. Without it the sweep is by name alone, which is what catches an
    orphan whose port we never knew.
    """
    def say(text: str) -> None:
        if not quiet:
            print(text)

    stopped = 0
    if engine:
        pid = find_server_pid(engine_port)
        if pid is not None and pid != os.getpid():
            say(yellow(f"Killing engine on {engine_port} (pid {pid})"))
            subprocess.run(["kill", str(pid)])
            stopped += 1

    patterns = SERVER_PATTERNS

    def skip(pid: int) -> bool:
        if pid == os.getpid():
            return True  # never sweep ourselves off the board
        if only_port is None:
            return False
        launched_on = _port_of(pid)
        return launched_on is not None and launched_on != only_port

    for pattern in patterns:
        for pid in _pids(pattern):
            if skip(pid):
                continue
            say(yellow(f"Killing {pattern} (pid {pid})"))
            subprocess.run(["kill", str(pid)])
            stopped += 1

    if stopped:
        time.sleep(2)

    remaining = 0
    for pattern in patterns:
        for pid in _pids(pattern):
            if skip(pid):
                continue
            say(red(f"Force-killing {pattern} (pid {pid})"))
            subprocess.run(["kill", "-9", str(pid)])
            remaining += 1

    if stopped == 0 and remaining == 0:
        say(green("No model servers running."))
    else:
        say(green(f"Stopped {stopped} server(s)."))
    return stopped

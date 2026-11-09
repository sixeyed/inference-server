"""The command line.

One entry point with subcommands, because the three things you do with this -
serve models on demand, run one in the foreground, stop whatever is running -
share a machine, a registry and a set of defaults. Three separate scripts meant
three separate argparse surfaces drifting apart.

    python -m inference engine            # the router on 8084
    python -m inference run qwen3.8       # one model, foreground
    python -m inference stop              # kill whatever is running
    python -m inference models            # what fits this Mac

After `pip install -e .` the same commands are `inference engine` and so on.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from dataclasses import asdict

from . import dashboard, models, process, tracing, usage
from .config import (
    AUTO_PULL,
    BIND_HOST,
    DEFAULT_API_KEY,
    DEFAULT_LISTEN_PORT,
    DEFAULT_LOG_DIR,
    DEFAULT_PORT,
    DEFAULT_SETTINGS_FILE,
    DEFAULT_TRACE_DIR,
    DEFAULT_USAGE_DB,
    MODEL_BIND,
)
from .config import apply_weights_env, weights_root_missing
from .console import blue, cyan, green, grey, red, yellow
from .machine import Options, Unsupported, detect_machine, profile_warning
from .process import prepare, serve
from .server import Engine, EngineHandler, EngineServer
from .supervisor import StartupFailed


def _engine_banner(engine: Engine, args) -> None:
    bar = "=" * 56
    live = [i for i in engine.infos if i.routable and not i.test_only]
    nofit = [i for i in engine.infos if not i.supported]
    print(blue(bar))
    print(blue("   inference engine - " + engine.machine.chip + " optimised"))
    print(blue(bar))
    print(cyan(f"   Listening:  {args.bind}:{engine.port}"))
    print(cyan(f"   Models on:  {engine.upstream_host}:{engine.upstream_port}"))
    print(cyan(f"   Registry:   {len(live)} available, {len(nofit)} no fit"))
    print(cyan(f"   Default:    {engine.default_model}"))
    print(cyan(f"   Trace:      {engine.tracer.path if not args.no_trace else 'off'}"))
    print(cyan(f"   Logs:       {args.log_dir}/"))
    print(blue(bar))
    print()
    print(f"  Point your client at http://{engine.machine.host}:{engine.port}/v1")
    print(grey("  Models start on demand - the first request blocks while the"))
    print(grey("  weights load (30s-2min warm; a first use pulls them first)."))
    print(grey("  Traces contain your source code - keep them out of git."))
    print()
    print(green("  Available: ") + ", ".join(i.alias for i in live))
    print()
    print(yellow("  Ctrl+C to stop") + grey(
        " (stops the model too; --leave-running keeps it)"))
    print(blue(bar))
    print()






def _add_thinking_args(parser: argparse.ArgumentParser) -> None:
    """The A/B switches, shared by `engine` and `run`.

    Both take them because the comparison this repo exists to run is "same
    weights, one flag different", and it has to be possible whichever way the
    model was started.
    """
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument("--thinking", dest="thinking", action="store_true",
                          default=None,
                          help="force reasoning on (overrides the model file)")
    thinking.add_argument("--no-thinking", dest="thinking", action="store_false",
                          help="force reasoning off")
    spec = parser.add_mutually_exclusive_group()
    spec.add_argument("--spec", dest="spec_decode", action="store_true",
                      default=None,
                      help="enable speculative decoding where the model has it")
    spec.add_argument("--no-spec", dest="spec_decode", action="store_false",
                      help="disable speculative decoding")
    parser.add_argument("--reasoning-effort",
                        choices=("low", "medium", "high", "xhigh"),
                        help="reasoning effort for models that expose levels "
                             "(Qwen 3.8 under MTPLX). Implies --thinking.")


def _add_sandbox_arg(parser: argparse.ArgumentParser) -> None:
    """Both `run` and `engine` take this, because both start model servers and
    the confinement must not depend on which one you used."""
    parser.add_argument(
        "--no-sandbox", dest="sandbox", action="store_false", default=True,
        help="don't confine the server with sandbox-exec. Not needed for a "
             "first download: missing weights are pulled before launch.")
    parser.add_argument(
        "--no-pull", dest="pull", action="store_false", default=AUTO_PULL,
        help="don't fetch missing weights before launching. Sandboxed, the "
             "launch is refused; with --no-sandbox the server downloads "
             "them itself")


def _add_engine_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--port", type=int, default=DEFAULT_LISTEN_PORT)
    parser.add_argument("--bind", default=BIND_HOST,
                        help="address to listen on (default 127.0.0.1 - this "
                             "machine only; 0.0.0.0 serves the network)")
    parser.add_argument("--model-bind", default=MODEL_BIND,
                        help="address model servers bind to")
    parser.add_argument("--upstream-port", type=int, default=DEFAULT_PORT,
                        help="where models are started and found")
    parser.add_argument("--model", help="preload this model at startup")
    parser.add_argument("--default-model", default=models.DEFAULT_MODEL)
    parser.add_argument("--on-unknown", choices=("current", "default", "404"),
                        default="current",
                        help="what to serve when the client sends an id we "
                             "don't know (default: whatever is loaded)")
    parser.add_argument("--trust-proxy", action="store_true",
                        help="take the client address from X-Forwarded-For - "
                             "only behind a proxy you run, like the ingress")
    parser.add_argument("--serve-deprecated", action="store_true",
                        help="also route to the retired models")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--trace-dir", default=DEFAULT_TRACE_DIR)
    parser.add_argument("--name", help="trace file name (default: timestamped)")
    parser.add_argument("--no-trace", action="store_true")
    parser.add_argument("--no-repair", action="store_true",
                        help="relay strictly verbatim, even when a malformed "
                             "tool call will 500 the rest of the session")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--usage-db", default="usage.db")
    parser.add_argument("--settings", default=DEFAULT_SETTINGS_FILE,
                        help="live proxy settings, editable from the dashboard")
    parser.add_argument("--no-usage", action="store_true")
    parser.add_argument("--no-ui", action="store_true")
    parser.add_argument("--start-timeout", type=float, default=1800.0)
    parser.add_argument("--leave-running", action="store_true",
                        help="don't stop the model when the engine exits")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--list", action="store_true",
                        help="list captured traces and exit")
    parser.add_argument("--ctx", type=int, help="override context for every launch")

    _add_thinking_args(parser)
    _add_sandbox_arg(parser)





def engine_command(args) -> int:

    if args.list:
        return tracing.list_traces(args.trace_dir)

    engine = Engine(args)

    if not args.no_usage:
        engine.attach_usage(usage.UsageDB(args.usage_db, engine.machine))

    if not args.no_ui:
        engine.dashboard = dashboard.Dashboard(engine)

    EngineHandler.engine = engine
    _engine_banner(engine, args)

    engine.supervisor.adopt(engine.index)

    if args.model:
        alias = engine.resolve(args.model) or args.model
        try:
            engine.supervisor.acquire(alias)
            engine.supervisor.release()
        except StartupFailed as exc:
            print(red(f"ERROR: {exc.message()}"), file=sys.stderr)
            return 1

    server = EngineServer((args.bind, args.port), EngineHandler)
    server.daemon_threads = True

    def _bye(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _bye)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\nStopping.")
    finally:
        server.server_close()
        engine.supervisor.shutdown()
        if engine.usage is not None:
            engine.usage.close()
        summary = trace_summary(engine.tracer, args.no_trace)
        if summary:
            print(summary)
    return 0


def trace_summary(tracer, no_trace: bool) -> str:
    """The line printed on the way out, or "" when there is nothing to say.

    Its own function because it runs in the shutdown `finally` - the one place
    where a mistake is invisible until you stop the server, and where any
    exception is raised after the model has already been killed, so nothing
    looks wrong until you read the log.
    """
    if no_trace or not os.path.exists(tracer.path):
        return ""
    size_mb = os.path.getsize(tracer.path) / 1024**2
    return (f"Captured {tracer.count} turns to "
            f"{tracer.path} ({size_mb:.1f}MB)\n")




# ============================================================
# run - one model, in the foreground
# ============================================================
def _list_models(machine, registry) -> None:
    infos = models.describe(machine, registry)
    live = [i for i in infos if not i.deprecated and not i.test_only]
    retired = [i for i in infos if i.deprecated]
    fixtures = [i for i in infos if i.test_only]

    # Which servers are installed. The first thing a new Mac is missing, and
    # without it every model reads "ok" and only fails when you pick it.
    from .engines import missing
    absent = {e: missing(e) for e in sorted({i.engine for i in live})}

    print()
    print(cyan(f"Models on {machine.chip} ({machine.cores} cores, {machine.ram_gb}GB)"))
    warning = profile_warning(machine)
    if warning:
        print(yellow(f"  {warning}"))
    print()
    for info in live:
        if not info.supported:
            status = red("no fit   ")
        elif absent.get(info.engine):
            status = yellow("no server")
        else:
            status = green("ok       ")
        print(f"  {status}  {info.alias:<18} {info.engine:<6} {grey(info.summary)}")

    needed = list(dict.fromkeys(msg for msg in absent.values() if msg))
    if needed:
        print()
        print(yellow("  Servers not installed - those models fail when picked:"))
        for msg in needed:
            print(yellow("    " + " ".join(msg.split())))

    if retired:
        print()
        print(grey("  Retired - still launchable here, not served by the engine:"))
        for info in retired:
            print(grey(f"          {info.alias:<18} {info.deprecated}"))

    if fixtures:
        print()
        print(grey("  Test only - routable by name, never listed by the engine:"))
        for info in fixtures:
            print(grey(f"          {info.alias:<18} {info.summary}"))

    print()
    print(grey("  Family defaults: " + ", ".join(
        f"{f}={m}" for f, m in models.FAMILY_DEFAULTS.items())))
    print()


def run_command(args) -> int:
    registry = models.load_all()
    machine = detect_machine()

    if args.model and args.family:
        print(red("pass a model or --family, not both"), file=sys.stderr)
        return 2

    alias = args.model or (
        models.FAMILY_DEFAULTS[args.family] if args.family else models.DEFAULT_MODEL
    )

    if alias not in registry:
        # Fall back to the registry's own resolution, which also knows former
        # aliases and HuggingFace repo ids - so a rename doesn't invalidate
        # whatever is in your shell history. Deprecated models included: this
        # is the front door that still launches them.
        index = models.build_index(models.describe(machine, registry),
                                   deprecated=True)
        resolved = models.resolve(alias, index)
        if resolved is None:
            print(red(f"Unknown model: {alias}"), file=sys.stderr)
            print(grey("Known: " + ", ".join(sorted(registry))), file=sys.stderr)
            return 2
        print(grey(f"{alias} -> {resolved}"), file=sys.stderr)
        alias = resolved

    module = registry[alias]

    opts = Options(
        port=args.port,
        thinking=args.thinking,
        spec_decode=args.spec_decode,
        ctx=args.ctx,
        reasoning_effort=args.reasoning_effort,
        dry_run=args.dry_run,
        sandbox=args.sandbox,
        bind=args.bind,
    )

    if args.plan:
        # Everything a launch would decide, and nothing launched. An engine
        # driving this Mac over SSH asks this before stopping the model it has.
        print(json.dumps(process.plan_launch(alias, module, machine, opts)))
        return 0

    def warn(message: str) -> None:
        print(yellow(f"WARNING: {message}"), file=sys.stderr)

    try:
        # The speculative-decoding capability gate lives in process.prepare() so
        # that the engine builds specs the same way this does - see the note
        # there on why the two must not drift.
        spec, _ = prepare(module, machine, opts, warn=warn)
        if args.model_path:
            if not hasattr(spec, "local_path"):
                print(red("--model-path only applies to llama-server models"),
                      file=sys.stderr)
                return 2
            spec.local_path = args.model_path
    except Unsupported as exc:
        print()
        print(red(f"ERROR: {exc}"))
        print(yellow("Run `python -m inference models` to see what fits this machine."))
        print()
        return 1

    deprecated = getattr(module, "DEPRECATED", None)
    if deprecated and not args.dry_run:
        print(yellow(f"DEPRECATED: {alias} - {deprecated}"), file=sys.stderr)

    if args.pull and not args.dry_run and not process.weights_are_local(spec):
        # Before the server, outside the sandbox - the server never downloads.
        code = _pull(alias, spec)
        if code:
            return code

    if args.detach:
        # Its own session with stdin closed, so the SSH call that asked for it
        # comes straight back and the server outlives the connection.
        launched = process.launch(alias, spec, machine, opts, log_dir=args.log_dir)
        print(json.dumps({"pid": launched.popen.pid,
                          "log_path": os.path.abspath(launched.log_path),
                          "engine": launched.engine}))
        return 0

    return serve(alias, spec, machine, opts)


def _pull(alias: str, spec) -> int:
    from .weights import PullError, pull
    print(grey(f"  pulling {alias} ({spec.hf_repo})"), file=sys.stderr)
    try:
        pull(spec, say=lambda line: print(grey(line), file=sys.stderr, flush=True))
    except KeyboardInterrupt:
        print(yellow(f"\n  pull of {alias} stopped - run it again to resume"),
              file=sys.stderr)
        return 130
    except (PullError, OSError) as exc:
        print(red(f"ERROR: could not pull {alias}: {exc}"), file=sys.stderr)
        return 1
    return 0


def pull_command(args) -> int:
    """Fetch a model's weights, or bring them up to the latest revision."""
    registry = models.load_all()
    machine = detect_machine()
    alias = args.model or (models.FAMILY_DEFAULTS[args.family] if args.family
                           else models.DEFAULT_MODEL)
    if alias not in registry:
        index = models.build_index(models.describe(machine, registry),
                                   deprecated=True)
        resolved = models.resolve(alias, index)
        if resolved is None:
            print(red(f"Unknown model: {alias}"), file=sys.stderr)
            return 2
        alias = resolved
    try:
        spec, _ = prepare(registry[alias], machine,
                          Options(thinking=args.thinking,
                                  spec_decode=args.spec_decode,
                                  reasoning_effort=args.reasoning_effort))
    except Unsupported as exc:
        print(red(f"ERROR: {exc}"), file=sys.stderr)
        return 1
    return _pull(alias, spec)


def models_command(args) -> int:
    _list_models(detect_machine(), models.load_all())
    return 0


def machine_command(args) -> int:
    """This machine as JSON - what an engine running elsewhere asks the Mac."""
    print(json.dumps(asdict(detect_machine())))
    return 0


def stop_command(args) -> int:
    from .process import stop_all
    stop_all(quiet=args.quiet, engine=args.engine,
             engine_port=args.engine_port, only_port=args.port)
    return 0


def status_command(args) -> int:
    """Who holds the model port, as JSON. With --pid, a server still loading -
    not yet bound - counts as alive, which the port alone would miss."""
    pid = process.find_server_pid(args.port)
    alive = pid is not None or (args.pid is not None and process.pid_alive(args.pid))
    print(json.dumps({"port": args.port, "pid": pid, "alive": alive,
                      "rss_mb": process.rss_mb_of(pid) if pid is not None else None}))
    return 0


# ============================================================
# Dispatch
# ============================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inference",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    engine = sub.add_parser(
        "engine", help="serve every model on one port, starting them on demand")
    _add_engine_args(engine)
    engine.set_defaults(func=engine_command)

    run = sub.add_parser("run", help="run one model in the foreground")
    run.add_argument("model", nargs="?",
                     help=f"model alias (default: {models.DEFAULT_MODEL})")
    run.add_argument("--family", choices=sorted(models.FAMILY_DEFAULTS),
                     help="launch that family's default model instead")
    run.add_argument("--port", type=int, default=DEFAULT_PORT)
    run.add_argument("--bind", default=MODEL_BIND,
                     help="address the server binds to (default 127.0.0.1)")
    run.add_argument("--ctx", type=int, help="override context length")
    run.add_argument("--model-path",
                     help="load this GGUF directly instead of resolving from "
                          "HuggingFace (the HF cache is checked automatically)")
    run.add_argument("--dry-run", action="store_true",
                     help="print the launch command, don't run it")
    run.add_argument("--plan", action="store_true",
                     help="print what a launch would do as JSON, launch nothing")
    run.add_argument("--detach", action="store_true",
                     help="start the server in the background, print its pid "
                          "and log as JSON, and exit")
    run.add_argument("--log-dir", default=DEFAULT_LOG_DIR,
                     help="where --detach writes the server's log")
    _add_thinking_args(run)
    _add_sandbox_arg(run)
    run.set_defaults(func=run_command)

    pull = sub.add_parser(
        "pull", help="fetch a model's weights, or update them to the latest")
    pull.add_argument("model", nargs="?",
                      help=f"model alias (default: {models.DEFAULT_MODEL})")
    pull.add_argument("--family", choices=sorted(models.FAMILY_DEFAULTS))
    _add_thinking_args(pull)
    pull.set_defaults(func=pull_command)

    stop = sub.add_parser("stop", help="stop the model server")
    stop.add_argument("--engine", action="store_true",
                      help="also stop the router (found by its port, not by name)")
    stop.add_argument("--engine-port", type=int, default=DEFAULT_LISTEN_PORT)
    stop.add_argument("--port", type=int, default=None,
                      help="only stop a server launched with this --port. "
                           "Without it every model server is swept, including "
                           "one you started on another port on purpose.")
    stop.add_argument("--quiet", action="store_true")
    stop.set_defaults(func=stop_command)

    status = sub.add_parser("status", help="who holds the model port, as JSON")
    status.add_argument("--port", type=int, default=DEFAULT_PORT)
    status.add_argument("--pid", type=int,
                        help="also count the server alive while this pid runs")
    status.set_defaults(func=status_command)

    listing = sub.add_parser("models", help="what this machine can run")
    listing.set_defaults(func=models_command)

    machine = sub.add_parser("machine", help="this machine's profile, as JSON")
    machine.set_defaults(func=machine_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    # Line-buffer stdout. Python block-buffers it when it isn't a terminal, so
    # `inference engine > engine.log` would swallow the banner and every
    # per-turn line until 8KB had accumulated - which on a quiet day is hours.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:  # pragma: no cover - very old Python
        pass

    apply_weights_env()

    args = build_parser().parse_args(argv)

    # Only the commands that start a server or fetch weights care. `models` and
    # `stop` still work with the drive unplugged, and saying so is more useful
    # than refusing.
    if args.command in ("run", "engine", "pull"):
        problem = weights_root_missing()
        if problem:
            print(red(f"ERROR: {problem}"), file=sys.stderr)
            return 1
    return args.func(args)

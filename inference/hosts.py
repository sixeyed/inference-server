"""Where model servers run, and how the engine reaches them.

The supervisor decides *when* to start and stop a model; a host decides *how*.
LocalHost is what the engine has always done - child processes on this Mac.
RemoteHost (remote.py) does the same things over SSH, for an engine that runs
somewhere the models cannot.
"""

from __future__ import annotations

from . import engines, process, weights


class LocalHost:
    """Model servers as child processes of this engine, on this machine."""

    remote = False

    def __init__(self, launcher=None, stopper=None, puller=None,
                 engine_check=None, bind: str = "127.0.0.1"):
        # Where to reach the model servers: the address they bind, unless
        # that is every interface, in which case loopback reaches them.
        self.address = "127.0.0.1" if bind in ("0.0.0.0", "::", "") else bind
        # Injected so tests can drive the state machine without 30GB of weights
        # on disk. The stopper especially: stop_all() matches on process name,
        # so a test using the real one would reach past a fake server and kill
        # whatever model this machine actually has loaded.
        self._launch = launcher or process.launch
        self._stop_all = stopper or process.stop_all
        # And the puller: a test that reached the real one would start
        # downloading tens of GB from HuggingFace.
        self._pull = puller or weights.pull
        # Is the server binary installed? Injected because the suite runs
        # without llama.cpp or MTPLX - in CI, and in the image.
        self._engine_check = engine_check or engines.resolve_engine

    def prepare(self, module, machine, opts):
        spec, _ = process.prepare(module, machine, opts)
        # Before anything else - and above all before a pull. A missing
        # llama-server found after the download is 30GB fetched for nothing.
        # Raises SystemExit with the install command, which the supervisor
        # turns into a refusal with the current model still serving.
        self._engine_check(spec)
        return spec

    def preflight(self, alias, spec, opts) -> str | None:
        """Why this launch cannot work, or None - asked before anything stops.

        A sandboxed launch with no weights on disk is a known failure: the
        server tries to fetch them and the sandbox denies the socket.
        """
        if opts.sandbox and not process.weights_are_local(spec):
            return (f"{spec.hf_repo} is not on disk and the egress sandbox "
                    f"blocks the download")
        return None

    def needs_pull(self, alias, spec, opts) -> bool:
        return not process.weights_are_local(spec)

    def pull(self, alias, spec, opts, say=print, cancel=None) -> None:
        """Fetch the weights here, with the network - see weights.py."""
        self._pull(spec, say=say, cancel=cancel)

    def launch(self, alias, spec, machine, opts, log_dir):
        return self._launch(alias, spec, machine, opts, log_dir)

    def stop_all(self, quiet: bool = False, only_port: int | None = None):
        return self._stop_all(quiet=quiet, only_port=only_port)

    def server_pid(self, port: int) -> int | None:
        return process.find_server_pid(port)

    def rss_mb(self, port: int) -> float | None:
        pid = process.find_server_pid(port)
        return process.rss_mb_of(pid) if pid is not None else None

"""Where model servers run, and how the engine reaches them.

The supervisor decides *when* to start and stop a model; a host decides *how*.
LocalHost is what the engine has always done - child processes on this Mac.
RemoteHost (remote.py) does the same things over SSH, for an engine that runs
somewhere the models cannot.
"""

from __future__ import annotations

from . import process, weights


class LocalHost:
    """Model servers as child processes of this engine, on this machine."""

    address = "127.0.0.1"
    remote = False

    def __init__(self, launcher=None, stopper=None, puller=None):
        # Injected so tests can drive the state machine without 30GB of weights
        # on disk. The stopper especially: stop_all() matches on process name,
        # so a test using the real one would reach past a fake server and kill
        # whatever model this machine actually has loaded.
        self._launch = launcher or process.launch
        self._stop_all = stopper or process.stop_all
        # And the puller: a test that reached the real one would start
        # downloading tens of GB from HuggingFace.
        self._pull = puller or weights.pull

    def prepare(self, module, machine, opts):
        spec, _ = process.prepare(module, machine, opts)
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

    def pull(self, alias, spec, opts, say=print) -> None:
        """Fetch the weights here, with the network - see weights.py."""
        self._pull(spec, say)

    def launch(self, alias, spec, machine, opts, log_dir):
        return self._launch(alias, spec, machine, opts, log_dir)

    def stop_all(self, quiet: bool = False, only_port: int | None = None):
        return self._stop_all(quiet=quiet, only_port=only_port)

    def server_pid(self, port: int) -> int | None:
        return process.find_server_pid(port)

    def rss_mb(self, port: int) -> float | None:
        pid = process.find_server_pid(port)
        return process.rss_mb_of(pid) if pid is not None else None

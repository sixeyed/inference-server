"""The engine's event log: what changed about a request, and when.

Most of this was already printed to the console, once per engine run, guarded
by a `warned_*` boolean on the Engine. Two problems with that, and this module
exists for both.

The boolean was set on the first occurrence and never cleared, so it reported
the *first* state and hid every later one. A VS Code update that renames a tool
between one turn and the next is exactly the case it swallowed: the engine had
already warned about the filter, so the drift that appeared afterwards was
never mentioned. Fingerprinting the event instead means a repeat is silent but
a change is not.

And a console line is gone when the terminal scrolls. The interesting question
after an update - "when did `create_file` stop being offered?" - needs the
answer to have outlived a restart.

One call drives both, so a machine running with `--no-usage` still warns.
"""

from __future__ import annotations

import json
import threading

from .console import yellow
from . import tracing


class Events:
    """Console line and stored row, deduped by content.

    The fingerprint is `detail`, not the message text: the text is written for
    a human and may well be identical across a real change.
    """

    def __init__(self, usage=None, quiet: bool = False):
        self.usage = usage
        self.quiet = quiet
        self._lock = threading.Lock()
        self._seen: dict[tuple[str, str | None], str] = {}

    def record(self, kind: str, client: str | None, text: str,
               detail: dict | None = None) -> None:
        """Always record, never dedup.

        For events that are an intervention rather than a state - each loop the
        engine breaks is its own occurrence, and collapsing two of them into
        one row would hide the second.
        """
        if not self.quiet:
            tracing.dim(f"{client or 'engine'}: {text}")
        if self.usage is not None:
            self.usage.record_event(kind, client, detail or {}, text)

    def note(self, kind: str, client: str | None, text: str,
             detail: dict | None = None, warn: bool = False) -> bool:
        """Record this event if it differs from the last of its kind.

        Returns whether it was recorded, which is what a caller needing to
        print more than one line checks before printing the rest.
        """
        fingerprint = json.dumps(detail or {}, sort_keys=True)
        key = (kind, client)
        with self._lock:
            if self._seen.get(key) == fingerprint:
                return False
            self._seen[key] = fingerprint

        if not self.quiet:
            line = f"{client or 'engine'}: {text}" if client else text
            print(yellow(f"  {line}")) if warn else tracing.dim(line)
        if self.usage is not None:
            self.usage.record_event(kind, client, detail or {}, text)
        return True

"""Grouping turns into sessions.

A pooled tok/s across every turn the box has ever served answers almost no
question worth asking. What you want is "how did that session go" - one agent
conversation, from the first prompt to wherever it stopped - because that is the
unit a model succeeds or fails at.

Nothing in the OpenAI request says which session it belongs to. Clients do not
send an id, and chat completions is stateless by design. But a session is still
recoverable from the shape of the traffic:

  * every turn in a session resends the whole conversation, so the *opening*
    message is identical across all of them and different between sessions
  * the conversation only grows within a session, so a drop in message count is
    a new one starting - which is what catches a restart whose opening happens
    to be identical

Verified against 230 captured turns: the opening hash changed exactly where the
session did, and the two message-count resets were both real restarts. MTPLX's
own `session_id` agreed across a 201-turn run, but it only exists for that one
engine, so it is a cross-check rather than the mechanism.
"""

from __future__ import annotations

import hashlib
import json
import threading


def opening_key(request: dict, client_ip: str | None = None) -> str:
    """A stable key for the conversation this request belongs to.

    The first message only: a system prompt plus the client's injected preamble
    is enough to separate sessions, and taking more would break the moment a
    client compacts its own history mid-run.
    """
    messages = (request or {}).get("messages") or []
    first = messages[0] if messages else {}
    payload = json.dumps([client_ip or "", first], sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


class SessionTracker:
    """Assigns session ids to turns as they arrive.

    In memory and per engine run. A restart begins fresh sessions, which is
    honest - the engine cannot know that a conversation predates it, and rows
    from the old run keep the id they were given.
    """

    def __init__(self):
        self._lock = threading.Lock()
        # key -> (session id, last message count seen)
        self._live: dict[str, tuple[str, int]] = {}
        self._counter = 0
        self._in_flight: set[str] = set()

    def identify(self, request: dict, client_ip: str | None = None) -> str:
        key = opening_key(request, client_ip)
        count = len((request or {}).get("messages") or [])

        with self._lock:
            existing = self._live.get(key)
            # A conversation that shrank is a new one wearing the same opening.
            if existing is None or count < existing[1]:
                self._counter += 1
                session = f"{key}-{self._counter}"
            else:
                session = existing[0]
            self._live[key] = (session, count)
            return session

    def claim(self, session: str) -> bool:
        """Mark a session as having a request in flight; False if it already has.

        MTPLX answers a second request on a session that is still generating
        with a 409 rather than queueing it. Two VS Code chats in one workspace
        share an opening, and a client can retry while its first attempt is
        still streaming - so the engine names a session upstream only while no
        other request is using that name, and otherwise leaves MTPLX to guess,
        which is what it did before the engine named sessions at all.
        """
        with self._lock:
            if session in self._in_flight:
                return False
            self._in_flight.add(session)
            return True

    def release(self, session: str) -> None:
        with self._lock:
            self._in_flight.discard(session)

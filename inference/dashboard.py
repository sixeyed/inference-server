"""The dashboard: static files, and the JSON the page polls for.

Kept apart from server.py because it is the one part of the system that is not
on the request path for a model - nothing here can make a chat completion slower
or wrong, and that separation is worth a file.

The routes claimed here are taken *before* the relay fallback, or they would be
forwarded upstream. Worth knowing: llama-server serves its own web UI at `/`, so
this shadows it. That is intended, and it is another reason the model port stays
directly reachable.
"""

from __future__ import annotations

import json
import mimetypes
import os

from . import about, prompts, toolsets, turns
from . import client_setup, clients

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# Served from disk on every request rather than cached in memory: the file is a
# few KB, and editing the UI while the engine runs should just work.
ALLOWED_SUFFIXES = (".html", ".css", ".js", ".svg", ".png", ".ico", ".woff2")


class Dashboard:
    def __init__(self, engine):
        self.engine = engine

    # --------------------------------------------------------
    def get(self, handler, route: str) -> bool:
        """True if this route belongs to the dashboard and has been answered."""
        if route == "/":
            return self._file(handler, "index.html")
        if route.startswith("/static/"):
            return self._file(handler, route[len("/static/"):])
        if route == "/api/state":
            return self._send(handler, self._state())
        if route == "/api/about":
            return self._send(handler, about.about())
        if route == "/api/models":
            handler._models()
            return True
        if route == "/api/stats":
            return self._send(handler, self._stats())
        if route == "/api/requests":
            limit = self._int(handler, "limit", 25, 200)
            offset = self._int(handler, "offset", 0, 1_000_000, floor=0)
            usage = self.engine.usage
            return self._send(handler, {
                "requests": usage.recent(limit, offset) if usage else [],
                "total": usage.count_requests() if usage else 0,
                "limit": limit,
                "offset": offset,
            })
        if route.startswith("/api/turn/"):
            usage = self.engine.usage
            try:
                request_id = int(route[len("/api/turn/"):])
            except ValueError:
                handler._raw(400, json.dumps({"error": "bad id"}).encode())
                return True
            row = usage.turn(request_id) if usage else None
            if row is None:
                return self._send(handler, {"turn": None, "trace": {
                    "found": False, "reason": "no such turn"}})
            return self._send(handler, {
                "turn": row,
                "trace": turns.load(self.engine.tracer_dir, row["trace_file"],
                                    row["trace_seq"], row["previous_messages"])})
        if route.startswith("/api/sessions/"):
            usage = self.engine.usage
            if usage is None:
                return self._send(handler, {"session": None, "turns": []})
            session_id = route[len("/api/sessions/"):]
            return self._send(handler, {
                "session": usage.session(session_id),
                "turns": usage.session_turns(session_id)})
        if route == "/api/sessions":
            limit = self._int(handler, "limit", 15, 100)
            usage = self.engine.usage
            return self._send(handler, {
                "sessions": usage.per_session(limit) if usage else []})
        if route == "/api/settings":
            return self._send(handler, self._settings())
        if route == "/api/clients":
            return self._send(handler, client_setup.payload(self.engine))
        if route == "/api/events":
            limit = self._int(handler, "limit", 100, 500)
            return self._send(handler, {"events": self._events(limit)})
        if route == "/api/memory":
            minutes = self._int(handler, "minutes", 30, 1440)
            usage = self.engine.usage
            return self._send(handler, {
                "series": usage.memory_series(minutes) if usage else []})
        return False

    def post(self, handler, route: str) -> bool:
        if route == "/api/settings":
            body = handler._read_body() or b"{}"
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                handler._raw(400, json.dumps({"error": "bad json"}).encode())
                return True
            if "prompt" in payload:
                cfg = payload["prompt"] or {}
                self.engine.settings.set_prompt(
                    cfg.get("client") or "", cfg.get("file"),
                    cfg.get("replace"))
            elif "loops" in payload:
                cfg = payload["loops"] or {}
                self.engine.settings.set_loop_threshold(
                    cfg.get("client") or "", cfg.get("count"))
            elif "toolset" in payload:
                cfg = payload["toolset"] or {}
                self.engine.settings.set_toolset(
                    cfg.get("client") or "", cfg.get("file"))
            else:
                self.engine.settings.set_stripped(payload.get("strip") or [])
            return self._send(handler, self._settings())
        return False  # /api/load and /api/unload are handled by the engine

    def _settings(self) -> dict:
        return {"params": self.engine.settings.describe(),
                "strip": sorted(self.engine.settings.stripped),
                "path": self.engine.settings.path,
                "prompts": self.engine.settings.prompts,
                "available_prompts": prompts.available(),
                "toolsets": self.engine.settings.toolsets,
                "loops": self.engine.settings.loops,
                "drift": (self.engine.usage.drift()
                          if self.engine.usage is not None else {}),
                "available_toolsets": toolsets.available(),
                "clients": self._clients_offered(),
                "contract_blocks": sorted(prompts.CONTRACT_BLOCKS)}

    def _events(self, limit: int) -> list[dict]:
        """The engine's own events, merged with failed loads.

        Load failures are not copied into `events` - they are already rows in
        `loads`, with their error and their log path. Merging on read means a
        failure can never go stale in a second copy of itself.
        """
        usage = self.engine.usage
        if usage is None:
            return []
        rows = usage.events(limit)
        for load in usage.loads(limit):
            if load.get("ok"):
                continue
            rows.append({
                "id": None, "ts": load.get("ts"), "kind": "load",
                "client": None,
                "text": f"{load.get('alias')} failed to start: "
                        f"{(load.get('error') or '').strip()}",
                "detail": {"alias": load.get("alias"),
                           "log_path": load.get("log_path")},
            })
        rows.sort(key=lambda row: row.get("ts") or "", reverse=True)
        return rows[:limit]

    def _clients_offered(self) -> list[str]:
        """The clients the Config tab gives rows to.

        The main coding clients always, so a fresh engine - a new pod, a new
        machine - can be set up for VS Code before VS Code has sent it anything.
        Then any other client that has sent traffic or already has settings,
        rather than every label the detector knows about.
        """
        settings = self.engine.settings
        others = set(settings.prompts) | set(settings.toolsets) | set(settings.loops)
        usage = self.engine.usage
        if usage is not None:
            rows = usage._query("SELECT DISTINCT client FROM requests "
                                "WHERE client IS NOT NULL ORDER BY client")
            others |= {r["client"] for r in rows}
        return list(clients.OFFERED) + sorted(others - set(clients.OFFERED))

    # --------------------------------------------------------
    def _state(self) -> dict:
        engine = self.engine
        machine = engine.machine
        status = engine.supervisor.status()
        status["rss_mb"] = (
            round(engine.supervisor.rss_mb() or 0, 1)
            if status["state"] == "ready" else None
        )
        status["machine"] = {
            "chip": machine.chip, "ram_gb": machine.ram_gb,
            "cores": machine.cores, "host": machine.host,
        }
        status["endpoint"] = client_setup.endpoints(engine)[0]["base_url"]
        return status

    def _stats(self) -> dict:
        usage = self.engine.usage
        if usage is None:
            return {"models": [], "preempted": {}}
        return {"models": usage.per_model(),
                "preempted": usage.preempted_counts()}

    # --------------------------------------------------------
    @staticmethod
    def _int(handler, key: str, default: int, cap: int, floor: int = 1) -> int:
        from urllib.parse import parse_qs, urlparse
        raw = parse_qs(urlparse(handler.path).query).get(key, [None])[0]
        try:
            return max(floor, min(cap, int(raw)))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _send(handler, payload) -> bool:
        handler._json(200, payload)
        return True

    def _file(self, handler, name: str) -> bool:
        # Resolve and confirm the result is still inside static/. Without this,
        # /static/../../etc/passwd is a file read on a box that answers to the
        # whole LAN.
        path = os.path.realpath(os.path.join(STATIC_DIR, name))
        if not path.startswith(STATIC_DIR + os.sep) and path != STATIC_DIR:
            handler._raw(403, json.dumps({"error": "no"}).encode())
            return True
        if not name.endswith(ALLOWED_SUFFIXES) or not os.path.isfile(path):
            handler._raw(404, json.dumps({"error": f"no {name}"}).encode())
            return True

        with open(path, "rb") as source:
            body = source.read()
        content_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
        handler._raw(200, body, content_type)
        return True

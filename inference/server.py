"""The HTTP front door: routing, relaying and recording.

Reads `model` off each request, makes sure that model is the one serving, and
forwards. Everything the old capture proxy did happens on the way through -
relayed verbatim, the stream timed, one JSONL record per turn.

The two places this is not a transparent relay are both deliberate and both
recorded: empty tool-call arguments, which poison the rest of a llama.cpp
session, and the `model` field, which some engines only answer to under a
different name. See prepare_upstream().

The CLI that drives this lives in cli.py.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import dashboard
from . import clients as clients_mod
from . import prompts as prompts_mod
from . import sessions as sessions_mod
from . import events as events_mod
from . import loops as loops_mod
from . import settings as settings_mod
from . import toolsets as toolsets_mod
from . import models
from . import tracing
from . import usage
from .config import (AUTO_PULL, CLIENT_ENDPOINTS, DEFAULT_API_KEY, DEFAULT_PORT,
                     DEFAULT_SETTINGS_FILE, MODEL_BIND, MODEL_HOST)
from .console import blue, cyan, green, grey, red, yellow
from .hosts import LocalHost
from .machine import Machine, Options, detect_machine, machine_from_profile
from .remote import ModelHostConfig, RemoteError, RemoteHost
from .sse import FirstTokenProbe, reassemble
from .supervisor import StartupFailed, State, Supervisor

DEFAULT_LISTEN_PORT = 8084
DEFAULT_TRACE_DIR = "traces"

# Headers we must not copy in either direction.
HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "content-length",
    "content-encoding", "host",
}

EXPLAIN_AUTH = {
    "mismatch": "the client sent an API key that this model server rejected. "
                "Set the client's key to the engine's, or leave it blank",
    "empty": "the client sent an empty Authorization header, which is worse "
             "than sending none - the engine only fills in a key when the "
             "header is absent entirely",
    "match": "the client's key matched the engine's, so the model server was "
             "started with a different one",
    "absent": "the engine supplied its own key and it was still rejected - the "
              "model server was started with a different one",
}

CHAT_PATHS = {"/v1/chat/completions", "/chat/completions"}
ANTHROPIC_PATHS = {"/v1/messages"}

# What MTPLX reads a session from, in its own order of precedence
# (engine_session.resolve_session_id). A client sending any of them has named
# its session itself, and the engine leaves it alone.
MTPLX_SESSION_HEADER = "X-MTPLX-Session-Id"
CLIENT_SESSION_HEADERS = {
    "x-mtplx-session-id", "x-session-affinity", "x-session-id",
    "x-openwebui-chat-id", "x-openwebui-user-id",
}


def _error(kind: str, message: str) -> bytes:
    """OpenAI-shaped, because that is what clients render."""
    return json.dumps({
        "error": {"message": message, "type": kind, "code": kind, "param": None}
    }).encode()


_WARNED_PROMPT_PATH = False
_WARNED_TOOLSET_PATH = False


def prepare_upstream(request: dict, current, repair: bool,
                     settings=None, client: str | None = None,
                     ) -> tuple[bytes, dict | None, dict]:
    """(body to send, untouched snapshot or None, what changed).

    The engine is a transparent relay except in two places, and both are
    recorded so the trace never lies about what the client actually sent:

      - empty tool-call arguments, which 500 the whole conversation upstream
      - the `model` field, because mlx_lm.server has no --alias and only
        answers to the full HuggingFace repo id, while the client sent an alias
      - whatever sampling parameters the settings say to strip, so the model's
        own tuned defaults apply instead of the client's
      - named blocks of the system prompt, when a replacement is configured for
        this client
      - the tool array, when a tool set is configured for this client
      - a run of identical tool calls in the history, when this client has a
        loop-collapse threshold set
    """
    toolset = settings.toolset_for(client) if settings is not None else None
    loop_at = settings.loop_threshold_for(client) if settings is not None else 0
    snapshot = (copy.deepcopy(request)
                if repair or current or toolset or loop_at else None)
    changed: dict = {}

    if repair:
        repaired = tracing.repair_tool_args(request)
        if repaired:
            changed["repaired_tool_args"] = repaired

    if settings is not None:
        stripped = settings.apply(request)
        if stripped:
            changed["stripped_params"] = stripped

        file, replace = settings.prompt_for(client)
        if file and replace:
            # The first system message, not messages[0] - a client is free to
            # put something else first, and rewriting the wrong message would
            # be silent.
            for message in request.get("messages") or []:
                if message.get("role") != "system":
                    continue
                body = message.get("content")
                if not isinstance(body, str):
                    break
                try:
                    # Read per turn rather than caching: editing a prompt file
                    # then takes effect on the next request instead of needing
                    # an engine restart, which is the whole point of iterating
                    # on one. set_prompt() stores a bare name, so ValueError
                    # here means settings.json was hand-edited into a path.
                    sections = prompts_mod.load(file)
                except ValueError as exc:
                    global _WARNED_PROMPT_PATH
                    if not _WARNED_PROMPT_PATH:
                        _WARNED_PROMPT_PATH = True
                        print(yellow(f"  bad prompt setting: {exc}"))
                    break
                rewritten, blocks = prompts_mod.rewrite(body, sections, replace)
                if blocks:
                    message["content"] = rewritten
                    changed["rewrote_prompt"] = {
                        "file": file, "blocks": blocks,
                        "chars_before": len(body), "chars_after": len(rewritten),
                    }
                break

        if toolset:
            try:
                # Read per turn, for the same reason the prompt file is: edit a
                # tool set mid-session and the next request uses it.
                allowed = toolsets_mod.load(toolset)
            except ValueError as exc:
                global _WARNED_TOOLSET_PATH
                if not _WARNED_TOOLSET_PATH:
                    _WARNED_TOOLSET_PATH = True
                    print(yellow(f"  bad toolset setting: {exc}"))
                allowed = []
            kept, dropped, unknown = toolsets_mod.apply(
                request.get("tools"), allowed)
            if dropped:
                request["tools"] = kept
                changed["filtered_tools"] = {
                    "file": toolset, "kept": len(kept),
                    "dropped": dropped, "unknown": unknown,
                    # What the set matched this turn. A later request missing
                    # one of these is a tool the client has stopped offering -
                    # which is the only kind of `unknown` worth an alert.
                    "matched": [(t.get("function") or {}).get("name")
                                for t in kept],
                }

        if loop_at:
            # After the tool filter, and on the messages rather than the tools:
            # a run of byte-identical calls with byte-identical results is a
            # model stuck re-deriving the same answer from a context that only
            # gains another copy of it each turn. See loops.py for why the
            # replacement carries the result and never the arguments.
            collapsed, runs = loops_mod.collapse(
                request.get("messages") or [], loop_at)
            if runs:
                request["messages"] = collapsed
                changed["collapsed_loops"] = runs

    sent = request.get("model")
    wanted = current.upstream_model_id
    if wanted and sent != wanted:
        request["model"] = wanted
        changed["rewrote_model"] = [sent, wanted]

    if not changed:
        snapshot = None
    return json.dumps(request).encode(), snapshot, changed


class EngineServer(ThreadingHTTPServer):
    """ThreadingHTTPServer without the traceback for a peer hanging up.

    socketserver prints a full traceback to stderr for any exception that
    escapes a handler - including a client resetting an idle keep-alive socket
    while the handler waits for its next request line. VS Code does that
    whenever it tears its connection pool down, and forty lines of
    ConnectionResetError on the console read as the engine crashing when
    nothing happened at all. Anything else still prints: a real handler bug
    must stay loud.
    """

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], (ConnectionResetError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)


class EngineHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # Set by main()
    engine = None  # the Engine instance

    def log_message(self, fmt, *args):  # silence the default access log
        pass

    # --------------------------------------------------------
    # Routing
    # --------------------------------------------------------
    def do_GET(self):
        route = self._route()
        if route in ("/v1/models", "/models"):
            return self._models()
        if route == "/health":
            return self._health()
        if route == "/status":
            return self._json(200, self.engine.supervisor.status())
        if self.engine.dashboard is not None:
            handled = self.engine.dashboard.get(self, route)
            if handled:
                return
        return self._relay("GET")

    def do_POST(self):
        route = self._route()
        if self.engine.dashboard is not None:
            handled = self.engine.dashboard.post(self, route)
            if handled:
                return
        if route in ("/load", "/api/load"):
            return self._load()
        if route in ("/unload", "/api/unload"):
            self.engine.supervisor.unload()
            return self._json(200, {"state": "idle"})
        return self._relay("POST")

    def do_DELETE(self):
        return self._relay("DELETE")

    def _route(self) -> str:
        path = self.path.split("?")[0]
        return path.rstrip("/") or "/"

    # --------------------------------------------------------
    # Engine-owned endpoints
    # --------------------------------------------------------
    def _models(self) -> None:
        """Every model this Mac can serve - not just the loaded one.

        The server behind us answers /v1/models with the single id it happens
        to be serving. Copying that would mean a client's model picker only
        ever offers the model already running, and no way to ask for another -
        which is the entire feature. So this is the registry's answer.
        """
        engine = self.engine
        loaded = engine.supervisor.status().get("model")
        data = []
        for info in engine.infos:
            if info.test_only:
                continue
            if not info.routable and not engine.serve_deprecated:
                continue
            data.append({
                "id": info.alias,
                "object": "model",
                "created": int(engine.started_at),
                "owned_by": info.engine,
                # Extra keys ride along; OpenAI clients ignore what they don't
                # know, and it makes `curl /v1/models | jq` worth typing.
                "_engine": info.engine,
                "_summary": info.summary,
                "_ctx": info.ctx,
                "_family": info.family,
                "_deprecated": info.deprecated,
                "_loaded": info.alias == loaded,
            })
        self._json(200, {"object": "list", "data": data})

    def _health(self) -> None:
        """200 whenever the engine is alive.

        The engine *is* healthy when idle - it can serve any model on demand -
        so the nuance lives in `state` rather than in the status code.
        """
        status = self.engine.supervisor.status()
        self._json(200, {"status": "ok", "state": status["state"],
                         "model": status["model"], "pulling": status.get("pulling"),
                         "port": self.engine.port})

    def _load(self) -> None:
        body = self._read_body()
        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError:
            payload = {}
        alias = self.engine.resolve(payload.get("model"))
        if alias is None:
            return self._raw(404, _error("model_not_found",
                                         f"unknown model {payload.get('model')!r}"))
        try:
            self.engine.supervisor.acquire(alias)
            self.engine.supervisor.release()
        except StartupFailed as exc:
            return self._raw(503, _error("model_start_failed", exc.message()))
        self._json(200, self.engine.supervisor.status())

    # --------------------------------------------------------
    # Relay
    # --------------------------------------------------------
    def _relay(self, method: str) -> None:
        engine = self.engine
        route = self._route()
        body = self._read_body()

        request_json = None
        if body:
            try:
                request_json = json.loads(body)
            except json.JSONDecodeError:
                pass

        is_chat = route in CHAT_PATHS and isinstance(request_json, dict)
        is_anthropic = route in ANTHROPIC_PATHS and isinstance(request_json, dict)

        if is_chat or is_anthropic:
            requested = request_json.get("model")
            alias = engine.resolve(requested)
            if alias is None:
                alias = engine.fallback(requested)
                if alias is None:
                    return self._raw(404, _error(
                        "model_not_found",
                        f"unknown model {requested!r}. Known: "
                        + ", ".join(engine.routable_aliases())))
            try:
                current = engine.supervisor.acquire(alias)
            except StartupFailed as exc:
                print(red(f"  {exc.alias}: {exc.reason}"), flush=True)
                return self._raw(503, _error("model_start_failed", exc.message()))
        else:
            # Anything else is relayed only if something is already serving.
            # A GET must never boot a 30GB model: llama.cpp clients poll /props
            # and /slots, and doing that would be absurd.
            status = engine.supervisor.status()
            if status["state"] != State.READY.value:
                return self._raw(503, _error(
                    "no_model_loaded",
                    "no model loaded - send a chat completion to start one"))
            try:
                current = engine.supervisor.acquire(status["model"])
            except StartupFailed as exc:
                return self._raw(503, _error("model_start_failed", exc.message()))
            alias, requested = status["model"], status["model"]

        epoch = current.epoch
        trace = is_chat or is_anthropic
        # From the history as the client sent it. prepare_upstream may collapse
        # a loop out of it, and a conversation that got shorter reads to the
        # tracker as a new one starting.
        session = (engine.sessions.identify(request_json, self.client_ip())
                   if trace else None)
        # MTPLX keeps a KV cache per session and takes the session from a
        # header, guessing from the prompt prefix only when there is none. VS
        # Code sends none it reads, and on 11 September the guess failed past
        # four times the shared system prompt: every turn became a new session
        # with its own copy of the context, and 25 turns took MTPLX from 36 GB
        # to its 96 GiB cap. So the engine names the conversation - unless the
        # client named it already, or another request of the same conversation
        # is still in flight, which MTPLX would 409.
        pinned = None
        if (session and current.engine.startswith("mtplx")
                and not self._client_names_session()
                and engine.sessions.claim(session)):
            pinned = session
        try:
            if is_anthropic and not current.engine.startswith("mtplx"):
                return self._raw(404, _error(
                    "unsupported_api",
                    f"/v1/messages is an MTPLX route; {alias} runs on "
                    f"{current.engine}, which serves the OpenAI API only"))
            self._forward(method, route, body, request_json, current,
                          epoch, alias, requested, trace=trace,
                          api="anthropic" if is_anthropic else "openai",
                          session=session, pinned=pinned)
        finally:
            if pinned:
                engine.sessions.release(pinned)
            engine.supervisor.release()

    def _forward(self, method, route, body, request_json, current, epoch,
                 alias, requested, trace: bool, api: str,
                 session: str | None = None, pinned: str | None = None) -> None:
        engine = self.engine
        snapshot, changed = None, {}

        # Identified before the rewrite, because the prompt to substitute is
        # chosen per client - and reused for the record so both agree.
        client = clients_mod.identify(request_json,
                                      self.headers.get("User-Agent"))
        if isinstance(request_json, dict) and api == "openai":
            body, snapshot, changed = prepare_upstream(
                request_json, current, engine.repair, engine.settings, client)
            if changed.get("repaired_tool_args"):
                tracing.note("repaired empty tool arguments: "
                             + ", ".join(changed["repaired_tool_args"]))
            # Deduped by content, not by a once-per-run flag: the flag
            # reported the first state and hid every later one, which is
            # precisely the case that matters - a client update changing what
            # it sends halfway through a session.
            if changed.get("rewrote_prompt"):
                rp = changed["rewrote_prompt"]
                engine.events.note(
                    "prompt", client,
                    f"rewriting system prompt from {rp['file']}: "
                    f"{', '.join(rp['blocks'])} "
                    f"({rp['chars_before']:,} -> {rp['chars_after']:,} chars)",
                    rp)
            if changed.get("filtered_tools"):
                ft = changed["filtered_tools"]
                text = (f"filtering tools from {ft['file']}: "
                        f"{ft['kept']} kept, {len(ft['dropped'])} dropped")
                if ft["unknown"]:
                    # Drift: the set names tools this client no longer sends,
                    # so the model has silently lost them. A rename in a client
                    # release looks exactly like this.
                    text += ("; set names tools not offered: "
                             + ", ".join(ft["unknown"]))
                engine.events.note("tools", client, text, ft,
                                   warn=bool(ft["unknown"]))
            if changed.get("collapsed_loops"):
                # Not deduped by content like the others - every intervention
                # is its own event, because each one is the engine changing a
                # live conversation and you want them all on the page.
                for run in changed["collapsed_loops"]:
                    engine.events.record(
                        "loop", client,
                        f"collapsed {run['repeats']} identical {run['tool']} "
                        f"calls ({run['messages_removed']} messages removed)",
                        run)
            if changed.get("stripped_params"):
                engine.events.note(
                    "strip", client,
                    "stripping client params: "
                    + ", ".join(changed["stripped_params"])
                    + " (model defaults apply)",
                    {"stripped": changed["stripped_params"]})
        elif isinstance(request_json, dict):
            body = json.dumps(request_json).encode()

        url = f"http://{engine.upstream_host}:{engine.upstream_port}{self.path}"
        headers = self._forward_headers()
        if pinned:
            headers[MTPLX_SESSION_HEADER] = pinned
        request = urllib.request.Request(
            url, data=body, headers=headers, method=method)

        started = time.perf_counter()
        try:
            response = urllib.request.urlopen(request, timeout=3600)
        except urllib.error.HTTPError as exc:
            payload = exc.read()
            self._raw(exc.code, payload)
            if exc.code in (401, 403):
                engine.warn_auth(exc.code, alias, current, self._client_auth())
            # Record it too. A failing request is the one you most want to
            # inspect, and dropping it leaves the trace ending just before
            # whatever broke.
            if trace:
                self._record(route, request_json,
                             {"_http_error": exc.code,
                              "_body": payload[:4000].decode("utf-8", "replace")},
                             False,
                             {"total_ms": round((time.perf_counter() - started) * 1000, 1)},
                             snapshot, changed, current, alias, requested, api,
                             preempted=False, client=client,
                             session=session, pinned=pinned)
            return
        except (urllib.error.URLError, TimeoutError) as exc:
            return self._upstream_gone(exc, epoch, started, route, request_json,
                                       snapshot, changed, current, alias,
                                       requested, api, trace, streamed=False,
                                       client=client, session=session,
                                       pinned=pinned)

        content_type = response.headers.get("Content-Type", "")
        streaming = "text/event-stream" in content_type

        preempted = False
        try:
            with response:
                if streaming:
                    raw, timing, gone = self._pump_stream(response, started)
                else:
                    raw, timing, gone = self._pump_buffered(response, started)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return self._upstream_gone(exc, epoch, started, route, request_json,
                                       snapshot, changed, current, alias,
                                       requested, api, trace, streamed=streaming,
                                       client=client, session=session,
                                       pinned=pinned)

        if engine.supervisor.preempted_since(epoch):
            preempted = True

        if not trace:
            return

        if streaming:
            response_json = reassemble(raw)
        else:
            try:
                response_json = json.loads(raw)
            except json.JSONDecodeError:
                response_json = {"_unparsed": raw[:2000].decode("utf-8", "replace")}

        self._record(route, request_json, response_json, streaming, timing,
                     snapshot, changed, current, alias, requested, api,
                     preempted, client, client_disconnected=gone,
                     session=session, pinned=pinned)

    def _upstream_gone(self, exc, epoch, started, route, request_json, snapshot,
                       changed, current, alias, requested, api, trace,
                       streamed: bool, client=None, session=None,
                       pinned=None) -> None:
        """The upstream vanished. Was that us, or did it crash?

        Both look identical from here, and conflating them makes the next real
        crash invisible - so the epoch decides. It changes only when a swap
        completed, which is exactly the deliberate case.
        """
        engine = self.engine
        preempted = engine.supervisor.preempted_since(epoch)
        timing = {"total_ms": round((time.perf_counter() - started) * 1000, 1)}

        if not streamed:
            kind = "model_swapped" if preempted else "upstream_unreachable"
            message = (f"model swapped to {engine.supervisor.status()['model']} "
                       f"mid-request" if preempted else f"upstream unreachable: {exc}")
            self._raw(503 if preempted else 502, _error(kind, message))
        # A streamed response has already had its headers sent and is
        # Connection: close, so simply stopping terminates it cleanly.

        if trace:
            self._record(route, request_json,
                         {"_interrupted": str(exc)}, streamed, timing,
                         snapshot, changed, current, alias, requested, api,
                         preempted, client, session=session, pinned=pinned)
        if not preempted:
            print(red(f"  upstream gone: {exc}"), flush=True)

    def _record(self, route, request_json, response_json, streaming, timing,
                snapshot, changed, current, alias, requested, api, preempted,
                client=None, client_disconnected=False, session=None,
                pinned=None):
        engine = self.engine
        routing = {
            "requested_model": requested,
            "alias": alias,
            "engine": current.engine,
            "upstream_model": current.upstream_model_id,
            "load_ms": round(current.load_ms, 1) or None,
            # Never the key - just whether one arrived and whether it matched.
            "client_auth": self._client_auth(),
            "client_ip": self.client_ip(),
            # Names only, never values - the values include the API key. What
            # a client actually sends is otherwise guesswork from its source.
            "client_headers": sorted({k.lower() for k in self.headers.keys()}),
            # The id MTPLX was told, so a trace turn lines up with MTPLX's own
            # request log. None when the engine did not name the session.
            "upstream_session": pinned,
        }
        seq = engine.tracer.record(
            path=route,
            request=request_json,
            response=response_json,
            streaming=streaming,
            timing=timing,
            original=snapshot,
            repaired=changed.get("repaired_tool_args"),
            rewrote_model=changed.get("rewrote_model"),
            stripped_params=changed.get("stripped_params"),
            rewrote_prompt=changed.get("rewrote_prompt"),
            filtered_tools=changed.get("filtered_tools"),
            collapsed_loops=changed.get("collapsed_loops"),
            client=client,
            routing=routing,
            preempted=preempted,
            client_disconnected=client_disconnected,
            api=api,
        )
        if engine.usage is not None:
            engine.usage.record_request(
                seq=seq, alias=alias, requested=requested, current=current,
                path=route, streaming=streaming, timing=timing,
                request=request_json, response=response_json,
                preempted=preempted, trace_file=engine.tracer.path,
                client_ip=self.client_ip(),
                session_id=session,
                # The header is read, never stored - only the label from it.
                client=client)

    # --------------------------------------------------------
    # HTTP plumbing
    # --------------------------------------------------------
    def _read_body(self) -> bytes | None:
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else None

    def client_ip(self) -> str:
        """The peer on the other end of the socket - or, behind a proxy the
        engine was told to trust, the client that proxy saw.

        The socket peer is the one thing a client cannot lie about, so it is
        the default. Behind the ingress every request arrives from the
        controller, and session identity - keyed on client IP - would merge
        every client into one; --trust-proxy is for exactly that.
        """
        if self.engine.trust_proxy:
            first = self.headers.get("X-Forwarded-For", "").split(",")[0].strip()
            if first:
                return first
        try:
            return self.client_address[0]
        except (AttributeError, IndexError):
            return ""

    def _client_names_session(self) -> bool:
        return any(key.lower() in CLIENT_SESSION_HEADERS
                   for key in self.headers.keys())

    def _client_auth(self) -> str:
        """How the client presented itself, without recording the credential.

        "absent" is the happy path for a client configured with no key at all;
        "mismatch" is the one that produces a 401 nobody can explain, so it is
        worth being able to see it in a trace afterwards. The key itself is
        never recorded - only whether it was there and whether it matched.
        """
        value = None
        for key, header in self.headers.items():
            if key.lower() == "authorization":
                value = header
                break
        if value is None:
            return "absent"

        raw = value.strip()
        parts = raw.split(None, 1)
        # "Bearer <token>", a bare token, or a scheme with nothing after it.
        if len(parts) == 2:
            token = parts[1].strip()
        elif raw.lower() in ("bearer", "basic", "token"):
            token = ""
        else:
            token = raw
        if not token:
            return "empty"
        return "match" if token == self.engine.api_key else "mismatch"

    def _forward_headers(self) -> dict:
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in HOP_BY_HOP
        }
        # One key for the whole system: the engine is started with it, launches
        # the model server with it, and clients send it. MTPLX genuinely checks
        # it - /health 401s without a bearer - while llama-server and
        # mlx_lm.server ignore it entirely.
        #
        # The client's header flows downstream untouched. If the two disagree we
        # want the 401, not a silent success against a key the user didn't think
        # they were using. Fill one in only when nothing was sent, so a client
        # configured with no key at all still works.
        if not any(k.lower() == "authorization" for k in headers):
            headers["Authorization"] = f"Bearer {self.engine.api_key}"
        return headers

    def _pump_stream(self, response,
                     started: float) -> tuple[bytes, dict, bool]:
        """Relay SSE chunk-by-chunk, timestamping the first tokens as they pass.

        Streamed responses have no length up front, so the connection is closed
        to signal the end rather than re-chunking on the way out.
        """
        self.send_response(response.status)
        for key, value in response.getheaders():
            if key.lower() not in HOP_BY_HOP:
                self.send_header(key, value)
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        chunks: list[bytes] = []
        probe = FirstTokenProbe()
        gone = False

        while True:
            chunk = response.read1(8192)
            if not chunk:
                break
            probe.feed(chunk, (time.perf_counter() - started) * 1000)
            chunks.append(chunk)
            try:
                self.wfile.write(chunk)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                # Client gave up; keep what we have. Remembered, because the
                # turn otherwise reads as an empty answer from the model - and
                # the usual cause is a client timing out on a long cold
                # prefill, which is the model working, not failing. Noticed
                # only on the next write, so up to one MTPLX keep-alive (5s)
                # after the client actually left.
                gone = True
                break

        total = (time.perf_counter() - started) * 1000
        timing = probe.as_dict()
        timing["total_ms"] = round(total, 1)
        return b"".join(chunks), timing, gone

    def _pump_buffered(self, response,
                       started: float) -> tuple[bytes, dict, bool]:
        raw = response.read()
        total = (time.perf_counter() - started) * 1000

        self.send_response(response.status)
        for key, value in response.getheaders():
            if key.lower() not in HOP_BY_HOP:
                self.send_header(key, value)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        gone = False
        try:
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            gone = True

        # Non-streamed: nothing is visible until it all arrives, so the only
        # honest first-token number is the same as the total.
        return raw, {"ttfb_ms": round(total, 1), "ttft_ms": None,
                     "ttfc_ms": None, "total_ms": round(total, 1)}, gone

    def _json(self, status: int, payload) -> None:
        self._raw(status, json.dumps(payload).encode(), "application/json")

    def _raw(self, status: int, payload: bytes,
             content_type: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass


# ============================================================
# The engine
# ============================================================
class Engine:
    def __init__(self, args):
        self.args = args
        self.host = self._model_host()
        self.machine = self._machine()
        self.registry = models.load_all()
        self.infos = models.describe(self.machine, self.registry)
        self.serve_deprecated = args.serve_deprecated
        self.index = models.build_index(self.infos, deprecated=args.serve_deprecated)

        self.port = args.port
        self.upstream_port = args.upstream_port
        self.api_key = args.api_key
        self.repair = not args.no_repair
        self.on_unknown = args.on_unknown
        self.trust_proxy = getattr(args, "trust_proxy", False)
        self.client_endpoints = CLIENT_ENDPOINTS or None
        self.default_model = args.default_model
        self.started_at = time.time()
        self._warned_unknown: set = set()
        self.sessions = sessions_mod.SessionTracker()
        self.settings = settings_mod.Settings(
            getattr(args, "settings", DEFAULT_SETTINGS_FILE))
        self._warned_auth: set = set()
        self.dashboard = None
        self.usage = None
        # Console-only until attach_usage() hands it somewhere to persist to,
        # so a run with --no-usage still warns about drift.
        self.events = events_mod.Events(None, quiet=args.quiet)

        opts = Options(port=args.upstream_port, thinking=args.thinking,
                       spec_decode=args.spec_decode, ctx=args.ctx,
                       reasoning_effort=getattr(args, 'reasoning_effort', None),
                       sandbox=getattr(args, "sandbox", True),
                       bind=getattr(args, "model_bind", None) or MODEL_BIND)
        self.supervisor = Supervisor(
            self.machine, self.registry, self.infos, opts,
            port=args.upstream_port, log_dir=args.log_dir,
            api_key=args.api_key, start_timeout=args.start_timeout,
            leave_running=args.leave_running, quiet=args.quiet,
            host=self.host, pull=getattr(args, "pull", AUTO_PULL))

        os.makedirs(args.trace_dir, exist_ok=True)
        name = args.name or f"trace-{datetime.now().strftime('%Y%m%d-%H%M%S')}.jsonl"
        if not name.endswith(".jsonl"):
            name += ".jsonl"
        self.tracer_dir = args.trace_dir
        self.tracer = tracing.TraceWriter(
            os.path.join(args.trace_dir, name),
            quiet=args.quiet, enabled=not args.no_trace)

    @property
    def upstream_host(self) -> str:
        """Where the model port is - this machine, or the model host."""
        return self.supervisor.host.address

    def _model_host(self):
        """LocalHost, unless config.json names a model host (remote.py)."""
        if not MODEL_HOST:
            return LocalHost(bind=getattr(self.args, "model_bind", None) or MODEL_BIND)
        config = ModelHostConfig.from_dict(MODEL_HOST)
        bind = getattr(self.args, "model_bind", None) or MODEL_BIND
        if bind in ("127.0.0.1", "localhost") and config.address not in (
                "host.docker.internal", "localhost", "127.0.0.1"):
            # Model servers on the Mac's loopback can't be reached from another
            # machine, and the symptom is a launch that never becomes ready.
            print(yellow(f"WARNING: model_bind is {bind}, so model servers on "
                         f"{config.address} only listen on its loopback and this "
                         f"engine cannot reach them. Set model_bind (and a real "
                         f"api_key) in the engine's config."), flush=True)
        return RemoteHost(config,
                          local_sha=os.environ.get("INFERENCE_GIT_SHA"),
                          on_drift=self._drift)

    def _machine(self) -> Machine:
        if not self.host.remote:
            return detect_machine()
        try:
            return Machine(**self.host.machine())
        except (RemoteError, TypeError) as exc:
            # A sleeping Mac must not crash-loop the engine. The profile is
            # enough to list models; the first launch finds out if it's awake.
            print(f"  WARNING: model host unavailable ({exc}) - using the "
                  f"{self.host.config.machine!r} profile", flush=True)
            return machine_from_profile(self.host.config.machine)

    def _drift(self, engine_sha: str, host_sha: str) -> None:
        self.events.note(
            "drift", None,
            f"model host checkout is at {host_sha} but this engine was built "
            f"from {engine_sha} - model files may differ",
            {"engine": engine_sha, "host": host_sha}, warn=True)

    def attach_usage(self, db) -> None:
        """Wire the stats database in, callbacks and all.

        One place, because the CLI and the test harness both do this and a
        callback added to one but not the other is a failure nobody notices
        until a row turns out to be missing - which is exactly how failed
        loads went unrecorded.
        """
        self.usage = db
        self.events.usage = db
        db.attach(self.supervisor)
        self.supervisor.on_load = db.record_load
        self.supervisor.on_unload = db.record_unload
        self.supervisor.on_failure = db.record_failure

    def routable_aliases(self) -> list[str]:
        return [i.alias for i in self.infos
                if i.routable or self.serve_deprecated]

    def resolve(self, name: str | None) -> str | None:
        return models.resolve(name, self.index)

    def fallback(self, requested: str | None) -> str | None:
        """What to serve when the client asked for something we don't know.

        Copilot and Cline both occasionally send an id from a stale profile. A
        404 reads to them as "endpoint broken" and ends the session, when the
        box has one obvious model to offer. So the default is to serve whatever
        is loaded, warn once, and let the trace record what was really asked
        for.
        """
        if self.on_unknown == "404":
            return None

        deprecated = self._deprecated_alias(requested)
        loaded = self.supervisor.status().get("model")
        target = loaded if (self.on_unknown == "current" and loaded) else self.default_model

        if requested not in self._warned_unknown:
            self._warned_unknown.add(requested)
            if deprecated:
                tracing.note(f"{requested} is deprecated ({deprecated}) "
                             f"-> serving {target}")
                tracing.dim("restart with --serve-deprecated to route to it")
            else:
                tracing.note(f"unknown model {requested!r} -> serving {target}")
        return target

    def warn_auth(self, code: int, alias: str, current, shape: str) -> None:
        """Say out loud that a request was rejected on its API key.

        The turn line already shows the status, but the *reason* is worth
        spelling out once: the engine passes the client's key through rather
        than substituting its own, so a 401 here means the client and the model
        server disagree about the key - not that the model is broken. Printed
        once per model, because a misconfigured client will retry forever.
        """
        seen = (code, alias, shape)
        if seen in self._warned_auth:
            return
        self._warned_auth.add(seen)
        tracing.note(f"{code} from {current.engine} serving {alias} - "
                     f"{EXPLAIN_AUTH.get(shape, 'the API key was rejected upstream')}")
        tracing.dim(f"the engine forwards the client's key unchanged; this model "
                    f"was started with --api-key {self.api_key!r}")

    def _deprecated_alias(self, requested: str | None) -> str | None:
        if not requested:
            return None
        full = models.build_index(self.infos, deprecated=True)
        alias = models.resolve(requested, full)
        info = next((i for i in self.infos if i.alias == alias), None)
        return info.deprecated if info else None

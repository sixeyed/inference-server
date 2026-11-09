"""A model server that isn't one.

The engine's behaviour is almost entirely about what it does around a model
server - starting one, deciding it is ready, killing it, labelling what happened
when it vanishes. None of that can be tested against a real model: the weights
are tens of GB and a load takes minutes.

So this impersonates one. It speaks the three engines' health and model routes
faithfully, including the bits that actually bite - MTPLX 401s a probe with no
bearer, llama-server answers `{"status":"ok"}` where MTPLX answers `{"ok":true}`
- and it can be told to be slow, to die during startup, or to die mid-stream,
which is the only way to test that a preempted turn is told apart from a crash.

Standard library only, like everything else here.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeModelServer:
    """An OpenAI-compatible server with a scripted personality.

    engine      "llama-server" | "mtplx" | "mlx_lm.server" - decides the health
                shape, whether a bearer is required, and whether `timings` is
                published.
    model_id    what /v1/models reports and what `model` must be.
    ready_after seconds of answering 503 before reporting healthy, so a cold
                start is a real wait rather than an instant one.
    """

    def __init__(self, model_id: str = "fake-model", engine: str = "llama-server",
                 api_key: str = "none", ready_after: float = 0.0,
                 chunks: list[str] | None = None, reasoning: list[str] | None = None,
                 tool_calls: list[dict] | None = None,
                 chunk_delay: float = 0.0, timings: bool = True,
                 empty_tool_args: bool = False):
        self.model_id = model_id
        self.engine = engine
        self.api_key = api_key
        self.ready_after = ready_after
        self.chunks = chunks if chunks is not None else ["hello", " world"]
        self.reasoning = reasoning or []
        self.tool_calls = tool_calls or []
        self.chunk_delay = chunk_delay
        self.timings = timings
        self.empty_tool_args = empty_tool_args

        self.started_at = time.monotonic()
        self.requests: list[dict] = []
        self.headers_seen: list[dict] = []
        self.die_mid_stream = False   # set to cut the next response in half
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.port = 0

    # --------------------------------------------------------
    def start(self, port: int = 0) -> int:
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def _authorised(self) -> bool:
                # Only MTPLX checks. The other two ignore the header entirely,
                # which is the difference the engine has to cope with.
                if fake.engine != "mtplx":
                    return True
                header = self.headers.get("Authorization", "")
                return header == f"Bearer {fake.api_key}"

            def _send(self, status, payload: bytes, ctype="application/json"):
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                path = self.path.split("?")[0].rstrip("/") or "/"
                if not self._authorised():
                    return self._send(401, b'{"error":"unauthorized"}')

                if path == "/health":
                    if fake.engine == "mlx_lm.server":
                        return self._send(404, b'{"error":"not found"}')
                    if time.monotonic() - fake.started_at < fake.ready_after:
                        return self._send(503, b'{"error":"loading"}')
                    body = ({"ok": True, "model": fake.model_id}
                            if fake.engine == "mtplx" else {"status": "ok"})
                    return self._send(200, json.dumps(body).encode())

                if path in ("/v1/models", "/models"):
                    if time.monotonic() - fake.started_at < fake.ready_after:
                        return self._send(503, b'{"error":"loading"}')
                    return self._send(200, json.dumps({
                        "object": "list",
                        "data": [{"id": fake.model_id, "object": "model"}],
                    }).encode())

                return self._send(404, b'{"error":"not found"}')

            def do_POST(self):
                path = self.path.split("?")[0].rstrip("/")
                if not self._authorised():
                    return self._send(401, b'{"error":"unauthorized"}')

                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    request = json.loads(raw)
                except json.JSONDecodeError:
                    request = {}
                fake.requests.append(request)
                fake.headers_seen.append(dict(self.headers))

                if path not in ("/v1/chat/completions", "/v1/messages"):
                    return self._send(404, b'{"error":"not found"}')

                if request.get("stream"):
                    return self._stream()
                return self._send(200, json.dumps(fake.completion()).encode())

            def _stream(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                for event in fake.events():
                    if fake.die_mid_stream:
                        try:
                            self.connection.close()
                        except OSError:
                            pass
                        return
                    try:
                        self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        return  # the engine dropped us, as a real one would
                    if fake.chunk_delay:
                        time.sleep(fake.chunk_delay)
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

        self._server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self.port

    def stop(self) -> None:
        """Go away the way a killed server goes away.

        ThreadingHTTPServer.shutdown() only stops the accept loop - in-flight
        handler threads carry on, so without this a "stopped" fake would happily
        finish streaming and the preemption tests would pass for the wrong
        reason. A real server that has been SIGTERMed drops the socket
        mid-sentence, so this does too.
        """
        self.die_mid_stream = True
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    # --------------------------------------------------------
    # What it says
    # --------------------------------------------------------
    def _timings(self) -> dict:
        return {"prompt_n": 20, "predicted_n": len(self.chunks),
                "prompt_ms": 100.0, "predicted_ms": 200.0,
                "prompt_per_second": 200.0, "predicted_per_second": 50.0}

    def events(self):
        """Reasoning first, then content, then tool calls, then usage.

        The order matters: ttft fires on the first token of any kind and ttfc
        only on a visible one, so a model that reasons before answering is the
        only way to produce a non-zero deferral.
        """
        base = {"id": "chatcmpl-fake", "object": "chat.completion.chunk",
                "model": self.model_id}
        for text in self.reasoning:
            yield {**base, "choices": [{"index": 0,
                                        "delta": {"reasoning_content": text}}]}
        for text in self.chunks:
            yield {**base, "choices": [{"index": 0, "delta": {"content": text}}]}
        for index, call in enumerate(self.tool_calls):
            yield {**base, "choices": [{"index": 0, "delta": {"tool_calls": [{
                "index": index, "id": f"call_{index}", "type": "function",
                "function": {"name": call["name"],
                             "arguments": "" if self.empty_tool_args
                             else json.dumps(call.get("arguments", {}))},
            }]}}]}
        final = {**base, "choices": [{"index": 0, "delta": {},
                                      "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 20,
                           "completion_tokens": len(self.chunks),
                           "total_tokens": 20 + len(self.chunks)}}
        if self.timings:
            final["timings"] = self._timings()
        yield final

    def completion(self) -> dict:
        message = {"role": "assistant", "content": "".join(self.chunks)}
        if self.reasoning:
            message["reasoning_content"] = "".join(self.reasoning)
        if self.tool_calls:
            message["tool_calls"] = [{
                "id": f"call_{i}", "type": "function",
                "function": {"name": c["name"],
                             "arguments": "" if self.empty_tool_args
                             else json.dumps(c.get("arguments", {}))},
            } for i, c in enumerate(self.tool_calls)]
        payload = {
            "id": "chatcmpl-fake", "object": "chat.completion",
            "model": self.model_id,
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": len(self.chunks),
                      "total_tokens": 20 + len(self.chunks)},
        }
        if self.timings:
            payload["timings"] = self._timings()
        return payload


class FakeLauncher:
    """Stands in for process.launch(), and records what it was asked to start.

    Also stands in for process.stop_all(), which matters more than it sounds:
    the real one matches on process name, so a test that used it would reach
    straight past these fakes and SIGTERM whatever model this machine actually
    has loaded.
    """

    def __init__(self, port: int, engine: str = "llama-server",
                 api_key: str = "none", fail: str | None = None,
                 ready_after: float = 0.0, **server_kwargs):
        self.port = port
        self.engine = engine
        self.api_key = api_key
        self.fail = fail            # message to die with, instead of serving
        self.ready_after = ready_after
        # Passed to every FakeModelServer this launcher makes, so a scenario
        # can ask for a slow stream, reasoning tokens, or broken tool calls.
        self.server_kwargs = server_kwargs
        self.started: list[str] = []
        self.stopped = 0
        # Repos "pulled" - recorded, never fetched. The real puller would start
        # downloading tens of GB from HuggingFace mid-test.
        self.pulled: list[str] = []
        self.pull_error: str | None = None
        self.server: FakeModelServer | None = None

    def pull(self, spec, say=print, cancel=None):
        if self.pull_error:
            from inference.weights import PullError
            raise PullError(self.pull_error)
        self.pulled.append(spec.hf_repo)
        return {"repo": spec.hf_repo}

    def launch(self, alias, spec, machine, opts, log_dir):
        from inference.process import ServerProcess
        self.started.append(alias)
        self.stop_all(quiet=True)

        model_id = getattr(spec, "alias", None) or getattr(spec, "model_id", None) \
            or getattr(spec, "hf_repo", None) or alias
        engine = type(spec).__name__.replace("ServerSpec", "").lower()
        kind = {"llama": "llama-server", "mtplx": "mtplx",
                "mlx": "mlx_lm.server"}.get(engine, "llama-server")

        import os
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{alias}-fake.log")
        with open(log_path, "w") as handle:
            handle.write(self.fail or "loading model\nready\n")

        if self.fail:
            popen = _DeadProcess()
        else:
            self.server = FakeModelServer(
                model_id=model_id, engine=kind, api_key=self.api_key,
                ready_after=self.ready_after, **self.server_kwargs)
            self.server.start(self.port)
            popen = _LiveProcess()

        return ServerProcess(alias=alias, engine=kind, port=self.port, argv=["fake"],
                             popen=popen, log_path=log_path,
                             started_at=time.monotonic())

    def stop_all(self, quiet: bool = False, engine: bool = False,
                 engine_port: int | None = None,
                 only_port: int | None = None) -> int:
        """Mirrors process.stop_all's signature.

        Kept in step deliberately: when the real one gained `only_port` and this
        did not, every swap raised TypeError inside a request handler and
        surfaced as a dropped connection rather than as a signature mismatch.
        """
        if self.server is not None:
            self.server.stop()
            self.server = None
            self.stopped += 1
            return 1
        return 0


# Deliberately not pid 0: getpgid(0) means "this process", so a fake with pid 0
# turns ServerProcess.terminate() into a suicide note. common.py guards against
# that now, but the fake should not be pretending to be something impossible.
_FAKE_PID = -1


class _LiveProcess:
    pid = _FAKE_PID

    def poll(self):
        return None


class _DeadProcess:
    pid = _FAKE_PID

    def poll(self):
        return 1

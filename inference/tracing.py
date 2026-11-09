"""Recording what actually went over the wire.

Chat completions is stateless - the client resends the whole conversation every
turn, and tool results (file contents, terminal output) are messages within it.
So a captured request is entirely self-contained: replaying it needs no
workspace, no repo, no client. That's the whole point.

Requests are recorded as the client sent them. Where the engine has to change
something on the way through - and there are exactly two places, empty tool-call
arguments and the `model` field - the untouched original is kept alongside, so
a trace never lies about what the client actually said.

Timings are measured at the proxy rather than taken from the server, so the
numbers mean the same thing for llama-server, mlx_lm.server and mtplx:

    ttfb_ms       first byte back from the server
    ttft_ms       first token of any kind - including reasoning
    ttfc_ms       first *visible* token - content or a tool call

ttfc_ms - ttft_ms is the deferral cost: how long a thinking model spends
reasoning before the client has anything to show.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from threading import Lock

from .console import grey, red, yellow


def repair_tool_args(request: dict) -> list[str]:
    """Replace empty tool-call arguments with `{}`.

    Some models emit `"arguments": ""` for a tool that needs none. The client
    echoes it back verbatim on the next turn, and llama.cpp then refuses the
    whole conversation:

        Failed to parse tool call arguments as JSON:
        parse error at line 1, column 1: attempting to parse an empty input

    which 500s every subsequent request and ends the session. `{}` is what the
    model meant, so this repairs it in flight. Mutates `request`; returns the
    names of the tools it touched, so the caller can record the change.
    """
    repaired = []
    for message in request.get("messages") or []:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            if function.get("arguments", "").strip() == "":
                function["arguments"] = "{}"
                repaired.append(function.get("name") or "?")
    return repaired


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def summarise(request: dict, response: dict) -> dict:
    """The few fields worth having without re-parsing the whole record."""
    messages = request.get("messages") or []
    tools = request.get("tools") or []
    message = (response.get("choices") or [{}])[0].get("message") or {}
    calls = message.get("tool_calls") or []
    usage = response.get("usage") or {}
    timings = response.get("timings") or {}

    # Two different quantities, easily confused:
    #
    #   usage.prompt_tokens   OpenAI semantics - the WHOLE input, cache included
    #   timings.prompt_n      llama.cpp - only the tokens actually processed
    #   timings.cache_n       the rest, restored from cache
    #
    # so prompt_n + cache_n == usage.prompt_tokens. `usage` only appears when
    # the client asks for it (stream_options.include_usage) and the proxy must
    # not add that - requests are recorded verbatim - so context falls back to
    # the sum. Adding cache_n to usage.prompt_tokens would count it twice.
    prefilled = timings.get("prompt_n")
    cache_n = timings.get("cache_n")
    context_tokens = usage.get("prompt_tokens")
    if context_tokens is None and prefilled is not None:
        context_tokens = prefilled + (cache_n or 0)

    return {
        "model": request.get("model"),
        "n_messages": len(messages),
        "n_tools": len(tools),
        "tool_names": [t.get("function", {}).get("name") for t in tools][:40],
        "request_chars": len(json.dumps(messages)),
        # Total conversation depth this request was served at.
        "context_tokens": context_tokens,
        # Tokens the server had to actually run through prefill this turn.
        "prefilled_tokens": prefilled,
        "completion_tokens": usage.get("completion_tokens",
                                       timings.get("predicted_n")),
        "cache_n": cache_n,
        "called_tools": [c.get("function", {}).get("name") for c in calls],
        "has_content": bool((message.get("content") or "").strip()),
        "has_reasoning": bool((message.get("reasoning_content") or "").strip()),
        "finish_reason": (response.get("choices") or [{}])[0].get("finish_reason"),
    }


class TraceWriter:
    """One JSONL file per engine run, one record per turn."""

    def __init__(self, path: str, quiet: bool = False, enabled: bool = True):
        self.path = path
        self.quiet = quiet
        self.enabled = enabled
        self._lock = Lock()
        self._seq = 0

    @property
    def count(self) -> int:
        return self._seq

    def record(self, *, path: str, request: dict, response: dict,
               streaming: bool, timing: dict, original: dict | None = None,
               repaired: list[str] | None = None,
               rewrote_model: list[str] | None = None,
               stripped_params: list[str] | None = None,
               rewrote_prompt: dict | None = None,
               filtered_tools: dict | None = None,
               client: str | None = None,
               collapsed_loops: list | None = None,
               routing: dict | None = None,
               preempted: bool = False,
               client_disconnected: bool = False,
               api: str = "openai") -> int:
        with self._lock:
            self._seq += 1
            seq = self._seq

        summary = summarise(request, response) if api == "openai" else None
        if summary is not None and routing:
            summary["upstream_model"] = routing.get("upstream_model")
        if summary is not None and rewrote_model:
            # `request` here is the body as sent upstream, so its `model` is the
            # rewritten one. The summary has to report what the *client* said:
            # list_traces, format_request and every replay tool read this field,
            # and an mlx model would otherwise show up under its HuggingFace
            # repo id in tooling that expects an alias.
            summary["model"] = rewrote_model[0]

        record = {
            "seq": seq,
            "ts": _now_iso(),
            "path": path,
            # Recorded rather than left to be re-derived. Filtering the tool
            # array removes the very names clients.identify recognises a client
            # by, so a stored request no longer identifies itself. The engine
            # identifies the client before it filters, so it already knows.
            "client": client,
            "stream": streaming,
            "api": api,
            "timing": timing,
            "request": request,
            "response": response,
            "summary": summary,
        }
        # Both mutations share one snapshot: what the client actually sent.
        if repaired:
            record["repaired_tool_args"] = repaired
        if rewrote_model:
            record["rewrote_model"] = rewrote_model
        if stripped_params:
            record["stripped_params"] = stripped_params
        if rewrote_prompt:
            record["rewrote_prompt"] = rewrote_prompt
        if filtered_tools:
            record["filtered_tools"] = filtered_tools
        if collapsed_loops:
            record["collapsed_loops"] = collapsed_loops
        if original is not None:
            record["request_original"] = original
        if routing:
            # Outside `timing` on purpose. A cold start is not a slow model,
            # and ttfc_ms has to keep meaning what it means in the traces
            # already on disk.
            record["routing"] = routing
        if preempted:
            record["preempted"] = True
        if client_disconnected:
            record["client_disconnected"] = True

        if self.enabled:
            with self._lock:
                with open(self.path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")

        if not self.quiet:
            self.print_turn(seq, record)
        return seq


    def print_turn(self, seq: int, record: dict) -> None:
        summary = record["summary"]
        timing = record["timing"]
        if summary is None:  # /v1/messages and friends
            print(f"  #{seq:<3}  {record['api']}  {record['path']}", flush=True)
            return

        bits = [
            f"#{seq:<3}",
            f"msgs={summary['n_messages']:<3}",
            f"tools={summary['n_tools']:<3}",
        ]
        if summary["context_tokens"] is not None:
            bits.append(f"ctx={summary['context_tokens']:<7}")
        if summary["prefilled_tokens"] is not None:
            bits.append(f"new={summary['prefilled_tokens']:<6}")
        if summary["cache_n"] is not None:
            bits.append(f"cached={summary['cache_n']:<7}")
        if summary["completion_tokens"] is not None:
            bits.append(f"out={summary['completion_tokens']:<5}")
        if timing.get("ttfc_ms"):
            bits.append(f"ttfc={timing['ttfc_ms'] / 1000:.1f}s")
        bits.append(f"wall={timing['total_ms'] / 1000:.1f}s")

        error = record.get("response", {}).get("_http_error")
        if record.get("preempted"):
            bits.append(yellow("-> PREEMPTED"))
        elif error:
            # Without this a 401 shows up as "EMPTY", which sends you looking
            # at the model when the problem is the request never reached it.
            label = f"-> HTTP {error}"
            if error in (401, 403):
                label += " unauthorized"
            bits.append(red(label))
        elif record.get("client_disconnected"):
            # Ahead of the tool calls and content: whatever the model produced
            # never reached the client. And not "EMPTY" - before the first
            # token it is nearly always a client timing out on a cold prefill,
            # which is the model working, not the model failing.
            label = "-> CLIENT DISCONNECTED"
            if not timing.get("ttft_ms"):
                label += " before first token"
            bits.append(yellow(label))
        elif summary["called_tools"]:
            bits.append("-> " + ",".join(filter(None, summary["called_tools"])))
        elif summary["has_reasoning"] and not summary["has_content"]:
            bits.append("-> REASONING ONLY")
        elif not summary["has_content"]:
            bits.append("-> EMPTY")
        print("  " + "  ".join(bits), flush=True)


def note(text: str) -> None:
    print(yellow(f"  {text}"), flush=True)


def dim(text: str) -> None:
    print(grey(f"  {text}"), flush=True)


def list_traces(trace_dir: str) -> int:
    if not os.path.isdir(trace_dir):
        print(f"No trace directory at {trace_dir}")
        return 1
    files = sorted(f for f in os.listdir(trace_dir) if f.endswith(".jsonl"))
    if not files:
        print(f"No traces in {trace_dir}")
        return 1

    for name in files:
        path = os.path.join(trace_dir, name)
        turns, seen, tools_seen, calls = 0, set(), 0, 0
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                turns += 1
                summary = record.get("summary") or {}
                if summary.get("model"):
                    seen.add(summary["model"])
                tools_seen = max(tools_seen, summary.get("n_tools") or 0)
                calls += len(summary.get("called_tools") or [])
        size_mb = os.path.getsize(path) / 1024**2
        print(
            f"  {name:<40} {turns:>4} turns  {calls:>4} tool calls  "
            f"{tools_seen:>3} tools  {size_mb:>6.1f}MB  {','.join(seen) or '?'}"
        )
    return 0

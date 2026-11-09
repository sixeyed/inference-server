"""Server-sent-event parsing shared by the capture proxy and the scenario runner.

Both need the same two things from a streamed completion: when the first token
arrived, and what the whole response was once it finished.
"""

from __future__ import annotations

import json
import re

# Probes run against raw chunks as they arrive, to timestamp the first token
# without parsing JSON in the hot path.
#
# `\s*:\s*` is load-bearing: llama.cpp emits compact JSON (`"content":"x"`) but
# mlx_lm.server is Python and emits `"content": "x"` with a space, so a literal
# `":"` silently never matches and every MLX model looks like it produced its
# first token with the last chunk.
#
# `[^"]` guards against the opening `"content":""` delta, which carries only the
# assistant role and is not a token.
RE_CONTENT = re.compile(rb'"content"\s*:\s*"[^"]')
RE_REASONING = re.compile(rb'"reasoning_content"\s*:\s*"[^"]')
RE_TOOL_CALL = re.compile(rb'"tool_calls"')

# A key can straddle a chunk boundary, so each scan re-reads this much of the
# previous chunk. Longer than the longest key we look for.
SCAN_OVERLAP = 32


class FirstTokenProbe:
    """Tracks when output first appeared, split by whether it was visible.

    ttfc - ttft is the deferral cost: time a thinking model spends reasoning
    before the client has anything to show a user.
    """

    def __init__(self):
        self.ttfb_ms: float | None = None
        self.ttft_ms: float | None = None  # any token, reasoning included
        self.ttfc_ms: float | None = None  # content or a tool call
        self._tail = b""

    def feed(self, chunk: bytes, elapsed_ms: float) -> None:
        if self.ttfb_ms is None:
            self.ttfb_ms = elapsed_ms
        if self.ttft_ms is not None and self.ttfc_ms is not None:
            return
        window = self._tail + chunk
        visible = RE_CONTENT.search(window) or RE_TOOL_CALL.search(window)
        if self.ttft_ms is None and (visible or RE_REASONING.search(window)):
            self.ttft_ms = elapsed_ms
        if self.ttfc_ms is None and visible:
            self.ttfc_ms = elapsed_ms
        self._tail = chunk[-SCAN_OVERLAP:]

    def as_dict(self) -> dict:
        return {
            "ttfb_ms": round(self.ttfb_ms, 1) if self.ttfb_ms else None,
            "ttft_ms": round(self.ttft_ms, 1) if self.ttft_ms else None,
            "ttfc_ms": round(self.ttfc_ms, 1) if self.ttfc_ms else None,
        }


# Fields the OpenAI streaming shape defines. Everything else on a chunk is an
# engine's own addition and gets carried through to the reassembled response.
_STREAM_STANDARD = {
    "id", "object", "created", "model", "choices", "usage", "timings",
    "system_fingerprint", "service_tier",
}


def reassemble(raw: bytes) -> dict:
    """Turn a streamed SSE body back into a non-streamed-shaped response."""
    content: list[str] = []
    reasoning: list[str] = []
    tool_calls: dict[int, dict] = {}
    finish_reason = None
    usage = None
    timings = None
    model = None
    # Carried through because they are only ever on the final chunk and they
    # are the whole per-request record for some engines - MTPLX puts its
    # 300-field diagnostic block on `mtplx_stats`, including decode rate, peak
    # memory and speculative-decoding acceptance. Dropping them here would mean
    # streamed turns, which is every real client, silently lost all of it.
    response_id = None
    extras: dict = {}
    chunks = 0

    for line in raw.split(b"\n"):
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if payload == b"[DONE]":
            continue
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue
        chunks += 1
        model = event.get("model", model)
        response_id = event.get("id") or response_id
        usage = event.get("usage") or usage
        timings = event.get("timings") or timings
        for key, value in event.items():
            # Anything the server bolts on beside the standard fields. Kept
            # whole rather than named, so a new engine's block survives too.
            if key not in _STREAM_STANDARD and value:
                extras[key] = value

        for choice in event.get("choices", []):
            finish_reason = choice.get("finish_reason") or finish_reason
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content.append(delta["content"])
            if delta.get("reasoning_content"):
                reasoning.append(delta["reasoning_content"])
            for call in delta.get("tool_calls") or []:
                index = call.get("index", 0)
                slot = tool_calls.setdefault(
                    index,
                    {"id": None, "type": "function",
                     "function": {"name": None, "arguments": ""}},
                )
                if call.get("id"):
                    slot["id"] = call["id"]
                function = call.get("function") or {}
                if function.get("name"):
                    slot["function"]["name"] = function["name"]
                if function.get("arguments"):
                    slot["function"]["arguments"] += function["arguments"]

    message = {"role": "assistant", "content": "".join(content)}
    if reasoning:
        message["reasoning_content"] = "".join(reasoning)
    if tool_calls:
        message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]

    result = {
        "id": response_id,
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "_reassembled_from_stream": True,
        "_stream_chunks": chunks,
    }
    if usage:
        result["usage"] = usage
    if timings:
        result["timings"] = timings
    result.update(extras)
    return result

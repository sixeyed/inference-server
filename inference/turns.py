"""Reading one turn back out of a trace.

The database holds what a turn cost; the JSONL holds what was actually said.
`trace_file` and `trace_seq` on every row join the two, and this is the join.

Two things shape what comes back.

**Only this turn's data.** Chat completions is stateless, so every request
carries the whole conversation - 256 messages and 66 tool schemas by turn 19 of
a real session, ~368KB against the ~4KB the same turn occupies in SQLite.
Rendering all of that to show one exchange is the wrong unit. What is new in
turn N is the tail beyond what turn N-1 sent: the tool results that came back,
and whatever the user typed. The database already records `n_messages` per turn,
so the boundary is known without reading the previous record.

**The trace may not be there.** It is gitignored, rotated per engine run, holds
source code, and `--no-trace` turns it off entirely. Every one of those is a
normal state rather than an error, so a missing trace reports why and the row's
metrics still stand on their own.
"""

from __future__ import annotations

import json
import os

# Per-message cap. A single tool result can be an entire file, and the point of
# this view is the shape of the exchange, not its contents.
MAX_CHARS = 4000


def _clip(text: str) -> tuple[str, bool]:
    if not isinstance(text, str) or len(text) <= MAX_CHARS:
        return text, False
    return text[:MAX_CHARS], True


def _message(raw: dict) -> dict:
    """One message, trimmed to what a reader needs."""
    content = raw.get("content")
    if isinstance(content, list):  # some clients send content parts
        content = " ".join(str(part.get("text", part)) for part in content)
    content, clipped = _clip(content or "")
    out = {"role": raw.get("role"), "content": content, "clipped": clipped}
    calls = []
    for call in raw.get("tool_calls") or []:
        function = call.get("function") or {}
        args, args_clipped = _clip(function.get("arguments") or "")
        calls.append({"name": function.get("name"), "arguments": args,
                      "clipped": args_clipped})
    if calls:
        out["tool_calls"] = calls
    if raw.get("tool_call_id"):
        out["tool_call_id"] = raw["tool_call_id"]
    return out


def load(trace_dir: str, trace_file: str | None, trace_seq: int | None,
         since: int = 0) -> dict:
    """The messages new in this turn, plus the reply.

    `since` is the previous turn's message count - everything from there on is
    what this turn added. Returns {"found": False, "reason": ...} rather than
    raising: a trace that has been cleaned out is an ordinary thing to find.
    """
    if not trace_file or trace_seq is None:
        return {"found": False,
                "reason": "this turn was recorded with tracing off"}

    path = os.path.join(trace_dir, trace_file)
    if not os.path.isfile(path):
        return {"found": False,
                "reason": f"{trace_file} is no longer on disk - traces are "
                          f"gitignored and rotate per engine run"}

    record = None
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    candidate = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if candidate.get("seq") == trace_seq:
                    record = candidate
                    break
    except OSError as exc:
        return {"found": False, "reason": f"could not read {trace_file}: {exc}"}

    if record is None:
        return {"found": False,
                "reason": f"turn {trace_seq} is not in {trace_file} - the file "
                          f"may have been replaced by a later run"}

    request = record.get("request") or {}
    messages = request.get("messages") or []
    response = record.get("response") or {}
    choice = (response.get("choices") or [{}])[0]
    reply = choice.get("message") or {}

    # Everything the client added since the previous turn. Guarded: if the
    # counts disagree - a client that compacted its own history - fall back to
    # the last few rather than showing nothing or showing everything.
    start = since if 0 < since <= len(messages) else max(0, len(messages) - 4)

    return {
        "found": True,
        "new_messages": [_message(m) for m in messages[start:]],
        "total_messages": len(messages),
        "skipped": start,
        "reply": _message(reply) if reply else None,
        "reasoning": _clip(reply.get("reasoning_content") or "")[0],
        "finish_reason": choice.get("finish_reason"),
        "tools": [(t.get("function") or {}).get("name")
                  for t in (request.get("tools") or [])],
        "params": {k: request[k] for k in
                   ("temperature", "top_p", "top_k", "max_tokens", "n", "stream")
                   if k in request},
        "stripped_params": record.get("stripped_params"),
        "rewrote_model": record.get("rewrote_model"),
        "repaired_tool_args": record.get("repaired_tool_args"),
        "preempted": bool(record.get("preempted")),
        "error": response.get("_http_error") or response.get("error"),
    }

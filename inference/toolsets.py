"""Cutting the tool list a client offers down to the ones it actually uses.

VS Code sends its whole tool surface on every turn - 80 definitions in the most
recent captured session, about 23,000 tokens of JSON schema. Across 15 trace
files it offered 103 distinct tools and the model called 18 of them. The other
85 are ~76% of that payload, resent in full on every request of every session,
and they cost twice: context the conversation cannot use, and 85 wrong answers
for the model to pick from when it decides what to call next.

This filters that array to a named set. The sets live in `inference/toolsets/`
as plain lists of names, one per line, so a set can be edited mid-session and
picked up on the next turn - the same reason `prompts` re-reads its files.

Deliberately not the same mechanism as `settings.strip`. That one removes
sampling parameters and protects `tools` precisely because dropping it would
change what the model was asked rather than how it answers. Filtering the array
*does* change what was asked. That is the point of it, so it is opt-in per
client, recorded in the trace, and never inferred.
"""

from __future__ import annotations

import os

TOOLSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "toolsets")


def parse(text: str) -> list[str]:
    """Tool names from a set file, in the order written, without repeats.

    `#` starts a comment anywhere on a line, so a set can carry its own notes
    about why a tool is on it - which is most of what makes one reviewable a
    month later.
    """
    names: list[str] = []
    for line in text.splitlines():
        name = line.split("#", 1)[0].strip()
        if name and name not in names:
            names.append(name)
    return names


def load(file: str) -> list[str]:
    """Names in a shipped tool set. Empty if it is not there.

    Same guard as `prompts.load`: the file names a shipped set, never a path
    into the filesystem, and this is reachable from an unauthenticated
    dashboard. Raise rather than quietly taking the basename - anything with a
    separator in it is either a bug in the caller or someone trying it on, and
    both want to be loud.
    """
    safe = file or ""
    if safe != os.path.basename(safe) or safe in ("", ".", ".."):
        raise ValueError(f"tool set must be a bare file name, got {file!r}")
    if not safe.endswith(".txt"):
        return []
    path = os.path.join(TOOLSET_DIR, safe)
    if not os.path.isfile(path):
        return []
    try:
        return parse(open(path, encoding="utf-8").read())
    except OSError:
        return []


def available() -> list[dict]:
    """The tool sets shipped with the server, for the dashboard's picker."""
    out = []
    if not os.path.isdir(TOOLSET_DIR):
        return out
    for name in sorted(os.listdir(TOOLSET_DIR)):
        if not name.endswith(".txt"):
            continue
        try:
            text = open(os.path.join(TOOLSET_DIR, name), encoding="utf-8").read()
        except OSError:
            continue
        names = parse(text)
        if not names:
            continue  # documentation, not a tool set
        out.append({"file": name, "count": len(names),
                    "summary": _summary(text), "tools": names})
    return out


def _summary(text: str) -> str:
    """First comment line - what the set is for."""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()[:160]
    return ""


def apply(tools: list | None,
          allowed: list[str]) -> tuple[list, list[str], list[str]]:
    """(tools to send, names dropped, allowed names the client did not offer).

    The client's own order is preserved. Reordering a tool array is a change to
    the request that buys nothing, and it would make two traces of the same
    session diff against each other for no reason.

    An allowlist that matches nothing is treated as a mistake rather than an
    instruction, and the client's tools go through untouched. A coding agent
    handed zero tools does not error - it quietly stops being an agent, and the
    only sign is a session that talks about work instead of doing it. Every
    name comes back in `unknown` in that case, which is what the caller warns
    on.
    """
    if not tools or not allowed:
        return tools or [], [], []

    wanted = set(allowed)
    offered = {(t.get("function") or {}).get("name") for t in tools}
    offered.discard(None)

    kept = [t for t in tools if (t.get("function") or {}).get("name") in wanted]
    unknown = [name for name in allowed if name not in offered]

    if not kept:
        return tools, [], unknown

    dropped = sorted(offered - wanted)
    return kept, dropped, unknown

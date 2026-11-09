"""Rewriting the system prompt a client sends.

VS Code ships a 15,000-character system prompt: an untagged preamble, then a
series of XML-ish blocks - `<instructions>`, `<toolUseInstructions>`,
`<editFileInstructions>`, `<memoryInstructions>`, and a second `<instructions>`
carrying that workspace's skills and agents. Most of it is Copilot's own
framing, written for whatever model Copilot expects, and none of it is under
your control from the client side.

This replaces named blocks on the way through and leaves the rest alone. That
matters more than it sounds: `<toolUseInstructions>` and `<editFileInstructions>`
describe the tool contract the *client* will parse the reply against, so
replacing them wholesale breaks tool calling. The safe targets are the preamble
and the first `<instructions>` - the behavioural half - which is why they are
what the shipped prompts address.

Only the *first* match of a tag is replaced. The second `<instructions>` block
holds skills and agents discovered from the workspace, so it is per-session data
rather than framing, and rewriting it would delete capability the client just
advertised.
"""

from __future__ import annotations

import os
import re

PROMPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")

# The untagged text before the first block. Named so it can be selected the
# same way as a real block.
PREAMBLE = "preamble"

# Blocks that describe the tool contract rather than behaviour. Replaceable, but
# the client parses replies against them, so the UI warns before you do.
CONTRACT_BLOCKS = frozenset({
    "toolUseInstructions", "editFileInstructions", "notebookInstructions",
    "outputFormatting",
})


def available() -> list[dict]:
    """The prompt files shipped with the server."""
    out = []
    if not os.path.isdir(PROMPT_DIR):
        return out
    for name in sorted(os.listdir(PROMPT_DIR)):
        if not name.endswith((".md", ".txt")):
            continue
        path = os.path.join(PROMPT_DIR, name)
        try:
            text = open(path, encoding="utf-8").read()
        except OSError:
            continue
        sections = parse(text)
        if not sections:
            continue  # documentation, not a prompt
        out.append({
            "file": name,
            "blocks": sorted(sections),
            "chars": len(text),
            "summary": _summary(text),
        })
    return out


def _summary(text: str) -> str:
    """First non-empty, non-tag line - what the prompt is for."""
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("<"):
            return line[:160]
    return ""


def parse(text: str) -> dict[str, str]:
    """A prompt file's sections, keyed by block name.

    `<preamble>` is the conventional name for the untagged opening of the
    client's prompt; every other section replaces the tag it is named for.
    """
    return {m.group(1): m.group(2).strip()
            for m in re.finditer(r"(?ms)^<([a-zA-Z_][\w-]*)>\n(.*?)\n</\1>", text)}


def load(file: str) -> dict[str, str]:
    """Sections of a shipped prompt file. Empty if it is not there."""
    # The file names a shipped prompt, never a path into the filesystem, and
    # this is reachable from an unauthenticated dashboard. Raise rather than
    # quietly taking the basename: anything with a separator in it is either a
    # bug in the caller or someone trying it on, and both want to be loud.
    safe = file or ""
    if safe != os.path.basename(safe) or safe in ("", ".", ".."):
        raise ValueError(f"prompt name must be a bare file name, got {file!r}")
    if not safe.endswith((".md", ".txt")):
        return {}
    path = os.path.join(PROMPT_DIR, safe)
    if not os.path.isfile(path):
        return {}
    try:
        return parse(open(path, encoding="utf-8").read())
    except OSError:
        return {}


def split(system: str) -> tuple[str, list[tuple[str, int, int]]]:
    """(preamble, [(tag, start, end)]) for a client's system message.

    Spans cover the whole element including its tags, so replacing one is a
    slice assignment. Only top-level elements are reported - a `<name>` inside
    a `<skill>` is not something to target.
    """
    blocks: list[tuple[str, int, int]] = []
    pos = 0
    while True:
        match = re.compile(r"(?m)^<([a-zA-Z_][\w-]*)>").search(system, pos)
        if not match:
            break
        tag = match.group(1)
        close = system.find(f"\n</{tag}>", match.end())
        if close == -1:
            pos = match.end()
            continue
        end = close + len(tag) + 4
        blocks.append((tag, match.start(), end))
        pos = end
    preamble = system[:blocks[0][1]] if blocks else system
    return preamble, blocks


def rewrite(system: str, sections: dict[str, str],
            replace: list[str]) -> tuple[str, list[str]]:
    """Replace the named blocks. Returns (new system message, what changed).

    Applied back-to-front so earlier spans stay valid while later ones are
    rewritten, and only the first occurrence of each tag is touched.
    """
    if not system or not sections or not replace:
        return system, []

    preamble, blocks = split(system)
    wanted = [name for name in replace if name in sections]
    changed: list[str] = []

    seen: set[str] = set()
    edits: list[tuple[int, int, str]] = []
    for tag, start, end in blocks:
        if tag in wanted and tag not in seen:
            seen.add(tag)
            edits.append((start, end, f"<{tag}>\n{sections[tag]}\n</{tag}>"))
            changed.append(tag)

    out = system
    for start, end, text in sorted(edits, reverse=True):
        out = out[:start] + text + out[end:]

    if PREAMBLE in wanted and preamble.strip():
        out = sections[PREAMBLE].rstrip() + "\n" + out[len(preamble):]
        changed.append(PREAMBLE)

    return out, sorted(changed)

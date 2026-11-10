"""Collapsing a repeated tool call out of the conversation on the way past.

The case this exists for: the model built a malformed Azure DevOps URL, got a
404, and re-issued the byte-identical command 37 times. By the last turn, 74 of
the 102 messages were that same wrong command and that same error. Each turn's
context differed from the last only by another copy of both, so the wrong
continuation stayed the likeliest one, and the session never recovered.

Sampling cannot fix that. Temperature was already stripped, so the model's own
0.7 applied, and replaying the turn still produced byte-identical output six
times out of six - the distribution in that context is peaked, not noisy. And
temperature is a per-token knob while the repetition is across turns: each turn
is a fresh sample from a context that genuinely leads to the same place.

The engine is the only component positioned to see it. Chat completions is
stateless, so the whole history arrives every turn, and the client - VS Code
here - did not stop it after thirty-seven tries.

What the fix had to look like was measured against the real model rather than
reasoned about. Replaying the stuck turn:

    keep one copy of the pair, add an explanatory note      0/3 escaped
    keep one copy, make the note directive                  2/3
    remove every copy, leave only a note                     3/3
    remove the calls, keep the shared result in the note     4/4

So the anchor is the command text, not the error. Delete the arguments and the
model re-derives the approach from what it knows - which it does correctly, as
the same model asked cold gets the URL right every time. Keep them and it
copies what is in front of it, however firmly the note tells it not to.

Hence the shape here: the run is replaced by a single message that reports what
came back, and never what was sent.
"""

from __future__ import annotations

import hashlib
import json

# How much of the repeated result to quote. Long enough for a stack trace or an
# HTTP error body, short enough that collapsing a loop over a large file read
# does not put the file back.
RESULT_CAP = 2000

NOTE = (
    "[engine] The assistant called {tool} {count} times in a row with "
    "byte-identical arguments. Every one of those calls returned the same "
    "result, which was:\n\n{result}\n\n"
    "Those {count} attempts have been removed from this transcript. That "
    "approach does not work - do not reconstruct or repeat the command that "
    "produced this result. Work out a different method."
)


def spans(messages: list) -> list[tuple[int, int]]:
    """Index ranges of each (assistant asks, tool answers) exchange.

    The pair is the unit, because one assistant message can call several tools
    and every `tool` message must keep the call it answers. Splitting a pair
    orphans a tool message, and llama.cpp fails the whole conversation over one
    rather than the single turn.
    """
    out: list[tuple[int, int]] = []
    i = 0
    while i < len(messages):
        message = messages[i]
        if message.get("role") == "assistant" and message.get("tool_calls"):
            j = i + 1
            while j < len(messages) and messages[j].get("role") == "tool":
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def fingerprint(messages: list, start: int, end: int) -> str:
    """What makes two exchanges 'the same'.

    Names, arguments, every result, and whatever the assistant said alongside
    the call. Deliberately *not* the tool-call id: the client mints a fresh one
    for every attempt, so including it would mean nothing ever matched.

    Including the results is what keeps a real poll safe. `az build show`
    returning "queued" and then "running" is progress, and progress must never
    look like a loop.
    """
    calls = [((c.get("function") or {}).get("name"),
              (c.get("function") or {}).get("arguments"))
             for c in messages[start].get("tool_calls") or []]
    results = [messages[k].get("content") for k in range(start + 1, end)]
    said = (messages[start].get("content") or "").strip()
    return hashlib.md5(
        json.dumps([calls, results, said], sort_keys=True).encode()).hexdigest()


def _note(messages: list, start: int, end: int, count: int) -> dict:
    """The one message a collapsed run leaves behind."""
    result = "\n".join((messages[k].get("content") or "")
                       for k in range(start + 1, end))
    if len(result) > RESULT_CAP:
        result = (result[:RESULT_CAP]
                  + f"\n... [truncated, {len(result):,} characters in total]")
    names = [(c.get("function") or {}).get("name") or "?"
             for c in messages[start].get("tool_calls") or []]
    return {"role": "user",
            "content": NOTE.format(tool=", ".join(names), count=count,
                                   result=result)}


def collapse(messages: list, threshold: int) -> tuple[list, list[dict]]:
    """(messages to send, one record per run collapsed).

    Every run is collapsed, not only one at the tail. That matters on the turn
    *after* an intervention works: the model emits a good call, the run stops
    being the last thing in the conversation, and a tail-only rule would hand
    the whole loop straight back. Collapsing wherever it appears keeps the
    intervention stable turn over turn - which also keeps the prefix cache
    hitting, since the collapsed form does not move.

    `threshold` of 0 disables it, which is how a client running a genuine poll
    turns it off from the Config page.
    """
    if not threshold or threshold < 2 or not messages:
        return messages, []

    found = spans(messages)
    if len(found) < threshold:
        return messages, []

    prints = [fingerprint(messages, a, b) for a, b in found]

    # Runs first, then rewrite back to front so earlier indices stay valid.
    runs = []
    i = 0
    while i < len(prints):
        j = i
        # Same fingerprint *and* back to back in the conversation. Spans skip
        # whatever isn't a tool exchange, so two identical calls with a user's
        # "try again" between them are neighbours in `found` - and collapsing
        # them would delete the user's message along with the repeats.
        while (j + 1 < len(prints) and prints[j + 1] == prints[i]
               and found[j + 1][0] == found[j][1]):
            j += 1
        if j - i + 1 >= threshold:
            runs.append((i, j))
        i = j + 1
    if not runs:
        return messages, []

    out = list(messages)
    info = []
    for i, j in reversed(runs):
        start = found[i][0]
        end = found[j][1]
        count = j - i + 1
        out[start:end] = [_note(messages, found[i][0], found[i][1], count)]
        info.append({
            "tool": ", ".join(
                (c.get("function") or {}).get("name") or "?"
                for c in messages[found[i][0]].get("tool_calls") or []),
            "repeats": count,
            "messages_removed": end - start - 1,
        })
    info.reverse()
    return out, info

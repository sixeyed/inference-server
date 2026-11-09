"""Proxy settings you can change without restarting, and that survive a restart.

The motivating case: VS Code sends `temperature: 0.1, top_p: 1` on every
request, which overrides whatever the model's own file tuned it to. On a long
agent session that near-greedy sampling can lock the model into re-emitting an
identical tool call forever - each turn's context differs only by an identical
assistant/tool pair, so a deterministic decoder re-derives the same answer.

Stripping the key is the fix, and it has to be reachable in the middle of an
incident: kill the session, flip temperature off in the dashboard, try again.
Hence live, and hence persisted - a setting you had to remember to pass on the
command line is one you will not have set when it matters.

Strip only, never override. Removing the key means the server falls back to what
that model's file chose, which keeps `models/*.py` the single source of truth
for sampling. Pinning a value here would put it in two places and leave the
trace as the only way to tell which won.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading

# What a client may send that overrides the server's own tuned sampling.
#
# Each entry is (key, what it does, why you might strip it). Only generation
# parameters: the structural fields are protected below, because removing them
# changes what was asked rather than how it was sampled.
STRIPPABLE: dict[str, str] = {
    "temperature": "Randomness. Clients that pin it low make decoding "
                   "near-deterministic, which is what locks a model into "
                   "repeating a tool call.",
    "top_p": "Nucleus sampling. A client sending 1.0 disables the model's own "
             "narrower setting.",
    "top_k": "Candidate cutoff.",
    "min_p": "Minimum probability floor.",
    "presence_penalty": "Discourages tokens already used in this response.",
    "frequency_penalty": "Scales the penalty by how often a token was used.",
    "repetition_penalty": "Non-standard, but some clients send it.",
    "max_tokens": "Response cap. Stripping it hands the server's much larger "
                  "limit back, so a client that caps low stops truncating.",
    "max_completion_tokens": "The newer spelling of max_tokens.",
    "seed": "Fixed seed. Guarantees the same answer to the same context, which "
            "is the opposite of what you want when a model is stuck.",
    "stop": "Extra stop strings.",
    "n": "How many completions to generate.",
    "reasoning_effort": "How hard a thinking model works before answering.",
    "logit_bias": "Per-token biasing.",
}

# Never removable. These are the request, not the sampling of it - dropping any
# of them would change what the model was asked rather than how it answers.
#
# `tools` has one deliberate exception, and it is not here: a per-client tool
# set (see `toolsets`) filters the array down to named tools. That does change
# what was asked, which is why it is a separate mechanism - opt-in per client,
# recorded in the trace - rather than a key you can tick off in this list.
PROTECTED = frozenset({
    "model", "messages", "tools", "tool_choice", "stream", "stream_options",
})


def _threshold(value) -> int:
    """A usable trigger count, or 0 for off.

    Anything unparseable becomes off rather than raising. This is reachable
    from the dashboard and from a hand-edited settings file, and a bad value
    should cost the feature, not the engine.
    """
    try:
        count = int(value)
    except (TypeError, ValueError):
        return 0
    return count if count >= 2 else 0


class Settings:
    """The live settings, backed by a JSON file.

    Read on every request and written from the dashboard, so both are behind
    one lock. Writes go through a temporary file and a rename, because a
    half-written settings file that fails to parse would silently revert every
    choice on the next restart.
    """

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._strip: set[str] = set()
        # client label -> {"file": ..., "replace": [...]}
        self._prompts: dict[str, dict] = {}
        # client label -> loop-collapse trigger count. 0 is off, which is how
        # a client running a genuine poll opts out from the Config page.
        self._loops: dict[str, int] = {}
        # client label -> tool set file name. Kept apart from _prompts rather
        # than folded into it: the two compose freely - any prompt with any
        # tool set - and clearing one must not take the other with it.
        self._toolsets: dict[str, str] = {}
        self.load()

    # --------------------------------------------------------
    def load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return  # no file yet, or an unreadable one: defaults stand
        with self._lock:
            self._strip = {key for key in data.get("strip") or []
                           if key in STRIPPABLE}
            self._prompts = {
                client: {"file": cfg.get("file"),
                         "replace": list(cfg.get("replace") or [])}
                for client, cfg in (data.get("prompts") or {}).items()
                if cfg.get("file")
            }
            self._toolsets = {
                client: os.path.basename(file)
                for client, file in (data.get("toolsets") or {}).items()
                if file
            }
            self._loops = {}
            for client, count in (data.get("loops") or {}).items():
                count = _threshold(count)
                if count:
                    self._loops[client] = count

    def save(self) -> None:
        with self._lock:
            payload = {"strip": sorted(self._strip),
                       "prompts": dict(self._prompts),
                       "toolsets": dict(self._toolsets),
                       "loops": dict(self._loops)}
        directory = os.path.dirname(os.path.abspath(self.path)) or "."
        os.makedirs(directory, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            "w", dir=directory, delete=False, encoding="utf-8")
        try:
            with handle:
                json.dump(payload, handle, indent=2)
                handle.write("\n")
            os.replace(handle.name, self.path)
        except OSError:
            os.unlink(handle.name)
            raise

    # --------------------------------------------------------
    @property
    def stripped(self) -> set[str]:
        with self._lock:
            return set(self._strip)

    def set_stripped(self, keys) -> set[str]:
        """Replace the strip list. Unknown and protected keys are ignored
        rather than rejected, so a stale dashboard cannot wedge the engine."""
        with self._lock:
            self._strip = {key for key in keys
                           if key in STRIPPABLE and key not in PROTECTED}
        self.save()
        return self.stripped

    def apply(self, request: dict) -> list[str]:
        """Remove the configured keys. Returns what was actually removed.

        Only reports keys that were present, so a trace records what changed
        about this request rather than what the settings happen to say.
        """
        removed = []
        for key in self.stripped:
            if key in request:
                del request[key]
                removed.append(key)
        return sorted(removed)

    def prompt_for(self, client: str | None) -> tuple[str | None, list[str]]:
        """(prompt file, blocks to replace) configured for this client."""
        with self._lock:
            cfg = self._prompts.get(client or "")
        if not cfg:
            return None, []
        return cfg.get("file"), list(cfg.get("replace") or [])

    def set_prompt(self, client: str, file: str | None,
                   replace: list[str] | None = None) -> dict:
        """Configure, or clear with file=None."""
        with self._lock:
            if not file:
                self._prompts.pop(client, None)
            else:
                self._prompts[client] = {
                    "file": os.path.basename(file),
                    "replace": sorted(set(replace or [])),
                }
            out = dict(self._prompts)
        self.save()
        return out

    @property
    def prompts(self) -> dict:
        with self._lock:
            return dict(self._prompts)

    def toolset_for(self, client: str | None) -> str | None:
        """The tool set file configured for this client, if any."""
        with self._lock:
            return self._toolsets.get(client or "")

    def set_toolset(self, client: str, file: str | None) -> dict:
        """Configure, or clear with file=None."""
        with self._lock:
            if not file:
                self._toolsets.pop(client, None)
            else:
                self._toolsets[client] = os.path.basename(file)
            out = dict(self._toolsets)
        self.save()
        return out

    @property
    def toolsets(self) -> dict:
        with self._lock:
            return dict(self._toolsets)

    def loop_threshold_for(self, client: str | None) -> int:
        """How many identical tool calls in a row before the engine steps in.
        0 means never."""
        with self._lock:
            return self._loops.get(client or "", 0)

    def set_loop_threshold(self, client: str, count) -> dict:
        """Configure, or turn off with 0. A count under 2 is refused rather
        than stored: 1 would collapse every tool call ever made."""
        count = _threshold(count)
        with self._lock:
            if count:
                self._loops[client] = count
            else:
                self._loops.pop(client, None)
            out = dict(self._loops)
        self.save()
        return out

    @property
    def loops(self) -> dict:
        with self._lock:
            return dict(self._loops)

    def describe(self) -> list[dict]:
        """The catalogue the dashboard renders."""
        stripped = self.stripped
        return [{"key": key, "why": why, "stripped": key in stripped}
                for key, why in STRIPPABLE.items()]

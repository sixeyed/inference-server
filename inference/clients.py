"""Working out which client sent a request.

Nothing in the OpenAI schema identifies the caller, and knowing it matters here
because the failure modes are client-shaped: VS Code pins temperature to 0.1 on
every request, opencode prefixes the model with its provider name, and a
hand-rolled harness sends five tools where an editor sends eighty. A tok/s
figure that mixes them is measuring the clients as much as the model.

Three signals, in order of how much they can be trusted:

  User-Agent    Direct, when a client sets a recognisable one. Read but never
                stored - only the label derived from it - because a header is
                the one place a token can turn up.

  Tool names    The strongest signal in practice. A client ships a fixed tool
                set, and some names are unmistakable: `copilot_getNotebookSummary`
                appeared in every VS Code request captured here, across tool
                counts from 51 to 80.

  Model prefix  opencode sends `local/qwen3.8`; the prefix is the provider it
                was configured with.

An unrecognised client is reported as `unknown` rather than guessed at. Adding
one is a line in SIGNATURES, which is the point of keeping this in a table.
"""

from __future__ import annotations

import re

# Tool names that identify a client outright. One is enough - matching on the
# whole set would break the moment a client adds a tool, which they do often.
SIGNATURES: dict[str, set[str]] = {
    # VS Code Copilot agent mode. The notebook and task tools are its own.
    "vscode": {"copilot_getNotebookSummary", "create_new_jupyter_notebook",
               "configure_notebook", "create_and_run_task",
               "configure_python_environment"},
    # Cline's plan/act loop. These are its distinctive verbs.
    "cline": {"attempt_completion", "plan_mode_respond",
              "ask_followup_question", "replace_in_file", "browser_action"},
}

# The clients the Config tab always offers rows for, so a fresh engine can be
# set up for them before they have sent it anything. Everything else appears
# once it has sent traffic or has settings.
OFFERED: tuple[str, ...] = ("vscode", "cline", "opencode")

# User-Agent substrings, checked lowercased.
USER_AGENTS: tuple[tuple[str, str], ...] = (
    ("vscode", "vscode"),
    ("code-oss", "vscode"),
    ("copilot", "vscode"),
    ("cline", "cline"),
    ("opencode", "opencode"),
    ("open-webui", "open-webui"),
    ("continue", "continue"),
    ("curl", "curl"),
    ("python-urllib", "harness"),
)

_PROVIDER_PREFIX = re.compile(r"^(local|opencode)/", re.I)


def identify(request: dict | None, user_agent: str | None = None) -> str:
    """A short label for whoever sent this request."""
    request = request or {}

    agent = (user_agent or "").lower()
    for needle, label in USER_AGENTS:
        if needle in agent:
            return label

    names = {(t.get("function") or {}).get("name")
             for t in (request.get("tools") or [])}
    names.discard(None)
    for label, markers in SIGNATURES.items():
        if names & markers:
            return label

    model = request.get("model") or ""
    if _PROVIDER_PREFIX.match(model):
        return "opencode"

    # No tools at all is a bare API call - the dashboard's own quick chat, a
    # curl, or a benchmark probe. Worth distinguishing from a client we simply
    # failed to recognise.
    if not names:
        return "api"
    return "unknown"

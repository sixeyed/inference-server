"""What a client needs to be pointed at this engine - for the Clients tab.

The addresses come from config (`client_endpoints`) because only whoever runs
the engine knows them: the Mac's LAN address, a name behind an ingress, a node
IP. The VS Code block is built here rather than in the page, so it has one
shape everywhere and can be tested.
"""

from __future__ import annotations

# VS Code's own bookkeeping sits on top of the prompt, so the input cap stays a
# little under the model's context: 249000 for 262K, which is what held up in
# use before this page existed.
INPUT_HEADROOM = 0.95
MAX_OUTPUT_TOKENS = 16000
FALLBACK_CTX = 32768


def endpoints(engine) -> list[dict]:
    """Where clients can reach this engine, in the order to offer them.

    With nothing configured, this machine's own address - all a client could
    use before the engine ran anywhere but the Mac.
    """
    configured = engine.client_endpoints
    if configured:
        return [{"label": e.get("label") or e["base_url"],
                 "base_url": e["base_url"],
                 "note": e.get("note") or ""} for e in configured]
    host = engine.machine.host
    return [{"label": host, "base_url": f"http://{host}:{engine.port}/v1", "note": ""}]


def live_models(infos) -> list[dict]:
    """Models a client should be offered: routable, and not a test fixture."""
    live = sorted((i for i in infos if i.routable and not i.test_only),
                  key=lambda i: i.alias)
    return [{"id": i.alias, "ctx": i.ctx, "vision": bool(i.vision)} for i in live]


def max_input_tokens(ctx: int | None) -> int:
    return int((ctx or FALLBACK_CTX) * INPUT_HEADROOM) // 1000 * 1000


def vscode_custom_models(endpoint: dict, models: list[dict], api_key: str) -> list[dict]:
    """The `github.copilot.chat.customModels` value for one address."""
    base = endpoint["base_url"].rstrip("/")
    return [{
        "name": endpoint["label"],
        "vendor": "customendpoint",
        # "none" really is the value: MTPLX checks for a key, the others ignore it.
        "apiKey": api_key,
        # None of the engines speak the Responses API.
        "apiType": "chat-completions",
        "models": [{
            "id": m["id"],
            "name": m["id"],
            "url": f"{base}/chat/completions",
            "toolCalling": True,  # required for agent mode
            "vision": m["vision"],
            "maxInputTokens": max_input_tokens(m["ctx"]),
            "maxOutputTokens": MAX_OUTPUT_TOKENS,
        } for m in models],
    }]


def payload(engine) -> dict:
    """Everything the Clients tab shows."""
    models = live_models(engine.infos)
    return {
        "endpoints": [dict(e, vscode=vscode_custom_models(e, models, engine.api_key))
                      for e in endpoints(engine)],
        "api_key": engine.api_key,
        "models": models,
    }

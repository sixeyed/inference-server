# Client setup

Point clients at the **inference engine** on `8084`, not at a model server. Then
the model picker lists everything the machine can run, and choosing one starts
it. Port `8083` stays directly reachable for `inference run` and your own
benchmarks.

The dashboard's **Clients** tab lists the addresses clients can use, each with
a ready-to-paste VS Code `github.copilot.chat.customModels` block for every live
model. The addresses come from `client_endpoints` in `config.json` - a list of
`label`, `base_url` and `note`; without it the tab offers this machine's own
`http://<host>:8084/v1`.

On the machine running the engine that is `http://127.0.0.1:8084/v1`. From
anywhere else, set `bind` (see [Configuration](configuration.md#configuration)) and use the
machine's address.

| Setting | Value |
|---|---|
| **Protocol** | OpenAI API compatible |
| **Base URL** | `http://<host>:8084/v1` — the engine |
| **Model ID** | the alias (`qwen3.8`, `kat-coder2.5`, …), for every backend. The engine rewrites it for engines that expect something else, so HF repo ids never appear in client config. `GET /v1/models` lists them, which is how most pickers fill themselves. |
| **API Key** | `none` |

One key for the whole system, and `none` really is the value. llama-server and
mlx_lm.server ignore the field entirely; MTPLX genuinely checks it, because it
requires a non-empty key on a non-localhost bind, and is started with this exact
string. The engine passes whatever the client sent straight through — if the two
disagree you get the 401, rather than a silent success against a key you didn't
think you were using — and fills one in only when the client sent nothing at all.

Change it with `api_key` in `config.json`, or `--api-key` on both `inference
engine` and `inference run`; going direct to
`8083` needs the same value.

## Open WebUI

1. In **Settings > Connections**, add an OpenAI-compatible connection
2. Base URL `http://<host>:<port>/v1`, API key `none`
3. The model appears by its alias

## Cline (VS Code)

1. Provider: **OpenAI Compatible**
2. Base URL `http://<host>:<port>/v1`, API key `none`
3. Model ID: the alias (or full HF repo id for MLX models)

For `muse-glimmer`, add `Reasoning strength: high` to Cline's custom instructions
— that's the only way to set its reasoning level.

## opencode

Edit `~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "local": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Local server",
      "options": { "baseURL": "http://127.0.0.1:8084/v1" },
      "models": {
        "qwen3.8": { "name": "Qwen 3.8 27B" },
        "kat-coder2.5": { "name": "KAT-Coder V2.5 (35B-A3B)" },
        "muse-glimmer": { "name": "Muse Glimmer 30B" },
        "FakeRockert543/gemma-4-31b-it-MLX-8bit": { "name": "Gemma 4 31B" }
      }
    }
  },
  "model": "local/qwen3.8"
}
```

## VS Code (custom model endpoint)

Add a `github.copilot.chat.customModels` array to your `settings.json`:

```json
[
  {
    "name": "http://127.0.0.1:8084/v1",
    "vendor": "customendpoint",
    "apiKey": "none",
    "apiType": "chat-completions",
    "models": [
      {
        "id": "kat-coder2.5",
        "name": "kat-coder2.5",
        "url": "http://127.0.0.1:8084/v1/chat/completions",
        "toolCalling": true,
        "vision": false,
        "maxInputTokens": 250000,
        "maxOutputTokens": 16000
      }
    ]
  }
]
```

- `apiType` is always `chat-completions` — none of the engines speak the OpenAI
  Responses API. MTPLX does serve Anthropic Messages at `/v1/messages`, but VS
  Code's custom-endpoint support doesn't use it.
- `apiKey` must be `"none"` for `qwen3.8` — MTPLX checks it, and an empty
  string is rejected. The other engines ignore the field, so `"none"` is safe
  everywhere and no client needs a per-model key.
- `toolCalling: true` is required for agent mode.
- `vision: true` works for `qwen3.8` and `muse-glimmer`; the rest are text-only.
- `maxInputTokens` should sit a bit under the server's `-c` context (262K here);
  250000 leaves headroom for VS Code's bookkeeping.

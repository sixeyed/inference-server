# inference-server

Local coding models on Apple Silicon, one at a time, on demand, behind one
OpenAI-compatible endpoint.

Point a coding agent at one URL and pick a model in its picker. The engine
fetches the weights if it has to, starts the right server for that model
(llama.cpp, MTPLX or mlx-lm) inside an egress sandbox, and relays. It records
every turn on the way through, so you can see which model is actually faster
on your work.

> [!NOTE]
> This repo is 100% Claude-coded (including these docs).

- **One endpoint, many models** — `GET /v1/models` lists everything that fits
  your Mac; asking for one loads it. One runs at a time, so switching cuts
  whatever is in flight. See [routing](docs/routing.md).
- **Your choice of models** — Qwen, KAT-Coder and Muse on llama.cpp and MTPLX,
  tuned per machine. See [backends](docs/backends.md), and
  [adding a model](docs/adding-a-model.md) is one Python file.
- **Sandboxed** — model servers run under `sandbox-exec` with outbound network
  denied. Weights are pulled by this repo's own code before launch, never by the
  server. See [the engine](docs/engine.md#egress-sandbox).
- **Local by default** — everything listens on `127.0.0.1` until you say
  otherwise. See [configuration](docs/configuration.md).
- **No dependencies** — the Python that ships with macOS is enough; nothing to
  `pip install`.

Requires an Apple Silicon Mac, plus the server for the models you want (MTPLX,
llama.cpp). [PREREQS.md](PREREQS.md#first-run-on-a-new-mac) has the first-run
steps for a new Mac.

## Quick start: Docker

The engine runs in a container and drives the Mac's model servers over SSH. The
models themselves can't run in a container — MTPLX and MLX are Apple Silicon
only — so they run natively, from a checkout of this repo on the Mac.

**1. On the Mac**, clone the repo, turn on Remote Login (System Settings > General
> Sharing), and authorise a key just for the engine:

```bash
git clone https://github.com/sixeyed/inference-server ~/inference-server
cd ~/inference-server

ssh-keygen -t ed25519 -f ~/.ssh/inference -C inference-server -N ''
cat ~/.ssh/inference.pub >> ~/.ssh/authorized_keys
ssh-keyscan -t ed25519 localhost 2>/dev/null | sed 's/^localhost/host.docker.internal/' > docker/known_hosts
```

**2. Configure the engine** — copy the template and set your user name and the
checkout path in `model_host`:

```bash
cp docker/config.example.json docker/config.json
```

```json
"model_host": {
  "ssh": "you@host.docker.internal",
  "address": "host.docker.internal",
  "checkout": "/Users/you/inference-server",
  "python": "/opt/homebrew/bin/python3"
}
```

**3. Run it:**

```bash
docker compose up -d
curl -s localhost:8084/health
```

The engine is on `http://127.0.0.1:8084/v1`, with a
[dashboard](docs/engine.md#dashboard) at `http://127.0.0.1:8084/`. The first request for a model pulls its weights on
the Mac — tens of GB for most of them — so do that ahead of time if you like:

```bash
python3 -m inference pull qwen3.8
```

To serve your network, or to run the engine on a different machine from the
Mac, see [the engine docs](docs/engine.md#running-in-a-container).

## Quick start: native

The same engine, straight from the checkout:

```bash
python3 -m inference engine          # 127.0.0.1:8084
python3 -m inference models          # what fits this Mac, and what to install
python3 -m inference pull <model>    # fetch weights ahead of time
```

Every setting has a default that works on a fresh clone; to change them see
[configuration](docs/configuration.md).

## VS Code

Add an OpenAI-compatible endpoint to your VS Code user `settings.json`, with
an entry in `models` for each model you want in the picker. The dashboard's **Clients** tab generates this block for every
model that fits your Mac, ready to paste.

```json
"github.copilot.chat.customModels": [
  {
    "name": "inference-server",
    "vendor": "customendpoint",
    "apiKey": "none",
    "apiType": "chat-completions",
    "models": [
      {
        "id": "qwen3.8",
        "name": "qwen3.8",
        "url": "http://127.0.0.1:8084/v1/chat/completions",
        "toolCalling": true,
        "vision": true,
        "maxInputTokens": 250000,
        "maxOutputTokens": 16000
      }
    ]
  }
]
```

- `apiKey` must be the string `"none"`, not empty — MTPLX checks it and rejects
  an empty key. The other servers ignore it.
- `apiType` is always `chat-completions`.
- `toolCalling: true` is required for agent mode.
- `maxInputTokens` should sit a little under the model's context (262K for
  `qwen3.8`).

Pick the model in Copilot's model picker and the engine loads it. Cline,
opencode and Open WebUI are in [docs/clients.md](docs/clients.md).

## Docs

| | |
|---|---|
| [docs/backends.md](docs/backends.md) | the models, the per-machine deployment, and the llama.cpp / MTPLX / mlx-lm servers |
| [docs/adding-a-model.md](docs/adding-a-model.md) | adding a new model: one Python file |
| [docs/engine.md](docs/engine.md) | running in a container, the egress sandbox and pulling weights, the dashboard, usage stats, tests |
| [docs/routing.md](docs/routing.md) | how requests are routed, models swapped, and the HTTP routes |
| [docs/configuration.md](docs/configuration.md) | `config.json`, environment variables, and the live proxy settings |
| [docs/clients.md](docs/clients.md) | setup for VS Code, Cline, opencode and Open WebUI |
| [docs/lessons-learned.md](docs/lessons-learned.md) | what running these models under real agents taught us |

## Licence

MIT — see [LICENSE](LICENSE).

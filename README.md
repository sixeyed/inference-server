# inference-server

Local coding models on Apple Silicon, one at a time, on demand, behind one
OpenAI-compatible endpoint.

Point a coding agent at one URL — VS Code Copilot, Cline, opencode, Open WebUI.
Pick a model in its picker and the engine fetches the weights if it has to,
starts the right server for that model (llama.cpp, MTPLX or mlx-lm) inside an
egress sandbox, and relays. It records every turn on the way through, so you can
see which model is actually faster on your work rather than on a benchmark.

Tooling and the versions this is known to work on: [PREREQS.md](PREREQS.md).

## Quick Start

```bash
git clone https://github.com/sixeyed/inference-server
cd inference-server
pip install -e .                     # the runtime itself needs no packages
inference engine                     # the engine, on 127.0.0.1:8084
```

`config.json` is optional — without it the weights go under your home directory
and everything listens on `127.0.0.1`, so a fresh clone serves this machine and
nothing else. To serve your network, or keep the weights on another disk, see
[Configuration](#configuration). The engine also runs in Docker — see
[Running in a container](#running-in-a-container).

Or straight from a checkout with nothing installed at all:

```bash
python3 -m inference engine
```

Point a client at `http://<host>:8084/v1`, pick any model in its picker, and the
engine starts it. It reads the `model` field off each request, and if that isn't
the model currently running it stops the current one, starts the right one, waits
for it to be healthy, and forwards. It also records every turn — see
[Performance Testing](#performance-testing) — and serves a dashboard at
`http://<host>:8084/`.

The first time you pick a model its weights are pulled from HuggingFace — tens
of GB for most of them, so that request takes a while. Do it ahead of time with
`inference pull <model>`. See [Egress sandbox](#egress-sandbox) for why the pull
is a separate step.

One model runs at a time, so **switching model cuts whatever is in flight**. That
is the intended trade: changing model means you want the new one.

To run a single model in the foreground instead — for a dry run, a benchmark, or
just to watch it boot:

```bash
inference pull kat-coder2.5          # fetch weights now, or update them
inference run                       # the default model (qwen3.8)
inference run kat-coder2.5          # a specific model
inference run --family qwen         # that family's default
inference run qwen3.8 --port 9000   # override the default port (8083)
inference models                    # what runs on this machine
inference run kat-coder2.5 --dry-run  # print the launch command, run nothing
inference stop                      # kill whatever is running
inference stop --engine             # ...and the router too
```

`inference run` detects the Mac, resolves the model's config for it, and execs the right
engine on port `8083`. The inference engine starts models on that same port, and
will adopt one you started by hand rather than restarting it.

Requires Python 3.10+ and no pip packages — the runtime is standard library only.
(The test suite needs two; see [Tests](#tests).)

### Layout

```
inference/                 # the package
  cli.py                   # one entry point: engine | run | pull | stop | status | models | machine
  config.py                # ports, API key, on-disk defaults
  console.py               # colour, and quoting a command for pasting back
  machine.py               # Machine, detect_machine, Options, Unsupported
  specs.py                 # one launch spec per engine, fully resolved
  engines.py               # find each executable, ask what it supports
  process.py               # launch, supervise, banner, stop
  weights.py               # pull weights before launch, outside the sandbox
  supervisor.py            # one model server: start, probe, stop, swap, adopt
  hosts.py                 # where model servers run: LocalHost, this machine
  remote.py                # RemoteHost: the same lifecycle, over SSH to a Mac
  server.py                # the HTTP front door: route, relay, record
  dashboard.py             # static files and the /api/* JSON the page polls
  usage.py                 # SQLite: one row per turn, and the aggregates
  tracing.py               # JSONL trace records, per-turn console line
  sse.py                   # SSE first-token probing and stream reassembly
  models/                  # one file per model - the whole deployment for it
  static/                  # the dashboard (vendored tailwind.css, no build)
  settings.py              # live proxy settings, editable from the dashboard
  prompts.py               # parse and substitute blocks of a client's prompt
  prompts/                 # the system prompts shipped for substitution

config.json                # this machine: paths, addresses, tuning (gitignored)
config.example.json        # the template, committed
settings.json              # live proxy settings, written by the dashboard

Dockerfile compose.yaml    # the engine as a container - see Running in a container
docker/                    # entrypoint and config template (config.json, known_hosts gitignored)

tests/                     # what pytest runs: no model, no network, seconds
  unit/                    # arithmetic, parsing, cache lookups
  features/                # behaviour specs (pytest-bdd) + steps/
  fakes.py                 # the fake model server both suites run against

docs/lessons-learned.md    # what running these models under agents taught us
logs/                      # model server output, one file per launch (gitignored)
```

`inference/` is a flat package on purpose rather than `src/inference/`: it means
`python -m inference engine` works from a bare checkout with nothing installed,
which is how you want to try this on a second Mac.

`tests/` is what CI runs — checks against a fake model server, with no weights
on disk and no network.

Each file in `inference/models/` is deliberately self-contained: HF repo, quant per
machine, context, sampling params, thinking switches and speculative decoding all
live in that one file, comments and all. There's duplication between them by
design — you should be able to open two model files side by side and see exactly
where they differ. The modules beside it hold only what is genuinely identical for
every model.

Adding a model is one new file in `inference/models/` — it's picked up
automatically, and needs two import lines:

```python
from ..machine import Machine, Options, Unsupported
from ..specs import LlamaServerSpec
```

Findings from running these models under real agent workloads — the failure
modes, what breaks and why, and how to measure it — are in
[docs/lessons-learned.md](docs/lessons-learned.md).

## Running in a container

The engine — routing, recording, the dashboard — also runs as a container:

```
ghcr.io/sixeyed/inference-server
```

The models don't come with it. MTPLX and MLX are Apple Silicon only, so the
models keep running natively on a Mac, and the containerised engine starts and
stops them over SSH by running a checkout of this repo on that Mac — `inference
run --plan`, `run --detach`, `pull`, `status`, `stop` and `machine`. Everything
that depends on the Mac — which llama.cpp build, whether the weights are on disk,
the sandbox — is still decided on the Mac.

`model_host` in the engine's config switches that on. Without it nothing changes.

### On the Mac that runs the models

Clone this repo on the Mac, and check that it runs: `python3 -m inference models`.
Then turn on Remote Login (System Settings > General > Sharing), and make a key
just for the engine:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/inference -C inference-server -N ''
PUB=$(cat ~/.ssh/inference.pub)
grep -qF "$PUB" ~/.ssh/authorized_keys || echo "$PUB" >> ~/.ssh/authorized_keys

# prove the key and the non-interactive PATH, with ~/.ssh/config bypassed
ssh -F /dev/null -i ~/.ssh/inference -o IdentitiesOnly=yes -o BatchMode=yes "$USER@localhost" \
  'cd ~/inference-server && PATH=/opt/homebrew/bin:/usr/local/bin:$PATH python3 -m inference machine'
```

Append the key directly rather than with `ssh-copy-id`, which checks the agent
instead of the file and can report success having copied nothing.

### The engine, on the same Mac

Pin the Mac's host key under the name the container reaches it by, and fill in
the config — your user name, and where the checkout is:

```bash
ssh-keyscan -t ed25519 localhost 2>/dev/null | sed 's/^localhost/host.docker.internal/' > docker/known_hosts
cp docker/config.example.json docker/config.json      # then edit model_host
docker compose up -d
curl -s localhost:8084/health
```

Stop a native `inference engine` first — both want 8084, and both would drive
the same model port. Compose publishes the engine on `127.0.0.1` only; change
the port mapping in `compose.yaml` to serve your network.

`/data` is a named Docker volume, so the container keeps its own `usage.db`,
traces and settings apart from a native engine's. It is not a bind mount of
the repo: SQLite's WAL index is shared memory, which does not work across Docker
Desktop's VM file share, and a bind-mounted `usage.db` read from the Mac while
the container has it open gets corrupted. Look at the container's data from
inside it — `docker compose exec inference ...`.

### The engine, somewhere else

Run the same image anywhere that can SSH to the Mac — a server, a cluster. Set
`ssh` and `address` in `model_host` to the Mac's hostname, and set `model_bind`
in the engine's config to an address on the Mac that the engine can reach
(`0.0.0.0` for all of them): the engine passes it to `inference run` on the Mac,
and by default model servers listen on loopback only. Behind a reverse proxy,
`--trust-proxy` (on in the image) takes each client's address from
`X-Forwarded-For`.

Test with `qwen3-tiny` — Qwen3 0.6B, 639MB, routable by name but never listed in
`/v1/models` — rather than loading 20GB next to whatever else the Mac is doing.

Keep the Mac's checkout on the commit the image was built from. `run --plan`
reports the checkout's sha, and a mismatch is recorded as a `drift` event,
because the model files in the two can differ.

## Models

| Model | Engine | Type | Params | Active | Best for |
|-------|--------|------|--------|--------|----------|
| [qwen3.8](https://huggingface.co/Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed) | MTPLX | Dense | 27B | 27B | **Default.** Best-in-class for its size; coding, agentic, vision |
| [muse-glimmer](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF) | llama.cpp | Dense | 30B | 30B | Agentic tasks — trained end-to-end for tool use and failure recovery |
| [kat-coder2.5](https://huggingface.co/Kwaipilot/KAT-Coder-V2.5-Dev) | llama.cpp | MoE | 35B | 3B | Autonomous SWE — 69.4% SWE-bench Verified |
| [qwen3.6-moe](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) | llama.cpp | MoE | 35B | 3B | Fast coding — beats 122B-A10B on SWE-bench at ~3× decode speed |

`kat-coder2.5` was called `kat-coder` until a V3 became likely enough to want the
version in the name. The old name still routes — model files can declare
`FORMER_ALIASES`, so a rename doesn't invalidate old traces or whatever is in
your shell history.

### Retired

Superseded, but kept rather than deleted: each file is the record of how earlier
measurements were configured, and reproducing one needs that exact quant, context
and sampling. `inference run` still launches any of them. The engine leaves them out of
`/v1/models` and won't route to one without `--serve-deprecated`.

| Model | Engine | Why |
|-------|--------|-----|
| [qwen3.6](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF) | llama.cpp | Superseded by qwen3.6-moe |
| [qwen3.5](https://huggingface.co/unsloth/Qwen3.5-122B-A10B-GGUF) | llama.cpp | Superseded by qwen3.6-moe |
| [qwen3.5-small](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF) | llama.cpp | Superseded by qwen3.6-moe |
| [qwen3-coder-next](https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF) | llama.cpp | Superseded by qwen3.6-moe and qwen3.8 |
| [gemma-4-31b](https://huggingface.co/FakeRockert543/gemma-4-31b-it-MLX-8bit) | mlx-lm | Superseded by qwen3.8 |
| [gemma-4-26b-a4b](https://huggingface.co/FakeRockert543/gemma-4-26b-a4b-it-MLX-8bit) | mlx-lm | Superseded by qwen3.8 |
| [ling-2.6-flash](https://huggingface.co/mlx-community/Ling-2.6-flash-mlx-5bit) | mlx-lm | Needs an unreleased mlx-lm fork for the `bailing_hybrid` arch |

No live model uses `mlx_lm.server` today. The engine still supports it — the
next MLX-only release will want it.

### The three new ones

**Qwen3.8-27B** — Alibaba's dense flagship for its size class. Outperforms
Qwen3.7-Plus overall at 27B params, with the gains concentrated in real-world
coding and office workflows. Natively multimodal, 256K context, and adds a
Developer role so it drops into Codex-style agent harnesses. Tool calling handles
nested objects properly now.

Served by **MTPLX** rather than llama.cpp — see [Speculative decoding](#speculative-decoding).
The llama.cpp path was too slow on this hardware, and its MTP option needed both a
third-party converted GGUF and a very recent llama.cpp build.

**Muse-Glimmer-30B** — Meta Superintelligence Labs, Apache-2.0, distilled from
Muse Spark and built specifically to run agentic workloads on consumer hardware.
Where most local models are chat models you push into tool use, this one was
trained for end-to-end task completion: long-horizon planning, schema-accurate
function calls, and recovery when a call fails. Ships with a DFlash drafter for
block-speculative decoding (see below).

**KAT-Coder-V2.5-Dev** — Kwaipilot, Apache-2.0. Built on Qwen3.6-35B-A3B, then
RL'd inside 100,000+ executable repository environments, so it was rewarded for
running commands and iterating to a green test rather than for plausible-looking
diffs. 69.4% on SWE-bench Verified from 3B active params. The RL stage explicitly
targeted excessive parallel tool calls and content repetition — the exact
failure modes that make local models painful under Cline.

## Hardware Configs

Detected per machine, then filled in from `config.json` — see
[Configuration](#configuration). The chip is detected; the address and tuning
below are what this repo's two boxes were measured at, and live in
`config.json` rather than in the source.

| | M4 Max | M1 Max | M2 |
|---|---|---|---|
| RAM | 128GB | 64GB | 24GB |
| Threads | 12 | 10 | 6 |
| Batch / ubatch | 4096 / 2048 | 2048 / 512 | 1024 / 512 |
| Prompt cache | 32GB | 16GB | 4GB |

### Why not BF16 on the M4 Max?

Muse-Glimmer BF16 is 58GB and Qwen3.8 BF16 is 55GB — both fit in 128GB. But a
dense model's decode speed is (model size / memory bandwidth), so BF16 roughly
halves tok/s for a quality difference you can't measure on coding work.
UD-Q8_K_XL is the ceiling worth using for the dense models here. The one
exception is `qwen3.5-small`, where 3B active params make BF16 cheap enough to be
worth it.

## Engines

### llama-server (llama.cpp)

```bash
brew install llama.cpp
```

GGUF files are auto-downloaded from HuggingFace on first run.

> **The three new models need a recent llama.cpp.** Homebrew's bottle can lag
> upstream by a long way — check `llama-server --version` against
> `brew info llama.cpp` and `brew upgrade llama.cpp` if you're behind. The
> harness detects a build too old for MTP/DFlash and disables speculative
> decoding with a warning rather than failing at launch, but architecture
> support for Muse Glimmer is not something it can work around.

### mtplx (MTPLX)

```bash
brew install youssofal/mtplx/mtplx
```

[MTPLX](https://github.com/youssofal/MTPLX) (Apache-2.0) is MLX-native and does
speculative decoding off the model's *own* MTP head — no second draft model in
RAM. Used for `qwen3.8`. Weights are MLX safetensors, downloaded on first run.

Two things it does differently from the other two engines:

- **Auth is not optional.** `--no-auth` only relaxes a *localhost* bind, and
  `model_bind` can open the bind up. So a key is always passed — but not a
  secret one: `inference run` passes `none`, the literal string
  the clients here already have in their API-key field. **No client config
  changes.** Note it must be that four-character string, not an empty one — an
  empty key is rejected like a missing one. Set `api_key` in `config.json` if
  you want a real one.
- **It appends a tok/s footer to responses by default.** That lands inside
  assistant content, so it would corrupt transcripts and anything parsing tool
  calls out of a reply. The harness always passes `--no-stats-footer`.

It also serves `/v1/messages` (Anthropic-compatible) alongside the OpenAI routes.

### mlx_lm.server (mlx-lm)

[mlx-lm](https://github.com/ml-explore/mlx-lm) runs MLX-format models natively on
Apple Silicon. Used for `ling-2.6-flash` (only published as MLX weights) and the
Gemma 4 models (PLE-safe MLX quants — see below). M4 Max only.

```bash
# The released mlx-lm doesn't yet include the bailing_hybrid arch Ling-2.6 needs
pip3 install --user --force-reinstall --no-deps \
  "git+https://github.com/ivanfioravanti/mlx-lm.git@add-ling-2.6-flash"
```

> **TODO:** swap back to `pip3 install --user mlx-lm` once
> [PR #1227](https://github.com/ml-explore/mlx-lm/pull/1227) is merged and released.

Unlike llama-server, mlx_lm.server has no `--alias` — clients must send the full
HuggingFace repo id in the request `model` field. `GET /v1/models` returns the id
the server is currently serving.

> **Gemma 4 PLE-safety:** standard `mlx-community`/`unsloth` Gemma 4 MLX quants
> corrupt Per-Layer Embedding layers (they quantize every layer including PLE /
> `ScaledLinear`) and produce garbage output. The Gemma model files enforce an
> allowlist of PLE-safe repos (`FakeRockert543/gemma-4-*-MLX-{4bit,8bit,bf16}`
> and `mlx-community/gemma-4-*-bf16`) and abort launch otherwise.

## Egress sandbox

Model servers are launched under `sandbox-exec` with outbound network denied.
This is on by default, for every engine and both launch paths.

The reasoning is that a model server needs the network exactly once — to
download weights. After that it only answers on its own port. So denying it
outbound sockets costs nothing, and it changes the guarantee from "I read the
source and found no exfiltration" to "it cannot, whatever the source says".
That matters most for `mtplx`, which is a third-party binary running arbitrary
code over your prompts.

The profile lives in `SANDBOX_PROFILE` (`inference/config.py`). Order is
load-bearing: `(allow default)`, then `(deny network-outbound)`, then narrow
re-allows for unix sockets, `network-bind` and loopback. A re-allow placed
before the deny is silently overridden by it — and a profile that denies
nothing looks exactly like one that works until you try to reach the network,
so `tests/unit/test_sandbox.py` pins the ordering.

**Weights are pulled before launch, not by the server.** Every engine will
download its weights at launch when they're missing — llama-server via `-hf`,
mlx_lm.server and MTPLX via the repo id — and the sandbox blocks exactly that.
Lifting the sandbox for a first run would hand the network to the one process
it exists to keep off it. So the download is a separate step, done before the
server starts:

- **GGUF and MLX models** are fetched by `inference/weights.py`, a standard
  library HuggingFace client in this repo. It fetches only the quant the model
  file asks for (GGUF repos hold every quant, often hundreds of GB together),
  writes the standard HuggingFace cache layout, resumes an interrupted download
  and checks each file's SHA-256. `HF_TOKEN` is used for gated repos.
- **MTPLX models** are fetched by `mtplx models --update`, MTPLX's own delta
  downloader. That runs with the network, but it only downloads; it never
  serves.

The server is then started sandboxed as usual, offline (`HF_HUB_OFFLINE=1`),
and never gets the network.

The engine does this by itself the first time a client picks a model that
isn't on disk. It pulls first, while the current model carries on serving, and
switches only once the weights have landed. A pull that fails leaves the
current model loaded. `inference run` does the same, and you can do it ahead of
time:

```bash
inference pull <model>        # fetch, or update to the latest revision
```

An interrupted pull is marked, so it reads as not on disk and the next launch
resumes it rather than loading half a model. `"auto_pull": false` in
`config.json` (or `--no-pull`) turns automatic pulling off. Then a sandboxed
launch with no weights is refused before anything is stopped, with the
`inference pull` command to run.

`--no-sandbox` is accepted by both `run` and `engine`, but nothing needs it.

**What it does not cover:** the built-in MTPLX browser chat UI at
`127.0.0.1:8083` pulls `marked.js` from a CDN. That fetch happens in your
browser, not in the sandbox. Irrelevant if you only use the API.

## Speculative decoding

Three of the models can predict several tokens per forward pass and verify them,
which is a straight throughput win at identical output quality.

| Model | Mechanism | Default | Notes |
|---|---|---|---|
| `muse-glimmer` | DFlash block-diffusion drafter | **on** | Drafter ships in the official repo (`dflash-kquant.gguf`). Block size 16, so `--spec-draft-n-max` is capped at 15. |
| `qwen3.8` | Native MTP (MTPLX) | **on** | The MTP head ships in the weights; the tuned depth (3) ships in the repo's `mtplx_runtime.json`. `--no-spec` drops to target-only AR against the *same* weights, which makes the A/B honest. |
| `kat-coder2.5` | MTP | off (`--spec`) | Same — swaps to `gbuzhf/KAT-Coder-V2.5-Dev-MTP-GGUF`. |

On the llama.cpp models MTP is worth roughly 1.4–2.2× decode throughput, and it
needs an MTP-converted GGUF because the prediction heads aren't in the standard
weights — which is why `--spec` changes the repo rather than just adding a flag,
and why it's opt-in where only a third-party conversion exists.

`qwen3.8` is the exception: MTPLX reads the MTP head straight out of the published
weights, so speculation is on by default and needs no repo swap. Drafts are
accepted by the probability-ratio rule with residual resampling, so the sampled
distribution matches unspeculated decode at any temperature — the speedup is not
paid for in output quality.

> **Read the vendor's numbers carefully.** The published 58.7 tok/s for
> `Qwen3.8-27B-MTPLX-Optimized-Speed` is an **M5 Max** with fans verified at max.
> The M4 Max here has no M5 TensorOps path (`mtplx hardware` says
> `M5 TensorOps eligible: false`), so expect less. Measure it.

> **And note the quant.** The llama.cpp setup ran UD-Q8_K_XL on the M4 Max, which
> is near-lossless. MTPLX Optimized Speed is a dynamic 4-bit build: KL 0.0220 to
> the bf16 original on the vendor's coding battery. That divergence is from the
> *quant*, not the speculation. `VARIANT = "quality"` in `models/qwen3_8.py`
> switches to the 8-bit build (30GB) if a comparison needs the old bar back.

```bash
inference run muse-glimmer            # DFlash on
inference run muse-glimmer --no-spec  # plain decode
inference run kat-coder2.5 --spec        # switch to the MTP weights
inference run qwen3.8                 # native MTP, on by default
inference run qwen3.8 --no-spec       # same weights, plain AR — the A/B
```

## Thinking / reasoning

Every model here handles this differently, which is half the reason each one has
its own file.

| Model | Control | Default here |
|---|---|---|
| `qwen3.8` | `--reasoning on/off` plus `--reasoning-effort` (xhigh/high/medium/low) — MTPLX server flags, not template kwargs | **off** |
| `qwen3.6`, `qwen3.6-moe`, `qwen3.5`, `qwen3.5-small` | `--reasoning-budget N` | `0` (off) |
| `kat-coder2.5` | `enable_thinking` / `preserve_thinking` via `--chat-template-kwargs` | **on**, preserved across turns |
| `muse-glimmer` | `Reasoning strength: <level>` line in the **system prompt** — no server flag | tells you to set `high` |
| `qwen3-coder-next` | n/a — not a thinking model | — |

`--thinking` / `--no-thinking` overrides the default for a run.

**Why most of these default to off:** under Copilot Agent and Cline, thinking-mode
qwen3 defers ("let me investigate...") rather than committing, and burns the
client's max-tool-calls-per-turn before producing a visible answer. Disabling
thinking makes the model commit; quality on multi-step planning hasn't been worse
in practice for this workflow.

**Why kat-coder2.5 is the exception:** removing exactly that behaviour is what its RL
stage was for, and its 69.4% SWE-bench number is a thinking-mode number. Turning
thinking off throws away the reason to run it. If the deferral loop does show up
under Cline, `--no-thinking` puts it back on the usual default (and switches to
the instruct sampling profile, which matters — see below).

**Sampling profiles follow thinking mode.** Qwen publishes two different sets and
using the wrong one is the usual cause of repetition loops: thinking wants
`temp=1.0, top_p=0.95, presence_penalty=0.0`; instruct wants
`temp=0.7, top_p=0.80, presence_penalty=1.5`. The model files switch between them
automatically when you flip `--thinking`.

A blank `content` field with non-empty `reasoning_content` in client responses is
the symptom of thinking eating all of `max_tokens` — raise `max_tokens`, lower the
reasoning effort, or turn thinking off.

## Inference engine

`inference engine` is the front door. It listens on `8084`, starts models on `8083`, and
is the only thing a client needs to know about.

```bash
inference engine                          # 8084 -> models on 8083
inference engine --model qwen3.8          # pre-warm one at startup
inference engine --serve-deprecated       # route to the retired models too
inference engine --no-ui                  # no dashboard
inference engine --list                   # list captured traces
```

**Models start on demand.** The first request for a cold model blocks while the
weights load — 30s to 2min warm, much longer on a first download. The request is
held open rather than refused, because no coding agent retries a 503 gracefully.
The console tails the server's own log while it loads, so a long wait doesn't
look like a hang, and the dashboard shows the same thing with a progress panel.

**Switching model kills the current one.** Anything streaming is cut. A swap only
happens because a human asked for a different model, and at that point the old
model's half-finished answer is worth nothing; waiting for it to drain would let
one wedged socket hold the box hostage. A cut turn is recorded as `PREEMPTED`
rather than as an error, and is excluded from the statistics — a turn that
stopped mid-generation has a real token count and a meaningless decode rate.

**`GET /v1/models` lists every available model, not just the loaded one.** That is
the whole point: a client's picker is populated from it, so choosing a cold model
is what triggers the load. The server behind the engine reports only what it is
currently serving; the engine's answer comes from the registry instead. Models
that don't fit this Mac, and retired ones, are filtered out.

**Aliases work for every backend.** `mlx_lm.server` has no `--alias` and only
answers to the full HuggingFace repo id, so the engine rewrites the `model` field
on the way out. You configure clients with `gemma-4-31b`, not
`FakeRockert543/gemma-4-31b-it-MLX-8bit`. The trace keeps both.

**An unknown model id is served rather than refused.** Copilot and Cline both
occasionally send an id from a stale profile, and a 404 reads to them as "endpoint
broken". The default is to serve whatever is loaded and print one warning;
`--on-unknown 404` is available when you want strictness for a benchmark.

**A launch that cannot work is refused before the current model is stopped.**
With the sandbox on and the weights not on disk, the download is blocked and the
launch is a known failure rather than a risk — so the engine says so and keeps
serving what it already had. Finding out after the swap costs a working session
to learn something knowable beforehand.

**A server you started by hand is adopted, not restarted.** If `inference run` already
has something on 8083, the engine picks it up, uses it, and never kills it —
including on its own shutdown.

| Route | |
|---|---|
| `GET /` | the dashboard |
| `GET /v1/models` | every available model |
| `GET /health` | engine health and current state (200 even when idle) |
| `GET /status` | model, engine, pid, load time, uptime, in-flight count |
| `POST /api/load` | pre-warm a model — `{"model": "qwen3.8"}` |
| `POST /api/unload` | stop the current model, free the memory |
| `GET`/`POST /api/settings` | the strip list, read and written by the Config tab |
| `POST /v1/chat/completions` | routed, relayed, recorded |
| `POST /v1/messages` | the same, for MTPLX's Anthropic-compatible route |

Model server output goes to `logs/<alias>-<timestamp>.log`. When a model fails to
start, the last 30 lines come back in the error body — Cline and Copilot both
render it, and it is the only place you will ever see `unknown flag --spec-type`.

## Dashboard

`http://<host>:8084/` — two tabs. **Dashboard** shows what is loaded, what is
available, memory, per-model statistics and a quick-chat smoke test.
**Config** lists the client parameters that can be stripped from incoming
requests (see [Stripping client parameters](#stripping-client-parameters)) and
sets the per-client system-prompt substitution (see
[Rewriting the system prompt](#rewriting-the-system-prompt)). It can drive the engine: clicking **Load** on a model starts it, and
says `Load (cuts current turn)` while a request is in flight, because it does.

Tailwind is **vendored**, not built. `static/tailwind.css` is generated once and
committed, so the page needs no network and `inference engine` needs no
toolchain. To regenerate after changing the markup:

```bash
# one-off: the standalone binary, no npm and no node
curl -sLo tailwindcss \
  https://github.com/tailwindlabs/tailwindcss/releases/latest/download/tailwindcss-macos-arm64
chmod +x tailwindcss

./tailwindcss -i static/input.css -o static/tailwind.css --minify
```

Forgetting to regenerate shows up as an unstyled element, not an error.

## Configuration

Three kinds of setting, kept apart because they change on different timescales.

**`inference/models/*.py` — how a model is deployed.** Quant per machine,
context, sampling, thinking switches, speculative decoding. One file tells you
the whole deployment for one model.

**`config.json` — where this machine is.** Paths, addresses, per-chip tuning.
Gitignored, because it describes your machine. Copy `config.example.json` and
edit. Omit anything you don't need and it falls back to a default that works on
a fresh clone.

```json
{
  "weights_root": "/Volumes/your-ssd/weights",
  "bind": "0.0.0.0",
  "api_key": "something-not-none",
  "llama_load_mode": "mmap+mlock",
  "machines": {
    "m4": { "chip": "M4 Max", "host": "192.168.1.10", "threads": 12,
            "batch": 4096, "ubatch": 2048, "cache_ram_mb": 32768 }
  }
}
```

| Key | Default | |
|---|---|---|
| `bind` | `127.0.0.1` | where the engine listens; `0.0.0.0` serves your network |
| `model_bind` | `127.0.0.1` | where model servers listen; only open it up when the engine runs on another machine |
| `auto_pull` | `true` | fetch missing weights before launch; `false` refuses instead |
| `weights_root` | `~` | where the HuggingFace and MTPLX caches go |
| `api_key` | `none` | the one key for clients and model servers |
| `machines.<chip>.host` | `127.0.0.1` | the address the dashboard tells clients to use |
| `client_endpoints` | this machine | the addresses the dashboard's Clients tab offers |
| `model_host` | none | drive a Mac's model servers over SSH — see [Running in a container](#running-in-a-container) |

Everything listens on loopback until you say otherwise. If you open `bind` up,
set a real `api_key` too: the default `none` is no auth at all.

The chip is still detected; `machines` only supplies what follows from it, so a
64GB M4 Max is a config change rather than a code change. Precedence is
environment variable, then `config.json`, then the built-in default —
`INFERENCE_WEIGHTS_ROOT`, `INFERENCE_BIND`, `INFERENCE_MODEL_BIND`,
`INFERENCE_AUTO_PULL`, `INFERENCE_CONFIG`, `HF_HOME` and `MTPLX_API_KEY` all
win over the file, because setting one is a deliberate act at the moment of
running something.

**`settings.json` — how the proxy behaves right now.** Written by the dashboard's
Config tab, reloaded on restart. See below.

### Weights on another volume

`weights_root` moves both caches at once: llama-server and mlx_lm.server resolve
through the HuggingFace layout (`HF_HOME`), and MTPLX keeps its own flat cache
beside it. The child process inherits the same environment, so the launcher and
the engine cannot disagree about where the weights are.

The MTPLX **session bank** stays on the internal disk deliberately. Unlike the
weights it is read *during* serving — restored on prefill, which is the current
bottleneck — and it is budgeted to grow to 48GB. It is the one thing here that
actually wants the fast disk.

`llama_load_mode` defaults to `mmap+mlock`: llama.cpp maps the GGUF then pins it
in RAM, so weights on a slower external disk are read once at load and never
paged back in mid-request. That costs locked RAM — drop to `auto` on a machine
smaller than 128GB.

If `weights_root` is on a volume that is not mounted the engine says so and
stops, rather than letting
`find_in_hf_cache()` return nothing and llama-server quietly start
re-downloading tens of gigabytes to a path that isn't there.

### Stripping client parameters

Clients send their own sampling settings, which override whatever the model's
file tuned it to — VS Code Copilot sends `temperature: 0.1, top_p: 1` on every
request. Near-greedy decoding is how an agent session gets stuck re-emitting an
identical tool call: each turn's context differs only by an identical
assistant/tool pair, so a deterministic decoder re-derives the same answer.

The dashboard's **Config** tab lists the parameters that can be stripped. Strip
one and the key is removed on the way through, so the model's own default
applies. It takes effect on the next request with no restart, and it is
remembered across restarts — a setting you had to remember to pass on the
command line is one you will not have set when it matters.

Strip only, never override: sampling stays defined in one place, the model's own
file. `model`, `messages`, `tools`, `tool_choice`, `stream` and `stream_options`
are protected, because removing those changes what was asked rather than how it
was sampled. The trace records `stripped_params` and keeps the client's original
values in `request_original`.

### Rewriting the system prompt

VS Code sends about 15,000 characters of Copilot framing on every turn. A local
model pays for that twice — in context, and in attention spent on instructions
written for a frontier model. The prompt from `traces/turn-1-request.json`
breaks down like this:

| offset | block | chars |
|---|---|---|
| 0 | *untagged preamble* | 565 |
| 565 | `<instructions>` | 2,257 |
| 2,823 | `<toolUseInstructions>` | 1,733 |
| 4,557 | `<editFileInstructions>` | 2,306 |
| 6,864 | `<notebookInstructions>` | 912 |
| 7,777 | `<outputFormatting>` | 575 |
| 8,353 | `<memoryInstructions>` | 1,929 |
| 10,284 | `<instructions>` | 4,383 |

The dashboard's **Config** tab substitutes named blocks of that, per client,
from the files in `inference/prompts/`. Pick a file and tick the blocks it
should replace; anything you don't tick is forwarded exactly as the client sent
it. Replacing the preamble and the first `<instructions>` with the shipped
`terse-coder.md` takes the prompt above from 15,126 to 13,120 characters.

Two things it deliberately does not do:

- **Only the first match of a tag is replaced.** That last row is not a
  duplicate — VS Code sends `<instructions>` twice, and the second one wraps the
  workspace's `<skills>` and `<agents>`. Replacing both would silently delete
  the skills you configured.
- **It never replaces a block the file has no section for.** Asking for a block
  the prompt doesn't contain is a no-op too, so a file written for VS Code does
  nothing surprising when Cline connects.

`<toolUseInstructions>` and `<editFileInstructions>` are flagged **contract** in
the UI: they describe the shape the client parses replies against, so replacing
one is how you break tool calling rather than how you shorten a prompt. They can
be replaced, but that is a thing to do on purpose.

Prompt files are read per request, so editing one takes effect on the next turn
without restarting the engine — which is the only way iterating on a prompt is
bearable. Each file is one system prompt with a `<name>` section per block it
replaces; see `inference/prompts/README.md`. The trace records `rewrote_prompt`
(the file, the blocks, and the character counts either side) and keeps the
client's original message in `request_original`, so a trace never claims the
model saw something it didn't.

### Filtering the tool list

The prompt is the smaller half of what a client sends. VS Code also offers its
whole tool surface on every turn — 80 definitions in the most recent captured
session, about 23,000 tokens of JSON schema, resent in full on every request of
every session.

Across the 15 trace files in `traces/` it offered **103 distinct tools and the
model called 18**:

| tool | calls | | tool | calls |
|---|---:|---|---|---:|
| `grep_search` | 436 | | `create_file` | 14 |
| `read_file` | 411 | | `fetch_webpage` | 14 |
| `run_in_terminal` | 304 | | `manage_todo_list` | 11 |
| `replace_string_in_file` | 95 | | `github_text_search` | 3 |
| `list_dir` | 35 | | `get_terminal_output` | 2 |
| `task_complete` | 29 | | `github_repo` | 2 |
| `file_search` | 20 | | `insert_edit_into_file` | 2 |
| `get_errors` | 15 | | `vscode_askQuestions` | 1 |
| `memory` | 15 | | `vscode_searchExtensions_internal` | 1 |

The other 85 are ~76% of the payload, and they cost twice: context the
conversation cannot use, and 85 wrong answers for the model to choose between
every time it decides what to call next. Notable dead weight — around 30
dotnet-SDK-management tools, the Pylance MCP set, every notebook and browser
tool, `runSubagent`, `vscode_renameSymbol`, `session_store_sql`.

A **tool set** names the tools a client may offer. The dashboard's **Config**
tab has one per client, beside the prompt picker: pick a set and the rest are
dropped on the way through. `inference/toolsets/vscode-core.txt` is those 18,
less the two called exactly once, plus three companions the traces would not
show until something goes wrong (`get_terminal_output` for a backgrounded
terminal, `insert_edit_into_file` for when `replace_string_in_file` fails,
`resolve_memory_file_uri` for `memory`). On a real captured turn that is 66
tools down to 16 — ~19,600 schema tokens down to ~6,900, about 30% off the whole
request.

An allowlist, not a patch: a name the client didn't send is reported in the
trace rather than added, because the engine has no definition to add. And a set
that matches nothing is treated as a typo — the client's tools go through
untouched, because zero tools doesn't make a coding agent fail, it makes it
quietly stop being an agent.

This is the one place the engine changes *what was asked* rather than how it is
sampled, which is why `tools` stays protected in the strip list and this is a
separate, opt-in, per-client mechanism. Sets are read per request, so editing
one takes effect on the next turn. The trace records `filtered_tools` (the file,
what was kept, what was dropped, and any name the client no longer offers) and
keeps the full array in `request_original`.

### Breaking a stuck tool-call loop

On 27 August a session asked about ARM builds got stuck. The model built a
malformed Azure DevOps URL — `[char]38` (`&`) where the first query parameter
needs `?` — and re-issued the byte-identical command **37 times**. By the last
turn, 74 of the 102 messages were that same wrong command and that same 404.

Sampling cannot fix this, and it is worth being precise about why:

- Temperature was already stripped, so the model's own 0.7 applied — not
  greedy. Replaying the stuck turn still gave byte-identical output **6/6**.
  The distribution in that context is peaked, not noisy.
- Temperature is a per-token knob; the repetition is *across turns*. Each turn
  is a fresh sample from a context that genuinely leads to the same place.
- `presence_penalty` has the same blind spot — it only sees within one
  generation.

It is not a knowledge problem either. The same model asked cold produces the
correct URL every time. It is context poisoning: the transcript contains 37
examples of the wrong answer, immediately before the question.

The engine is the only component positioned to fix it. Chat completions is
stateless, so the whole history arrives every turn, and the client — VS Code
here — did not stop it after thirty-seven tries.

**What the fix had to look like was measured, not guessed.** Replaying the
stuck turn against the real model:

| what the model is left with | escapes the loop |
|---|---|
| one copy of the call, plus an explanatory note | 0 / 3 |
| one copy of the call, plus a *directive* note | 2 / 3 |
| no copy, note only | 3 / 3 |
| **no copy, with the shared result kept in the note** | **4 / 4** |

The anchor is the command text, not the error. An example of the wrong command
in front of the model is an example it copies, however firmly the note tells it
not to. So the run is replaced by a single message reporting what came back and
never what was sent — on the real turn, 102 messages down to 29.

Set a trigger count per client on the **Config** tab; `0` is off. Two things
keep a genuine poll safe: the fingerprint covers the tool *results*, so a poll
whose output moves on (`queued` → `running`, an elapsed time, a percentage) is
never a match at any threshold; and the count is there for one whose output
doesn't. Runs are collapsed wherever they appear, not only at the tail —
otherwise the moment the model escapes, its new call makes the run non-trailing
and the whole loop comes back on the next turn.

The trace records `collapsed_loops` and keeps the full history in
`request_original`, and every intervention gets its own row on the Logs page —
undeduped, unlike the other events, because each one is the engine changing a
live conversation.

### The Logs page

The engine changes things on the way past — it strips parameters, rewrites
prompt blocks, filters tools, repairs tool arguments. All of that used to be a
console line printed once per engine run, guarded by a boolean set on the first
occurrence and never cleared. That reported the *first* state and hid every
later one, which is backwards: the case worth seeing is the one that appears
halfway through a session.

The **Logs** tab records an event when something *changes*, keyed on content
rather than a flag. A long session with steady settings writes nothing; a
client update that changes what it sends writes a line. Rows live in
`usage.db`, so they survive the restart you do straight after an update, and
failed model loads are merged in on read from `loads` rather than copied, so a
failure can't go stale in a second copy. There is no retention job — deduping
makes this tens of rows a week, and a cleaner would only delete the history you
opened the page to read.

**Drift** is the line to act on, and it is narrower than "the set names a tool
the client didn't send". A set entry the client has *never* offered has simply
never applied — reporting it would light the indicator permanently, which is
the same as not having one. Drift is a tool the client **did** offer and has
stopped offering: the model has silently lost it, and the guard doesn't fire
because every other name still matches. That is what a rename in a client
release looks like from here — and VS Code really does rename tools, having
moved the whole Pylance family from `mcp_pylance_mcp_s_*` to
`mcp_provides_tool_*` between 20 and 26 August in the captures here.

The Config tab shows the same thing against the client it affects, next to the
tool set that asks for it. The engine can't detect drift for a tool it has
never seen, which is honest: on a fresh `usage.db` there is nothing to have
drifted from.

## Client Setup

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
anywhere else, set `bind` (see [Configuration](#configuration)) and use the
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

### Open WebUI

1. In **Settings > Connections**, add an OpenAI-compatible connection
2. Base URL `http://<host>:<port>/v1`, API key `none`
3. The model appears by its alias

### Cline (VS Code)

1. Provider: **OpenAI Compatible**
2. Base URL `http://<host>:<port>/v1`, API key `none`
3. Model ID: the alias (or full HF repo id for MLX models)

For `muse-glimmer`, add `Reasoning strength: high` to Cline's custom instructions
— that's the only way to set its reasoning level.

### opencode

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

### VS Code (custom model endpoint)

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

## Performance Testing

### Capturing real agent traffic

The engine records every turn it serves. There is nothing to start separately —
just work normally with a client pointed at `8084`.

```bash
inference engine                 # records to traces/trace-<timestamp>.jsonl
inference engine --list          # what got captured
inference engine --no-trace      # don't
```

Chat completions is stateless — the client resends the whole conversation every
turn, and tool results (file contents, terminal output) are messages within it.
So **a captured request is entirely self-contained**: replaying it needs no
workspace, no repo and no client. Requests are recorded exactly as received, so a
replay sends the model precisely what the real client sent.

Timings are measured at the engine, so they mean the same thing for
llama-server, mtplx and mlx_lm.server:

| | |
|---|---|
| `ttfb_ms` | first byte back from the server |
| `ttft_ms` | first token of any kind — **including reasoning** |
| `ttfc_ms` | first *visible* token — content or a tool call |

`ttfc_ms - ttft_ms` is the deferral cost: how long a thinking model spends
reasoning before the client has anything to show. That turns the "thinking mode
stalls under agent harnesses" problem into a number you can compare per model and
per reasoning level.

The engine also records `cache_n` from llama.cpp's `timings`, which is how you
tell a genuine warm turn (small delta on a reused prefix) from a cache miss.

Each record also carries `routing` — what the client asked for, what it resolved
to, and how long a cold start took if that turn triggered one. `load_ms` sits
outside `timing` deliberately: a cold start is not a slow model, and `ttfc_ms`
has to keep meaning what it means in every trace already on disk.

> ⚠️ Traces contain whatever source code was in the session. `traces/` is
> gitignored — keep them out of any repo.

### Usage stats

Alongside the traces, the engine writes one row per turn to `usage.db` (SQLite,
gitignored). The traces are what you read to find out *what happened*; this is
what you query to find out *which model is faster*. `trace_file` and `trace_seq`
join the two, so any row can be taken back to the exact exchange.

```bash
sqlite3 usage.db 'SELECT alias, engine, rate_source, COUNT(*) AS turns,
  ROUND(SUM(output_tokens)*1000.0/SUM(decode_ms),1) AS pooled_tps,
  ROUND(MIN(decode_tps),1), ROUND(MAX(decode_tps),1), ROUND(MAX(ttft_ms))
  FROM requests WHERE ok = 1 AND preempted = 0 GROUP BY alias, engine;'
```

Two things decide whether the numbers mean anything:

**Rates are pooled, never averaged.** A model's tok/s is total tokens over total
decode time. Averaging per-request rates over-weights short turns, and a session
full of one-line answers would flatter a model that is slow on real work.

**`rate_source` has to be read alongside any tok/s figure.** llama.cpp and MTPLX
both publish a `timings` block with their own decode timer, so those rows say
`server`. Anything without one falls back to wall-minus-TTFT arithmetic, which
includes queueing and is not the same measurement — those say `client`. The
dashboard shows the badge for the same reason.

Preempted turns — cut by a model swap — are recorded, shown in the recent feed,
and excluded from every aggregate.

For MTPLX there is more, because it puts its entire per-request diagnostic record
on the response as `mtplx_stats`: peak memory, server-side TTFT, cache hits, and
the speculative-decoding acceptance rate (`mtp_accept`), which is the number that
tells you whether MTP is actually paying off.

This measures whatever you happened to do. For anything conclusive, control
the workload.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

`tests/features/` is the behaviour suite (pytest-bdd) and `tests/unit/` the plain
unit tests. Neither needs a model, a network or a real port — they run in a couple of
seconds against `fakes.py`, a fake model server that impersonates each engine
including the parts that actually bite: MTPLX 401s a probe with no bearer,
llama-server answers `{"status":"ok"}` where MTPLX answers `{"ok":true}`, and it
can be told to die mid-stream, which is the only way to check that a preempted
turn is told apart from a crash.

The supervisor takes its launcher **and its stopper** as arguments so the suite
can substitute both. That second one matters: `stop_all()` matches on process
name, so a test using the real one would reach straight past the fakes and
SIGTERM whatever model this machine actually has loaded.

These two packages are the repo's only dependencies, and they are test-only.
`inference engine` still needs nothing but the standard library.

## References

- [llama.cpp](https://github.com/ggerganov/llama.cpp) — the inference engine
- [mlx-lm](https://github.com/ml-explore/mlx-lm) — MLX inference for Apple Silicon
- [Unsloth GGUF quants](https://huggingface.co/unsloth) — the quantizations used here
- [Unsloth MTP guide](https://unsloth.ai/docs/models/mtp) — speculative decoding flags

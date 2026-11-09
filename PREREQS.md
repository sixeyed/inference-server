# Prerequisites

Everything this repo needs, and the versions it is currently known to work on.

Versions below are what the setup was last verified against (2026-09-05), not
minimums — where a real minimum is known, it says so. To check your own machine
against the list, see [Verifying](#verifying) at the end.

## First run on a new Mac

The minimum, in order:

1. **Command Line Tools.** On a new Mac the first `git` or `python3` asks to
   install them; say yes. They bring `git` and Python 3.9 — and that Python is
   enough. The engine needs nothing else from Python: no `pip install`, no venv.
2. **Homebrew**, from [brew.sh](https://brew.sh).
3. **The server for the models you want.** The default model, `qwen3.8`, runs on
   MTPLX; most of the others run on llama.cpp:

   ```bash
   brew install youssofal/mtplx/mtplx     # qwen3.8
   brew install llama.cpp                 # kat-coder2.5, qwen3.6-moe, muse-glimmer
   ```

4. **Disk for the weights** — 15–40 GB per model, pulled on first use into your
   home directory unless `weights_root` says otherwise.

Docker is optional. It's only for running the engine in a container, and the
Mac still needs everything above, because that's where the models run.

What happens if something is missing:

| Missing | What you see |
|---|---|
| Command Line Tools | macOS offers to install them the first time you run `git` or `python3` |
| a model's server (`mtplx`, `llama-server`) | the engine still starts and lists every model that fits. Picking one whose server isn't installed fails straight away with the install command — `llama-server not found. Install with: brew install llama.cpp`. Nothing is downloaded, and the model already loaded keeps serving |
| the weights | pulled on first use, while the request waits — see [docs/engine.md](docs/engine.md#egress-sandbox). Pull ahead of time with `python3 -m inference pull <model>` |
| Docker | only the container quick start needs it; run natively with `python3 -m inference engine` |

`python3 -m inference models` shows what fits this Mac, and which server each
model needs.

## Hardware and OS

| | verified on |
|---|---|
| Machine | Apple M4 Max, 128 GB unified memory; also run on M1 Max and M2 |
| OS | macOS 26.4.1, arm64 |
| Disk | room for the weights you use — 15–40 GB per model |

Apple Silicon is not optional. Every engine here is Metal-backed; there is no
CUDA or CPU-only path in this repo.

Unified memory is the binding constraint on which models fit, not disk. The
registry works this out per machine — `models.describe()` marks a model
`no fit` rather than letting it OOM at load. Profiles for M1 Max and M2 machines ship
in `inference/config.py` for the same reason: the same model files, different
verdicts about what will run.

> **M3 and M5 are assumptions, not measurements.** The chip is read from its
> brand string (`Apple M4 Max`), and M3 and M5 are assumed to report the same
> way. An M5 is tuned like an M4 and an M3 like an M2 — the nearest older
> generation with a profile — but neither has been run on real hardware. Any
> other chip falls back to the `unknown` profile, which picks small quants and
> short contexts so that something fits. See
> [docs/backends.md](docs/backends.md#hardware-configs).

## Python

| | version | notes |
|---|---|---|
| `python3` | 3.9.6 (macOS Command Line Tools) to 3.14 | `requires-python = ">=3.9"` — the suite passes on both ends |

**The engine has no runtime dependencies, and that is a design constraint
rather than an accident** — `python -m inference engine` has to work from a
bare checkout on a fresh Mac. `pyproject.toml` says `dependencies = []` and it
should stay that way. Anything you are tempted to add belongs in
`requirements-dev.txt` or in an engine.

Note the brew-installed `mtplx` builds its own venv on `python@3.13`,
independently of the `python3` above. The two do not need to agree.

## Inference engines

Install all three only if you want the whole registry; each model names the one
it needs.

### llama.cpp — `llama-server`

```bash
brew install llama.cpp
```

Currently `version: 0.3.0 (build 10621, commit c1d0e7a00)`.

**llama.cpp changed its version scheme at 0.3.0.** It used to report
`version: 10470 (34af94cd9)`; it now reports `version: 0.3.0 (build 10621, …)`.
`engines.llama_build()` handles both, preferring `build N`. This is display
only — the capability gates use `llama_supports()`, which greps `--help` for
specific flags and is scheme-agnostic.

Speculative decoding needs a build new enough to know `--spec-draft-n-max` and
the `draft-mtp` / `draft-dflash` modes. All three are present on 0.3.0. The
gate matters because MTP models also need a different set of weights, so
support has to be settled *before* 30 GB is downloaded.

### MTPLX — `mtplx`

```bash
brew install youssofal/mtplx/mtplx
```

Currently **2.10.2**. Serves `qwen3.8` with native multi-token-prediction
speculative decoding.

Minimum in practice is **2.9.2**. Before that, MTPLX rewrote agent transcripts
by default — compacting tool results and trimming file reads on the way to the
model. That produced a session where the model correctly reported *"the outputs
were compacted"* and was misdiagnosed as confabulation, because the proxy trace
records what the client sent and what MTPLX returned, and the rewriting happened
in between. 2.9.2 (#282) made passthrough the default, gated behind
`MTPLX_AGENT_REWRITES`. Leave that unset.

Two further fixes worth having: 2.9.1 (#310) fixed agent sessions truncating
near 19,000 tokens, and 2.10.2 (#415) fixed large prompts being admitted
optimistically and then stalling.

`mtplx stop --port 8083` will *not* stop a model the engine started — the engine
spawns `python -m mtplx.server.openai` directly rather than through the MTPLX
daemon, so `stop` reports the port is "in use, but not by an MTPLX server".
Stop the engine instead; a clean `TERM` takes the model with it.

### MLX — `mlx_lm.server`

```bash
pip3 install --user mlx-lm
```

Currently `mlx-lm` **0.31.3** on `mlx` **0.31.2**.

A `--user` install puts the binary outside the default `PATH`. On this machine
`mlx_lm.server` is *not* on `PATH`; `engines.find_mlx_server()` falls back to
`~/Library/Python/{3.13,3.12,3.11,3.10}/bin/mlx_lm.server` and resolves it at
`~/Library/Python/3.10/bin/mlx_lm.server`. Nothing to fix, but `which
mlx_lm.server` returning nothing does not mean it is missing.

## Supporting tools

| tool | version | needed for |
|---|---|---|
| Homebrew | 6.0.21 | installing llama.cpp and mtplx |
| git | 2.51.2 | the repo |
| Node.js | v25.1.0 | dashboard JS tests only (`tests/unit/test_dashboard_js.py`) |
| sqlite3 | 3.51.0 | reading `usage.db` by hand; the engine uses Python's built-in module |

Node is genuinely optional — those tests `skipif(shutil.which("node") is None)`.
They exist because a top-level error in `app.js` aborts the whole script while
the page still renders from static HTML, so every panel silently stays empty.
That happened once and was caught by eye, which is not a test strategy.

Taps in use: `youssofal/mtplx`.

No `huggingface-cli` is needed. Weights are pulled by `inference/weights.py`,
a standard library HuggingFace client, and MTPLX packs by `mtplx models
--update` — see [docs/engine.md](docs/engine.md#egress-sandbox). Set `HF_TOKEN`
for gated repos.

## Development and testing

```bash
pip3 install -r requirements-dev.txt
```

| package | version | declared |
|---|---|---|
| pytest | 9.1.1 | `>=8.0` |
| pytest-bdd | 8.1.0 | `>=7.0` |

```bash
python3 -m pytest tests -q      # ~350 tests, about a minute, no model needed
```

## Storage

| path | what |
|---|---|
| `<weights_root>/.cache/huggingface/hub` | GGUF and MLX weights (`<weights_root>/huggingface/hub` when `weights_root` is not `~`) |
| `<weights_root>/.mtplx/models` | MTPLX model packs (`<weights_root>/mtplx`) |
| `~/.mtplx/session-bank` | MTPLX SSD prefix/KV cache — budget up to 100 GB |
| `usage.db` | request history (gitignored, WAL mode) |

`weights_root` defaults to your home directory; set it in `config.json` to keep
the weights on another disk. If it is on a volume that is not mounted,
`weights_root_missing()` says so rather than letting `llama-server` quietly
re-download tens of gigabytes to a path that is not there.

The MTPLX session bank stays on the boot disk on purpose — it is read on
prefill, so it wants the fast disk — and it grows with every distinct
conversation prefix up to `--ssd-session-cache-max-size`. MTPLX evicts the
oldest at the cap.

`traces/`, `logs/` and `usage.db*` are gitignored. Traces contain whatever
source code was in the session.

## Models

The registry finds these automatically; each `inference/models/*.py` carries its
own weights URL and install notes. `routable` reflects what fits *this* machine.

| alias | engine | ctx | fits M4 Max |
|---|---|---|---|
| `qwen3.8` | mtplx | 262144 | yes — the default |
| `kat-coder2.5` | llama.cpp | 262144 | yes |
| `qwen3.6-moe` | llama.cpp | 262144 | yes |
| `muse-glimmer` | llama.cpp | 131072 | yes |
| `qwen3.5`, `qwen3.5-small`, `qwen3.6`, `qwen3-coder-next` | llama.cpp | — | no fit |
| `gemma-4-31b`, `gemma-4-26b-a4b`, `ling-2.6-flash` | mlx-lm | — | no fit |

`qwen3.8` runs the `Optimized-Speed` pack with a pre-quantized MTP sidecar
(`mtp_sidecar: int4-g64-prequantized`, 239 MB). Before 2026-08-20 that sidecar
shipped as bf16 (849 MB) and MTPLX quantized it at load.

## Clients

| client | version | notes |
|---|---|---|
| VS Code | 1.134.0 | `github.copilot.chat.customModels` — see [docs/clients.md](docs/clients.md) |
| Claude Code | 2.1.259 | |

VS Code needs `apiType: "chat-completions"`, `apiKey: "none"` (MTPLX checks it
and rejects an empty string), and `maxInputTokens` a little under the server's
context.

## Verifying

```bash
sw_vers && uname -m && sysctl -n machdep.cpu.brand_string
llama-server --version
mtplx --version
python3 -c 'import mlx_lm, mlx.core; print(mlx_lm.__version__, mlx.core.__version__)'
python3 -V && python3 -m pytest --version
python3 -c "import sys;sys.path.insert(0,'.');from inference import engines;\
print(engines.find_llama_server(), engines.find_mlx_server())"
python3 -m inference models     # what fits this machine
```

What is outdated, and what the latest is:

```bash
brew update && brew outdated --verbose
```

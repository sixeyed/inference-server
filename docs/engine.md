# The engine

What runs between your client and the model: where it can run, how it keeps the
model servers off the network, what it records and how it's tested. For the
request routing itself see [routing.md](routing.md); for every setting see
[configuration.md](configuration.md).

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
and by default model servers listen on loopback only. Opening `model_bind` puts
the model servers on your network, so set a real `api_key` too — the engine
checks it on every request, and MTPLX is started with it. Behind a reverse
proxy you run, set `trust_proxy: true` (or `INFERENCE_TRUST_PROXY=1`) to take
each client's address from `X-Forwarded-For`. It is off in the image: with no
proxy in front, any client could claim any address.

Test with `qwen3-tiny` — Qwen3 0.6B, 639MB, routable by name but never listed in
`/v1/models` — rather than loading 20GB next to whatever else the Mac is doing.

Keep the Mac's checkout on the commit the image was built from. `run --plan`
reports the checkout's sha, and a mismatch is recorded as a `drift` event,
because the model files in the two can differ.

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
python3 -m inference pull <model>   # fetch, or update to the latest revision
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

## Dashboard

`http://<host>:8084/` — two tabs. **Dashboard** shows what is loaded, what is
available, memory, per-model statistics and a quick-chat smoke test.
**Config** lists the client parameters that can be stripped from incoming
requests (see [Stripping client parameters](configuration.md#stripping-client-parameters)) and
sets the per-client system-prompt substitution (see
[Rewriting the system prompt](configuration.md#rewriting-the-system-prompt)). It can drive the engine: clicking **Load** on a model starts it, and
says `Load (cuts current turn)` while a request is in flight, because it does.

The footer shows which build is running: the name, version and repository
link from the checkout's `pyproject.toml`, and the commit - `INFERENCE_GIT_SHA`
in the image, `git rev-parse` in a checkout. The same JSON is at `/api/about`.

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

## Recording and usage stats

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
unit tests. Neither needs a model, a network or a real port — they run in about a
minute against `fakes.py`, a fake model server that impersonates each engine
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

## Code layout

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
  about.py                 # name, version, repository and commit, for the footer
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

tests/                     # what pytest runs: no model, no network, about a minute
  unit/                    # arithmetic, parsing, cache lookups
  features/                # behaviour specs (pytest-bdd) + steps/
  fakes.py                 # the fake model server both suites run against

docs/                      # backends, engine, routing, configuration, clients,
                           # adding-a-model, lessons-learned
PREREQS.md                 # what to install, and the versions verified
logs/                      # model server output, one file per launch (gitignored)
```

`inference/` is a flat package on purpose rather than `src/inference/`: it means
`python -m inference engine` works from a bare checkout with nothing installed,
which is how you want to try this on a second Mac.

`tests/` is what CI runs — checks against a fake model server, with no weights
on disk and no network.

# Routing

How the engine turns one endpoint into many models.

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

**Everything that can fail is settled before the current model is stopped.**
Missing weights are pulled first, while the current model keeps serving (see
[pulling weights](engine.md#egress-sandbox)). A model that doesn't fit, a
missing engine binary, a pull that fails, a model host that can't be reached —
each is refused with the old model still loaded. Finding out after the swap
costs a working session to learn something knowable beforehand.

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

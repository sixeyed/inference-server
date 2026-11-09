# Backends: models and the servers that run them

The engine doesn't run models itself. It starts one model server at a time —
llama.cpp, MTPLX or mlx-lm, whichever the model needs — and relays to it. This
page covers which models are registered, how each is deployed per machine, and
what each server does differently.

Each model is one self-contained file in `inference/models/`: HF repo, quant per
machine, context, sampling, thinking switches and speculative decoding, comments
and all. Adding one is a new file there — see
[adding-a-model.md](adding-a-model.md).

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

The chip is detected from its brand string (`sysctl machdep.cpu.brand_string`,
e.g. `Apple M4 Max`) and mapped to a profile key. Model files index their
quant and context on that key, and the profile supplies the machine tuning
below. The defaults live in `inference/config.py`; `machines` in `config.json`
overrides them per key — see [Configuration](configuration.md#configuration).

| | M4 Max | M1 Max | M2 |
|---|---|---|---|
| Profile key | `m4` | `m1` | `m2` |
| RAM | 128GB | 64GB | 24GB |
| Threads | 12 | 10 | 6 |
| Batch / ubatch | 4096 / 2048 | 2048 / 512 | 1024 / 512 |
| Prompt cache | 32GB | 16GB | 4GB |

Those three were measured on real machines. Other chips follow the same pattern
on assumption:

| Chip | Profile key | |
|---|---|---|
| M5 | `m4` | **assumed** — tuned like an M4 |
| M3 | `m2` | **assumed** — tuned like an M2 |
| anything else | `unknown` | small quants and short contexts, so that something fits |

> **M3 and M5 are assumptions, not measurements.** Neither has been run on real
> hardware. The mapping assumes they report `Apple M3 …` and `Apple M5 …` the way
> the others do, and that the nearest older generation with a profile is a safe
> starting point. Like the existing keys, a generation's profile assumes one RAM
> tier — the M4 profile is a 128GB M4 Max — so an M5 with less memory may be
> offered quants that don't fit. The mapping is `CHIP_GENERATIONS` in
> `inference/machine.py`. The dashboard and `inference models` still show the
> real chip name. If you run one, tune `machines` in `config.json`, and please
> open an issue with what worked.

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

## References

- [llama.cpp](https://github.com/ggerganov/llama.cpp) — the inference engine
- [mlx-lm](https://github.com/ml-explore/mlx-lm) — MLX inference for Apple Silicon
- [Unsloth GGUF quants](https://huggingface.co/unsloth) — the quantizations used here
- [Unsloth MTP guide](https://unsloth.ai/docs/models/mtp) — speculative decoding flags

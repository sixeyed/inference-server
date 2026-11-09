"""Muse-Glimmer-30B - dense 30B + perception encoder, built for local agents.

Meta Superintelligence Labs' Apache-2.0 agentic model, distilled from Muse Spark
and explicitly targeted at running on consumer hardware rather than in a cloud.
Where most local models are chat models you push into tool use, this one was
trained for end-to-end task completion: long-horizon planning, schema-accurate
function calls, and recovery when a tool call fails.

Engine:  llama-server (llama.cpp)
Weights: https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF
Docs:    https://unsloth.ai/docs/models/muse-glimmer
"""

from ..machine import Machine, Options, Unsupported
from ..specs import LlamaServerSpec

ALIAS = "muse-glimmer"
FAMILY = "muse"
SUMMARY = "Dense 30B agentic multimodal from Meta - tool use + failure recovery"

HF_REPO = "unsloth/Muse-Glimmer-30B-GGUF"

# ------------------------------------------------------------
# Quant per machine.
#
# BF16 is 58GB and would fit on the M4 Max, but a 30B dense model is memory
# bandwidth bound on decode: BF16 costs roughly 2x the tok/s of Q8 for a quality
# difference you can't measure on agentic work. UD-Q8_K_XL at 32.3GB is the
# right ceiling here, leaving plenty of room for a 256K KV cache and the 3.85GB
# vision projector.
# ------------------------------------------------------------
# Context is capped at the training length. llama.cpp caps it anyway - "the
# slot context (262144) exceeds the training context of the model (131072) -
# capping" - so asking for more only made the banner and the dashboard report a
# number the model never had.
MACHINES = {
    "m4": dict(
        hf_file="Muse-Glimmer-30B-UD-Q8_K_XL.gguf",  # 32.3GB
        ctx=131072,
        predict=32768,
        parallel=1,
    ),
    "m1": dict(
        hf_file="Muse-Glimmer-30B-UD-Q6_K_XL.gguf",  # 26.3GB
        ctx=65536,
        predict=16384,
        parallel=1,
    ),
    "m2": dict(
        hf_file="Muse-Glimmer-30B-UD-Q3_K_XL.gguf",  # 13.4GB
        ctx=16384,
        predict=8192,
        parallel=1,
    ),
    "unknown": dict(
        hf_file="Muse-Glimmer-30B-UD-Q4_K_XL.gguf",  # 15.9GB
        ctx=16384,
        predict=8192,
        parallel=1,
    ),
}

# Meta's published defaults. Note top_k=64, much wider than the qwen family's 20.
SAMPLING = dict(
    temp=1.0, top_p=0.95, top_k=64, min_p=0.0, presence_penalty=0.0, repeat_penalty=1.0
)

# ------------------------------------------------------------
# Reasoning.
#
# Muse Glimmer does NOT take a chat-template kwarg. Reasoning strength is set by
# putting a line in the *system prompt*:
#
#     Reasoning strength: low | medium | high | xhigh
#
# Meta recommends high/xhigh for complex tasks. There is no server-side flag to
# force it, so the `--thinking` switch here only changes what the banner tells
# you to paste into your client's system prompt. Set it in Cline / VS Code /
# opencode yourself.
# ------------------------------------------------------------
THINKING_DEFAULT = True
REASONING_STRENGTH_ON = "high"
REASONING_STRENGTH_OFF = "low"

# ------------------------------------------------------------
# DFlash speculative decoding (ON by default).
#
# DFlash is a block-diffusion drafter that proposes a whole block of tokens in
# one forward pass, which the 30B then verifies - so output is identical to
# running without it, just faster. Unlike the MTP setups for qwen3.8 and
# kat-coder, the drafter ships in the *same* official repo (dflash-kquant.gguf,
# 1.63GB), so there's no third-party weight risk and no reason to leave it off.
#
# It was trained with a block size of 16 (1 anchor + 15 proposed), so
# --spec-draft-n-max above 15 is clamped. llama.cpp resolves the drafter out of
# the -hf repo when --spec-type draft-dflash is set, so there's no separate -md
# path to pass. Needs a recent llama.cpp (Muse Glimmer support, PR #26841);
# disable with `--no-spec` if your build predates it.
# ------------------------------------------------------------
SPEC_DEFAULT = True
DFLASH_N_MAX = 15

# Vision: mmproj-Muse-Glimmer-30B-BF16.gguf (3.85GB) - the perception encoder is
# core to the model, so this is on.
VISION = True


def spec(machine: Machine, opts: Options) -> LlamaServerSpec:
    cfg = MACHINES.get(machine.key)
    if cfg is None:
        raise Unsupported(f"muse-glimmer has no config for {machine.chip}")

    thinking = THINKING_DEFAULT if opts.thinking is None else opts.thinking
    use_dflash = SPEC_DEFAULT if opts.spec_decode is None else opts.spec_decode

    strength = REASONING_STRENGTH_ON if thinking else REASONING_STRENGTH_OFF

    spec_args: list[str] = []
    notes = [("Reasoning", f'put "Reasoning strength: {strength}" in your system prompt')]

    if use_dflash:
        spec_args = [
            "--spec-type", "draft-dflash",
            "--spec-draft-n-max", str(DFLASH_N_MAX),
        ]
        notes.append(("Speculative", f"DFlash, n-max={DFLASH_N_MAX}"))
    else:
        notes.append(("Speculative", "off"))

    return LlamaServerSpec(
        alias=ALIAS,
        hf_repo=HF_REPO,
        hf_file=cfg["hf_file"],
        ctx=opts.ctx or cfg["ctx"],
        predict=cfg["predict"],
        parallel=cfg["parallel"],
        vision=VISION,
        spec_args=spec_args,
        notes=notes,
        **SAMPLING,
    )

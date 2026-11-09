"""Qwen3.5-122B-A10B - the big MoE. Large context, broad knowledge.

Only the M4 Max can hold the 122B. Smaller machines fall back to a different
checkpoint entirely (35B-A3B on the M1 Max, 9B dense on the M2) - which is why
the repo differs per machine here and not just the quant.

Engine:  llama-server (llama.cpp)
Weights: https://huggingface.co/unsloth/Qwen3.5-122B-A10B-GGUF
"""

from ..machine import Machine, Options, Unsupported
from ..specs import LlamaServerSpec

ALIAS = "qwen3.5"
FAMILY = "qwen"
SUMMARY = "MoE 122B/10B active - large context, broad knowledge"

# Retired. `inference run` still launches this - the file is the record of how
# earlier measurements were configured, and reproducing one needs this exact quant, context
# and sampling. The engine leaves it out of /v1/models and won't route to it
# without --serve-deprecated.
DEPRECATED = "superseded by qwen3.6-moe"

HF_REPO = "unsloth/Qwen3.5-122B-A10B-GGUF"

MACHINES = {
    "m4": dict(
        hf_repo=HF_REPO,
        hf_file="UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL-00001-of-00003.gguf",  # 77GB
        ctx=262144,
        predict=32768,
        parallel=1,
    ),
    "m1": dict(
        # 122B won't fit in 64GB - drop to the 35B-A3B checkpoint
        hf_repo="unsloth/Qwen3.5-35B-A3B-GGUF",
        hf_file="Qwen3.5-35B-A3B-UD-Q6_K_XL.gguf",  # 32GB
        ctx=65536,
        predict=16384,
        parallel=2,
    ),
    "m2": dict(
        # 9B at a high quant beats 35B at a low one in 24GB
        hf_repo="unsloth/Qwen3.5-9B-GGUF",
        hf_file="Qwen3.5-9B-UD-Q8_K_XL.gguf",  # 13GB
        ctx=16384,
        predict=8192,
        parallel=1,
    ),
    "unknown": dict(
        hf_repo="unsloth/Qwen3.5-9B-GGUF",
        hf_file="Qwen3.5-9B-UD-Q8_K_XL.gguf",
        ctx=16384,
        predict=8192,
        parallel=1,
    ),
}

SAMPLING = dict(
    temp=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repeat_penalty=1.0
)

# See qwen3_6.py for why thinking is off by default.
REASONING_BUDGET = 0
THINKING_BUDGET_ON = 1000

VISION = False


def spec(machine: Machine, opts: Options) -> LlamaServerSpec:
    cfg = MACHINES.get(machine.key)
    if cfg is None:
        raise Unsupported(f"qwen3.5 has no config for {machine.chip}")

    budget = REASONING_BUDGET if not opts.thinking else THINKING_BUDGET_ON

    return LlamaServerSpec(
        alias=ALIAS,
        hf_repo=cfg["hf_repo"],
        hf_file=cfg["hf_file"],
        ctx=opts.ctx or cfg["ctx"],
        predict=cfg["predict"],
        parallel=cfg["parallel"],
        vision=VISION,
        extra_args=["--reasoning-budget", str(budget)],
        notes=[
            ("Reasoning", "disabled" if budget == 0 else f"capped at {budget} tokens"),
        ],
        **SAMPLING,
    )

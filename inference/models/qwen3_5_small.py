"""Qwen3.5-35B-A3B - lightweight, fast inference.

BF16 on the M4 Max: at 3B active params the extra bandwidth cost of full
precision is small, and it's the one place in this repo where running unquantised
is affordable.

Engine:  llama-server (llama.cpp)
Weights: https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF
"""

from ..machine import Machine, Options, Unsupported
from ..specs import LlamaServerSpec

ALIAS = "qwen3.5-small"
FAMILY = "qwen"
SUMMARY = "MoE 35B/3B active - lightweight, fast inference"

# Retired. run.py still launches this - the file is the record of how the runs
# in docs/ were configured, and reproducing one needs this exact quant, context
# and sampling. The engine leaves it out of /v1/models and won't route to it
# without --serve-deprecated.
DEPRECATED = "superseded by qwen3.6-moe"

HF_REPO = "unsloth/Qwen3.5-35B-A3B-GGUF"

MACHINES = {
    "m4": dict(
        hf_repo=HF_REPO,
        hf_file="BF16/Qwen3.5-35B-A3B-BF16-00001-of-00002.gguf",  # 69GB
        ctx=131072,
        predict=32768,
        parallel=2,
    ),
    "m1": dict(
        hf_repo="unsloth/Qwen3.5-9B-GGUF",
        hf_file="Qwen3.5-9B-UD-Q8_K_XL.gguf",  # 13GB
        ctx=65536,
        predict=16384,
        parallel=2,
    ),
    "m2": None,  # 9B is already the smallest useful size - use qwen3.5
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
        raise Unsupported(
            f"qwen3.5-small has no config for {machine.chip}. "
            "This model is deprecated - see DEPRECATED."
        )

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

"""Qwen3.6-35B-A3B - MoE, 3B active. Fast coding.

Beats the 122B-A10B on SWE-bench Verified at roughly 3x the decode speed, which
made it the default before kat-coder (same base model, RL'd for agents)
showed up.

Engine:  llama-server (llama.cpp)
Weights: https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF
"""

from ..machine import Machine, Options, Unsupported
from ..specs import LlamaServerSpec

ALIAS = "qwen3.6-moe"
FAMILY = "qwen"
SUMMARY = "MoE 35B/3B active - fast coding, beats 122B-A10B on SWE-bench"

HF_REPO = "unsloth/Qwen3.6-35B-A3B-GGUF"

MACHINES = {
    "m4": dict(
        hf_file="Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf",  # 38.5GB - near-lossless
        ctx=262144,
        predict=32768,
        parallel=1,
    ),
    "m1": dict(
        hf_file="Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf",  # 32GB - Q8 too tight in 64GB
        ctx=65536,
        predict=16384,
        parallel=2,
    ),
    "m2": None,  # 35B-A3B won't fit usefully in 24GB
    "unknown": dict(hf_file=None, ctx=16384, predict=8192, parallel=1),
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
            f"qwen3.6-moe is too large for {machine.chip} ({machine.ram_gb}GB)."
        )

    budget = REASONING_BUDGET if not opts.thinking else THINKING_BUDGET_ON

    return LlamaServerSpec(
        alias=ALIAS,
        hf_repo=HF_REPO,
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

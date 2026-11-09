"""Qwen3-Coder-Next - 80B-A3B MoE, code-specialised.

Not a thinking model, so there's no reasoning budget to set - and its sampling
profile is deliberately looser than the rest of the qwen family (temp 1.0,
top_k 40, min_p 0.01) per Qwen's published coder settings.

Engine:  llama-server (llama.cpp)
Weights: https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF
"""

from ..machine import Machine, Options, Unsupported
from ..specs import LlamaServerSpec

ALIAS = "qwen3-coder-next"
FAMILY = "qwen"
SUMMARY = "MoE 80B/3B active - code-specialised, no thinking mode"

# Retired. `inference run` still launches this - the file is the record of how
# earlier measurements were configured, and reproducing one needs this exact quant, context
# and sampling. The engine leaves it out of /v1/models and won't route to it
# without --serve-deprecated.
DEPRECATED = "superseded by qwen3.6-moe and qwen3.8"

HF_REPO = "unsloth/Qwen3-Coder-Next-GGUF"

MACHINES = {
    "m4": dict(
        hf_file="UD-Q8_K_XL/Qwen3-Coder-Next-UD-Q8_K_XL-00001-of-00003.gguf",  # 86GB
        ctx=262144,
        predict=65536,
        parallel=1,
    ),
    "m1": dict(
        hf_file="Qwen3-Coder-Next-Q4_0.gguf",  # 45GB
        ctx=65536,
        predict=16384,
        parallel=1,
    ),
    "m2": None,  # won't fit in 24GB
    "unknown": dict(hf_file=None, ctx=16384, predict=8192, parallel=1),
}

SAMPLING = dict(
    temp=1.0, top_p=0.95, top_k=40, min_p=0.01, presence_penalty=0.0, repeat_penalty=1.0
)

VISION = False


def spec(machine: Machine, opts: Options) -> LlamaServerSpec:
    cfg = MACHINES.get(machine.key)
    if cfg is None:
        raise Unsupported(
            f"qwen3-coder-next is too large for {machine.chip} ({machine.ram_gb}GB). "
            "This model is deprecated - see DEPRECATED."
        )

    return LlamaServerSpec(
        alias=ALIAS,
        hf_repo=HF_REPO,
        hf_file=cfg["hf_file"],
        ctx=opts.ctx or cfg["ctx"],
        predict=cfg["predict"],
        parallel=cfg["parallel"],
        vision=VISION,
        notes=[("Reasoning", "n/a (not a thinking model)")],
        **SAMPLING,
    )

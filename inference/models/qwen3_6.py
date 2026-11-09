"""Qwen3.6-27B - dense 27B, general coding and agentic work.

The previous generation's dense flagship. Superseded by qwen3.8 on this box, but
kept because it's the model the Cline analysis in docs/ was run against.

Engine:  llama-server (llama.cpp)
Weights: https://huggingface.co/unsloth/Qwen3.6-27B-GGUF
"""

from ..machine import Machine, Options, Unsupported
from ..specs import LlamaServerSpec

ALIAS = "qwen3.6"
FAMILY = "qwen"
SUMMARY = "Dense 27B - general coding, agentic tasks"

# Retired. run.py still launches this - the file is the record of how the runs
# in docs/ were configured, and reproducing one needs this exact quant, context
# and sampling. The engine leaves it out of /v1/models and won't route to it
# without --serve-deprecated.
DEPRECATED = "superseded by qwen3.6-moe; kept for the Cline analysis in docs/"

HF_REPO = "unsloth/Qwen3.6-27B-GGUF"

MACHINES = {
    "m4": dict(
        hf_file="Qwen3.6-27B-UD-Q8_K_XL.gguf",  # 35GB - best available quant
        ctx=131072,
        predict=32768,
        parallel=1,
    ),
    "m1": dict(
        hf_file="Qwen3.6-27B-Q8_0.gguf",  # 29GB - Q8 fits well in 64GB
        ctx=65536,
        predict=16384,
        parallel=1,
    ),
    "m2": None,  # 27B dense won't fit usefully in 24GB
    "unknown": dict(hf_file=None, ctx=16384, predict=8192, parallel=1),
}

SAMPLING = dict(
    temp=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repeat_penalty=1.0
)

# Thinking off. Under Copilot Agent and Cline, thinking-mode qwen3 defers
# ("let me investigate") rather than committing, exhausting the client's
# max-tool-calls-per-turn before producing an answer. reasoning_budget 0 makes
# thinking activate and immediately terminate, so the model goes straight to
# content. Set REASONING_BUDGET to 1000 for a capped budget, or -1 for
# unrestricted.
REASONING_BUDGET = 0
THINKING_BUDGET_ON = 1000

VISION = False


def spec(machine: Machine, opts: Options) -> LlamaServerSpec:
    cfg = MACHINES.get(machine.key)
    if cfg is None:
        raise Unsupported(
            f"qwen3.6 is too large for {machine.chip} ({machine.ram_gb}GB). "
            "Try: inference run qwen3.6-moe"
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

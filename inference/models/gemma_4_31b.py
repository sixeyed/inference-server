"""Gemma 4 31B - dense, MLX. The one that actually commits to tool use in Cline.

Raw throughput is no better than qwen3.6, but in Cline it follows a multi-step
plan through to the end instead of stalling after the first tool call. See
docs/gemma-4-31b-cline-perf-analysis.md.

Engine:  mlx_lm.server (mlx-lm)
Weights: https://huggingface.co/FakeRockert543/gemma-4-31b-it-MLX-8bit

PLE safety: the standard mlx-community / unsloth Gemma 4 MLX quants list every
layer in config.json's `quantization` block, including the Per-Layer Embedding
(ScaledLinear) layers. Quantising those corrupts them and the model emits
garbage. FakeRockert543's quants skip them; bf16 has nothing to corrupt. The
allowlist below is enforced before launch.
"""

import re

from ..machine import Machine, Options, Unsupported
from ..specs import MlxServerSpec

ALIAS = "gemma-4-31b"
FAMILY = "gemma"
SUMMARY = "Dense 31B - best Cline behaviour, commits to multi-step tool use"

# Retired. run.py still launches this - the file is the record of how the runs
# in docs/ were configured, and reproducing one needs this exact quant, context
# and sampling. The engine leaves it out of /v1/models and won't route to it
# without --serve-deprecated.
DEPRECATED = "superseded by qwen3.8; kept for the Cline analysis in docs/"

HF_REPO = "FakeRockert543/gemma-4-31b-it-MLX-8bit"  # ~33GB
CLIENT_MODEL_ID = HF_REPO  # mlx_lm.server has no --alias

MAX_TOKENS = 262144
TEMP = 1.0
TOP_P = 0.95
PREFILL_STEP_SIZE = 8192
PROMPT_CACHE_BYTES = 16 * 1024**3

PLE_SAFE_PATTERNS = [
    r"^FakeRockert543/gemma-4-.+-MLX-(4bit|8bit|bf16)$",
    r"^mlx-community/gemma-4-.+-bf16$",
]

SUPPORTED_MACHINES = {"m4"}


def _ple_safe(repo: str) -> bool:
    return any(re.match(p, repo) for p in PLE_SAFE_PATTERNS)


def spec(machine: Machine, opts: Options) -> MlxServerSpec:
    if machine.key not in SUPPORTED_MACHINES:
        raise Unsupported(
            f"gemma-4-31b needs the M4 Max (128GB). "
            f"Detected: {machine.chip} ({machine.ram_gb}GB)"
        )
    if not _ple_safe(HF_REPO):
        raise Unsupported(
            f"{HF_REPO} is not on the PLE-safe allowlist - standard MLX Gemma 4 "
            "quants corrupt Per-Layer Embeddings"
        )

    return MlxServerSpec(
        hf_repo=HF_REPO,
        max_tokens=opts.ctx or MAX_TOKENS,
        temp=TEMP,
        top_p=TOP_P,
        prefill_step_size=PREFILL_STEP_SIZE,
        prompt_cache_bytes=PROMPT_CACHE_BYTES,
        notes=[
            ("Client model id", CLIENT_MODEL_ID),
            ("PLE-safe", "yes (PLE-aware quantization)"),
        ],
    )

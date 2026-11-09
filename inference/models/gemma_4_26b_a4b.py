"""Gemma 4 26B-A4B - MoE, ~4B active per token, MLX.

The cheaper sibling of gemma-4-31b. Same PLE-safety constraint applies.

Engine:  mlx_lm.server (mlx-lm)
Weights: https://huggingface.co/FakeRockert543/gemma-4-26b-a4b-it-MLX-8bit
"""

import re

from ..machine import Machine, Options, Unsupported
from ..specs import MlxServerSpec

ALIAS = "gemma-4-26b-a4b"
FAMILY = "gemma"
SUMMARY = "MoE 26B/4B active - lighter Gemma 4 for the same Cline behaviour"

# Retired. `inference run` still launches this - the file is the record of how
# earlier measurements were configured, and reproducing one needs this exact quant, context
# and sampling. The engine leaves it out of /v1/models and won't route to it
# without --serve-deprecated.
DEPRECATED = "superseded by qwen3.8; the lighter half of the Gemma 4 comparison"

HF_REPO = "FakeRockert543/gemma-4-26b-a4b-it-MLX-8bit"  # ~29GB
CLIENT_MODEL_ID = HF_REPO  # mlx_lm.server has no --alias

MAX_TOKENS = 262144
TEMP = 1.0
TOP_P = 0.95
PREFILL_STEP_SIZE = 4096
PROMPT_CACHE_BYTES = 16 * 1024**3

# See gemma_4_31b.py for why this allowlist exists.
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
            f"gemma-4-26b-a4b needs the M4 Max (128GB). "
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

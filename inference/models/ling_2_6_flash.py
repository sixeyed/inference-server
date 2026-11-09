"""Ling-2.6-flash - 104B-A7.4B MoE, MLX only. M4 Max only.

Published only as MLX weights, so this one runs on mlx_lm.server rather than
llama-server. 72GB at 5-bit means the M4 Max is the only machine here that can
hold it.

Engine:  mlx_lm.server (mlx-lm)
Weights: https://huggingface.co/mlx-community/Ling-2.6-flash-mlx-5bit

Install note: the released mlx-lm doesn't yet include the `bailing_hybrid` arch
this model needs. Until PR #1227 lands:

    pip3 install --user --force-reinstall --no-deps \\
      "git+https://github.com/ivanfioravanti/mlx-lm.git@add-ling-2.6-flash"
"""

from ..machine import Machine, Options, Unsupported
from ..specs import MlxServerSpec

ALIAS = "ling-2.6-flash"
FAMILY = "ling"
SUMMARY = "MoE 104B/7.4B active - fast agentic tasks (M4 Max only)"

# Retired. `inference run` still launches this - the file is the record of how
# earlier measurements were configured, and reproducing one needs this exact quant, context
# and sampling. The engine leaves it out of /v1/models and won't route to it
# without --serve-deprecated.
DEPRECATED = "needs an unreleased mlx-lm fork for the bailing_hybrid arch"

HF_REPO = "mlx-community/Ling-2.6-flash-mlx-5bit"  # ~72GB

# mlx_lm.server has no --alias: clients must send this full repo id as `model`.
CLIENT_MODEL_ID = HF_REPO

MAX_TOKENS = 262144
TEMP = 0.7
TOP_P = 0.8
PREFILL_STEP_SIZE = 4096
PROMPT_CACHE_BYTES = 16 * 1024**3  # KV cache reuse across requests

SUPPORTED_MACHINES = {"m4"}


def spec(machine: Machine, opts: Options) -> MlxServerSpec:
    if machine.key not in SUPPORTED_MACHINES:
        raise Unsupported(
            f"ling-2.6-flash needs the M4 Max (128GB) - 5-bit weights are ~72GB. "
            f"Detected: {machine.chip} ({machine.ram_gb}GB)"
        )

    return MlxServerSpec(
        hf_repo=HF_REPO,
        max_tokens=opts.ctx or MAX_TOKENS,
        temp=TEMP,
        top_p=TOP_P,
        prefill_step_size=PREFILL_STEP_SIZE,
        prompt_cache_bytes=PROMPT_CACHE_BYTES,
        notes=[("Client model id", CLIENT_MODEL_ID)],
    )

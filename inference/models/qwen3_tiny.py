"""Qwen3-0.6B - a test fixture, not a model anyone should work with.

Exists so the engine can be exercised end to end - cold start, relay, swap,
stop, and in a container over SSH - without loading tens of GB beside whatever
else the Mac is doing. 639MB of weights, about 1GB resident at 8k context,
ready in seconds.

Engine:  llama-server (llama.cpp)
Weights: https://huggingface.co/Qwen/Qwen3-0.6B-GGUF
"""

from ..machine import Machine, Options
from ..specs import LlamaServerSpec

ALIAS = "qwen3-tiny"
FAMILY = "qwen"
SUMMARY = "Qwen3 0.6B - test fixture for end-to-end checks, not for real work"

# Routable by name, but left out of /v1/models so no client picker offers it.
TEST_ONLY = True

HF_REPO = "Qwen/Qwen3-0.6B-GGUF"
HF_FILE = "Qwen3-0.6B-Q8_0.gguf"  # 639MB


def spec(machine: Machine, opts: Options) -> LlamaServerSpec:
    # The same on every machine - the point of it is that it fits anywhere.
    return LlamaServerSpec(
        alias=ALIAS,
        hf_repo=HF_REPO,
        hf_file=HF_FILE,
        ctx=opts.ctx or 8192,
        predict=1024,
        parallel=1,
        temp=0.7, top_p=0.8, top_k=20,
        # Thinking off, like the other Qwen files: short replies, fast checks.
        extra_args=["--reasoning-budget", "0"],
        notes=[("Purpose", "test fixture")],
    )

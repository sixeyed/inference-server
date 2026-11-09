"""KAT-Coder-V2.5-Dev - 35B-A3B MoE, trained inside real repositories.

Kwaipilot's agentic coding model, Apache-2.0. Built on Qwen3.6-35B-A3B, then SFT
on 127K examples and RL'd inside 100,000+ executable repository environments -
so it was rewarded for actually running commands, reading files and iterating to
a green test, not for producing plausible-looking diffs. 69.4% on SWE-bench
Verified, from a model with 3B active params.

The RL stage specifically targeted the failure modes that make local models
painful in Cline: excessive parallel tool calls and content repetition.

Engine:  llama-server (llama.cpp)
Weights: https://huggingface.co/bartowski/Kwaipilot_KAT-Coder-V2.5-Dev-GGUF
Card:    https://huggingface.co/Kwaipilot/KAT-Coder-V2.5-Dev
"""

from ..machine import Machine, Options, Unsupported
from ..specs import LlamaServerSpec

ALIAS = "kat-coder2.5"
FAMILY = "kat"

# The alias used to be a bare "kat-coder". A V3 will land and the two need to be
# tellable apart, so the version is in the name now. Traces and client
# configs written before the rename still say the old one, and the
# registry keeps routing it here - see models/__init__.py.
FORMER_ALIASES = ("kat-coder",)
SUMMARY = "35B-A3B agentic coding MoE - 69.4% SWE-bench Verified, 3B active"

HF_REPO = "bartowski/Kwaipilot_KAT-Coder-V2.5-Dev-GGUF"

# ------------------------------------------------------------
# Quant per machine.
#
# 3B active params means decode is cheap and the constraint is purely "does the
# whole 35B fit in RAM alongside the KV cache". On the M4 Max that means Q8_0
# outright (36.9GB). On the M1 Max, Q6_K at 30GB leaves enough for 64K context.
# The M2's 24GB can't hold this at any quant worth using - the 35B would have to
# drop to IQ3 and still crowd out the KV cache, at which point qwen3.5's 9B
# fallback is the better machine for that job.
# ------------------------------------------------------------
MACHINES = {
    "m4": dict(
        hf_file="Kwaipilot_KAT-Coder-V2.5-Dev-Q8_0.gguf",  # 36.9GB
        ctx=262144,
        predict=32768,
        parallel=1,
    ),
    "m1": dict(
        hf_file="Kwaipilot_KAT-Coder-V2.5-Dev-Q6_K.gguf",  # 30.1GB
        ctx=65536,
        predict=16384,
        parallel=2,
    ),
    "m2": None,  # 35B won't fit usefully in 24GB
    "unknown": dict(
        hf_file="Kwaipilot_KAT-Coder-V2.5-Dev-Q4_K_M.gguf",  # 21.4GB
        ctx=16384,
        predict=8192,
        parallel=1,
    ),
}

# Qwen3.6 lineage, so the same two sampling profiles apply.
SAMPLING_THINKING = dict(
    temp=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repeat_penalty=1.0
)
SAMPLING_INSTRUCT = dict(
    temp=0.7, top_p=0.80, top_k=20, min_p=0.0, presence_penalty=1.5, repeat_penalty=1.0
)

# ------------------------------------------------------------
# Thinking - ON by default, which is the opposite of every other model here.
#
# The rest of the qwen family runs with thinking off, because thinking-mode qwen
# defers instead of committing and exhausts Cline's per-turn tool budget. That
# behaviour is exactly what KAT-Coder's RL stage was built to remove, and its
# 69.4% SWE-bench number is a thinking-mode number - turning thinking off here
# throws away the thing that makes this model worth running.
#
# `preserve_thinking` keeps earlier reasoning traces across turns, which the
# model card recommends for multi-turn agentic loops.
#
# If you do see the old deferral loop under Cline, `--no-thinking` puts it back
# on the usual default (and switches to the instruct sampling profile).
# ------------------------------------------------------------
THINKING_DEFAULT = True
PRESERVE_THINKING = True

# ------------------------------------------------------------
# Multi-token prediction (opt-in, `--spec`).
#
# Same deal as qwen3.8: needs an MTP-converted GGUF, so `--spec` swaps repo.
# On a 3B-active MoE the decode is already fast, so MTP matters less here than
# on the dense models - but it's ~1.4-2x on top of an already quick model, and
# UD-Q6_K at 30GB fits the M4 Max fine. Needs llama.cpp with PR #22673.
# ------------------------------------------------------------
SPEC_DEFAULT = False
MTP_REPO = "gbuzhf/KAT-Coder-V2.5-Dev-MTP-GGUF"
MTP_FILES = {
    "m4": "Kwaipilot_KAT-Coder-V2.5-Dev-MTP-UD-Q6_K.gguf",  # 30.0GB
    "m1": "Kwaipilot_KAT-Coder-V2.5-Dev-MTP-UD-Q5_K_S.gguf",  # 25.5GB
    "unknown": "Kwaipilot_KAT-Coder-V2.5-Dev-MTP-UD-Q4_K_XL.gguf",  # 22.9GB
}
MTP_DRAFT_N_MAX = 2

# Text-only model - no projector to fetch.
VISION = False


def spec(machine: Machine, opts: Options) -> LlamaServerSpec:
    cfg = MACHINES.get(machine.key)
    if cfg is None:
        raise Unsupported(
            f"kat-coder2.5 is too large for {machine.chip} "
            f"({machine.ram_gb}GB)."
        )

    thinking = THINKING_DEFAULT if opts.thinking is None else opts.thinking
    use_mtp = SPEC_DEFAULT if opts.spec_decode is None else opts.spec_decode

    sampling = SAMPLING_THINKING if thinking else SAMPLING_INSTRUCT

    repo = MTP_REPO if use_mtp else HF_REPO
    hf_file = MTP_FILES[machine.key] if use_mtp else cfg["hf_file"]

    template_kwargs = (
        '{"enable_thinking":true,"preserve_thinking":true}'
        if thinking and PRESERVE_THINKING
        else '{"enable_thinking":true}'
        if thinking
        else '{"enable_thinking":false}'
    )
    extra = ["--chat-template-kwargs", template_kwargs]
    spec_args: list[str] = []
    notes = [
        ("Thinking", "on (preserved across turns)" if thinking else "off"),
    ]

    if use_mtp:
        spec_args = [
            "--spec-type", "draft-mtp",
            "--spec-draft-n-max", str(MTP_DRAFT_N_MAX),
        ]
        notes.append(("Speculative", f"MTP, n-max={MTP_DRAFT_N_MAX}"))
    else:
        notes.append(("Speculative", "off (--spec enables MTP)"))

    return LlamaServerSpec(
        alias=ALIAS,
        hf_repo=repo,
        hf_file=hf_file,
        ctx=opts.ctx or cfg["ctx"],
        predict=cfg["predict"],
        parallel=cfg["parallel"],
        vision=VISION,
        extra_args=extra,
        spec_args=spec_args,
        notes=notes,
        **sampling,
    )

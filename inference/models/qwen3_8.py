"""Qwen3.8-27B on MTPLX - native multi-token prediction, MLX, Apple Silicon.

The same Qwen3.8-27B as before, served by MTPLX instead of llama.cpp. MTPLX
drafts several tokens ahead using the model's *own* MTP head - no second draft
model sitting in RAM - and verifies each drafted block in one batched forward
pass. Drafts are accepted by the probability-ratio rule with residual
resampling, so what gets sampled is what the model would have sampled without
speculation, at any temperature.

That is why this replaced the llama.cpp setup. The old `--spec` path needed a
third-party MTP-converted GGUF *and* a llama.cpp new enough to know
`--spec-type draft-mtp`, and the plain path was simply too slow.

Engine:  mtplx serve (MLX)
Weights: https://huggingface.co/Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed
Project: https://github.com/youssofal/MTPLX  (Apache-2.0)
Install: brew install youssofal/mtplx/mtplx

Two caveats worth keeping in view when reading numbers off this model:

Speed - the vendor's 58.7 tok/s coding figure is an M5 Max with fans verified
at max. This rig's M4 Max has no M5 TensorOps path (`mtplx hardware` reports
`M5 TensorOps eligible: false`), so expect less. Measure it, don't assume it.

Quality - the llama.cpp setup ran UD-Q8_K_XL on the M4 Max, which is
near-lossless. Optimized Speed is a dynamic 4-bit build: KL 0.0220 to the bf16
original on the vendor's coding battery. That divergence is from the *quant*,
not from the speculation, which is exact. If a comparison needs the old quality
bar back, set VARIANT = "quality" - 30.4GB fits the M4 Max with room to spare.
"""

from ..machine import Machine, Options, Unsupported
from ..specs import MtplxServerSpec

ALIAS = "qwen3.8"
FAMILY = "qwen"
SUMMARY = "Dense 27B multimodal on MTPLX - native MTP speculative decode"

# Accepts images. MTPLX specs carry no vision flag, so it is declared here like
# the llama.cpp models' VISION; the Clients tab tells VS Code from this.
VISION = True

# run.py reads this to skip the llama.cpp speculative-decoding capability gate,
# which asks `llama-server --help` about flags that mean nothing here. MTP is
# native to MTPLX and needs no support check.
ENGINE = "mtplx"

# ------------------------------------------------------------
# Variant.
#
# All three are the same weights forged three ways, and all three carry the MTP
# head and the tuned draft settings. Sizes are the full repo download.
#
#   speed    21.3GB  dynamic 4-bit, 8-bit embeddings/head/GDN out-projs/last 8
#                    MLPs, bf16 norms + MTP head.  KL 0.0220.  The vendor's
#                    recommendation for coding on a 32GB+ Mac.
#   quality  30.4GB  8-bit. Closest match to the old UD-Q8_K_XL baseline.
#   bare     16.9GB  flat 4-bit. Fastest, and the loosest against the original.
#
# M1 and M2 don't run bf16 well, so those machines take the -FP16 sibling: the
# quantized bytes are identical, the remaining float tensors are fp16 instead
# of bf16. Same layout, same depth, same context.
# ------------------------------------------------------------
VARIANTS = {
    "speed": "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed",
    "quality": "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality",
    "bare": "Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed",
}
VARIANT = "speed"
FP16_SUFFIX = "-FP16"

# ------------------------------------------------------------
# Per machine.
#
# Peak unified memory is 23.6GB measured, near enough flat across variants at
# this quant level, and the vendor floor is a 32GB Mac. The M2's 24GB can't
# hold it plus a KV cache, so this model is off that machine entirely - which
# is a change from the llama.cpp setup, where a 3-bit GGUF just about fitted.
# ------------------------------------------------------------
MACHINES = {
    "m4": dict(ctx=262144, max_tokens=32768, fp16=False),
    "m1": dict(ctx=65536, max_tokens=16384, fp16=True),
    "m2": None,  # 24GB, and peak is 23.6GB before any KV cache
    "unknown": dict(ctx=32768, max_tokens=8192, fp16=False),
}
MIN_RAM_GB = 32

# ------------------------------------------------------------
# Sampling - the official Qwen3.8 contract, and what the vendor measured with.
#
# MTPLX has no min-p and no repeat-penalty; it has presence and frequency
# penalties instead, both defaulting to 0. The llama.cpp setup leaned on
# presence_penalty=1.5 to keep the non-thinking path out of repetition loops,
# so that carries over to the instruct profile.
# ------------------------------------------------------------
SAMPLING_THINKING = dict(
    temp=1.0, top_p=0.95, top_k=20, presence_penalty=0.0, frequency_penalty=0.0
)
SAMPLING_INSTRUCT = dict(
    temp=0.7, top_p=0.80, top_k=20, presence_penalty=1.5, frequency_penalty=0.0
)

# ------------------------------------------------------------
# Thinking - and this is the real behavioural win over the GGUF path.
#
# The chat template shipped in the GGUF accepted only xhigh/medium/low for
# reasoning_effort. `none` made the template raise and llama-server 500'd, so
# reasoning could not actually be switched off on this model - `low` was the
# floor, and the rest of the qwen family runs with it off precisely because
# thinking-mode qwen defers instead of committing and burns Cline's per-turn
# tool budget.
#
# MTPLX takes reasoning as a server flag rather than through the template, and
# `--reasoning off` is a real setting. So THINKING_DEFAULT = False now means
# off, not "as close to off as the template will allow".
#
# preserve-thinking stays on `auto`, which MTPLX resolves to preserved for
# Qwen 3.8 - that model's trained contract, unlike Qwen 3.6's scoped one.
# ------------------------------------------------------------
THINKING_DEFAULT = False
REASONING_EFFORT_ON = "medium"  # MTPLX's coding default for this model
PRESERVE_THINKING = "auto"
REASONING_PARSER = "qwen3"

# ------------------------------------------------------------
# MTP.
#
# On by default - it is the entire point of this engine. `--no-spec` maps to
# `--no-mtp`, which keeps the same loaded runtime and drops to target-only
# autoregressive decode. That makes the A/B a one-flag change against identical
# weights, which is the honest way to measure what MTP is actually buying.
#
# Depth is left unset on purpose: the tuned depth (3) and draft settings ship
# inside the repo's mtplx_runtime.json and MTPLX reads them on load. Pin DEPTH
# only to override that, and use `mtplx tune` to find the value for this Mac
# rather than guessing.
# ------------------------------------------------------------
# ------------------------------------------------------------
# Response cap for thinking mode.
#
# With reasoning on, this model occasionally runs away, in two shapes. Over one
# session:
#
#   7,687 tokens / 231s - 7,442 of them reasoning. A thinking runaway.
#   9,395 tokens / 273s - zero reasoning and zero content, ending in two
#                         grep_search calls of ~250 characters. ttft and ttfc
#                         were both 277s: the client saw nothing for four and a
#                         half minutes.
#
# Nothing bounded either - MTPLX's default is 32768 and VS Code sends no
# max_tokens - so a cap is worth having. The value is the whole question.
#
# 2048 was tried first, from the output distribution (median 116, 90th
# percentile under 430). It ended a session. The turn that hit it spent 2,014
# of its 2,048 tokens reasoning, reached a sound conclusion - ask the user
# before making a design-level change - started writing the answer, and was
# cut 159 characters in, mid-sentence, with no tool call. The client had
# nothing to act on and stopped.
#
# That is the trap: with reasoning on, max_tokens is a *shared* budget.
# Reasoning spends it first and the answer takes what is left, so a tight cap
# converts a slow turn into a turn that produces nothing at all - strictly
# worse than the stall it was meant to prevent. MTPLX has no separate reasoning
# limit (only effort levels and this total), so there is no setting that bounds
# thinking without also bounding the reply.
#
# 6144 is the compromise, and every number in it comes from observed turns:
#   - above the largest legitimate turn seen (4,700: 4 tool calls, 4,387
#     reasoning tokens), so real work is not truncated
#   - above 2,014 reasoning plus a full answer, the case that broke 2048
#   - below both runaways (7,687 and 9,395), so those are still cut
#   - roughly three minutes at this model's ~34 tok/s, against 273s observed
#
# Only applies with reasoning on, which is where both runaways happened. The
# non-thinking profile fails differently - repetition at ~89 tokens a turn -
# and a cap does nothing about that.
MAX_TOKENS_THINKING = 6144

SPEC_DEFAULT = True
DEPTH = None

# `mtplx inspect` reports recommended_profile: turbo for this artifact.
PROFILE = "turbo"

# Concurrent-batching preset aimed at coding-agent traffic, which is what the
# scenarios in tests/ generate.
BATCHING_PRESET = "agent"


def spec(machine: Machine, opts: Options) -> MtplxServerSpec:
    cfg = MACHINES.get(machine.key)
    if cfg is None or machine.ram_gb < MIN_RAM_GB:
        raise Unsupported(
            f"qwen3.8 on MTPLX needs a {MIN_RAM_GB}GB+ Mac - peak unified memory "
            f"is 23.6GB before the KV cache. Detected: {machine.chip} "
            f"({machine.ram_gb}GB)."
        )

    # An explicit effort implies reasoning on - asking for "low" while
    # reasoning is off is not a combination that means anything.
    effort = getattr(opts, "reasoning_effort", None)
    thinking = THINKING_DEFAULT if opts.thinking is None else opts.thinking
    if effort and opts.thinking is not False:
        thinking = True
    use_mtp = SPEC_DEFAULT if opts.spec_decode is None else opts.spec_decode

    sampling = SAMPLING_THINKING if thinking else SAMPLING_INSTRUCT
    repo = VARIANTS[VARIANT] + (FP16_SUFFIX if cfg["fp16"] else "")

    notes = [
        ("Variant", f"{VARIANT} ({repo.rsplit('/', 1)[1]})"),
        ("Precision", "fp16 floats (M1/M2 build)" if cfg["fp16"] else "bf16 floats"),
        ("Reasoning",
         f"on, effort={effort or REASONING_EFFORT_ON}" if thinking else "off"),
        ("Max tokens", f"{MAX_TOKENS_THINKING:,} (capped for thinking runaways)"
         if thinking else f"{cfg['max_tokens']:,}"),
        (
            "Speculative",
            f"native MTP, depth={DEPTH or 'from mtplx_runtime.json'}"
            if use_mtp
            else "off (--no-spec: target-only AR)",
        ),
        ("Profile", PROFILE),
        ("Client model id", ALIAS),
    ]

    return MtplxServerSpec(
        hf_repo=repo,
        model_id=ALIAS,
        profile=PROFILE,
        context_window=opts.ctx or cfg["ctx"],
        max_tokens=MAX_TOKENS_THINKING if thinking else cfg["max_tokens"],
        mtp=use_mtp,
        depth=DEPTH,
        reasoning="on" if thinking else "off",
        reasoning_effort=(effort or REASONING_EFFORT_ON) if thinking else "auto",
        reasoning_parser=REASONING_PARSER,
        preserve_thinking=PRESERVE_THINKING,
        batching_preset=BATCHING_PRESET,
        notes=notes,
        **sampling,
    )

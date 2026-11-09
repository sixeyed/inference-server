"""One fully-resolved launch per engine.

A spec is the whole command a model server will be started with, decided before
anything runs. Three engines means three shapes, and the differences between them
are load-bearing rather than incidental - which id clients must send, whether
auth is checked, what the sampling flags are even called.

`Machine`, `Options` and `Unsupported` are re-exported here because every model
file needs them alongside its spec class, and asking model authors for two import
lines to add one file works against the point of the `models/` layout.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .config import (
    DEFAULT_API_KEY,
    LLAMA_LOAD_MODE,
    MTPLX_CACHE_DIR,
    MTPLX_SESSION_CACHE_DIR,
)
from .machine import Machine, Options, Unsupported

__all__ = [
    "LlamaServerSpec", "MlxServerSpec", "MtplxServerSpec",
    "Machine", "Options", "Unsupported", "find_in_hf_cache",
]


@dataclass
class LlamaServerSpec:
    """A fully-resolved llama-server launch."""

    alias: str
    hf_repo: str
    hf_file: str | None = None
    local_path: str | None = None  # explicit -m override, set by run.py
    ctx: int = 32768
    predict: int = 32768
    parallel: int = 1

    # Sampling
    temp: float = 0.7
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    presence_penalty: float = 0.0
    repeat_penalty: float = 1.0

    vision: bool = False  # False => --no-mmproj (skip the projector download)
    extra_args: list[str] = field(default_factory=list)
    # Speculative-decoding args, kept separate so they can be dropped when the
    # installed llama.cpp is too old to know them - they're a pure speed
    # optimisation, so falling back to plain decode is better than not starting.
    spec_args: list[str] = field(default_factory=list)
    notes: list[tuple[str, str]] = field(default_factory=list)  # banner rows

    def argv(self, machine: Machine, port: int,
             host: str = "127.0.0.1") -> list[str]:
        local = self.local_path or find_in_hf_cache(self.hf_repo, self.hf_file)
        source = ["-m", local] if local else ["-hf", self.hf_repo]
        args = source + [
            "--alias", self.alias,
            "--host", host,
            "--port", str(port),
            "-c", str(self.ctx),
            "-b", str(machine.batch),
            "-ub", str(machine.ubatch),
            "-t", str(machine.threads),
            "-n", str(self.predict),
            "-ngl", "999",
            "--parallel", str(self.parallel),
            "--load-mode", LLAMA_LOAD_MODE,
            "--flash-attn", "on",
            "--cache-ram", str(machine.cache_ram_mb),
            "-ctxcp", "128",
            "--jinja",
            "--temp", str(self.temp),
            "--top-p", str(self.top_p),
            "--top-k", str(self.top_k),
            "--min-p", str(self.min_p),
            "--presence-penalty", str(self.presence_penalty),
            "--repeat-penalty", str(self.repeat_penalty),
        ]
        if self.hf_file and not local:
            args += ["--hf-file", self.hf_file]
        if not self.vision:
            args.append("--no-mmproj")
        args += self.extra_args
        args += self.spec_args
        return args


@dataclass
class MlxServerSpec:
    """A fully-resolved mlx_lm.server launch. No --alias support: clients must
    send the full HF repo id in the request `model` field."""

    hf_repo: str
    max_tokens: int = 262144
    temp: float = 1.0
    top_p: float = 0.95
    prefill_step_size: int = 4096
    prompt_cache_bytes: int = 16 * 1024**3
    extra_args: list[str] = field(default_factory=list)
    notes: list[tuple[str, str]] = field(default_factory=list)

    def argv(self, machine: Machine, port: int,
             host: str = "127.0.0.1") -> list[str]:
        return [
            "--model", self.hf_repo,
            "--host", host,
            "--port", str(port),
            "--max-tokens", str(self.max_tokens),
            "--temp", str(self.temp),
            "--top-p", str(self.top_p),
            "--prefill-step-size", str(self.prefill_step_size),
            "--prompt-cache-bytes", str(self.prompt_cache_bytes),
            "--trust-remote-code",
        ] + self.extra_args


@dataclass
class MtplxServerSpec:
    """A fully-resolved `mtplx serve` launch.

    MTPLX is MLX-native and does speculative decoding off the model's own MTP
    head, so there is no draft model to configure and no llama.cpp build to
    check - `--mtp` is the default and the tuned depth ships in the model repo's
    mtplx_runtime.json.

    Two things differ from the other two engines and both are load-bearing:

    `--model-id` sets the id clients send, so this behaves like llama-server's
    `--alias` rather than mlx_lm.server's "send the full HF repo id".

    Auth is not optional off localhost. `--no-auth` only relaxes localhost
    binds; any other bind wants a key, so one is always passed here -
    see DEFAULT_API_KEY for why it is a known string rather than a secret.
    """

    hf_repo: str
    model_id: str
    profile: str = "turbo"
    context_window: int = 262144
    max_tokens: int = 32768

    # Sampling. MTPLX has no min-p and no repeat-penalty.
    temp: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    mtp: bool = True
    depth: int | None = None  # None = use the model's tuned mtplx_runtime.json

    reasoning: str = "auto"  # auto | on | off
    reasoning_effort: str = "auto"  # auto | low | medium | high | xhigh
    reasoning_parser: str = "qwen3"
    preserve_thinking: str = "auto"  # auto | on | off | scoped
    tool_prompt_mode: str = "native"  # render tools from the model's own template
    batching_preset: str = "agent"

    api_key: str = DEFAULT_API_KEY
    cache_dir: str = MTPLX_CACHE_DIR
    session_cache_dir: str = MTPLX_SESSION_CACHE_DIR
    download: bool = True  # fetch the weights on first launch if not cached
    extra_args: list[str] = field(default_factory=list)
    notes: list[tuple[str, str]] = field(default_factory=list)

    def argv(self, machine: Machine, port: int,
             host: str = "127.0.0.1") -> list[str]:
        args = [
            "serve",
            "--model", self.hf_repo,
            "--model-id", self.model_id,
            "--host", host,
            "--port", str(port),
            "--api-key", self.api_key,
            # Weights on the external volume; the session bank stays internal
            # because it is read on prefill, not just at load.
            "--cache-dir", self.cache_dir,
            "--ssd-session-cache-dir", self.session_cache_dir,
            "--profile", self.profile,
            "--context-window", str(self.context_window),
            "--max-tokens", str(self.max_tokens),
            "--default-temperature", str(self.temp),
            "--default-top-p", str(self.top_p),
            "--default-top-k", str(self.top_k),
            "--default-presence-penalty", str(self.presence_penalty),
            "--default-frequency-penalty", str(self.frequency_penalty),
            "--reasoning", self.reasoning,
            "--reasoning-effort", self.reasoning_effort,
            "--reasoning-parser", self.reasoning_parser,
            "--preserve-thinking", self.preserve_thinking,
            "--tool-prompt-mode", self.tool_prompt_mode,
            "--batching-preset", self.batching_preset,
            # MTPLX appends a visible tok/s footer to returned text by default.
            # That lands inside assistant content, so it corrupts transcripts
            # and anything parsing tool calls out of the reply.
            "--no-stats-footer",
        ]
        args += ["--mtp"] if self.mtp else ["--no-mtp"]
        if self.depth is not None:
            args += ["--depth", str(self.depth)]
        if self.download:
            args.append("--download")
        return args + self.extra_args


def find_in_hf_cache(hf_repo: str, hf_file: str | None) -> str | None:
    """Locate an already-downloaded GGUF in the HuggingFace cache.

    llama.cpp keeps its own cache (~/Library/Caches/llama.cpp) and won't look in
    the HF one, so `-hf` happily re-downloads tens of GB of weights that are
    already on disk from `huggingface-cli` or an MLX run. If the exact quant is
    there, pass it by path instead.

    Only exact filename matches count - picking a different quant than the model
    file asked for would silently change what's being measured.
    """
    if not hf_file:
        return None
    # Split files are referenced by their first part; -m loads the rest itself.
    name = os.path.basename(hf_file)

    cache = os.environ.get("HF_HOME")
    hub = (os.path.join(cache, "hub") if cache
           else os.path.expanduser("~/.cache/huggingface/hub"))
    repo_dir = os.path.join(hub, "models--" + hf_repo.replace("/", "--"))
    snapshots = os.path.join(repo_dir, "snapshots")
    if not os.path.isdir(snapshots):
        return None

    for revision in sorted(os.listdir(snapshots)):
        candidate = os.path.join(snapshots, revision, name)
        if os.path.exists(candidate):  # follows the symlink into blobs/
            return candidate
    return None

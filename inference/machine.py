"""The machine this is running on, and the knobs that follow from it.

Machine-level settings (threads, batch sizes, the prompt-cache budget) live here
because they are a property of the Mac, not of the model. Model-level settings -
quant, context, sampling - live in that model's own file under `models/`.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass

from .config import DEFAULT_PORT, MACHINE_PROFILES, MODEL_BIND


class Unsupported(Exception):
    """Raised by a model's spec() when it won't fit the detected machine."""


@dataclass(frozen=True)
class Machine:
    key: str  # "m4" | "m1" | "m2" | "unknown" - the key model files index on
    chip: str
    ram_gb: int
    cores: int
    host: str  # LAN address clients connect to
    threads: int
    batch: int
    ubatch: int
    cache_ram_mb: int  # llama-server --cache-ram, scaled to RAM


def _sysctl(name: str) -> str:
    return subprocess.run(
        ["sysctl", "-n", name], capture_output=True, text=True, check=True
    ).stdout.strip()



def machine_from_profile(key: str) -> Machine:
    """A Machine built from config alone, for when there is no Mac to ask.

    Used off-Mac: by an engine in a container while its model host cannot be
    reached, and by the test suite inside the image. RAM and cores come from
    the profile, because models decide whether they fit on them.
    """
    if key not in MACHINE_PROFILES:
        key = "unknown"
    profile = MACHINE_PROFILES[key]
    return Machine(
        key=key,
        chip=profile.get("chip") or key,
        ram_gb=int(profile.get("ram_gb", 0)),
        cores=int(profile.get("cores", 0)),
        host=profile["host"],
        threads=profile["threads"],
        batch=profile["batch"],
        ubatch=profile["ubatch"],
        cache_ram_mb=profile["cache_ram_mb"],
    )


def detect_machine() -> Machine:
    """The Mac this is running on, and the settings that follow from it.

    The chip is detected; everything that follows from it - LAN address,
    threads, batch sizes, the prompt-cache budget - comes from
    config.MACHINE_PROFILES, which config.json overrides per chip. That keeps
    one box's address and tuning out of the source, and lets a differently
    sized Mac of the same chip be configured rather than coded.
    """
    if sys.platform != "darwin":
        # Nothing to detect off a Mac. INFERENCE_MACHINE names the profile to
        # stand in - the test stage of the image sets it, where the engine
        # never launches a model itself.
        key = os.environ.get("INFERENCE_MACHINE")
        if key:
            return machine_from_profile(key)
        raise SystemExit("This harness targets Apple Silicon macOS.")

    brand = _sysctl("machdep.cpu.brand_string")
    cores = int(_sysctl("machdep.cpu.core_count"))
    ram_gb = round(int(_sysctl("hw.memsize")) / (1024**3))

    key = next((k.lower() for k in ("M4", "M2", "M1") if k in brand), "unknown")
    profile = dict(MACHINE_PROFILES.get(key, MACHINE_PROFILES["unknown"]))

    return Machine(
        key=key,
        # A profile may name the chip more precisely than the brand string
        # ("M4 Max"), but an unconfigured machine should still say what it
        # actually is rather than an empty string.
        chip=profile.get("chip") or brand,
        ram_gb=ram_gb,
        cores=cores,
        host=profile["host"],
        threads=profile["threads"],
        batch=profile["batch"],
        ubatch=profile["ubatch"],
        cache_ram_mb=profile["cache_ram_mb"],
    )


@dataclass
class Options:
    """Per-run overrides. Every model's spec() takes these; models ignore the
    ones that don't apply to them."""

    port: int = DEFAULT_PORT
    thinking: bool | None = None  # None = use the model file's default
    spec_decode: bool | None = None  # None = use the model file's default
    # Reasoning effort for models that expose levels - Qwen 3.8 takes
    # low/medium/high/xhigh under MTPLX. None = the model file's choice.
    # Setting it implies thinking on: an effort with reasoning off is not a
    # combination that means anything.
    reasoning_effort: str | None = None
    ctx: int | None = None  # override context length
    dry_run: bool = False
    # Confine the server with sandbox-exec so it cannot open outbound sockets.
    # On by default: a server whose weights are already on disk never needs the
    # network, so the safe setting is the one you get without thinking about it.
    # First-time downloads happen before launch, outside it - see weights.py.
    sandbox: bool = True
    # Where the model server binds. Loopback unless something off this machine
    # has to reach it directly - see config.MODEL_BIND.
    bind: str = MODEL_BIND

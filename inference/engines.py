"""Finding each engine's executable, and asking it what it can do.

The capability gates matter more than they look. Speculative decoding needs both
a recent llama.cpp *and*, for MTP models, an entirely different set of weights -
so whether the installed build supports it has to be settled before a spec is
built, not after 30GB has been downloaded.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess

from .console import red
from .specs import LlamaServerSpec, MlxServerSpec, MtplxServerSpec


def find_llama_server() -> str:
    exe = shutil.which("llama-server")
    if not exe:
        raise SystemExit(
            red("llama-server not found.") + " Install with: brew install llama.cpp"
        )
    return exe


_llama_help: str | None = None


def llama_help(exe: str) -> str:
    global _llama_help
    if _llama_help is None:
        result = subprocess.run(
            [exe, "--help"], capture_output=True, text=True, timeout=30
        )
        _llama_help = result.stdout + result.stderr
    return _llama_help


def llama_build(exe: str) -> int | None:
    """Installed llama.cpp build number, e.g. 10621. None if unparseable.

    Two shapes, because llama.cpp changed scheme at 0.3.0:

        version: 10470 (34af94cd9)                      before
        version: 0.3.0 (build 10621, commit c1d0e7a00)  after

    `build N` is tried first. Anchoring on `version:` alone reads the newer
    form as build 0 - `0.3.0` starts with a zero - and a warning telling you to
    upgrade from a build number that never existed is worse than one that says
    `?`. Hence None rather than 0 when neither shape matches.

    Display only. The capability gates ask `--help` about specific flags (see
    llama_supports), which is why the scheme change cost a warning string and
    not speculative decoding.
    """
    result = subprocess.run(
        [exe, "--version"], capture_output=True, text=True, timeout=30
    )
    text = result.stdout + result.stderr
    match = re.search(r"build\s+(\d+)", text) or re.search(r"version:\s*(\d+)", text)
    return int(match.group(1)) if match else None


def llama_supports(exe: str, token: str) -> bool:
    """Is `token` (a flag, or a value inside a flag's enum) in --help output?

    Both matter: `--spec-type` has existed for a while, but the `draft-mtp` and
    `draft-dflash` modes the new models need are much newer additions to its
    enum, so checking the flag alone would give a false positive.
    """
    return token in llama_help(exe)


def resolve_spec_decode(requested: bool | None) -> tuple[bool | None, str | None]:
    """Force speculative decoding off if the installed llama.cpp can't do it.

    Done before the model file builds its spec, because models that use MTP have
    to load a *different* GGUF repo for it - and there's no point downloading
    30GB of MTP weights for a build that will ignore the prediction heads.
    """
    if requested is False:
        return requested, None
    exe = shutil.which("llama-server")
    if exe is None or llama_supports(exe, "--spec-draft-n-max"):
        return requested, None
    if requested is None:
        return False, None  # model wanted it by default; drop it quietly
    return False, (
        f"--spec ignored - this llama.cpp (build {llama_build(exe) or '?'}) has no "
        "MTP/DFlash speculative decoding. Upgrade with: brew upgrade llama.cpp"
    )


def check_spec_support(exe: str, spec_args: list[str]) -> tuple[list[str], str | None]:
    """Return (usable spec args, warning). Drops the args entirely if this build
    doesn't know them, since speculative decoding is only ever a speed win."""
    if not spec_args:
        return [], None

    missing = [
        token
        for token in spec_args
        if (token.startswith("--") or token.startswith("draft-"))
        and not llama_supports(exe, token)
    ]
    if not missing:
        return spec_args, None

    build = llama_build(exe)
    return [], (
        f"speculative decoding disabled - this llama.cpp "
        f"(build {build if build else '?'}) has no {', '.join(missing)}. "
        "Upgrade with: brew upgrade llama.cpp"
    )


def find_mlx_server() -> str:
    exe = shutil.which("mlx_lm.server")
    if exe:
        return exe
    # pip --user installs land outside the default PATH
    for version in ("3.13", "3.12", "3.11", "3.10"):
        candidate = os.path.expanduser(f"~/Library/Python/{version}/bin/mlx_lm.server")
        if os.path.exists(candidate):
            return candidate
    raise SystemExit(
        red("mlx_lm.server not found.") + " Install with: pip3 install --user mlx-lm"
    )


def find_mtplx() -> str:
    exe = shutil.which("mtplx")
    if exe:
        return exe
    raise SystemExit(
        red("mtplx not found.")
        + " Install with: brew install youssofal/mtplx/mtplx"
    )


def resolve_engine(spec) -> tuple[str, str, str | None]:
    """(executable, engine name, warning) for a spec.

    Splitting this out of serve() is what lets the engine start the same model
    the same way without also inheriting serve()'s banner and its blocking
    subprocess.call - the two front doors have to agree on the command, or the
    same alias means different things depending on how you launched it.

    Mutates spec.spec_args and spec.notes when the installed llama.cpp is too
    old for the speculative-decoding flags, exactly as before.
    """
    warning = None
    if isinstance(spec, LlamaServerSpec):
        exe, engine = find_llama_server(), "llama-server"
        spec.spec_args, warning = check_spec_support(exe, spec.spec_args)
        if warning:
            spec.notes = [
                (label, "off (build too old)" if label == "Speculative" else value)
                for label, value in spec.notes
            ]
    elif isinstance(spec, MlxServerSpec):
        exe, engine = find_mlx_server(), "mlx_lm.server"
    elif isinstance(spec, MtplxServerSpec):
        exe, engine = find_mtplx(), "mtplx serve"
    else:
        raise TypeError(f"unknown spec type: {type(spec)!r}")
    return exe, engine, warning

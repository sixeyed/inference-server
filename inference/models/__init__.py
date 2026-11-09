"""Model registry.

One module per model. Each one must expose:

    ALIAS    str   - the name you pass to run.py, and the id clients send
    FAMILY   str   - grouping for `run.py --family`
    SUMMARY  str   - one line for `run.py --list`
    spec(machine, opts) -> LlamaServerSpec | MlxServerSpec | MtplxServerSpec
                     - or raise Unsupported if it won't fit that machine

and may expose:

    ENGINE          str   - "mtplx" etc, when it isn't llama-server. Read before
                            the spec exists, to skip the llama.cpp gate.
    CLIENT_MODEL_ID str   - the id the *server* answers to, when that isn't the
                            alias. mlx_lm.server has no --alias, so its models
                            only respond to the full HuggingFace repo id.
    FORMER_ALIASES  tuple - names this model used to have. Old traces, old
                            results and old client configs still say them.
    DEPRECATED      str   - why it was retired. Keeps `run.py` able to launch it
                            while keeping it out of the router (see engine.py).
    VISION          bool  - accepts images. llama.cpp models also carry it on
                            the spec; MTPLX models have only this.
    TEST_ONLY       bool  - a fixture for exercising the engine end to end.
                            Routes by name like any model, but is left out of
                            /v1/models so no client picker ever offers it.

Modules are discovered by scanning this package, so adding a model is one new
file and nothing else.
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from types import ModuleType

# What `run.py` launches with no arguments, and what the engine falls back to.
DEFAULT_MODEL = "qwen3.8"

# Family default - what `run.py --family qwen` launches, in listing order.
#
# Only families with a live model in them. gemma and ling were dropped when
# their only members were deprecated: offering a family that can't launch
# anything is worse than not offering it.
FAMILY_DEFAULTS = {
    "qwen": "qwen3.8",
    "muse": "muse-glimmer",
    "kat": "kat-coder2.5",
}


def load_all() -> dict[str, ModuleType]:
    """alias -> module, for every model file in this package."""
    found: dict[str, ModuleType] = {}
    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{__name__}.{info.name}")
        alias = getattr(module, "ALIAS", None)
        if alias is None:
            continue
        if alias in found:
            raise RuntimeError(
                f"duplicate ALIAS {alias!r} in {info.name} and {found[alias].__name__}"
            )
        found[alias] = module
    return found


# ============================================================
# Describing what's available
# ============================================================
@dataclass(frozen=True)
class ModelInfo:
    """One model, resolved against one machine.

    Built by actually calling the model's spec(), so `supported` is the same
    judgement the launcher makes rather than a second opinion that can drift
    from it.
    """

    alias: str
    family: str
    summary: str
    engine: str
    supported: bool
    reason: str | None          # the Unsupported message, when it won't fit
    deprecated: str | None      # the DEPRECATED reason, when it's retired
    client_ids: tuple[str, ...]  # every name a client might legitimately send
    hf_repo: str | None
    ctx: int | None
    test_only: bool = False     # TEST_ONLY - routable, never listed
    vision: bool = False        # VISION, or the llama.cpp spec's flag

    @property
    def routable(self) -> bool:
        return self.supported and not self.deprecated


def _engine_of(module: ModuleType, spec) -> str:
    if spec is not None:
        return type(spec).__name__.replace("ServerSpec", "").lower()
    return getattr(module, "ENGINE", "llama-server")


def _ctx_of(spec) -> int | None:
    """Context window, whatever each engine calls it."""
    for attr in ("ctx", "context_window", "max_tokens"):
        value = getattr(spec, attr, None)
        if value is not None:
            return value
    return None


def describe(machine, registry: dict[str, ModuleType] | None = None,
             ) -> list[ModelInfo]:
    """Every model, with its spec built once so we know whether it fits.

    Shared by `run.py --list` and the engine's GET /v1/models, so the picker in
    VS Code shows exactly what the command line shows.
    """
    from ..machine import Options, Unsupported

    registry = registry if registry is not None else load_all()
    infos = []

    for alias in sorted(registry):
        module = registry[alias]
        spec, supported, reason = None, True, None
        try:
            spec = module.spec(machine, Options())
        except Unsupported as exc:
            supported, reason = False, str(exc)

        ids = [alias]
        ids += list(getattr(module, "FORMER_ALIASES", ()))
        client_id = getattr(module, "CLIENT_MODEL_ID", None)
        if client_id:
            ids.append(client_id)
        hf_repo = getattr(spec, "hf_repo", None)
        if hf_repo:
            ids.append(hf_repo)

        infos.append(ModelInfo(
            alias=alias,
            family=getattr(module, "FAMILY", ""),
            summary=getattr(module, "SUMMARY", ""),
            engine=_engine_of(module, spec),
            supported=supported,
            reason=reason,
            deprecated=getattr(module, "DEPRECATED", None),
            client_ids=tuple(dict.fromkeys(ids)),  # ordered, deduplicated
            hf_repo=hf_repo,
            ctx=_ctx_of(spec) if spec is not None else None,
            test_only=bool(getattr(module, "TEST_ONLY", False)),
            vision=bool(getattr(module, "VISION", getattr(spec, "vision", False))),
        ))
    return infos


# ============================================================
# Working out which model a client meant
# ============================================================
def build_index(infos: list[ModelInfo],
                deprecated: bool = False) -> dict[str, str]:
    """Lowercased client-visible name -> alias.

    Deprecated models are left out unless asked for, which is how the router
    declines to serve them while `run.py` still can.
    """
    index: dict[str, str] = {}
    for info in infos:
        if not info.supported:
            continue
        if info.deprecated and not deprecated:
            continue
        for name in info.client_ids:
            index.setdefault(name.lower(), info.alias)
    return index


def resolve(name: str | None, index: dict[str, str]) -> str | None:
    """A client's `model` string -> an alias, or None if we've never heard of it.

    Handles the three ways the same model legitimately arrives: the alias, a
    former alias (traces and client configs outlive renames), and the full
    HuggingFace repo id (mlx_lm.server has no --alias, so its clients are
    configured that way).

    Provider prefixes are stripped because opencode and friends send
    `local/qwen3.8`. The prefix carries no information here - there is only one
    provider - so dropping it is safe, and not dropping it means those clients
    fall through to the unknown-model path on every single request.

    Returns None rather than guessing. The caller owns the fallback policy,
    because "serve something anyway" is right for a chat client and wrong for a
    benchmark.
    """
    if not name:
        return None
    candidate = name.strip().lower()
    if candidate in index:
        return index[candidate]
    if "/" in candidate:
        # Only strip a single leading provider segment; HF repo ids are
        # `owner/model` and must be tried whole first, which they were above.
        _, _, tail = candidate.partition("/")
        if tail in index:
            return index[tail]
    return None

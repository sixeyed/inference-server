"""Egress confinement, and the fact that both front doors get it.

A model server needs the network once, to fetch weights. After that it only
answers on its own port, so denying it outbound sockets costs nothing and is
the difference between "no exfiltration found in the source" and "none is
possible". The risk this guards against is subtler than forgetting to sandbox:
sandboxing one launch path and not the other, so the same alias is confined or
not depending on whether you typed `run` or went through the router.
"""

import sys

import pytest

from inference.config import SANDBOX_PROFILE
from inference.machine import Machine, Options
from inference.process import build_argv, weights_are_local
from inference.specs import LlamaServerSpec, MlxServerSpec, MtplxServerSpec

# sandbox-exec is macOS. The image's test stage has none - and the engine in
# the image never launches a model itself; the Mac it drives does.
pytestmark = pytest.mark.skipif(sys.platform != "darwin",
                                reason="sandbox-exec is macOS only")

MACHINE = Machine(key="m4", chip="M4 Max", ram_gb=128, cores=16,
                  host="127.0.0.1", threads=12, batch=4096, ubatch=2048,
                  cache_ram_mb=32768)

SPECS = [
    LlamaServerSpec(alias="x", hf_repo="o/r", hf_file="x.gguf",
                    local_path="/tmp/x.gguf"),
    MlxServerSpec(hf_repo="o/r"),
    MtplxServerSpec(hf_repo="o/r", model_id="x"),
]


def _argv(spec, **kw):
    argv, _, warnings = build_argv(spec, MACHINE, Options(**kw))
    return argv, warnings


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: type(s).__name__)
def test_every_engine_is_sandboxed_by_default(spec):
    argv, _ = _argv(spec)
    assert argv[0].endswith("sandbox-exec")
    assert argv[1] == "-p"
    assert argv[2] == SANDBOX_PROFILE


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: type(s).__name__)
def test_no_sandbox_leaves_the_command_bare(spec):
    argv, _ = _argv(spec, sandbox=False)
    assert "sandbox-exec" not in argv[0]


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: type(s).__name__)
def test_the_real_command_survives_wrapping(spec):
    """The engine's own argv must be intact after the prefix, or the sandbox
    would silently change how the model is configured."""
    wrapped, _ = _argv(spec)
    bare, _ = _argv(spec, sandbox=False)
    assert wrapped[3:] == bare


def test_profile_denies_outbound_before_reallowing():
    """Order is load-bearing: a re-allow placed before the deny is overridden
    by it, and a profile that denies nothing looks identical to one that works
    until you actually try to reach the network."""
    deny = SANDBOX_PROFILE.index("(deny network-outbound)")
    assert SANDBOX_PROFILE.index("(allow default)") < deny
    for reallow in ("(allow network-bind)", "(allow network-inbound)"):
        assert SANDBOX_PROFILE.index(reallow) > deny


def test_warns_when_the_weights_would_have_to_be_downloaded(monkeypatch):
    monkeypatch.setattr("inference.process.weights_are_local", lambda spec: False)
    _, warnings = _argv(MtplxServerSpec(hf_repo="o/r", model_id="x"))
    assert any("inference pull" in w for w in warnings), warnings


def test_no_download_warning_when_the_weights_are_there(monkeypatch):
    monkeypatch.setattr("inference.process.weights_are_local", lambda spec: True)
    _, warnings = _argv(MtplxServerSpec(hf_repo="o/r", model_id="x"))
    assert not any("inference pull" in w for w in warnings), warnings


def test_no_download_warning_when_not_sandboxed(monkeypatch):
    """Without the sandbox the download works, so the hint would be noise."""
    monkeypatch.setattr("inference.process.weights_are_local", lambda spec: False)
    _, warnings = _argv(MtplxServerSpec(hf_repo="o/r", model_id="x"), sandbox=False)
    assert not any("inference pull" in w for w in warnings), warnings


def test_mtplx_weights_are_not_looked_for_in_the_huggingface_cache(monkeypatch):
    """`mtplx inspect` leaves a metadata-only stub in the HF cache. Treating
    that as "downloaded" would suppress the warning for a model whose 20GB of
    weights are not there."""
    monkeypatch.setattr("os.path.isdir", lambda path: "huggingface" in path)
    assert not weights_are_local(MtplxServerSpec(hf_repo="o/r", model_id="x"))

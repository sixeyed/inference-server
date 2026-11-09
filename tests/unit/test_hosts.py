"""The local host - what the engine has always done, behind one object."""

import pytest

from inference import hosts, supervisor
from inference.machine import Options
from inference.specs import LlamaServerSpec

SPEC = LlamaServerSpec(alias="x", hf_repo="o/r", hf_file="x.gguf")


def test_a_sandboxed_launch_with_no_weights_is_refused(monkeypatch):
    monkeypatch.setattr(hosts.process, "weights_are_local", lambda spec: False)
    refusal = hosts.LocalHost().preflight("x", SPEC, Options(sandbox=True))
    assert refusal == "o/r is not on disk and the egress sandbox blocks the download"


def test_without_the_sandbox_nothing_is_refused(monkeypatch):
    monkeypatch.setattr(hosts.process, "weights_are_local", lambda spec: False)
    assert hosts.LocalHost().preflight("x", SPEC, Options(sandbox=False)) is None


def test_injected_launcher_and_stopper_are_used():
    calls = []
    host = hosts.LocalHost(launcher=lambda *a: calls.append("launch"),
                           stopper=lambda quiet, only_port: calls.append(("stop", only_port)))
    host.launch("x", SPEC, None, Options(), "logs")
    host.stop_all(quiet=True, only_port=9000)
    assert calls == ["launch", ("stop", 9000)]


def test_the_probe_asks_the_host_it_is_given(monkeypatch):
    seen = []

    def fake_get(url, api_key, timeout):
        seen.append(url)
        return 200, b""

    monkeypatch.setattr(supervisor, "_get", fake_get)
    assert supervisor.probe(SPEC, 8083, "none", host="mac.example.com")[0] == "ready"
    assert seen == ["http://mac.example.com:8083/health"]


def test_a_missing_engine_is_found_before_any_pull():
    """A llama.cpp model with no llama-server installed must fail at prepare -
    before the supervisor asks whether to pull 30GB of weights it cannot run."""
    from inference import models
    from inference.machine import machine_from_profile

    def missing(spec):
        raise SystemExit("llama-server not found. Install with: brew install llama.cpp")
    pulled = []
    host = hosts.LocalHost(puller=lambda spec, say: pulled.append(spec),
                           engine_check=missing)
    module = models.load_all()["qwen3-tiny"]
    with pytest.raises(SystemExit, match="brew install llama.cpp"):
        host.prepare(module, machine_from_profile("m4"), Options(sandbox=True))
    assert pulled == []


@pytest.mark.parametrize("bind, address", [
    ("127.0.0.1", "127.0.0.1"), ("0.0.0.0", "127.0.0.1"), ("192.168.1.10", "192.168.1.10")])
def test_the_model_servers_are_probed_where_they_bind(bind, address):
    assert hosts.LocalHost(bind=bind).address == address

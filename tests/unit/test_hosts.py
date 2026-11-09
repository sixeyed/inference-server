"""The local host - what the engine has always done, behind one object."""

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

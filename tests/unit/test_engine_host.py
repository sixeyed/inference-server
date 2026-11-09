"""Which host the engine drives, and who it thinks the client is."""

import types

from inference import cli, hosts, remote, server

MODEL_HOST = {"ssh": "user@mac.example.com", "address": "mac.example.com",
              "checkout": "/Users/user/inference-server", "machine": "m4"}

MAC = {"key": "m4", "chip": "M4 Max", "ram_gb": 96, "cores": 16, "host": "localhost",
       "threads": 12, "batch": 4096, "ubatch": 2048, "cache_ram_mb": 32768}


def _engine(tmp_path, *flags):
    args = cli.build_parser().parse_args([
        "engine", "--port", "0",
        "--trace-dir", str(tmp_path / "traces"), "--log-dir", str(tmp_path / "logs"),
        "--usage-db", str(tmp_path / "usage.db"), "--settings", str(tmp_path / "settings.json"),
        "--no-ui", "--quiet", *flags])
    return server.Engine(args)


def test_no_model_host_means_this_machine(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MODEL_HOST", None, raising=False)
    engine = _engine(tmp_path)
    assert isinstance(engine.supervisor.host, hosts.LocalHost)
    assert engine.upstream_host == "127.0.0.1"


def test_a_model_host_sends_models_to_the_mac(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MODEL_HOST", MODEL_HOST, raising=False)
    monkeypatch.setattr(remote.RemoteHost, "machine", lambda self: MAC)
    engine = _engine(tmp_path)
    assert isinstance(engine.supervisor.host, remote.RemoteHost)
    assert engine.upstream_host == "mac.example.com"
    assert engine.machine.ram_gb == 96


def test_an_unreachable_mac_falls_back_to_the_profile(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MODEL_HOST", MODEL_HOST, raising=False)

    def asleep(self):
        raise remote.RemoteError("ssh: connect to host mac.example.com port 22: Operation timed out")

    monkeypatch.setattr(remote.RemoteHost, "machine", asleep)
    engine = _engine(tmp_path)
    assert engine.machine.key == "m4"
    assert "qwen3.8" in [i.alias for i in engine.infos if i.routable]


def _handler(monkeypatch, trust_proxy, headers):
    monkeypatch.setattr(server.EngineHandler, "engine",
                        types.SimpleNamespace(trust_proxy=trust_proxy), raising=False)
    handler = server.EngineHandler.__new__(server.EngineHandler)
    handler.headers = headers
    handler.client_address = ("10.244.1.9", 50123)
    return handler


def test_forwarded_for_is_ignored_unless_the_proxy_is_trusted(monkeypatch):
    handler = _handler(monkeypatch, False, {"X-Forwarded-For": "192.168.1.50"})
    assert handler.client_ip() == "10.244.1.9"


def test_behind_a_trusted_proxy_the_first_forwarded_address_is_the_client(monkeypatch):
    handler = _handler(monkeypatch, True, {"X-Forwarded-For": "192.168.1.50, 10.244.0.1"})
    assert handler.client_ip() == "192.168.1.50"


def test_a_trusted_proxy_that_sends_nothing_leaves_the_socket_peer(monkeypatch):
    assert _handler(monkeypatch, True, {}).client_ip() == "10.244.1.9"

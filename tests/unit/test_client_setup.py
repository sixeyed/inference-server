"""What the Config and Clients tabs offer, from the engine's side."""

import types

from inference import client_setup, clients, dashboard, models, server
from inference.machine import Machine
from inference.models import ModelInfo

M4 = Machine(key="m4", chip="M4 Max", ram_gb=128, cores=16, host="192.168.1.10",
             threads=12, batch=4096, ubatch=2048, cache_ram_mb=32768)

HTTPS = {"label": "inference.example.com (HTTPS)", "base_url": "https://inference.example.com/v1",
         "note": "private CA"}
HTTP = {"label": "192.168.1.20 (HTTP)", "base_url": "http://192.168.1.20:8084/v1",
        "note": "LAN only"}


def _info(alias, *, supported=True, deprecated=None, test_only=False, ctx=262144,
          vision=False):
    return ModelInfo(alias=alias, family="", summary="", engine="llama",
                     supported=supported, reason=None, deprecated=deprecated,
                     client_ids=(alias,), hf_repo=None, ctx=ctx,
                     test_only=test_only, vision=vision)


class FakeUsage:
    def __init__(self, seen):
        self.seen = seen

    def _query(self, sql, params=()):
        return [{"client": c} for c in sorted(self.seen)]


def _engine(seen=(), toolsets=None, prompts=None, loops=None, endpoints=None, infos=()):
    return types.SimpleNamespace(
        settings=types.SimpleNamespace(prompts=prompts or {}, toolsets=toolsets or {},
                                       loops=loops or {}),
        usage=FakeUsage(seen) if seen is not None else None,
        machine=M4, port=8084, api_key="none", infos=list(infos),
        client_endpoints=endpoints)


# ---- Config tab: which clients get rows ------------------------------------
def test_the_main_coding_clients_are_offered_before_any_traffic():
    offered = dashboard.Dashboard(_engine(seen=[]))._clients_offered()
    assert offered == list(clients.OFFERED) == ["vscode", "cline", "opencode"]


def test_seen_and_configured_clients_join_without_duplicates():
    engine = _engine(seen=["curl", "vscode"], toolsets={"harness": "vscode-core.txt"},
                     loops={"cline": 5})
    assert dashboard.Dashboard(engine)._clients_offered() == [
        "vscode", "cline", "opencode", "curl", "harness"]


def test_no_usage_database_still_offers_the_main_clients():
    assert dashboard.Dashboard(_engine(seen=None))._clients_offered()[:3] == [
        "vscode", "cline", "opencode"]


# ---- Clients tab ------------------------------------------------------------
def test_the_configured_addresses_are_offered_in_order():
    payload = client_setup.payload(_engine(endpoints=[HTTPS, HTTP]))
    assert [e["base_url"] for e in payload["endpoints"]] == [
        "https://inference.example.com/v1", "http://192.168.1.20:8084/v1"]
    assert payload["endpoints"][0]["note"] == "private CA"
    assert payload["api_key"] == "none"


def test_with_no_configured_addresses_this_machine_is_the_address():
    payload = client_setup.payload(_engine(endpoints=None))
    assert [(e["label"], e["base_url"]) for e in payload["endpoints"]] == [
        ("192.168.1.10", "http://192.168.1.10:8084/v1")]


def test_only_live_models_are_offered_to_clients():
    infos = [_info("qwen3.8", vision=True), _info("muse-glimmer", ctx=131072, vision=True),
             _info("qwen3-tiny", test_only=True), _info("gemma-4-31b", deprecated="old"),
             _info("too-big", supported=False)]
    payload = client_setup.payload(_engine(infos=infos))
    assert payload["models"] == [
        {"id": "muse-glimmer", "ctx": 131072, "vision": True},
        {"id": "qwen3.8", "ctx": 262144, "vision": True}]


def test_the_vscode_block_is_ready_to_paste():
    infos = [_info("qwen3.8", vision=True), _info("muse-glimmer", ctx=131072, vision=True),
             _info("kat-coder2.5")]
    endpoint = client_setup.payload(_engine(endpoints=[HTTP], infos=infos))["endpoints"][0]
    [provider] = endpoint["vscode"]
    assert provider["name"] == "192.168.1.20 (HTTP)"
    assert provider["vendor"] == "customendpoint"
    assert provider["apiKey"] == "none"
    assert provider["apiType"] == "chat-completions"
    by_id = {m["id"]: m for m in provider["models"]}
    assert list(by_id) == ["kat-coder2.5", "muse-glimmer", "qwen3.8"]
    qwen = by_id["qwen3.8"]
    assert qwen == {"id": "qwen3.8", "name": "qwen3.8",
                    "url": "http://192.168.1.20:8084/v1/chat/completions",
                    "toolCalling": True, "vision": True,
                    "maxInputTokens": 249000, "maxOutputTokens": 16000}
    assert by_id["muse-glimmer"]["maxInputTokens"] == 124000
    assert by_id["kat-coder2.5"]["vision"] is False


def test_a_trailing_slash_on_the_base_url_does_not_double_up():
    endpoint = dict(HTTP, base_url="http://192.168.1.20:8084/v1/")
    [provider] = client_setup.payload(_engine(endpoints=[endpoint], infos=[_info("qwen3.8")]))[
        "endpoints"][0]["vscode"]
    assert provider["models"][0]["url"] == "http://192.168.1.20:8084/v1/chat/completions"


def test_the_header_shows_the_first_configured_address():
    engine = _engine(endpoints=[HTTPS, HTTP])
    engine.supervisor = types.SimpleNamespace(status=lambda: {"state": "idle"},
                                              rss_mb=lambda: None)
    assert dashboard.Dashboard(engine)._state()["endpoint"] == "https://inference.example.com/v1"


# ---- where the facts come from ------------------------------------------------
def test_vision_comes_from_the_model_files():
    infos = {i.alias: i for i in models.describe(M4)}
    assert infos["qwen3.8"].vision is True
    assert infos["muse-glimmer"].vision is True
    assert infos["kat-coder2.5"].vision is False
    assert infos["qwen3-tiny"].vision is False


def test_the_engine_reads_client_endpoints_from_config(monkeypatch, tmp_path):
    from inference import cli
    monkeypatch.setattr(server, "MODEL_HOST", None)
    monkeypatch.setattr(server, "CLIENT_ENDPOINTS", [HTTPS, HTTP], raising=False)
    args = cli.build_parser().parse_args([
        "engine", "--port", "0", "--trace-dir", str(tmp_path / "t"),
        "--log-dir", str(tmp_path / "l"), "--usage-db", str(tmp_path / "u.db"),
        "--settings", str(tmp_path / "s.json"), "--no-ui", "--quiet"])
    assert server.Engine(args).client_endpoints == [HTTPS, HTTP]

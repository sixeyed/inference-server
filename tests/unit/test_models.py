"""Model registry rules that are not about name resolution."""

from inference import models
from inference.machine import Machine

M4 = Machine(key="m4", chip="M4 Max", ram_gb=128, cores=16, host="localhost",
             threads=12, batch=4096, ubatch=2048, cache_ram_mb=32768)


def test_the_tiny_model_is_test_only_and_still_routable():
    infos = {info.alias: info for info in models.describe(M4)}
    tiny = infos["qwen3-tiny"]
    assert tiny.test_only
    assert tiny.routable
    assert models.build_index(list(infos.values()))["qwen3-tiny"] == "qwen3-tiny"


def test_no_real_model_is_test_only():
    assert [i.alias for i in models.describe(M4) if i.test_only] == ["qwen3-tiny"]

"""The engine's API key reaches the model server that checks it.

MTPLX is the one server that checks a key. It used to be started with the key
from config.json whatever --api-key the engine was given, so the engine probed
and relayed with one key while the server expected another.
"""

import time

from inference import cli, models, process, usage
from inference.machine import Options, machine_from_profile
from inference.supervisor import State, Supervisor


def test_an_mtplx_spec_is_built_with_the_runs_key():
    machine = machine_from_profile("m4")
    module = models.load_all()["qwen3.8"]
    spec, _ = process.prepare(module, machine, Options(api_key="s3cret"))
    assert spec.api_key == "s3cret"
    argv = spec.argv(machine, 8083)
    assert argv[argv.index("--api-key") + 1] == "s3cret"


def test_a_server_that_takes_no_key_is_built_as_before():
    machine = machine_from_profile("m4")
    module = models.load_all()["qwen3-tiny"]
    spec, _ = process.prepare(module, machine, Options(api_key="s3cret"))
    assert not hasattr(spec, "api_key")


def test_run_takes_an_api_key():
    args = cli.build_parser().parse_args(["run", "qwen3.8", "--api-key", "s3cret"])
    assert args.api_key == "s3cret"


def test_a_model_name_that_is_not_a_string_resolves_to_nothing():
    index = models.build_index(models.describe(machine_from_profile("m4"),
                                               models.load_all()))
    assert models.resolve(42, index) is None
    assert models.resolve({"id": "qwen3.8"}, index) is None


# ------------------------------------------------------------
class _SlowSupervisor:
    """rss_mb() takes a while - like an SSH call to a model host."""

    def __init__(self):
        self.calls = 0

    def rss_mb(self):
        self.calls += 1
        time.sleep(0.05)
        return 100.0


def test_a_swap_stops_the_previous_sampler(tmp_path):
    db = usage.UsageDB(str(tmp_path / "u.db"), sample_interval=0.01)
    db.attach(_SlowSupervisor())
    db._start_sampling("old")
    time.sleep(0.03)        # the old sampler is inside rss_mb()
    db._start_sampling("new")
    time.sleep(0.3)
    db._stop_sampling()
    time.sleep(0.1)
    late = db._query("SELECT COUNT(*) AS n FROM memory WHERE alias = 'old' "
                     "AND ts > (SELECT MIN(ts) FROM memory WHERE alias = 'new')")
    assert late[0]["n"] <= 1  # at most the sample already in flight
    db.close()


def test_a_server_that_died_is_recorded_as_unloaded(tmp_path):
    machine = machine_from_profile("m4")
    registry = models.load_all()
    sup = Supervisor(machine, registry, models.describe(machine, registry),
                     Options(port=8083), port=8083, log_dir=str(tmp_path), quiet=True)
    unloaded = []
    sup.on_unload = unloaded.append

    class Dead:
        alias, engine, adopted, upstream_model_id = "qwen3-tiny", "llama", False, None
    sup.current, sup.state = Dead(), State.READY
    sup._alive = lambda: False
    sup.status()
    assert [c.alias for c in unloaded] == ["qwen3-tiny"]
    assert sup.current is None

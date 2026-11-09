"""The machine, when there is no Mac to ask."""

import json

import pytest

from inference import cli, machine


def test_off_mac_the_named_profile_stands_in(monkeypatch):
    monkeypatch.setattr(machine.sys, "platform", "linux")
    monkeypatch.setenv("INFERENCE_MACHINE", "m4")
    detected = machine.detect_machine()
    assert detected.key == "m4"
    # qwen3.8 refuses a machine under its RAM floor, so a profile with no RAM
    # would quietly drop the default model from the model list.
    assert detected.ram_gb == 128


def test_off_mac_with_no_profile_still_refuses(monkeypatch):
    monkeypatch.setattr(machine.sys, "platform", "linux")
    monkeypatch.delenv("INFERENCE_MACHINE", raising=False)
    with pytest.raises(SystemExit):
        machine.detect_machine()


def test_an_unknown_profile_name_is_the_unknown_profile():
    assert machine.machine_from_profile("m9").key == "unknown"


def test_machine_command_prints_json(monkeypatch, capsys):
    monkeypatch.setattr(cli, "detect_machine",
                        lambda: machine.machine_from_profile("m4"))
    assert cli.main(["machine"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["key"] == "m4" and printed["cores"] == 16


# ---------------------------------------- chip generations, measured and assumed
@pytest.mark.parametrize("brand, key, chip", [
    ("Apple M4 Max", "m4", "M4 Max"),        # measured: the profile names it
    ("Apple M1 Max", "m1", "M1 Max"),
    ("Apple M2", "m2", "M2"),
    ("Apple M5 Max", "m4", "Apple M5 Max"),  # assumed: tuned like an M4, says M5
    ("Apple M3 Pro", "m2", "Apple M3 Pro"),  # assumed: tuned like an M2, says M3
    ("Apple M9 Ultra", "unknown", "Apple M9 Ultra"),
])
def test_each_chip_generation_maps_to_a_profile(monkeypatch, brand, key, chip):
    values = {"machdep.cpu.brand_string": brand, "machdep.cpu.core_count": "12",
              "hw.memsize": str(64 * 1024**3)}
    monkeypatch.setattr(machine.sys, "platform", "darwin")
    monkeypatch.setattr(machine, "_sysctl", lambda name: values[name])
    detected = machine.detect_machine()
    assert (detected.key, detected.chip) == (key, chip)

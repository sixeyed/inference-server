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

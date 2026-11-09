"""Reading a build number out of `llama-server --version`.

llama.cpp changed its version scheme at 0.3.0. It used to report

    version: 10470 (34af94cd9)

and now reports

    version: 0.3.0 (build 10621, commit c1d0e7a00)

A parser anchored on `version:` reads the new form as build 0, because `0.3.0`
starts with a zero. That is only ever displayed - the capability gates ask
`--help` about specific flags and do not care - so it degraded a warning
message rather than breaking a gate. It is still wrong, and a wrong build
number in a warning telling you to upgrade is worse than no number at all.
"""
import re

from inference import engines


def _build(text, monkeypatch):
    monkeypatch.setattr(engines.subprocess, "run",
                        lambda *a, **k: type("R", (), {"stdout": text,
                                                       "stderr": ""})())
    return engines.llama_build("llama-server")


def test_the_scheme_llama_cpp_uses_now(monkeypatch):
    assert _build("version: 0.3.0 (build 10621, commit c1d0e7a00)\n",
                  monkeypatch) == 10621


def test_the_scheme_it_used_before(monkeypatch):
    """Traces and results in this repo were produced under it, so a checkout
    pointed at an older llama.cpp has to keep reporting truthfully."""
    assert _build("version: 10470 (34af94cd9)\n", monkeypatch) == 10470


def test_a_dev_build(monkeypatch):
    assert _build("version: 0.1.1-dev (build 10470, commit 34af94cd9)\n",
                  monkeypatch) == 10470


def test_something_unparseable_is_none_rather_than_zero(monkeypatch):
    """None prints as '?'. Zero prints as a build number that never existed."""
    assert _build("llama-server, a friendly binary\n", monkeypatch) is None

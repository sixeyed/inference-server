"""The dashboard's JavaScript must at least execute.

A top-level error - a `let` read before its declaration, a missing element -
aborts the whole script. The page still renders, because the HTML is static, so
it looks fine while every panel stays empty. That happened once and was only
caught by eye, which is not a test strategy.
"""

import os
import shutil
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(os.path.dirname(os.path.dirname(HERE)),
                   "inference", "static", "app.js")


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_app_js_runs_without_a_top_level_error():
    result = subprocess.run(
        ["node", os.path.join(HERE, "domcheck.js"), APP],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr[-1500:]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_app_js_parses():
    result = subprocess.run(["node", "--check", APP],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr[-800:]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_every_data_attribute_has_a_listener():
    """A row rendered with data-x but no querySelectorAll("[data-x]") is dead.

    That is not a crash and not a failed request - the element simply does
    nothing when clicked, which no server-side check can see. It has happened
    once already, with the turn rows on the session page.
    """
    import re

    source = open(APP).read()
    rendered = set(re.findall(r'data-([a-z-]+)="\$\{', source))
    bound = set(re.findall(r'querySelectorAll\("\[data-([a-z-]+)\]"\)', source))
    orphans = rendered - bound
    assert not orphans, f"rendered but never bound: {sorted(orphans)}"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_stored_timestamps_parse():
    """Stored timestamps carry "+00:00"; appending "Z" makes them invalid.

    That produced "NaNh NaNm" in the session header. Anything parsing one has
    to go through parseTs.
    """
    script = (
        "const iso='2026-08-20T12:21:43+00:00';"
        "function parseTs(s){if(!s)return null;"
        "const d=new Date(/[Z+]|-\\\\d\\\\d:\\\\d\\\\d$/.test(s)?s:s+'Z');"
        "return isNaN(d)?null:d;}"
        "if(!parseTs(iso))throw new Error('offset form failed');"
        "if(!parseTs('2026-08-20T12:21:43'))throw new Error('naive form failed');"
        "if(isNaN(new Date(iso+'Z'))===false)throw new Error('the old bug is gone?');"
    )
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-500:]


def test_only_parse_ts_appends_a_z():
    """Everything else must go through parseTs.

    Appending "Z" to a timestamp that already ends in "+00:00" yields an
    invalid date, which renders as NaN rather than failing loudly. parseTs does
    it conditionally; nowhere else should do it at all.
    """
    offenders = [line.strip() for line in open(APP)
                 if '+ "Z"' in line and "test(iso)" not in line]
    assert not offenders, offenders

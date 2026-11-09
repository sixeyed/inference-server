"""about.py - the dashboard footer's name, version, repository and commit."""

from inference import about

PYPROJECT = '''
[project]
name = "inference-server"
version = "1.2.3"
description = "x"

[project.urls]
Repository = "https://github.com/o/r"

[tool.other]
version = "9.9.9"
'''


def test_the_footer_fields_come_from_pyproject():
    assert about.parse_pyproject(PYPROJECT) == {
        "name": "inference-server", "version": "1.2.3",
        "repository": "https://github.com/o/r"}


def test_a_version_in_another_table_is_not_the_project_version():
    assert about.parse_pyproject('[tool.x]\nversion = "9"\n')["version"] is None


def test_no_pyproject_is_all_none_not_an_error():
    assert about.parse_pyproject("") == {
        "name": None, "version": None, "repository": None}


def test_this_checkout_reports_its_own_pyproject(monkeypatch):
    monkeypatch.setenv("INFERENCE_GIT_SHA", "abc1234")
    about.about.cache_clear()
    try:
        info = about.about()
    finally:
        about.about.cache_clear()
    assert info["name"] and info["version"] and info["sha"] == "abc1234"


def test_an_image_built_without_a_sha_asks_git(monkeypatch):
    monkeypatch.setenv("INFERENCE_GIT_SHA", "unknown")
    monkeypatch.setattr(about, "checkout_sha", lambda: "def5678")
    about.about.cache_clear()
    try:
        assert about.about()["sha"] == "def5678"
    finally:
        about.about.cache_clear()

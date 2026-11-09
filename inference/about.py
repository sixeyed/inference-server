"""What this engine is: the name, version and repository from the checkout's
pyproject.toml, and the commit it was built from - for the dashboard footer.

Read from pyproject.toml rather than written into the package, because more than
one repository ships this same package under its own name and version, and the
package has to stay identical between them. The container image copies
pyproject.toml in for the same reason.

Parsed by hand: tomllib is 3.11+, the floor here is 3.9, and three string keys
do not justify a dependency.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache

from .process import checkout_sha

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_SECTION = re.compile(r"^\s*\[([^\]]+)\]\s*$")
_STRING = re.compile(r'^\s*([A-Za-z0-9_-]+)\s*=\s*"([^"]*)"')
_WANTED = {("project", "name"): "name",
           ("project", "version"): "version",
           ("project.urls", "Repository"): "repository"}


def parse_pyproject(text: str) -> dict:
    """name, version and repository - each None when the file does not say."""
    found = dict.fromkeys(_WANTED.values())
    section = None
    for line in text.splitlines():
        header = _SECTION.match(line)
        if header:
            section = header.group(1).strip()
            continue
        pair = _STRING.match(line)
        if pair and (section, pair.group(1)) in _WANTED:
            found[_WANTED[(section, pair.group(1))]] = pair.group(2)
    return found


@lru_cache(maxsize=1)
def about() -> dict:
    """Read once: none of it changes while the engine runs."""
    try:
        with open(os.path.join(ROOT, "pyproject.toml"), encoding="utf-8") as handle:
            info = parse_pyproject(handle.read())
    except OSError:
        info = parse_pyproject("")
    # The image bakes the sha in at build time (and says "unknown" if the
    # build did not pass one); a checkout asks git.
    sha = os.environ.get("INFERENCE_GIT_SHA")
    info["sha"] = sha if sha and sha != "unknown" else checkout_sha()
    return info

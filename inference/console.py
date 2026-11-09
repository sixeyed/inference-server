"""Terminal output: colour, and quoting a command so it can be pasted back.

Colour is switched off when stdout is not a terminal or NO_COLOR is set, so
piping the engine to a log file yields plain text.
"""

from __future__ import annotations

import os
import sys

_TTY = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text


def blue(t):
    return _c("34", t)


def cyan(t):
    return _c("36", t)


def green(t):
    return _c("32", t)


def yellow(t):
    return _c("33", t)


def red(t):
    return _c("31", t)


def grey(t):
    return _c("90", t)


class Unsupported(Exception):
    """Raised by a model's spec() when it won't fit the detected machine."""


def quote(arg: str) -> str:
    """Quote an argv element only when it needs it, so a printed command can be
    copied straight back into a shell."""
    return f"'{arg}'" if any(ch in arg for ch in " {}\"") else arg

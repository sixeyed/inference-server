"""The suite runs against the built-in defaults, never this machine's config.

Loaded before anything under tests/ imports `inference`, which is when
inference.config reads config.json and the environment - once, at import. A
checkout's own config.json (LAN binds, a weights drive, tuned profiles) or an
INFERENCE_* variable in the shell would otherwise change what the tests see,
so they would pass in one checkout and fail in another.
"""

import os

os.environ["INFERENCE_CONFIG"] = os.path.join(os.path.dirname(__file__),
                                              "no-such-config.json")
for name in [n for n in os.environ if n.startswith("INFERENCE_")]:
    # INFERENCE_MACHINE is how the image's test stage stands in a profile off a
    # Mac, so it stays; everything else is this shell's own setting.
    if name not in ("INFERENCE_CONFIG", "INFERENCE_MACHINE"):
        del os.environ[name]

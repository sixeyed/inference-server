"""Deployment settings: the built-in defaults, and the file that overrides them.

Two kinds of value live here and they behave differently.

**Environment** - where the weights are, what this machine is called on the LAN,
how many threads it has - varies per box and belongs to whoever runs it, not to
the repo. Those come from `config.json`, which is gitignored; the defaults below
are the fallback when there is no file, and `config.example.json` is the
committed template.

**Policy** - the sandbox profile, the on-disk layout - is the same everywhere and
stays in code, where the comments explaining it can live next to it.

Precedence is environment variable, then `config.json`, then the default below.
An environment variable is a deliberate act at the moment of running something,
so it wins over a file written earlier.

Model-specific settings are not here at all: those live in that model's own file
under `models/`, which is the whole point of the `models/` layout.
"""

from __future__ import annotations

import json
import os

# The file itself. Kept out of git because it carries LAN addresses and a path
# to somebody's disk - see config.example.json for the template.
CONFIG_FILE = os.environ.get(
    "INFERENCE_CONFIG",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "config.json"))


def _load() -> dict:
    try:
        with open(CONFIG_FILE, encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        # Loud, but not fatal. Running with silently-default settings after
        # someone typo'd a path is how you spend an afternoon wondering why the
        # weights are being re-downloaded.
        print(f"WARNING: {CONFIG_FILE} could not be read ({exc}); using defaults")
        return {}


_CONFIG = _load()


def _setting(key: str, default, env: str | None = None):
    if env and os.environ.get(env):
        return os.environ[env]
    return _CONFIG.get(key, default)


# ============================================================
# Where the weights live
# ============================================================
# The default is the home directory, which is where HuggingFace and MTPLX put
# things if nobody says otherwise - so a fresh clone works with no config file.
# Setting weights_root to an external volume is a capacity decision rather than
# a speed one: a 27B model is 20GB and the internal disk is the scarce one
# (measured on an M4 Max: internal 6.29 GB/s, external 3.85 GB/s).
#
# The cost is confined to model load. MTPLX reads safetensors into unified
# memory and closes them; llama.cpp mmaps GGUF and could page back in under
# memory pressure, which is what LLAMA_LOAD_MODE below exists to prevent.
WEIGHTS_ROOT = os.path.expanduser(
    _setting("weights_root", "~", env="INFERENCE_WEIGHTS_ROOT"))

# llama-server and mlx_lm.server both resolve through the HuggingFace cache
# layout, so one variable moves both. find_in_hf_cache() reads HF_HOME and the
# child process inherits it, so the launcher and the engine cannot disagree
# about where the weights are.
HF_HOME = os.path.expanduser(
    _setting("hf_home", os.path.join(WEIGHTS_ROOT, ".cache", "huggingface")
             if WEIGHTS_ROOT == os.path.expanduser("~")
             else os.path.join(WEIGHTS_ROOT, "huggingface")))

# MTPLX keeps its own flat cache and ignores the HuggingFace one.
MTPLX_CACHE_DIR = os.path.expanduser(
    _setting("mtplx_cache_dir",
             os.path.join(WEIGHTS_ROOT, ".mtplx", "models")
             if WEIGHTS_ROOT == os.path.expanduser("~")
             else os.path.join(WEIGHTS_ROOT, "mtplx")))

# The session bank stays on the internal disk deliberately. Unlike the weights
# it is read *during* serving - restored on prefill - and prefill is the current
# bottleneck. It is also budgeted to grow to 48GB, so it is not on the external
# for capacity reasons either: it is the one thing here that wants the fast disk.
MTPLX_SESSION_CACHE_DIR = os.path.expanduser(
    _setting("mtplx_session_cache_dir", "~/.mtplx/session-bank"))

# llama.cpp deprecated --mmap/--mlock in favour of --load-mode. `mmap+mlock`
# maps the file then pins it in RAM, so weights on a slower external disk are
# read once at load and never paged back in mid-request. That costs locked RAM,
# which is affordable on 128GB and is not on a smaller machine - drop to `auto`
# there.
LLAMA_LOAD_MODE = _setting("llama_load_mode", "mmap+mlock")


def apply_weights_env() -> None:
    """Point the HuggingFace cache at the configured root.

    setdefault, not assignment - an HF_HOME already in the environment is a
    deliberate act by whoever set it, and silently overriding it would move
    someone's cache out from under them.
    """
    os.environ.setdefault("HF_HOME", HF_HOME)


def weights_root_missing() -> str | None:
    """An error message if the weights volume is not mounted, else None.

    Worth failing on rather than letting through: with the volume absent,
    find_in_hf_cache() returns None and LlamaServerSpec falls back to
    `-hf <repo>`, which silently starts re-downloading tens of gigabytes to a
    path that is not there.
    """
    if os.path.isdir(WEIGHTS_ROOT):
        return None
    if not WEIGHTS_ROOT.startswith("/Volumes/"):
        # Not a drive that can be unplugged - just a directory a fresh setup
        # has not made yet. Make it, and the first pull fills it.
        try:
            os.makedirs(WEIGHTS_ROOT, exist_ok=True)
            return None
        except OSError as exc:
            return f"weights root {WEIGHTS_ROOT} could not be created ({exc})"
    volume = os.path.join("/Volumes", WEIGHTS_ROOT.split("/")[2])
    mounted = os.path.ismount(volume)
    detail = ("the volume is mounted but the directory is missing"
              if mounted else f"{volume} is not mounted")
    return (f"weights root {WEIGHTS_ROOT} is unavailable - {detail}. "
            f"Plug the drive in, or set weights_root in {CONFIG_FILE}.")


# ============================================================
# Per-machine profile
# ============================================================
# Keyed by the chip detect_machine() finds. The host is what clients on the LAN
# connect to; the rest is tuning measured on these specific boxes - see docs/.
#
# Overridden per key from config.json, so a differently-sized Mac of the same
# chip changes a file rather than the code. Anything absent falls back to these.
# ram_gb and cores are only read off-Mac (machine_from_profile) - on a Mac they
# are detected.
MACHINE_PROFILES: dict[str, dict] = {
    "m4": dict(chip="M4 Max", host="localhost", threads=12,
               batch=4096, ubatch=2048, cache_ram_mb=32768,
               ram_gb=128, cores=16),
    "m1": dict(chip="M1 Max", host="localhost", threads=10,
               batch=2048, ubatch=512, cache_ram_mb=16384,
               ram_gb=64, cores=10),
    "m2": dict(chip="M2", host="localhost", threads=6,
               batch=1024, ubatch=512, cache_ram_mb=4096,
               ram_gb=24, cores=8),
    "unknown": dict(chip="", host="localhost", threads=6,
                    batch=1024, ubatch=512, cache_ram_mb=4096,
                    ram_gb=0, cores=0),
}
def _apply_machine_overrides() -> None:
    # A function rather than a module-level loop, whose variables would be
    # module globals only when config.json names machines - indistinguishable,
    # to test_exit_summary's scope check, from a name that was never imported.
    for key, overrides in (_CONFIG.get("machines") or {}).items():
        MACHINE_PROFILES.setdefault(key, dict(MACHINE_PROFILES["unknown"]))
        MACHINE_PROFILES[key].update(overrides)


_apply_machine_overrides()


# ============================================================
# Ports and auth
# ============================================================
# Where model servers listen. One model runs at a time, so they all share it.
DEFAULT_PORT = int(_setting("model_port", 8083))

# Where the engine listens. Kept off the model port so `run` and any benchmark
# can still talk to a model server directly.
DEFAULT_LISTEN_PORT = int(_setting("engine_port", 8084))

# Where the engine listens, and where model servers bind. Loopback by default:
# a fresh clone serves this machine and nothing else. Set `bind` to 0.0.0.0 to
# serve the network (and set a real api_key when you do). `model_bind` only
# needs opening up when the engine reaches the model servers from another
# machine - an engine in a container on a different box, say.
BIND_HOST = _setting("bind", "127.0.0.1", env="INFERENCE_BIND")
MODEL_BIND = _setting("model_bind", "127.0.0.1", env="INFERENCE_MODEL_BIND")

# Take each client's address from X-Forwarded-For. Only behind a proxy you run
# (an ingress): anywhere else a client can claim any address it likes, and the
# address is part of how sessions are told apart.
TRUST_PROXY = str(_setting("trust_proxy", False, env="INFERENCE_TRUST_PROXY")
                  ).strip().lower() in ("1", "true", "yes", "on")

# Fetch missing weights before a launch, outside the sandbox, rather than
# refusing it - see weights.py. Off brings back the refusal.
AUTO_PULL = str(_setting("auto_pull", True, env="INFERENCE_AUTO_PULL")
                ).strip().lower() not in ("0", "false", "no", "off")

# MTPLX refuses a non-localhost bind without an API key, so one is always
# passed - which keeps `bind: 0.0.0.0` working. It requires *a* key, not a secret one -
# so this is literally the string the clients here already send in their
# API-key field, which keeps every client config working untouched.
#
# The default is deliberately not a secret, and with it the engine checks
# nothing. Set a real api_key (config.json or INFERENCE_API_KEY) and the engine
# enforces it: every request but /health and the dashboard's static files
# needs `Authorization: Bearer <key>`. See EngineHandler._authorized.
DEFAULT_API_KEY = _setting("api_key", "none", env="INFERENCE_API_KEY")

# The key that means "no auth". Kept apart from DEFAULT_API_KEY, which is
# whatever this machine's config says.
NO_AUTH_KEY = "none"


# ============================================================
# Model host
# ============================================================
# Where model servers run when that is not this machine - an engine in a
# container or a pod, driving a Mac over SSH. None means here, which is how the
# engine ran before it could run anywhere else. Keys: see remote.ModelHostConfig.
MODEL_HOST = _CONFIG.get("model_host")

# The addresses the dashboard's Clients tab offers, in order: a list of
# {"label", "base_url", "note"}. None means this machine's own address - see
# client_setup.py.
CLIENT_ENDPOINTS = _CONFIG.get("client_endpoints")


# ============================================================
# Policy - the same everywhere, so it stays in code
# ============================================================
# Egress sandbox.
#
# A model server needs the network exactly once - to download weights. After
# that it only ever answers on its own port, so denying it outbound sockets
# costs nothing and turns "I read the source and found no exfiltration" into
# "it cannot, whatever the source says". That matters for engines like MTPLX
# that are third-party binaries running arbitrary code on your prompts.
#
# Passed to `sandbox-exec -p`. Order matters: (allow default) first, then the
# deny, then the narrow re-allows:
#   unix sockets   - the dynamic loader and IPC go through /private/var/run
#   network-bind   - listening on HOST:PORT is a bind, not an outbound
#   localhost      - loopback callers, and the server talking to itself
SANDBOX_PROFILE = """(version 1)
(allow default)
(deny network-outbound)
(allow network-outbound (regex #"^/private/var/run/"))
(allow network-bind)
(allow network-inbound)
(allow network-outbound (remote ip "localhost:*"))
"""

DEFAULT_TRACE_DIR = "traces"
DEFAULT_LOG_DIR = "logs"
DEFAULT_USAGE_DB = "usage.db"

# Live proxy settings, editable from the dashboard and reloaded on restart.
# Distinct from config.json: this is behaviour you change mid-session, that one
# is where the machine is.
DEFAULT_SETTINGS_FILE = "settings.json"

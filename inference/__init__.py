"""Run local coding models on Apple Silicon, one at a time, on demand.

The engine is the front door: it reads the `model` field off each request,
starts that model if it isn't the one running, and forwards. Everything a model
needs to be deployed - HF repo, quant per machine, context, sampling, thinking
switches, speculative decoding - lives in one file under `models/`, so adding
the next one is a single new file and nothing else.
"""

from .config import DEFAULT_API_KEY, DEFAULT_LISTEN_PORT, DEFAULT_PORT

__all__ = ["DEFAULT_API_KEY", "DEFAULT_LISTEN_PORT", "DEFAULT_PORT"]

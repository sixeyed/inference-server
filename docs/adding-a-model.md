# Adding a new model

A model is one Python file in `inference/models/`. The registry finds it by
scanning the package, so there is nothing to register. Add the file, and the
model shows up in `inference models`, in `GET /v1/models` and in every client
picker.

Each file is deliberately self-contained: the HuggingFace repo, the quant for
each machine, context, sampling, thinking switches and speculative decoding all
live in that one file, with the comments explaining why. There's duplication
between model files by design. Open two side by side and you see exactly where
two deployments differ.

## 1. Pick the engine

| Engine | Use it for | Spec |
|---|---|---|
| llama.cpp (`llama-server`) | GGUF quants — most models | `LlamaServerSpec` |
| MTPLX (`mtplx serve`) | MLX models packed for MTPLX's multi-token prediction | `MtplxServerSpec` |
| mlx-lm (`mlx_lm.server`) | plain MLX models | `MlxServerSpec` |

If a GGUF exists, start with llama.cpp. It's the best supported of the three,
and the only one that can load a single quant out of a repo that holds many.
[backends.md](backends.md#engines) has what each engine does differently.

## 2. Write the file

The smallest real example is `inference/models/qwen3_tiny.py`, the 0.6B test
fixture. A new llama.cpp model looks like this:

```python
"""My-Model-14B - one line on what it is and why it's here.

Engine:  llama-server (llama.cpp)
Weights: https://huggingface.co/someone/My-Model-14B-GGUF
"""

from ..machine import Machine, Options, Unsupported
from ..specs import LlamaServerSpec

ALIAS = "my-model"            # what clients send, and what you pass to `inference run`
FAMILY = "mine"               # grouping for `inference run --family`
SUMMARY = "14B dense - what it is good for, in one line"

HF_REPO = "someone/My-Model-14B-GGUF"

# The quant and context per machine, keyed by profile: m4, m1, m2, and unknown
# for anything else. M5 Macs use m4 and M3 Macs use m2 - an assumption, see
# backends.md. Leave a key out and the model is listed as "no fit" there.
MACHINES = {
    "m4": dict(hf_file="My-Model-14B-Q8_0.gguf", ctx=131072),
    "m1": dict(hf_file="My-Model-14B-Q6_K.gguf", ctx=65536),
    "unknown": dict(hf_file="My-Model-14B-Q4_K_M.gguf", ctx=16384),
}


def spec(machine: Machine, opts: Options) -> LlamaServerSpec:
    cfg = MACHINES.get(machine.key)
    if cfg is None:
        raise Unsupported(f"{ALIAS} has no config for {machine.chip}")
    return LlamaServerSpec(
        alias=ALIAS,
        hf_repo=HF_REPO,
        hf_file=cfg["hf_file"],
        ctx=opts.ctx or cfg["ctx"],
        temp=0.7, top_p=0.95, top_k=20,   # the model card's recommended sampling
    )
```

### What a model file must have

| Name | |
|---|---|
| `ALIAS` | the id clients send and the name for `inference run`. Unique — a duplicate fails at load. |
| `FAMILY` | grouping for `inference run --family` |
| `SUMMARY` | one line, shown in `inference models` and the dashboard |
| `spec(machine, opts)` | returns the engine's spec for this machine, or raises `Unsupported` if the model won't fit |

`spec()` is called for every model whenever the registry is listed, so keep it
cheap and free of side effects. Whether a model fits is decided by calling it,
so raise `Unsupported` with a reason rather than returning something that will
fail at load.

### What it may have

| Name | |
|---|---|
| `ENGINE` | `"mtplx"` or `"mlx"` when the model isn't llama.cpp. It's read before the spec exists, to skip the llama.cpp capability gate. |
| `CLIENT_MODEL_ID` | the id the *server* answers to, when that isn't the alias. mlx-lm has no `--alias`, so set this to `HF_REPO` for mlx-lm models. Clients still use the alias; the engine rewrites it. |
| `VISION` | `True` if the model takes images. |
| `FORMER_ALIASES` | a tuple of names this model used to have, so a rename doesn't break old client configs and traces. |
| `DEPRECATED` | why it was retired. The model drops out of `/v1/models` and routing, but `inference run` can still launch it. |
| `TEST_ONLY` | routable by name, but never listed. For fixtures like `qwen3-tiny`. |

## 3. Per engine

**llama.cpp.** `hf_file` names the exact quant, and it's an exact match: the
engine never substitutes a different quant, because that would silently change
what you're running. For a split GGUF, name the first part
(`...-00001-of-00003.gguf`); the pull fetches every part and llama-server loads
the rest. Speculative-decoding flags go in `spec_args`, not `extra_args`. If
the installed llama.cpp is too old to know them, they're dropped and the model
still starts.

**MTPLX.** Set `ENGINE = "mtplx"` and return
`MtplxServerSpec(hf_repo=..., model_id=ALIAS, ...)`. `model_id` makes MTPLX
answer to the alias, the way `--alias` does for llama.cpp. `qwen3_8.py` is the
worked example.

**mlx-lm.** Set `ENGINE = "mlx"`, `CLIENT_MODEL_ID = HF_REPO`, and return
`MlxServerSpec(hf_repo=..., ...)`.

## 4. Try it

```bash
python3 -m inference models                  # is it listed, and does it fit?
python3 -m inference run my-model --dry-run  # the exact launch command, nothing run
python3 -m inference pull my-model           # fetch the weights
python3 -m inference run my-model            # run it in the foreground
```

Then start the engine. The model is in the picker of any client pointed at it,
with no other change. Leave the egress sandbox on: the pull happens before
launch, outside it, so the model server never needs the network (see
[engine.md](engine.md#egress-sandbox)).

Run the suite before you commit: `python3 -m pytest -q`. The engine tests list
the registry, which imports every model file and calls its `spec()` for this
machine, so a file that fails to import or build fails there rather than at
load. `inference models` is the quickest check for the other machines: a
missing entry in `MACHINES` shows as "no fit" rather than an error.

## Defaults, renames and retirement

- **The default model** — the one `inference run` launches with no arguments,
  and the engine's fallback — is `DEFAULT_MODEL` in
  `inference/models/__init__.py`. A family's default is in `FAMILY_DEFAULTS`
  beside it.
- **Renaming** — change `ALIAS` and add the old name to `FORMER_ALIASES`.
- **Retiring** — set `DEPRECATED` to the reason, rather than deleting the file.
  The file records how earlier measurements were configured, and reproducing
  one needs that exact quant, context and sampling.

"""Locating weights on disk.

The sandbox guard refuses a launch when the weights are absent, so a wrong
answer here is not cosmetic: a false negative makes a working model
unloadable, and a false positive starts a server that then tries to download
through a sandbox that denies it.
"""

import os

from inference import process
from inference.specs import LlamaServerSpec, MlxServerSpec, MtplxServerSpec


def test_mtplx_looks_in_the_cache_the_spec_names(tmp_path):
    """Not a hardcoded ~/.mtplx.

    The cache moved to an external volume and the check went on reading the old
    path, so qwen3.8 read as missing and the guard refused a model that was
    sitting right there. The spec carries the directory the server is actually
    launched with; that is the one to ask.
    """
    cache = tmp_path / "mtplx"
    repo = "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed"
    spec = MtplxServerSpec(hf_repo=repo, model_id="qwen3.8", cache_dir=str(cache))

    assert not process.weights_are_local(spec), "nothing there yet"
    (cache / repo.replace("/", "--")).mkdir(parents=True)
    assert process.weights_are_local(spec), "present in the spec's own cache_dir"


def test_mtplx_ignores_the_huggingface_cache(tmp_path, monkeypatch):
    # `mtplx inspect` leaves a metadata-only stub in the HF cache for a repo
    # whose weights are not downloaded. Trusting it would report present.
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    repo = "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed"
    stub = tmp_path / "hf" / "hub" / f"models--{repo.replace('/', '--')}"
    stub.mkdir(parents=True)

    spec = MtplxServerSpec(hf_repo=repo, model_id="qwen3.8",
                           cache_dir=str(tmp_path / "mtplx"))
    assert not process.weights_are_local(spec)


def test_llama_wants_the_exact_quant(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    repo, wanted = "unsloth/Muse-Glimmer-30B-GGUF", "Muse-Glimmer-30B-UD-Q8_K_XL.gguf"
    snapshot = (tmp_path / "hub" / f"models--{repo.replace('/', '--')}"
                / "snapshots" / "abc")
    snapshot.mkdir(parents=True)

    spec = LlamaServerSpec(alias="muse-glimmer", hf_repo=repo, hf_file=wanted)
    assert not process.weights_are_local(spec)

    # A different quant present is not the one the model file asked for.
    (snapshot / "Muse-Glimmer-30B-UD-Q4_K_XL.gguf").touch()
    assert not process.weights_are_local(spec)

    (snapshot / wanted).touch()
    assert process.weights_are_local(spec)


def test_mlx_wants_a_complete_snapshot_not_just_the_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    repo = "FakeRockert543/gemma-4-31b-it-MLX-8bit"
    spec = MlxServerSpec(hf_repo=repo)
    assert not process.weights_are_local(spec)

    root = tmp_path / "hub" / f"models--{repo.replace('/', '--')}"
    (root / "blobs").mkdir(parents=True)
    assert not process.weights_are_local(spec), "a bare folder is a partial download"

    (root / "refs").mkdir()
    (root / "refs" / "main").write_text("abc")
    (root / "snapshots" / "abc").mkdir(parents=True)
    (root / "snapshots" / "abc" / "config.json").touch()
    assert process.weights_are_local(spec)

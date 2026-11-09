"""find_in_hf_cache: exact filename matches only."""

import os

from inference import specs


def _make_snapshot(root, repo, filename):
    path = os.path.join(root, "hub", "models--" + repo.replace("/", "--"),
                        "snapshots", "abc123")
    os.makedirs(path, exist_ok=True)
    target = os.path.join(path, filename)
    open(target, "w").close()
    return target


def test_finds_an_exact_match(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    expected = _make_snapshot(str(tmp_path), "unsloth/Qwen3.6-27B-GGUF",
                              "Qwen3.6-27B-UD-Q8_K_XL.gguf")
    found = specs.find_in_hf_cache("unsloth/Qwen3.6-27B-GGUF",
                                    "Qwen3.6-27B-UD-Q8_K_XL.gguf")
    assert found == expected


def test_a_different_quant_is_not_a_match(tmp_path, monkeypatch):
    # Picking a different quant than the model file asked for would silently
    # change what is being measured, which is worse than re-downloading.
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    _make_snapshot(str(tmp_path), "unsloth/Qwen3.6-27B-GGUF",
                   "Qwen3.6-27B-Q4_K_M.gguf")
    assert specs.find_in_hf_cache("unsloth/Qwen3.6-27B-GGUF",
                                   "Qwen3.6-27B-UD-Q8_K_XL.gguf") is None


def test_no_filename_means_no_lookup(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert specs.find_in_hf_cache("unsloth/Qwen3.6-27B-GGUF", None) is None


def test_a_missing_repo_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert specs.find_in_hf_cache("nobody/nothing", "x.gguf") is None

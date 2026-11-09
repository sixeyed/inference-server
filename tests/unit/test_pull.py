"""weights.py - fetching weights outside the sandbox, against a fake hub."""

import hashlib
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from inference import process, weights
from inference.specs import LlamaServerSpec, MlxServerSpec, find_in_hf_cache

SHA = "0123456789abcdef0123456789abcdef01234567"
GGUF = b"GGUF" + bytes(range(256)) * 400       # ~100KB, LFS
README = b"# a model card\n"                    # plain git blob


def _sibling(name, data, lfs=True):
    if lfs:
        return {"rfilename": name, "size": len(data),
                "lfs": {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}}
    return {"rfilename": name, "size": len(data), "blobId": "b" + str(len(data))}


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """A fake huggingface.co: the revision API and /resolve/, with Range."""
    files = {
        "README.md": (README, False),
        "m-Q4_K_M.gguf": (GGUF, True),
        "m-Q8_0.gguf": (GGUF[::-1], True),
        "BF16/m-BF16-00001-of-00002.gguf": (GGUF[:500], True),
        "BF16/m-BF16-00002-of-00002.gguf": (GGUF[500:], True),
    }
    state = {"requests": [], "files": files, "corrupt": set()}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            state["requests"].append((self.path, self.headers.get("Range")))
            if self.path.startswith("/api/models/o/m/revision/main"):
                body = json.dumps({"sha": SHA, "siblings": [
                    _sibling(n, d, lfs) for n, (d, lfs) in files.items()]}).encode()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(body)
                return
            prefix = f"/o/m/resolve/{SHA}/"
            name = self.path[len(prefix):].replace("%2F", "/")
            if not self.path.startswith(prefix) or name not in files:
                self.send_error(404)
                return
            data = files[name][0]
            if name in state["corrupt"]:
                data = b"X" * len(data)
            start = 0
            rng = self.headers.get("Range")
            if rng:
                start = int(rng.split("=")[1].rstrip("-"))
                self.send_response(206)
            else:
                self.send_response(200)
            self.send_header("Content-Length", str(len(data) - start))
            self.end_headers()
            self.wfile.write(data[start:])

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(weights, "HF_ENDPOINT", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    yield state
    server.shutdown()


def _llama(hf_file):
    return LlamaServerSpec(alias="m", hf_repo="o/m", hf_file=hf_file)


def test_only_the_configured_quant_is_fetched(hub):
    weights.pull(_llama("m-Q4_K_M.gguf"), say=lambda _: None)
    fetched = [p for p, _ in hub["requests"] if "/resolve/" in p]
    assert fetched == [f"/o/m/resolve/{SHA}/m-Q4_K_M.gguf"]


def test_the_result_is_a_standard_hf_cache_entry(hub):
    spec = _llama("m-Q4_K_M.gguf")
    weights.pull(spec, say=lambda _: None)
    path = find_in_hf_cache("o/m", "m-Q4_K_M.gguf")
    assert path and open(path, "rb").read() == GGUF
    assert os.path.islink(path), "snapshots link into blobs/, as huggingface_hub does"
    repo = weights.hf_repo_dir("o/m")
    assert open(os.path.join(repo, "refs", "main")).read() == SHA
    assert process.weights_are_local(spec)


def test_a_split_gguf_brings_every_part(hub):
    weights.pull(_llama("BF16/m-BF16-00001-of-00002.gguf"), say=lambda _: None)
    snap = os.path.join(weights.hf_repo_dir("o/m"), "snapshots", SHA, "BF16")
    assert sorted(os.listdir(snap)) == ["m-BF16-00001-of-00002.gguf",
                                         "m-BF16-00002-of-00002.gguf"]


def test_an_mlx_repo_is_fetched_whole(hub):
    weights.pull(MlxServerSpec(hf_repo="o/m"), say=lambda _: None)
    snap = os.path.join(weights.hf_repo_dir("o/m"), "snapshots", SHA)
    assert os.path.exists(os.path.join(snap, "README.md"))
    assert os.path.exists(os.path.join(snap, "BF16", "m-BF16-00002-of-00002.gguf"))


def test_a_second_pull_downloads_nothing(hub):
    spec = _llama("m-Q4_K_M.gguf")
    weights.pull(spec, say=lambda _: None)
    hub["requests"].clear()
    weights.pull(spec, say=lambda _: None)
    assert not [p for p, _ in hub["requests"] if "/resolve/" in p]


def test_an_interrupted_download_resumes(hub):
    spec = _llama("m-Q4_K_M.gguf")
    blob = os.path.join(weights.hf_repo_dir("o/m"), "blobs",
                        hashlib.sha256(GGUF).hexdigest())
    os.makedirs(os.path.dirname(blob))
    with open(blob + ".incomplete", "wb") as handle:
        handle.write(GGUF[:1000])
    weights.pull(spec, say=lambda _: None)
    assert ("/o/m/resolve/" + SHA + "/m-Q4_K_M.gguf", "bytes=1000-") in hub["requests"]
    assert open(blob, "rb").read() == GGUF


def test_a_corrupt_download_is_rejected_and_reads_as_missing(hub):
    spec = _llama("m-Q4_K_M.gguf")
    hub["corrupt"].add("m-Q4_K_M.gguf")
    with pytest.raises(weights.PullError, match="sha256"):
        weights.pull(spec, say=lambda _: None)
    assert find_in_hf_cache("o/m", "m-Q4_K_M.gguf") is None
    assert weights.is_pulling(spec), "the marker stays until a pull completes"
    assert not process.weights_are_local(spec)


def test_a_quant_the_repo_does_not_have_is_an_error(hub):
    with pytest.raises(weights.PullError, match="no file"):
        weights.pull(_llama("m-Q2_K.gguf"), say=lambda _: None)


def test_no_quant_configured_is_an_error_not_a_guess():
    with pytest.raises(weights.PullError, match="no quant"):
        weights.select_files(_llama(None), [])


def test_the_token_is_not_sent_on_to_the_cdn(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    request = weights._request("https://huggingface.co/o/m/resolve/x/f")
    assert request.unredirected_hdrs.get("Authorization") == "Bearer hf_secret"
    assert "Authorization" not in request.headers


def test_a_sandboxed_server_is_told_it_is_offline():
    from inference.machine import Options
    assert process.server_env(Options(sandbox=True))["HF_HUB_OFFLINE"] == "1"
    assert process.server_env(Options(sandbox=False)) is None

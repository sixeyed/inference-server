"""Fetching model weights - the one step that needs the network.

Model servers run inside an egress sandbox (config.SANDBOX_PROFILE), which is
the point: they are third-party binaries running arbitrary code on your prompts,
and "it cannot open a socket" beats "I read the source". But that same sandbox
blocks the HuggingFace download a first launch needs.

So the download is split off and done here, before the server starts, by code
in this repo rather than by the engine:

  GGUF and MLX repos   a stdlib HuggingFace client below, writing the standard
                       HF cache layout (blobs/, snapshots/, refs/) so
                       find_in_hf_cache(), llama-server -m and mlx_lm.server
                       all find the files without being told.

  MTPLX packs          `mtplx models --update`, MTPLX's own delta downloader.
                       Run with the network, but it only downloads - it never
                       serves - and MTPLX owns the pack format, which it
                       rewrites on upgrade, so writing it from outside would be
                       a fight.

Either way the server itself is then started sandboxed as usual, and never gets
the network. Still standard library only - pyproject's dependencies stay [].

A download in progress is marked, so an interrupted one reads as "not on disk"
and the next launch resumes it rather than loading half a model.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

from .specs import LlamaServerSpec, MlxServerSpec, MtplxServerSpec

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
MARKER = ".inference-pulling"
_CHUNK = 8 * 1024 * 1024
_SPLIT = re.compile(r"^(?P<stem>.*)-\d{5}-of-(?P<total>\d{5})\.gguf$")


class PullError(Exception):
    """A pull that did not complete. The partial download is kept for resume."""


# ------------------------------------------------------------
# Where things go
# ------------------------------------------------------------
def hf_hub_dir() -> str:
    hf_home = os.environ.get("HF_HOME")
    return (os.path.join(hf_home, "hub") if hf_home
            else os.path.expanduser("~/.cache/huggingface/hub"))


def hf_repo_dir(repo: str) -> str:
    return os.path.join(hf_hub_dir(), "models--" + repo.replace("/", "--"))


def mtplx_pack_dir(spec: MtplxServerSpec) -> str:
    return os.path.join(os.path.expanduser(spec.cache_dir),
                        spec.hf_repo.replace("/", "--"))


def marker_path(spec) -> str:
    """Where the in-progress marker for this spec's weights lives.

    Beside the MTPLX pack rather than in it: `mtplx models --update` owns that
    directory and validates what is in it.
    """
    if isinstance(spec, MtplxServerSpec):
        return mtplx_pack_dir(spec) + MARKER
    return os.path.join(hf_repo_dir(spec.hf_repo), MARKER)


def is_pulling(spec) -> bool:
    return os.path.exists(marker_path(spec))


# ------------------------------------------------------------
# HuggingFace, standard library only
# ------------------------------------------------------------
def _token() -> str | None:
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(name):
            return os.environ[name].strip()
    path = os.path.join(os.environ.get("HF_HOME")
                        or os.path.expanduser("~/.cache/huggingface"), "token")
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read().strip() or None
    except OSError:
        return None


def _request(url: str, headers: dict | None = None) -> urllib.request.Request:
    request = urllib.request.Request(url, headers={
        "User-Agent": "inference-server", **(headers or {})})
    token = _token()
    if token:
        # Unredirected: /resolve/ answers with a redirect to a signed CDN URL,
        # and the token has no business going there.
        request.add_unredirected_header("Authorization", f"Bearer {token}")
    return request


def repo_info(repo: str, revision: str = "main") -> dict:
    """The revision's commit sha and its files, with sizes and blob ids."""
    url = (f"{HF_ENDPOINT}/api/models/{repo}/revision/"
           f"{urllib.parse.quote(revision, safe='')}?blobs=true")
    try:
        with urllib.request.urlopen(_request(url), timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        hint = (" - a gated or private repo needs HF_TOKEN"
                if exc.code in (401, 403) else "")
        raise PullError(f"{repo}: HuggingFace answered {exc.code}{hint}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise PullError(f"{repo}: could not reach {HF_ENDPOINT} ({exc})") from exc


def select_files(spec, siblings: list[dict]) -> list[dict]:
    """The files this spec loads - one quant of a GGUF repo, all of an MLX one.

    A GGUF repo holds every quant, often hundreds of GB together; only the
    one the model file asked for is fetched. Split GGUFs are named by their
    first part, and llama-server loads the rest from beside it.
    """
    if not isinstance(spec, LlamaServerSpec):
        return siblings
    if not spec.hf_file:
        raise PullError(f"{spec.hf_repo}: no quant is configured for this "
                        "machine, so there is no one file to fetch")
    wanted = spec.hf_file
    split = _SPLIT.match(wanted)
    if split:
        prefix = split.group("stem") + "-"
        suffix = f"-of-{split.group('total')}.gguf"
        chosen = [s for s in siblings if s["rfilename"].startswith(prefix)
                  and s["rfilename"].endswith(suffix)]
    else:
        chosen = [s for s in siblings if s["rfilename"] == wanted]
    if not chosen:
        raise PullError(f"{spec.hf_repo} has no file {wanted}")
    return chosen


def _blob_id(sibling: dict) -> tuple[str, int | None, str | None]:
    """(cache blob name, expected size, sha256 to verify or None)."""
    lfs = sibling.get("lfs")
    if lfs:
        return lfs["sha256"], lfs.get("size"), lfs["sha256"]
    return sibling["blobId"], sibling.get("size"), None


def _download(url: str, dest: str, size: int | None, sha256: str | None,
              progress) -> None:
    """Fetch url to dest, resuming a partial `.incomplete`, then verify."""
    partial = dest + ".incomplete"
    have = os.path.getsize(partial) if os.path.exists(partial) else 0
    digest = hashlib.sha256() if sha256 else None
    if digest and have:
        with open(partial, "rb") as handle:
            for block in iter(lambda: handle.read(_CHUNK), b""):
                digest.update(block)

    if size is None or have < size:
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            response = urllib.request.urlopen(_request(url, headers), timeout=60)
        except urllib.error.HTTPError as exc:
            raise PullError(f"{url}: HTTP {exc.code}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise PullError(f"{url}: {exc}") from exc
        with response:
            if have and response.status != 206:
                # The server ignored the Range - start again rather than append
                # a second copy onto the first.
                have = 0
                digest = hashlib.sha256() if sha256 else None
            with open(partial, "ab" if have else "wb") as out:
                for block in iter(lambda: response.read(_CHUNK), b""):
                    out.write(block)
                    if digest:
                        digest.update(block)
                    have += len(block)
                    progress(len(block))

    if size is not None and os.path.getsize(partial) != size:
        raise PullError(f"{os.path.basename(dest)}: got "
                        f"{os.path.getsize(partial)} bytes, expected {size}")
    if digest and digest.hexdigest() != sha256:
        os.remove(partial)
        raise PullError(f"{os.path.basename(dest)}: sha256 mismatch - "
                        "the partial download was discarded")
    os.replace(partial, dest)


def pull_hf(spec, say=print) -> dict:
    """Fetch a GGUF or MLX repo's files into the HuggingFace cache."""
    repo = spec.hf_repo
    info = repo_info(repo)
    sha = info["sha"]
    files = select_files(spec, info.get("siblings") or [])

    repo_dir = hf_repo_dir(repo)
    blobs = os.path.join(repo_dir, "blobs")
    snapshot = os.path.join(repo_dir, "snapshots", sha)
    os.makedirs(blobs, exist_ok=True)
    marker = marker_path(spec)
    with open(marker, "w", encoding="utf-8") as handle:
        handle.write(sha)

    todo = []
    for sibling in files:
        blob_name, size, sha256 = _blob_id(sibling)
        blob = os.path.join(blobs, blob_name)
        if not os.path.exists(blob):
            todo.append((sibling, blob, size, sha256))
    total = sum(item[2] or 0 for item in todo)
    meter = _Meter(repo, total, say)

    for sibling, blob, size, sha256 in todo:
        name = sibling["rfilename"]
        url = (f"{HF_ENDPOINT}/{repo}/resolve/{sha}/"
               f"{urllib.parse.quote(name)}")
        meter.file = name
        _download(url, blob, size, sha256, meter.add)

    # Links only once every blob is whole, so a snapshot never points at a
    # file that is still arriving.
    for sibling in files:
        blob_name, _, _ = _blob_id(sibling)
        link = os.path.join(snapshot, sibling["rfilename"])
        os.makedirs(os.path.dirname(link), exist_ok=True)
        if not os.path.lexists(link):
            os.symlink(os.path.relpath(os.path.join(blobs, blob_name),
                                       os.path.dirname(link)), link)

    refs = os.path.join(repo_dir, "refs")
    os.makedirs(refs, exist_ok=True)
    with open(os.path.join(refs, "main"), "w", encoding="utf-8") as handle:
        handle.write(sha)
    os.remove(marker)
    meter.done()
    return {"repo": repo, "revision": sha, "files": len(files),
            "downloaded_bytes": meter.seen}


# ------------------------------------------------------------
# MTPLX
# ------------------------------------------------------------
def pull_mtplx(spec: MtplxServerSpec, say=print, runner=None) -> dict:
    """`mtplx models --update` for this pack - first download or delta."""
    exe = shutil.which("mtplx")
    if not exe:
        raise PullError("mtplx is not installed (brew install mtplx)")
    cache_dir = os.path.expanduser(spec.cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    marker = marker_path(spec)
    with open(marker, "w", encoding="utf-8") as handle:
        handle.write(spec.hf_repo)

    argv = [exe, "models", "--cache-dir", cache_dir,
            "--update", spec.hf_repo, "--progress-json"]
    meter = _Meter(spec.hf_repo, 0, say)
    tail: list[str] = []
    proc = (runner or subprocess.Popen)(
        argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        tail = (tail + [line.rstrip()])[-10:]
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") == "progress":
            meter.file = event.get("file") or meter.file
            meter.add(int(event.get("delta_bytes") or 0))
    code = proc.wait()
    if code != 0:
        raise PullError(f"mtplx models --update {spec.hf_repo} exited {code}: "
                        + " | ".join(tail[-3:]))
    os.remove(marker)
    meter.done()
    return {"repo": spec.hf_repo, "downloaded_bytes": meter.seen}


# ------------------------------------------------------------
# The front door
# ------------------------------------------------------------
def pull(spec, say=print) -> dict:
    """Bring this spec's weights up to date on disk. Network, no sandbox."""
    if isinstance(spec, MtplxServerSpec):
        return pull_mtplx(spec, say)
    if isinstance(spec, (LlamaServerSpec, MlxServerSpec)):
        if isinstance(spec, LlamaServerSpec) and spec.local_path:
            raise PullError(f"{spec.local_path} is a local file - nothing to pull")
        return pull_hf(spec, say)
    raise PullError(f"don't know how to fetch weights for {type(spec).__name__}")


class _Meter:
    """Progress, at most every few seconds - a 30GB pull is a long time."""

    def __init__(self, repo: str, total: int, say, every: float = 5.0):
        self.repo, self.total, self.say, self.every = repo, total, say, every
        self.seen = 0
        self.file = ""
        self.started = self.last = time.monotonic()

    def add(self, n: int) -> None:
        self.seen += n
        now = time.monotonic()
        if now - self.last >= self.every:
            self.last = now
            rate = self.seen / max(now - self.started, 1e-6) / 1e6
            pct = f" {100 * self.seen / self.total:.0f}%" if self.total else ""
            self.say(f"     pulling {self.repo}{pct} - {self.seen / 1e9:.1f} GB "
                     f"at {rate:.0f} MB/s ({self.file})")

    def done(self) -> None:
        elapsed = time.monotonic() - self.started
        self.say(f"     pulled {self.repo} - {self.seen / 1e9:.1f} GB "
                 f"in {elapsed:.0f}s")

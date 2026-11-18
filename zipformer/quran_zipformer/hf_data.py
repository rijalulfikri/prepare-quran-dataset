"""Fetch a prepared `data_dir` from tar shards on a Hugging Face dataset repo.

The shards are made by `vastai/pack_data_to_hf.py` from an already prepared
data_dir, so extraction reproduces it byte for byte (same cut ids and audio,
which an exact sampler resume relies on). Much faster than syncing ~300 k
small files from Drive, whose API is rate-limited per request.

Each shard is downloaded, checked against the sha256 in `index.json`,
extracted, and deleted; a marker per shard makes re-runs skip finished ones.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

log = logging.getLogger(__name__)

INDEX_NAME = "index.json"
STATE_DIR = ".hf_shards"


class ShardVerificationError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def _download(repo: str, filename: str, local_dir: Path, token: str | None) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(repo, filename, repo_type="dataset", local_dir=local_dir, token=token, force_download=True)
    )


def fetch_hf_shards(repo: str, data_dir: Path, token: str | None = None, workers: int = 4) -> int:
    """Download, verify and extract every shard of `repo` into `data_dir`.

    Returns the number of shards extracted by this call (0 if all were done).
    Raises `ShardVerificationError` if a shard fails its sha256 check.
    """
    # hf_xet: more parallel range requests per file (default is tuned for laptops)
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    token = token or os.environ.get("HUGGINGFACE_TOKEN") or None
    data_dir = Path(data_dir)
    state = data_dir / STATE_DIR
    dl_root = state / "dl"
    state.mkdir(parents=True, exist_ok=True)

    index = json.loads(_download(repo, INDEX_NAME, dl_root, token).read_text())
    shards = index["shards"]
    todo = [s for s in shards if _marker_sha(state, s["name"]) != s["sha256"]]
    log.info(
        "hf data %s: %d shards (%.1f GiB, %d files), %d to fetch",
        repo, len(shards), index["total_bytes"] / 2**30, index["num_files"], len(todo),
    )

    def fetch(shard: dict) -> None:
        tar = _download(repo, shard["path"], dl_root, token)
        try:
            if _sha256(tar) != shard["sha256"]:
                raise ShardVerificationError(f"{shard['name']} failed sha256 verification")
            subprocess.run(["tar", "-xf", str(tar), "-C", str(data_dir)], check=True)
        finally:
            tar.unlink(missing_ok=True)
        (state / f"{shard['name']}.done").write_text(shard["sha256"])
        log.info("hf data: %s extracted (%d files)", shard["name"], shard["num_files"])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in [pool.submit(fetch, s) for s in todo]:
            future.result()
    shutil.rmtree(dl_root, ignore_errors=True)
    return len(todo)


def _marker_sha(state: Path, name: str) -> str | None:
    marker = state / f"{name}.done"
    return marker.read_text().strip() if marker.exists() else None

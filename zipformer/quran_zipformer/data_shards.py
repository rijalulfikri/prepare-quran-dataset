"""Fetch a prepared `data_dir` from tar shards on Hugging Face or Google Drive.

The shards are made by `vastai/pack_data_to_hf.py` from an already prepared
data_dir and mirrored byte for byte to Drive by
`vastai/mirror_shards_to_drive.py`, so both sources share one `index.json`
and extraction reproduces data_dir exactly (same cut ids and audio, which an
exact sampler resume relies on). Much faster than syncing ~300 k small files
from Drive, whose API is rate-limited per request.

Each shard is downloaded, checked against the sha256 in `index.json`,
extracted, and deleted; a marker per shard (its sha256, identical on every
source) makes re-runs skip finished ones, even after switching source.
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


class HfSource:
    """Shards in a Hugging Face dataset repo."""

    def __init__(self, repo: str, token: str | None = None) -> None:
        self.repo = repo
        self.token = token or os.environ.get("HUGGINGFACE_TOKEN") or None
        self.name = f"hf:{repo}"
        # hf_xet: more parallel range requests per file (default is tuned for laptops)
        os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

    def download(self, filename: str, local_dir: Path) -> Path:
        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(
                self.repo, filename, repo_type="dataset", local_dir=local_dir, token=self.token, force_download=True
            )
        )


class RcloneSource:
    """Shards in an rclone `remote:path` folder (e.g. Google Drive)."""

    def __init__(self, remote_dir: str) -> None:
        self.remote_dir = remote_dir.rstrip("/")
        self.name = remote_dir

    def download(self, filename: str, local_dir: Path) -> Path:
        dst = Path(local_dir) / filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["rclone", "copyto", f"{self.remote_dir}/{filename}", str(dst), "--multi-thread-streams", "8"],
            check=True, capture_output=True, text=True,
        )
        return dst


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def fetch_shards(source, data_dir: Path, workers: int = 4) -> int:
    """Download, verify and extract every shard of `source` into `data_dir`.

    Returns the number of shards extracted by this call (0 if all were done).
    Raises `ShardVerificationError` if a shard fails its sha256 check, and
    whatever the source raises if it is unreachable or incomplete.
    """
    data_dir = Path(data_dir)
    state = data_dir / STATE_DIR
    dl_root = state / "dl"
    state.mkdir(parents=True, exist_ok=True)

    index = json.loads(source.download(INDEX_NAME, dl_root).read_text())
    shards = index["shards"]
    todo = [s for s in shards if _marker_sha(state, s["name"]) != s["sha256"]]
    log.info(
        "data shards %s: %d shards (%.1f GiB, %d files), %d to fetch",
        source.name, len(shards), index["total_bytes"] / 2**30, index["num_files"], len(todo),
    )

    def fetch(shard: dict) -> None:
        tar = source.download(shard["path"], dl_root)
        try:
            if _sha256(tar) != shard["sha256"]:
                raise ShardVerificationError(f"{shard['name']} from {source.name} failed sha256 verification")
            subprocess.run(["tar", "-xf", str(tar), "-C", str(data_dir)], check=True)
        finally:
            tar.unlink(missing_ok=True)
        (state / f"{shard['name']}.done").write_text(shard["sha256"])
        log.info("data shards: %s extracted (%d files)", shard["name"], shard["num_files"])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in [pool.submit(fetch, s) for s in todo]:
            future.result()
    shutil.rmtree(dl_root, ignore_errors=True)
    return len(todo)


def shard_sources(cfg) -> list[HfSource | RcloneSource]:
    """Configured sources in priority order: Hugging Face, then Drive."""
    sources: list[HfSource | RcloneSource] = []
    if getattr(cfg, "hf_data_repo", None):
        sources.append(HfSource(cfg.hf_data_repo))
    if getattr(cfg, "drive_data_shards_path", None):
        sources.append(RcloneSource(cfg.drive_data_shards_path))
    return sources


def _marker_sha(state: Path, name: str) -> str | None:
    marker = state / f"{name}.done"
    return marker.read_text().strip() if marker.exists() else None

# zipformer/tests/test_data_shards.py
import hashlib
import json
import shutil
import subprocess
import types
from pathlib import Path

import pytest

from quran_zipformer import data_shards


def _make_repo(tmp_path: Path) -> tuple[Path, dict]:
    """A fake source dir: two tar shards of a small data_dir plus index.json."""
    src = tmp_path / "src"
    for rel, body in {"audio/m/1.flac": b"a" * 10, "audio/m/2.flac": b"b" * 20, "tokens.txt": b"t"}.items():
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_bytes(body)
    repo = tmp_path / "repo"
    (repo / "shards").mkdir(parents=True)
    shards = []
    for i, members in enumerate([["audio/m/1.flac", "audio/m/2.flac"], ["tokens.txt"]]):
        tar = repo / "shards" / f"shard-{i:05d}.tar"
        subprocess.run(["tar", "-cf", str(tar), "-C", str(src), *members], check=True)
        shards.append({
            "name": tar.name, "path": f"shards/{tar.name}", "num_files": len(members),
            "size_bytes": tar.stat().st_size, "sha256": hashlib.sha256(tar.read_bytes()).hexdigest(),
        })
    index = {"num_files": 3, "total_bytes": sum(s["size_bytes"] for s in shards), "shards": shards}
    (repo / "index.json").write_text(json.dumps(index))
    return repo, index


class _DirSource:
    """Copies files out of a local directory, recording what was requested."""

    def __init__(self, root: Path) -> None:
        self.root, self.name, self.downloads = root, f"dir:{root}", []

    def download(self, filename, local_dir):
        self.downloads.append(filename)
        dst = Path(local_dir) / filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(self.root / filename, dst)
        return dst


@pytest.fixture
def source(tmp_path):
    repo, _ = _make_repo(tmp_path)
    return _DirSource(repo)


def test_fetch_extracts_all_shards_and_cleans_up(tmp_path, source):
    data = tmp_path / "data"
    assert data_shards.fetch_shards(source, data) == 2
    assert (data / "audio/m/2.flac").read_bytes() == b"b" * 20
    assert (data / "tokens.txt").read_bytes() == b"t"
    assert not (data / data_shards.STATE_DIR / "dl").exists()


def test_rerun_skips_finished_shards(tmp_path, source):
    data = tmp_path / "data"
    data_shards.fetch_shards(source, data)
    source.downloads.clear()
    assert data_shards.fetch_shards(source, data) == 0
    assert source.downloads == ["index.json"]


def test_markers_carry_over_between_sources(tmp_path, source):
    """Same bytes on HF and Drive: shards finished from one source are not refetched from the other."""
    data = tmp_path / "data"
    data_shards.fetch_shards(source, data)
    other = _DirSource(source.root)
    assert data_shards.fetch_shards(other, data) == 0


def test_corrupt_shard_raises_and_leaves_no_marker(tmp_path, source):
    with open(source.root / "shards/shard-00001.tar", "ab") as f:
        f.write(b"corrupt")
    data = tmp_path / "data"
    with pytest.raises(data_shards.ShardVerificationError):
        data_shards.fetch_shards(source, data, workers=1)
    assert not (data / data_shards.STATE_DIR / "shard-00001.tar.done").exists()


def test_rclone_source_fetches_from_local_remote(tmp_path, source):
    if shutil.which("rclone") is None:
        pytest.skip("rclone not installed")
    data = tmp_path / "data"
    assert data_shards.fetch_shards(data_shards.RcloneSource(str(source.root)), data) == 2
    assert (data / "audio/m/1.flac").read_bytes() == b"a" * 10


def test_sources_in_priority_order():
    cfg = types.SimpleNamespace(hf_data_repo="u/data", drive_data_shards_path="remote:shards")
    assert [s.name for s in data_shards.shard_sources(cfg)] == ["hf:u/data", "remote:shards"]
    cfg = types.SimpleNamespace(hf_data_repo=None, drive_data_shards_path=None)
    assert data_shards.shard_sources(cfg) == []

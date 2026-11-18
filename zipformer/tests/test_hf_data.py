# zipformer/tests/test_hf_data.py
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from quran_zipformer import hf_data


def _make_repo(tmp_path: Path) -> tuple[Path, dict]:
    """A fake repo dir: two tar shards of a small data_dir plus index.json."""
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


@pytest.fixture
def fake_hub(tmp_path, monkeypatch):
    repo, index = _make_repo(tmp_path)
    downloads = []

    def download(repo_id, filename, local_dir, token):
        downloads.append(filename)
        dst = Path(local_dir) / filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(repo / filename, dst)
        return dst

    monkeypatch.setattr(hf_data, "_download", download)
    return repo, index, downloads


def test_fetch_extracts_all_shards_and_cleans_up(tmp_path, fake_hub):
    data = tmp_path / "data"
    assert hf_data.fetch_hf_shards("u/data", data) == 2
    assert (data / "audio/m/2.flac").read_bytes() == b"b" * 20
    assert (data / "tokens.txt").read_bytes() == b"t"
    assert not (data / hf_data.STATE_DIR / "dl").exists()


def test_rerun_skips_finished_shards(tmp_path, fake_hub):
    _, _, downloads = fake_hub
    data = tmp_path / "data"
    hf_data.fetch_hf_shards("u/data", data)
    downloads.clear()
    assert hf_data.fetch_hf_shards("u/data", data) == 0
    assert downloads == ["index.json"]


def test_corrupt_shard_raises_and_leaves_no_marker(tmp_path, fake_hub):
    repo, _, _ = fake_hub
    with open(repo / "shards/shard-00001.tar", "ab") as f:
        f.write(b"corrupt")
    data = tmp_path / "data"
    with pytest.raises(hf_data.ShardVerificationError):
        hf_data.fetch_hf_shards("u/data", data, workers=1)
    assert not (data / hf_data.STATE_DIR / "shard-00001.tar.done").exists()

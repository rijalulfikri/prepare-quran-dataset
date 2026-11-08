import logging
import shutil
import threading

import pytest
import torch

from prepare_quran_dataset.zipformer import checkpointing as ck
from prepare_quran_dataset.zipformer.drive_backup import DriveBackup, Rclone, RcloneError

pytestmark = pytest.mark.skipif(shutil.which("rclone") is None, reason="rclone not installed")


@pytest.fixture
def remote(tmp_path, monkeypatch):
    monkeypatch.setenv("RCLONE_CONFIG_TESTLOCAL_TYPE", "local")
    (tmp_path / "remote").mkdir()
    return f"testlocal:{tmp_path / 'remote'}"


def _save(d, kind, b, e):
    return ck.save_checkpoint_atomic(
        {"x": torch.randn(1000)},
        d / ck.ckpt_name(kind, b, e),
        {"kind": kind, "batch_idx_train": b, "epoch": e, "batches_consumed_in_epoch": b},
    )


def test_upload_then_prune_remote(tmp_path, remote):
    local = tmp_path / "local"
    local.mkdir()
    bk = DriveBackup(remote, keep_last_k=2, timeout=60)
    for b in (2000, 4000, 6000):
        _save(local, "batch", b, 1)
        bk.submit(local / f"checkpoint-{b}.pt")
        assert bk.flush(60)
    names = sorted(e["Name"] for e in Rclone().lsjson(remote))
    assert names == [
        "checkpoint-4000.meta.json",
        "checkpoint-4000.pt",
        "checkpoint-6000.meta.json",
        "checkpoint-6000.pt",
    ]
    assert bk.last_error is None


def test_lsjson_missing_dir_is_empty(tmp_path, remote):
    assert Rclone().lsjson(f"{remote}/does-not-exist") == []


def test_cat_missing_file_raises(tmp_path, remote):
    with pytest.raises(RcloneError):
        Rclone().cat(f"{remote}/nope.meta.json")


def test_download_latest_into_empty_dir_and_verify(tmp_path, remote):
    src = tmp_path / "src"
    src.mkdir()
    dst = tmp_path / "dst"
    dst.mkdir()
    bk = DriveBackup(remote, keep_last_k=3, timeout=60)
    _save(src, "batch", 2000, 1)
    _save(src, "epoch", 4000, 1)
    for p in ("checkpoint-2000.pt", "epoch-1.pt"):
        bk.submit(src / p)
    assert bk.flush(60)
    m = bk.latest_remote()
    assert m.kind == "epoch" and m.batch_idx_train == 4000
    p = bk.download(m, dst)
    assert p == dst / "epoch-1.pt"
    assert ck.verify_sha256(p, ck.read_meta(p))


def test_download_corrupt_remote_raises_and_cleans(tmp_path, remote):
    src = tmp_path / "src"
    src.mkdir()
    dst = tmp_path / "dst"
    dst.mkdir()
    bk = DriveBackup(remote, keep_last_k=3, timeout=60)
    m = _save(src, "batch", 2000, 1)
    bk.submit(src / "checkpoint-2000.pt")
    assert bk.flush(60)
    remote_pt = tmp_path / "remote" / "checkpoint-2000.pt"
    data = bytearray(remote_pt.read_bytes())
    data[-1] ^= 0xFF
    remote_pt.write_bytes(bytes(data))
    with pytest.raises(RcloneError):
        bk.download(m, dst)
    assert list(dst.iterdir()) == []


def test_download_missing_epochs(tmp_path, remote):
    src = tmp_path / "src"
    src.mkdir()
    dst = tmp_path / "dst"
    dst.mkdir()
    bk = DriveBackup(remote, keep_last_k=5, timeout=60)
    for e in (1, 2, 3):
        _save(src, "epoch", 1000 * e, e)
        bk.submit(src / f"epoch-{e}.pt")
    _save(src, "batch", 3500, 3)
    bk.submit(src / "checkpoint-3500.pt")
    assert bk.flush(60)
    shutil.copy(src / "epoch-2.pt", dst / "epoch-2.pt")
    got = bk.download_missing_epochs(dst)
    assert sorted(p.name for p in got) == ["epoch-1.pt", "epoch-3.pt"]
    assert all(ck.verify_sha256(p, ck.read_meta(p)) for p in got)
    assert not (dst / "checkpoint-3500.pt").exists()


def test_remote_pt_without_meta_is_not_latest(tmp_path, remote):
    src = tmp_path / "src"
    src.mkdir()
    bk = DriveBackup(remote, keep_last_k=3, timeout=60)
    _save(src, "batch", 2000, 1)
    bk.submit(src / "checkpoint-2000.pt")
    assert bk.flush(60)
    _save(src, "batch", 4000, 1)
    Rclone().copyto(str(src / "checkpoint-4000.pt"), f"{remote}/checkpoint-4000.pt", 60)  # death before meta
    assert bk.latest_remote().batch_idx_train == 2000


def test_prune_remote_keeps_newer_meta_less_pt(tmp_path, remote):
    src = tmp_path / "src"
    src.mkdir()
    bk = DriveBackup(remote, keep_last_k=1, timeout=60)
    _save(src, "batch", 2000, 1)
    _save(src, "batch", 4000, 1)
    bk.submit(src / "checkpoint-2000.pt")
    assert bk.flush(60)
    Rclone().copyto(str(src / "checkpoint-4000.pt"), f"{remote}/checkpoint-4000.pt", 60)  # mid-upload elsewhere
    bk.prune_remote()
    names = sorted(e["Name"] for e in Rclone().lsjson(remote))
    assert names == ["checkpoint-2000.meta.json", "checkpoint-2000.pt", "checkpoint-4000.pt"]


def test_latest_remote_empty(tmp_path, remote):
    assert DriveBackup(remote, keep_last_k=3, timeout=60).latest_remote() is None


def test_upload_failure_is_a_warning(tmp_path, caplog):
    bk = DriveBackup("nonexistent-remote:x", keep_last_k=3, timeout=10)
    _save(tmp_path, "batch", 2000, 1)
    with caplog.at_level(logging.WARNING):
        bk.submit(tmp_path / "checkpoint-2000.pt")
        assert bk.flush(30) and bk.last_error is not None
    assert any(r.levelname == "WARNING" for r in caplog.records)


def test_newer_batch_replaces_pending_batch_but_not_epoch(tmp_path, remote):
    src = tmp_path / "src"
    src.mkdir()
    bk = DriveBackup(remote, keep_last_k=10, timeout=60)
    real_copyto = bk._rclone.copyto
    gate = threading.Event()
    started = threading.Event()
    copied = []
    lock = threading.Lock()

    def gated_copyto(src_path, dst, timeout):
        with lock:
            first = not copied
            copied.append(str(src_path))
        if first:
            started.set()
            assert gate.wait(30)
        real_copyto(src_path, dst, timeout)

    bk._rclone.copyto = gated_copyto
    _save(src, "batch", 2000, 1)
    _save(src, "epoch", 3000, 1)
    _save(src, "batch", 4000, 2)
    _save(src, "batch", 6000, 2)
    bk.submit(src / "checkpoint-2000.pt")
    assert started.wait(30)  # worker is now blocked inside the first upload
    bk.submit(src / "epoch-1.pt")
    bk.submit(src / "checkpoint-4000.pt")
    bk.submit(src / "checkpoint-6000.pt")
    assert not bk.flush(0.2)  # still blocked -> flush times out
    gate.set()
    assert bk.flush(60)
    uploaded = {p.rsplit("/", 1)[-1] for p in copied if p.endswith(".pt")}
    assert uploaded == {"checkpoint-2000.pt", "epoch-1.pt", "checkpoint-6000.pt"}
    assert not any("checkpoint-4000" in p for p in copied)
    assert bk.last_error is None

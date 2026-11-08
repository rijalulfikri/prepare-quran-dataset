import json
import os
import subprocess
import sys

import torch

from prepare_quran_dataset.zipformer import checkpointing as ck


def _save(d, kind, b, e, n=0):
    return ck.save_checkpoint_atomic(
        {"x": torch.arange(4)},
        d / ck.ckpt_name(kind, b, e),
        {"kind": kind, "batch_idx_train": b, "epoch": e, "batches_consumed_in_epoch": n},
    )


def test_meta_written_after_pt_and_sha_matches(tmp_path):
    m = _save(tmp_path, "batch", 2000, 1, 2000)
    pt = tmp_path / "checkpoint-2000.pt"
    assert ck.read_meta(pt) == m and ck.verify_sha256(pt, m)
    assert ck.meta_path(pt).name == "checkpoint-2000.meta.json"
    assert not list(tmp_path.glob("*.tmp"))


def test_verify_sha256_detects_corruption(tmp_path):
    m = _save(tmp_path, "batch", 2000, 1, 2000)
    pt = tmp_path / "checkpoint-2000.pt"
    data = bytearray(pt.read_bytes())
    data[-1] ^= 0xFF
    pt.write_bytes(bytes(data))
    assert not ck.verify_sha256(pt, m)


def test_read_meta_invalid(tmp_path):
    pt = tmp_path / "checkpoint-2000.pt"
    assert ck.read_meta(pt) is None
    ck.meta_path(pt).write_text("{not json")
    assert ck.read_meta(pt) is None
    ck.meta_path(pt).write_text(json.dumps({"kind": "batch"}))
    assert ck.read_meta(pt) is None


def test_incomplete_checkpoints_are_ignored(tmp_path):
    _save(tmp_path, "batch", 2000, 1, 2000)
    (tmp_path / "checkpoint-4000.pt").write_bytes(b"truncated")  # no meta
    _save(tmp_path, "batch", 6000, 1, 6000)
    (tmp_path / "checkpoint-6000.pt").write_bytes(b"short")  # size mismatch
    assert ck.latest_complete(tmp_path)[1].batch_idx_train == 2000


def test_latest_complete_empty_dir(tmp_path):
    assert ck.latest_complete(tmp_path) is None


def test_epoch_sorts_after_batch_at_same_step(tmp_path):
    _save(tmp_path, "batch", 8000, 1, 8000)
    _save(tmp_path, "epoch", 8000, 1, 8000)
    assert ck.latest_complete(tmp_path)[0].name == "epoch-1.pt"


def test_prune_per_kind(tmp_path):
    for b in (2000, 4000, 6000, 8000):
        _save(tmp_path, "batch", b, 1, b)
    for e in (1, 2, 3, 4):
        _save(tmp_path, "epoch", 10000 * e, e)
    (tmp_path / "checkpoint-9000.pt.tmp").write_bytes(b"x")
    (tmp_path / "checkpoint-9000.meta.json.tmp").write_bytes(b"x")
    ck.prune(tmp_path, keep_last_k=3)
    assert sorted(p.name for p in tmp_path.glob("checkpoint-*.pt")) == [
        "checkpoint-4000.pt",
        "checkpoint-6000.pt",
        "checkpoint-8000.pt",
    ]
    assert sorted(p.name for p in tmp_path.glob("epoch-*.pt")) == ["epoch-2.pt", "epoch-3.pt", "epoch-4.pt"]
    assert not list(tmp_path.glob("*.tmp"))
    assert not (tmp_path / "checkpoint-2000.meta.json").exists()
    assert not (tmp_path / "epoch-1.meta.json").exists()


def test_prune_meta_less_pt_older_deleted_newer_kept(tmp_path):
    _save(tmp_path, "batch", 2000, 1, 2000)
    (tmp_path / "checkpoint-1000.pt").write_bytes(b"old orphan")
    (tmp_path / "checkpoint-4000.pt").write_bytes(b"possibly in progress")
    deleted = ck.prune(tmp_path, keep_last_k=3)
    assert tmp_path / "checkpoint-1000.pt" in deleted
    assert (tmp_path / "checkpoint-4000.pt").exists()
    assert (tmp_path / "checkpoint-2000.pt").exists()


def test_dump_cli(tmp_path):
    p = torch.nn.Parameter(torch.ones(3))
    opt = torch.optim.Adam([p], lr=0.01)
    p.grad = torch.ones(3)
    opt.step()
    state = {
        "model": {"w": p.detach()},
        "optimizer": opt.state_dict(),
        "batch_idx_train": 2000,
        "epoch": 1,
        "batches_consumed_in_epoch": 2000,
        "sampler": {"pos": 7},
        "wandb_run_id": "abc123",
    }
    pt = tmp_path / "checkpoint-2000.pt"
    torch.save(state, pt)
    bare = tmp_path / "bare.pt"
    torch.save({"model": {}}, bare)

    def run(path):
        out = subprocess.run(
            [sys.executable, "-m", "prepare_quran_dataset.zipformer.checkpointing", "dump", str(path)],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONWARNINGS": "ignore"},
        )
        return json.loads(out.stdout)

    d = run(pt)
    assert d["batch_idx_train"] == 2000 and d["epoch"] == 1 and d["batches_consumed_in_epoch"] == 2000
    assert d["lrs"] == [0.01]
    assert d["sampler_state"] == {"pos": 7} and d["wandb_run_id"] == "abc123"
    assert isinstance(d["optimizer_sha256"], str) and len(d["optimizer_sha256"]) == 64
    assert run(pt)["optimizer_sha256"] == d["optimizer_sha256"]

    b = run(bare)
    assert b["batch_idx_train"] is None and b["lrs"] == [] and b["optimizer_sha256"] is None

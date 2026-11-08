"""The "strong resume" tests: exact state and data position across processes and boxes."""

import json
import logging
import shutil
import subprocess
import sys

import pytest

pytest.importorskip("k2")

from prepare_quran_dataset.zipformer import checkpointing as ck  # noqa: E402
from prepare_quran_dataset.zipformer import train_loop as tl  # noqa: E402

needs_rclone = pytest.mark.skipif(shutil.which("rclone") is None, reason="rclone not installed")


@pytest.fixture(autouse=True)
def _no_wandb(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)


def _dump(pt):  # fresh interpreter = "new process"
    out = subprocess.run(
        [sys.executable, "-m", "prepare_quran_dataset.zipformer.checkpointing", "dump", str(pt)],
        check=True, capture_output=True, text=True,
    ).stdout
    return json.loads(out)


def test_state_identical_after_reload_in_new_process(tmp_path, tiny_cfg, tiny_data_dir_big, tokens):
    # tiny_cfg: tiny dims, save_every_n_batches=3, num_workers=2, max_duration=4, num_epochs=1, cpu
    tiny_cfg.output_dir = str(tmp_path / "out")
    snap = tl.run_steps_for_test(tiny_cfg, tiny_data_dir_big, tokens, n_batches=5)
    assert len(snap["cut_ids_per_batch"]) == 5
    pt = ck.latest_complete(tmp_path / "out")[0]
    assert pt.name == "checkpoint-3.pt"
    d = _dump(pt)
    assert d["batch_idx_train"] == snap["batch_idx_train"] == 3
    assert d["batches_consumed_in_epoch"] == 3
    assert d["lrs"] == pytest.approx(snap["lrs"], rel=0, abs=0)
    assert d["optimizer_sha256"] == snap["optimizer_sha256"]
    assert d["sampler_state"] == snap["sampler_state"]
    assert d["sampler_state"] is not None


def test_resumed_run_sees_exact_next_batches(tmp_path, tiny_cfg, tiny_data_dir_big, tokens):
    tiny_cfg.output_dir = str(tmp_path / "out")
    ref = tl.run_steps_for_test(tiny_cfg, tiny_data_dir_big, tokens, n_batches=6)["cut_ids_per_batch"]
    shutil.rmtree(tmp_path / "out")
    first = tl.run_steps_for_test(tiny_cfg, tiny_data_dir_big, tokens, n_batches=3)  # "box dies" after checkpoint at 3
    assert first["cut_ids_per_batch"] == ref[:3]
    saved_lrs = _dump(ck.latest_complete(tmp_path / "out")[0])["lrs"]
    resumed = subprocess.run(
        [sys.executable, "-m", "prepare_quran_dataset.zipformer.train_loop",
         "--test-resume", "--config", str(tmp_path / "out/config.yml"),
         "--data-dir", str(tiny_data_dir_big), "--n-batches", "3"],
        check=True, capture_output=True, text=True,
    )
    r = json.loads(resumed.stdout.splitlines()[-1])
    assert r["start_batch_idx_train"] == 3
    assert r["cut_ids_per_batch"] == ref[3:6]  # no replay, no skip, same order
    assert r["first_lrs"] == saved_lrs  # LR continues, not restarted by warmup
    assert r["batch_idx_train"] == 6
    assert "resume: source=local checkpoint=checkpoint-3.pt epoch=1 batch_idx_train=3" in resumed.stderr


@needs_rclone
def test_new_box_pulls_newest_checkpoint_from_drive(tmp_path, tiny_cfg, tiny_data_dir_big, tokens, local_remote, caplog):
    # local_remote: "testlocal:<tmp>/drive" via RCLONE_CONFIG_TESTLOCAL_TYPE=local
    tiny_cfg.output_dir = str(tmp_path / "boxA")
    tiny_cfg.rclone_checkpoint_path = local_remote
    tl.run_steps_for_test(tiny_cfg, tiny_data_dir_big, tokens, n_batches=6)  # saves at 3 and 6, uploads
    tiny_cfg.output_dir = str(tmp_path / "boxB")  # empty dir = new box
    with caplog.at_level(logging.INFO, logger=tl.__name__):
        pt = tl.resolve_resume(tiny_cfg, tl.make_backup(tiny_cfg))
    assert pt is not None and pt.parent == tmp_path / "boxB"
    meta = ck.read_meta(pt)
    assert meta.batch_idx_train == 6 and ck.verify_sha256(pt, meta)
    assert "resume: source=drive checkpoint=checkpoint-6.pt epoch=1 batch_idx_train=6" in caplog.text


@needs_rclone
def test_local_newer_than_drive_wins(tmp_path, tiny_cfg, tiny_data_dir_big, tokens, local_remote, caplog):
    # upload only the batch-3 checkpoint, keep batch-6 local-only; resolve_resume must return the local batch-6
    tiny_cfg.rclone_checkpoint_path = local_remote
    tl.run_steps_for_test(tiny_cfg, tiny_data_dir_big, tokens, n_batches=3)
    tiny_cfg.rclone_checkpoint_path = None
    snap = tl.run_steps_for_test(tiny_cfg, tiny_data_dir_big, tokens, n_batches=3)
    assert (snap["start_batch_idx_train"], snap["batch_idx_train"]) == (3, 6)
    tiny_cfg.rclone_checkpoint_path = local_remote
    backup = tl.make_backup(tiny_cfg)
    assert backup.latest_remote().batch_idx_train == 3
    with caplog.at_level(logging.INFO, logger=tl.__name__):
        pt = tl.resolve_resume(tiny_cfg, backup)
    assert pt == tmp_path / "out" / "checkpoint-6.pt"
    assert "resume: source=local checkpoint=checkpoint-6.pt epoch=1 batch_idx_train=6" in caplog.text


@needs_rclone
def test_remote_listing_failure_falls_back_to_local(tmp_path, tiny_cfg, tiny_data_dir_big, tokens, caplog):
    tl.run_steps_for_test(tiny_cfg, tiny_data_dir_big, tokens, n_batches=3)
    tiny_cfg.rclone_checkpoint_path = "nonexistent-remote:x"
    with caplog.at_level(logging.INFO, logger=tl.__name__):
        pt = tl.resolve_resume(tiny_cfg, tl.make_backup(tiny_cfg))
    assert pt == tmp_path / "out" / "checkpoint-3.pt"
    assert any(r.levelname == "WARNING" for r in caplog.records)
    assert "resume: source=local" in caplog.text


def test_resolve_resume_nothing(tmp_path, tiny_cfg, caplog):
    with caplog.at_level(logging.INFO, logger=tl.__name__):
        assert tl.resolve_resume(tiny_cfg, None) is None
    assert "resume: source=none checkpoint=- epoch=1 batch_idx_train=0" in caplog.text

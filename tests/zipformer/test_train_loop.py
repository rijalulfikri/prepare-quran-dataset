import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import torch

pytest.importorskip("k2")

from prepare_quran_dataset.zipformer import checkpointing as ck  # noqa: E402
from prepare_quran_dataset.zipformer import train_loop as tl  # noqa: E402
from prepare_quran_dataset.zipformer.datamodule import train_dataloader  # noqa: E402
from prepare_quran_dataset.zipformer.model import build_model  # noqa: E402

CPU = torch.device("cpu")


@pytest.fixture(autouse=True)
def _no_wandb(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)


def test_encode_targets_handles_empty(tokens):
    tgt, lens = tl.encode_targets(["ب", "", "بب"], tokens, CPU)
    assert lens.tolist() == [1, 0, 2] and tgt.numel() == 3
    assert tgt.dtype == lens.dtype == torch.int64
    assert tgt.tolist() == tokens.encode("ب") + tokens.encode("بب")


def test_ctc_loss_finite_with_silence_cut(tiny_model, tokens):
    feats = torch.randn(2, 120, 80)
    lens = torch.tensor([120, 100])
    tgt, tl_ = tl.encode_targets(["ب", ""], tokens, CPU)
    loss, n = tl.compute_ctc_loss(tiny_model, feats, lens, tgt, tl_)
    assert torch.isfinite(loss) and n > 0
    assert int(n) == 220


def test_make_checkpoint_keys(tiny_model, cfg):
    opt, sched = tl.build_optimizer(tiny_model, cfg)
    ckpt = tl.make_checkpoint(tiny_model, opt, sched, tl.TrainState(wandb_run_id="r1"), sampler=None, cfg=cfg)
    assert set(ckpt) == {
        "model", "optimizer", "scheduler", "batch_idx_train", "epoch",
        "batches_consumed_in_epoch", "sampler", "wandb_run_id", "config",
    }
    assert ckpt["sampler"] is None and ckpt["wandb_run_id"] == "r1"
    assert ckpt["config"] == cfg.model_dump(mode="json")


def test_restore_sets_zipformer_batch_count(tiny_model, make_tiny_model, cfg):
    opt, sched = tl.build_optimizer(tiny_model, cfg)
    st = tl.TrainState(epoch=2, batch_idx_train=5000, batches_consumed_in_epoch=10)
    ckpt = tl.make_checkpoint(tiny_model, opt, sched, st, sampler=None, cfg=cfg)
    m2 = make_tiny_model()
    o2, s2 = tl.build_optimizer(m2, cfg)
    st2 = tl.restore(m2, o2, s2, ckpt)
    assert st2.batch_idx_train == 5000
    assert (st2.epoch, st2.batches_consumed_in_epoch) == (2, 10)
    counts = {mod.batch_count for mod in m2.modules() if hasattr(mod, "batch_count")}
    # recipe get_adjusted_batch_count: batch_idx_train * max_duration / ref_duration(600)
    assert counts == {5000 * cfg.max_duration / 600}


def test_restore_refuses_other_frame_rate(tiny_model, tiny_cfg_factory):
    cfg2 = tiny_cfg_factory(output_downsampling_factor=2)
    opt, sched = tl.build_optimizer(tiny_model, cfg2)
    ckpt = tl.make_checkpoint(
        tiny_model, opt, sched, tl.TrainState(), sampler=None,
        cfg=tiny_cfg_factory(output_downsampling_factor=1),
    )
    m2 = build_model(cfg2, 44)
    before = {k: v.clone() for k, v in m2.state_dict().items()}
    with pytest.raises(tl.IncompatibleCheckpoint):
        tl.restore(m2, opt, sched, ckpt)
    assert all(torch.equal(before[k], v) for k, v in m2.state_dict().items())


def test_frame_budget_abort(tiny_cfg, tiny_data_dir_infeasible, tokens):
    # tiny_data_dir_infeasible: 10 speech cuts, 2 of them 0.3 s with 10-char repeated labels
    tiny_cfg.max_dropped_fraction = 0.1
    with pytest.raises(tl.FrameBudgetError):
        tl.run_training(tiny_cfg, tiny_data_dir_infeasible, tokens, CPU)
    dropped = json.loads((Path(tiny_cfg.output_dir) / "dropped_cuts.json").read_text())
    assert dropped["factor"] == 1 and dropped["speed"] == 1.1
    assert sorted(dropped["train"]) == sorted(dropped["dev"]) == ["moshaf_6.0__8", "moshaf_6.0__9"]
    assert not list(ck.list_complete(tiny_cfg.output_dir))


def test_epoch_checkpoint_resumes_next_epoch(tiny_model, cfg):
    opt, sched = tl.build_optimizer(tiny_model, cfg)
    st = tl.TrainState(epoch=3, batch_idx_train=9000, batches_consumed_in_epoch=3000)
    ckpt = tl.make_checkpoint(tiny_model, opt, sched, st, sampler=None, cfg=cfg)
    ckpt["kind"] = "epoch"
    st2 = tl.restore(tiny_model, opt, sched, ckpt)
    assert (st2.epoch, st2.batches_consumed_in_epoch, st2.sampler_state) == (4, 0, None)
    assert st2.batch_idx_train == 9000


class _FakeBackup:
    def __init__(self, pending):
        self.pending, self.submitted, self.flushed = set(pending), [], []
        self.consecutive_failures, self.last_error = 0, None

    def submit(self, pt):
        self.submitted.append(pt)

    def pending_paths(self):
        return set(self.pending)

    def flush(self, timeout):
        self.flushed.append(timeout)
        return True


def test_save_and_backup_never_prunes_pending_upload(tmp_path, tiny_model, tiny_cfg):
    tiny_cfg.keep_last_k = 1
    opt, sched = tl.build_optimizer(tiny_model, tiny_cfg)
    out = tmp_path / "out"
    first = tl.save_and_backup(
        tl.make_checkpoint(tiny_model, opt, sched, tl.TrainState(batch_idx_train=3), None, tiny_cfg),
        "batch", tiny_cfg, None,
    )
    assert first == out / "checkpoint-3.pt" and ck.read_meta(first).kind == "batch"
    backup = _FakeBackup(pending={first})  # checkpoint-3 still uploading
    second = tl.save_and_backup(
        tl.make_checkpoint(tiny_model, opt, sched, tl.TrainState(batch_idx_train=6), None, tiny_cfg),
        "batch", tiny_cfg, backup,
    )
    assert backup.submitted == [second]
    assert first.exists() and second.exists()
    backup.pending.clear()
    third = tl.save_and_backup(
        tl.make_checkpoint(tiny_model, opt, sched, tl.TrainState(batch_idx_train=9), None, tiny_cfg),
        "batch", tiny_cfg, backup,
    )
    assert [p.name for p, _ in ck.list_complete(out)] == [third.name]


def test_graceful_stop_saves_flushes_and_raises(tiny_cfg, tiny_data_dir_big, tokens, tiny_model):
    tiny_cfg.save_every_n_batches = 1000
    out = tiny_cfg.output_dir
    opt, sched = tl.build_optimizer(tiny_model, tiny_cfg)
    loader = train_dataloader(tiny_cfg, tiny_data_dir_big, epoch=1)
    stop = threading.Event()
    stop.set()
    backup = _FakeBackup(pending=())
    st = tl.TrainState()
    with pytest.raises(tl.TrainingInterrupted):
        tl.train_one_epoch(model=tiny_model, opt=opt, sched=sched, loader=loader, st=st, cfg=tiny_cfg,
                           tokens=tokens, device=CPU, logger=tl.NullLogger(), backup=backup, stop_flag=stop)
    del loader
    path, meta = ck.latest_complete(out)
    assert path.name == "checkpoint-1.pt" and meta.batches_consumed_in_epoch == 1
    assert backup.submitted == [path] and backup.flushed == [tiny_cfg.shutdown_flush_seconds]
    assert torch.load(path, weights_only=False)["sampler"] is not None


def test_install_signal_handlers_sets_flag():
    old_term, old_int = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)
    try:
        stop = threading.Event()
        tl.install_signal_handlers(stop)
        os.kill(os.getpid(), signal.SIGTERM)
        assert stop.wait(5)
        stop.clear()
        os.kill(os.getpid(), signal.SIGINT)  # first SIGINT: graceful
        assert stop.wait(5)
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
    finally:
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGINT, old_int)


def test_make_logger_without_wandb_key_uses_tensorboard(tiny_cfg, caplog):
    with caplog.at_level("WARNING"):
        logger, run_id = tl.make_logger(tiny_cfg, None)
    try:
        assert isinstance(logger, tl.TensorBoardLogger) and run_id is None
        logger.log({"train/loss": 1.0}, step=1)
    finally:
        logger.close()
    assert any("WANDB_API_KEY" in r.getMessage() for r in caplog.records)
    assert list((Path(tiny_cfg.output_dir) / "tb").glob("events.*"))


def test_average_last_epochs(tmp_path, tiny_model, tiny_cfg):
    opt, sched = tl.build_optimizer(tiny_model, tiny_cfg)
    states = []
    for e in (1, 2, 3):
        with torch.no_grad():
            for p in tiny_model.parameters():
                p.add_(1.0)
        ckpt = tl.make_checkpoint(tiny_model, opt, sched, tl.TrainState(epoch=e, batch_idx_train=10 * e), None, tiny_cfg)
        states.append({k: v.clone() for k, v in ckpt["model"].items()})
        tl.save_and_backup(ckpt, "epoch", tiny_cfg, None)
    out = tl.average_last_epochs(tmp_path / "out", 2)
    assert out == tmp_path / "out" / "averaged-2.pt"
    avg = torch.load(out, weights_only=False)
    assert avg["averaged_from"] == ["epoch-2.pt", "epoch-3.pt"]
    assert avg["config"] == tiny_cfg.model_dump(mode="json")
    name = next(k for k, v in states[0].items() if v.is_floating_point())
    assert torch.allclose(avg["model"][name], (states[1][name] + states[2][name]) / 2)


def test_run_training_end_to_end_then_skip_when_complete(tiny_cfg, tiny_data_dir, tokens, monkeypatch):
    out = Path(tiny_cfg.output_dir)
    avg = tl.run_training(tiny_cfg, tiny_data_dir, tokens, CPU)
    assert avg == out / "averaged-1.pt" and avg.exists()
    assert (out / "config.yml").exists() and (out / "dropped_cuts.json").exists()
    assert (out / "epoch-1.pt").exists()
    lines = [json.loads(x) for x in (out / "dev_metrics.jsonl").read_text().splitlines()]
    assert [x["epoch"] for x in lines] == [1]
    assert {"per_phonemes", "silence_per", "num_speech", "num_silence", "batch_idx_train"} <= set(lines[0])
    epoch_meta = ck.read_meta(out / "epoch-1.pt")
    assert lines[0]["batch_idx_train"] == epoch_meta.batch_idx_train > 0

    # all epochs complete: no training, dev eval re-run for the missing epoch, averaged again
    (out / "dev_metrics.jsonl").unlink()
    avg.unlink()

    def boom(*a, **k):
        raise AssertionError("must not train when all epochs are complete")

    monkeypatch.setattr(tl, "train_one_epoch", boom)
    assert tl.run_training(tiny_cfg, tiny_data_dir, tokens, CPU) == avg and avg.exists()
    lines = [json.loads(x) for x in (out / "dev_metrics.jsonl").read_text().splitlines()]
    assert [x["epoch"] for x in lines] == [1]


# ---------------------------------------------------------------- review fix round 1


def _inf_loss(model, feats, feat_lens, targets, target_lengths):
    return torch.tensor(float("inf")), feat_lens.sum()


def _run_epoch(tiny_cfg, data_dir, tokens, model, max_batches=None):
    opt, sched = tl.build_optimizer(model, tiny_cfg)
    loader = train_dataloader(tiny_cfg, data_dir, epoch=1)
    st = tl.TrainState()
    try:
        tl._train_batches(model, opt, sched, loader, st, tiny_cfg, tokens, CPU, tl.NullLogger(), None, None,
                          max_batches=max_batches)
    finally:
        del loader
    return st


def test_divergence_raises_before_saving(tiny_cfg, tiny_data_dir_big, tokens, tiny_model, monkeypatch):
    tiny_cfg.save_every_n_batches = tl.MAX_NONFINITE_BATCHES  # a save is due at the 20th bad batch
    monkeypatch.setattr(tl, "compute_ctc_loss", _inf_loss)
    with pytest.raises(tl.DivergedTraining, match="20 consecutive"):
        _run_epoch(tiny_cfg, tiny_data_dir_big, tokens, tiny_model)
    assert ck.latest_complete(tiny_cfg.output_dir) is None


def test_nonfinite_streak_resets_on_finite_loss(tiny_cfg, tiny_data_dir_big, tokens, tiny_model, monkeypatch):
    tiny_cfg.save_every_n_batches = 1000
    real = tl.compute_ctc_loss
    calls = []

    def flaky(*args):
        calls.append(1)
        return real(*args) if len(calls) == 15 else _inf_loss(*args)  # 14 bad, 1 good, then bad

    monkeypatch.setattr(tl, "compute_ctc_loss", flaky)
    st = _run_epoch(tiny_cfg, tiny_data_dir_big, tokens, tiny_model, max_batches=30)
    assert st.batch_idx_train == 30 and st.nonfinite_streak == 15


def test_nan_weights_are_never_checkpointed(tiny_cfg, tiny_data_dir_big, tokens, tiny_model):
    tiny_cfg.save_every_n_batches = 1
    with torch.no_grad():
        next(tiny_model.parameters()).fill_(float("nan"))
    with pytest.raises(tl.DivergedTraining, match="non-finite weights"):
        _run_epoch(tiny_cfg, tiny_data_dir_big, tokens, tiny_model, max_batches=2)
    assert ck.latest_complete(tiny_cfg.output_dir) is None


def test_torn_dev_metrics_line_is_skipped(tiny_cfg, tiny_data_dir, tokens, caplog):
    out = Path(tiny_cfg.output_dir)
    out.mkdir(parents=True)
    (out / "dev_metrics.jsonl").write_text('{"epoch": 1, "per_phonemes": 0.5}\n{"epoch": 2, "per_ph')  # crash mid-write
    with caplog.at_level("WARNING"):
        assert tl._evaluated_epochs(out) == {1}
    assert any("unparsable" in r.getMessage() for r in caplog.records)
    tl.run_training(tiny_cfg, tiny_data_dir, tokens, CPU)  # start path must not choke on it
    lines = (out / "dev_metrics.jsonl").read_text().splitlines()
    assert len(lines) == 3 and json.loads(lines[-1])["epoch"] == 1  # new row not glued onto the torn one
    assert tl._evaluated_epochs(out) == {1}


def test_stop_during_dev_eval_flushes_and_stops(tiny_cfg, tiny_data_dir, tokens, monkeypatch):
    tiny_cfg.num_epochs = 2
    flags, flushed = [], []
    monkeypatch.setattr(tl, "install_signal_handlers", flags.append)
    real_dev_eval = tl._dev_eval

    def dev_eval_then_sigterm(*args, **kwargs):
        row = real_dev_eval(*args, **kwargs)
        flags[0].set()  # SIGTERM arrives while the epoch is being evaluated
        return row

    monkeypatch.setattr(tl, "_dev_eval", dev_eval_then_sigterm)
    backup = _FakeBackup(pending=())
    backup.flush = lambda timeout: flushed.append(timeout) or True
    monkeypatch.setattr(tl, "make_backup", lambda cfg: backup)
    with pytest.raises(tl.TrainingInterrupted):
        tl.run_training(tiny_cfg, tiny_data_dir, tokens, CPU)
    path, meta = ck.latest_complete(tiny_cfg.output_dir)
    assert path.name == "epoch-1.pt" and meta.kind == "epoch"
    assert backup.submitted[-1] == path
    assert flushed == [tiny_cfg.shutdown_flush_seconds]


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT], ids=["SIGTERM", "SIGINT"])
def test_signal_to_process_group_stops_gracefully(tmp_path, tiny_cfg, tiny_data_dir_big, sig):
    # preemption (SIGTERM) / Ctrl-C (SIGINT) signal the whole process group, DataLoader workers
    # included; a worker killed by SIGTERM makes torch's SIGCHLD handler crash the main loop
    tiny_cfg.num_epochs = 50
    tiny_cfg.save_every_n_batches = 2
    tiny_cfg.to_yaml(tmp_path / "cfg.yml")
    out = Path(tiny_cfg.output_dir)
    env = {k: v for k, v in os.environ.items() if k != "WANDB_API_KEY"}
    cmd = [sys.executable, "-m", "prepare_quran_dataset.zipformer.train_loop", "--config", str(tmp_path / "cfg.yml"),
           "--data-dir", str(tiny_data_dir_big), "--device", "cpu"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            start_new_session=True, env=env)
    try:
        deadline = time.monotonic() + 90
        while ck.read_meta(out / "checkpoint-2.pt") is None:  # workers up, batches flowing
            assert proc.poll() is None and time.monotonic() < deadline, "training did not reach batch 2"
            time.sleep(0.1)
        os.killpg(proc.pid, sig)
        _, stderr = proc.communicate(timeout=90)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    assert "exited unexpectedly" not in stderr
    assert proc.returncode == 130, stderr[-3000:]
    stopped_at = int(re.search(r"stopped at batch_idx_train=(\d+)", stderr).group(1))
    path, meta = ck.latest_complete(out)
    assert path.name == f"checkpoint-{stopped_at}.pt" and meta.batch_idx_train == stopped_at >= 2


def test_backup_health():
    from types import SimpleNamespace

    assert tl.backup_health(SimpleNamespace(consecutive_failures=0, last_error="old")) == "backup: ok"
    assert (
        tl.backup_health(SimpleNamespace(consecutive_failures=4, last_error="quota exceeded"))
        == "backup: FAILING (4 consecutive, last error quota exceeded)"
    )


def test_finish_epoch_logs_backup_health(monkeypatch, caplog):
    from types import SimpleNamespace

    for name in ("_check_finite_weights", "save_and_backup", "make_checkpoint", "_dev_eval"):
        monkeypatch.setattr(tl, name, lambda *a, **k: None)
    st = tl.TrainState(epoch=2)
    backup = SimpleNamespace(consecutive_failures=3, last_error="quota exceeded")
    with caplog.at_level("INFO", logger=tl.__name__):
        tl._finish_epoch(None, None, None, st, None, backup, None, None, None, None)
        tl._finish_epoch(None, None, None, st, None, None, None, None, None, None)
    assert "epoch 2 finished; backup: FAILING (3 consecutive, last error quota exceeded)" in caplog.text
    assert "epoch 3 finished" not in caplog.text  # no backup configured -> no health line
    assert st.epoch == 4

# zipformer/tests/test_cli.py
import types
from pathlib import Path

import pytest

import train_zipformer as tz


def test_stage_expansion():
    assert tz.stages_to_run("all") == ["prepare", "train", "eval", "export"]
    assert tz.stages_to_run("export") == ["export"]


def test_defaults():
    a = tz.parse_args([])
    assert a.config.endswith("configs/train_config.yml") and a.stage == "all" and not a.push_to_hub


@pytest.fixture
def shards_unavailable(monkeypatch):
    """The real config sets shard sources; make every shard fetch fail without network."""
    from quran_zipformer import data_shards

    def fail(*args, **kwargs):
        raise OSError("no network in tests")

    monkeypatch.setattr(data_shards, "fetch_shards", fail)


def test_prepare_without_rclone_fails_with_message(monkeypatch, capsys, shards_unavailable):
    monkeypatch.setattr(tz.shutil, "which", lambda name: None)

    def no_subprocess(*args, **kwargs):
        raise AssertionError("subprocess must not run when rclone is missing")

    monkeypatch.setattr(tz.subprocess, "run", no_subprocess)
    code = tz.main(["--stage", "prepare"])
    assert code != 0
    assert "rclone" in capsys.readouterr().err


def test_prepare_without_remote_fails(monkeypatch, capsys, shards_unavailable):
    monkeypatch.setattr(tz.shutil, "which", lambda name: "/usr/bin/rclone")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return tz.subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(tz.subprocess, "run", fake_run)
    code = tz.main(["--stage", "prepare"])
    assert code != 0
    assert "rclone" in capsys.readouterr().err
    assert all(cmd[:2] != ["rclone", "sync"] for cmd in calls)


def test_train_interrupted_exits_143(monkeypatch):
    from quran_zipformer.train_loop import TrainingInterrupted

    def interrupted(cfg, args):
        raise TrainingInterrupted("stopped at batch 7")

    monkeypatch.setattr(tz, "stage_train", interrupted)
    assert tz.main(["--stage", "train", "--device", "cpu"]) == 143


class _Sup:
    def __init__(self, input_type):
        self.custom = {"input_type": input_type}


class _Cut:
    def __init__(self, supervisions):
        self.supervisions = supervisions

    def load_audio(self):
        return [[0.0]]


def _parity_wavs_over(monkeypatch, cuts):
    from quran_zipformer import datamodule

    monkeypatch.setattr(datamodule, "load_cuts", lambda data_dir, split: cuts)
    return tz._parity_wavs(types.SimpleNamespace(data_dir="d"))


def test_parity_wavs_without_speech_cuts_raises(monkeypatch):
    with pytest.raises(RuntimeError, match="no speech cuts"):
        _parity_wavs_over(monkeypatch, [_Cut([]), _Cut([_Sup("silence")])])


def test_parity_wavs_skips_silence_and_unsupervised_cuts(monkeypatch, caplog):
    wavs = _parity_wavs_over(monkeypatch, [_Cut([]), _Cut([_Sup("silence")]), _Cut([_Sup("speech")])])
    assert len(wavs) == 1
    assert "only 1 test speech cuts" in caplog.text


def test_prepare_syncs_with_drive_tuned_flags(monkeypatch):
    monkeypatch.setattr(tz, "check_rclone", lambda: None)
    calls = []
    monkeypatch.setattr(tz.subprocess, "run", lambda cmd, **kwargs: calls.append(cmd))
    cfg = types.SimpleNamespace(
        rclone_remote_path="gdrive:data", data_dir="data-zipformer", hf_data_repo=None, drive_data_shards_path=None
    )
    tz.stage_prepare(cfg, tz.parse_args([]))
    assert calls == [RCLONE_SYNC]


RCLONE_SYNC = [
    "rclone", "sync", "gdrive:data", "data-zipformer", "--progress",
    "--transfers", "32", "--checkers", "32", "--fast-list",
    "--drive-pacer-min-sleep", "10ms", "--drive-pacer-burst", "200",
    "--exclude", "/checkpoints/**",
]


def _prepare_with_shards(monkeypatch, failing_sources=()):
    """Run stage_prepare with HF + Drive shard sources; names in `failing_sources` raise."""
    from quran_zipformer import data_shards

    monkeypatch.setattr(tz, "check_rclone", lambda: None)
    calls, tried = [], []
    monkeypatch.setattr(tz.subprocess, "run", lambda cmd, **kwargs: calls.append(cmd))

    def fetch(source, data_dir):
        tried.append(source.name)
        if source.name in failing_sources:
            raise OSError(f"{source.name} down")

    monkeypatch.setattr(data_shards, "fetch_shards", fetch)
    cfg = types.SimpleNamespace(
        rclone_remote_path="gdrive:data", data_dir="data-zipformer",
        hf_data_repo="u/data", drive_data_shards_path="gdrive:shards",
    )
    tz.stage_prepare(cfg, tz.parse_args([]))
    return tried, calls


def test_prepare_uses_hf_shards_first(monkeypatch):
    tried, calls = _prepare_with_shards(monkeypatch)
    assert tried == ["hf:u/data"]
    assert calls == []


def test_prepare_falls_back_to_drive_shards_when_hf_fails(monkeypatch, caplog):
    tried, calls = _prepare_with_shards(monkeypatch, failing_sources={"hf:u/data"})
    assert tried == ["hf:u/data", "gdrive:shards"]
    assert calls == []
    assert "trying the next source" in caplog.text


def test_prepare_falls_back_to_rclone_sync_when_all_shards_fail(monkeypatch, caplog):
    tried, calls = _prepare_with_shards(monkeypatch, failing_sources={"hf:u/data", "gdrive:shards"})
    assert tried == ["hf:u/data", "gdrive:shards"]
    assert calls == [RCLONE_SYNC]
    assert "falling back to rclone sync" in caplog.text


class _DotenvLoaded(Exception):
    pass


def _raise_loaded(*args, **kwargs):
    raise _DotenvLoaded


def test_main_loads_dotenv_first(monkeypatch):
    monkeypatch.setattr(tz, "load_dotenv", _raise_loaded)
    with pytest.raises(_DotenvLoaded):
        tz.main(["--stage", "train", "--config", "does-not-exist.yml"])


def test_train_loop_main_loads_dotenv_first(monkeypatch):
    pytest.importorskip("k2")
    from quran_zipformer import train_loop as tl

    monkeypatch.setattr(tl, "load_dotenv", _raise_loaded)
    with pytest.raises(_DotenvLoaded):
        tl.main(["--config", "does-not-exist.yml"])


# ---------------------------------------------------------------- eval sets on old checkpoints


def test_evalsets_stage_runs_prepare_first_and_is_not_in_all():
    assert tz.stages_to_run("evalsets") == ["prepare", "evalsets"]
    assert "evalsets" not in tz.stages_to_run("all")
    assert tz.parse_args(["--stage", "evalsets", "--checkpoint", "a.pt", "--checkpoint", "r:b.pt"]).checkpoint == ["a.pt", "r:b.pt"]


def test_evalsets_without_checkpoint_fails(monkeypatch):
    monkeypatch.setattr(tz, "stage_prepare", lambda cfg, args: None)
    assert tz.main(["--stage", "evalsets", "--device", "cpu"]) == 1


def test_local_checkpoint_downloads_a_remote_once(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)
        Path(cmd[3]).write_text("pt")

    monkeypatch.setattr(tz, "check_rclone", lambda: None)
    monkeypatch.setattr(tz.subprocess, "run", fake_run)
    src = "remote:runs/v0-results/averaged-3.pt"
    assert tz._local_checkpoint(src, tmp_path) == tmp_path / "averaged-3.pt"
    assert tz._local_checkpoint(src, tmp_path) == tmp_path / "averaged-3.pt"
    assert len(calls) == 1 and calls[0][:3] == ["rclone", "copyto", src]
    local = tmp_path / "local.pt"
    local.write_text("x")
    assert tz._local_checkpoint(str(local), tmp_path / "cache") == local
    with pytest.raises(FileNotFoundError):
        tz._local_checkpoint("missing.pt", tmp_path)


@pytest.fixture
def eval_box(tmp_path, tiny_cfg_factory, tiny_data_dir_eval, fake_qdat):
    """A tiny config with every eval set over a copy of `tiny_data_dir_eval`, plus a tiny model checkpoint."""
    import shutil

    import torch

    pytest.importorskip("k2")
    from quran_zipformer.model import build_model

    data = tmp_path / "data"
    shutil.copytree(tiny_data_dir_eval, data)
    cfg = tiny_cfg_factory(
        data_dir=str(data), output_dir=str(tmp_path / "out"), num_workers=0, max_duration=10, avg_last_k=1, keep_last_k=1,
        dev_unseen_cuts_per_moshaf=2, eval_qdat=True, stress_sets=["vol-20db", "reverb", "speed1.25"],
    )
    yml = tmp_path / "cfg.yml"
    cfg.to_yaml(yml)
    pt = tmp_path / "epoch-30.pt"
    torch.save({"model": build_model(cfg, 44).state_dict(), "epoch": 30}, pt)
    return cfg, yml, pt


SET_KEYS = {"per_phonemes", "silence_per", "num_speech", "num_silence",
            *(f"{n}/{k}" for n in ("dev_unseen", "dev_unseen_vol-20db", "dev_unseen_reverb", "dev_unseen_speed1.25", "qdat")
              for k in ("per_phonemes", "num_speech"))}


def test_evalsets_scores_an_old_checkpoint_without_training(eval_box, monkeypatch):
    import json

    cfg, yml, pt = eval_box
    monkeypatch.setattr(tz, "stage_prepare", lambda cfg, args: None)
    monkeypatch.setattr(tz, "stage_train", lambda cfg, args: pytest.fail("must not train"))
    assert tz.main(["--config", str(yml), "--stage", "evalsets", "--checkpoint", str(pt), "--device", "cpu"]) == 0
    row = json.loads((Path(cfg.output_dir) / "eval_sets" / "epoch-30.json").read_text())
    assert row["checkpoint"] == str(pt) and row["epoch"] == 30
    assert SET_KEYS <= set(row)
    assert row["qdat/num_speech"] == 2 and row["dev_unseen/num_speech"] == 6


def test_eval_stage_writes_eval_set_results_for_the_averaged_model(eval_box, monkeypatch):
    import json
    import shutil

    cfg, yml, pt = eval_box
    out = Path(cfg.output_dir)
    out.mkdir(parents=True)
    shutil.copy(pt, out / "averaged-1.pt")
    (out / "test_results.json").write_text("{}")  # test split already scored: only the eval sets run
    assert tz.main(["--config", str(yml), "--stage", "eval", "--device", "cpu"]) == 0
    row = json.loads((out / tz.EVAL_SETS_RESULTS).read_text())
    assert row["checkpoint"] == "averaged-1.pt" and SET_KEYS <= set(row)
    assert (out / "test_results.json").read_text() == "{}"

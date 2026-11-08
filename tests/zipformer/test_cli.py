# tests/zipformer/test_cli.py
import types

import pytest

import train_zipformer as tz


def test_stage_expansion():
    assert tz.stages_to_run("all") == ["prepare", "train", "eval", "export"]
    assert tz.stages_to_run("export") == ["export"]


def test_defaults():
    a = tz.parse_args([])
    assert a.config.endswith("configs/train/zipformer/train_config.yml") and a.stage == "all" and not a.push_to_hub


def test_prepare_without_rclone_fails_with_message(monkeypatch, capsys):
    monkeypatch.setattr(tz.shutil, "which", lambda name: None)

    def no_subprocess(*args, **kwargs):
        raise AssertionError("subprocess must not run when rclone is missing")

    monkeypatch.setattr(tz.subprocess, "run", no_subprocess)
    code = tz.main(["--stage", "prepare"])
    assert code != 0
    assert "rclone" in capsys.readouterr().err


def test_prepare_without_remote_fails(monkeypatch, capsys):
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
    from prepare_quran_dataset.zipformer.train_loop import TrainingInterrupted

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
    from prepare_quran_dataset.zipformer import datamodule

    monkeypatch.setattr(datamodule, "load_cuts", lambda data_dir, split: cuts)
    return tz._parity_wavs(types.SimpleNamespace(data_dir="d"))


def test_parity_wavs_without_speech_cuts_raises(monkeypatch):
    with pytest.raises(RuntimeError, match="no speech cuts"):
        _parity_wavs_over(monkeypatch, [_Cut([]), _Cut([_Sup("silence")])])


def test_parity_wavs_skips_silence_and_unsupervised_cuts(monkeypatch, caplog):
    wavs = _parity_wavs_over(monkeypatch, [_Cut([]), _Cut([_Sup("silence")]), _Cut([_Sup("speech")])])
    assert len(wavs) == 1
    assert "only 1 test speech cuts" in caplog.text

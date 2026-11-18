# zipformer/tests/test_cli.py
import types

import pytest

import train_zipformer as tz


def test_stage_expansion():
    assert tz.stages_to_run("all") == ["prepare", "train", "eval", "export"]
    assert tz.stages_to_run("export") == ["export"]


def test_defaults():
    a = tz.parse_args([])
    assert a.config.endswith("configs/train_config.yml") and a.stage == "all" and not a.push_to_hub


@pytest.fixture
def hf_unavailable(monkeypatch):
    """The real config sets hf_data_repo; make the HF fetch fail without network."""
    from quran_zipformer import hf_data

    def fail(*args, **kwargs):
        raise OSError("no network in tests")

    monkeypatch.setattr(hf_data, "fetch_hf_shards", fail)


def test_prepare_without_rclone_fails_with_message(monkeypatch, capsys, hf_unavailable):
    monkeypatch.setattr(tz.shutil, "which", lambda name: None)

    def no_subprocess(*args, **kwargs):
        raise AssertionError("subprocess must not run when rclone is missing")

    monkeypatch.setattr(tz.subprocess, "run", no_subprocess)
    code = tz.main(["--stage", "prepare"])
    assert code != 0
    assert "rclone" in capsys.readouterr().err


def test_prepare_without_remote_fails(monkeypatch, capsys, hf_unavailable):
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
    cfg = types.SimpleNamespace(rclone_remote_path="gdrive:data", data_dir="data-zipformer", hf_data_repo=None)
    tz.stage_prepare(cfg, tz.parse_args([]))
    assert calls == [RCLONE_SYNC]


RCLONE_SYNC = [
    "rclone", "sync", "gdrive:data", "data-zipformer", "--progress",
    "--transfers", "32", "--checkers", "32", "--fast-list",
    "--drive-pacer-min-sleep", "10ms", "--drive-pacer-burst", "200",
    "--exclude", "/checkpoints/**",
]


def _prepare_with_hf(monkeypatch, fetch):
    from quran_zipformer import hf_data

    monkeypatch.setattr(tz, "check_rclone", lambda: None)
    calls = []
    monkeypatch.setattr(tz.subprocess, "run", lambda cmd, **kwargs: calls.append(cmd))
    monkeypatch.setattr(hf_data, "fetch_hf_shards", fetch)
    cfg = types.SimpleNamespace(rclone_remote_path="gdrive:data", data_dir="data-zipformer", hf_data_repo="u/data")
    tz.stage_prepare(cfg, tz.parse_args([]))
    return calls


def test_prepare_uses_hf_shards_without_rclone(monkeypatch):
    fetched = []
    calls = _prepare_with_hf(monkeypatch, lambda repo, data_dir: fetched.append((repo, str(data_dir))))
    assert fetched == [("u/data", "data-zipformer")]
    assert calls == []


def test_prepare_falls_back_to_rclone_when_hf_fails(monkeypatch, caplog):
    def fail(repo, data_dir):
        raise OSError("hub down")

    calls = _prepare_with_hf(monkeypatch, fail)
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

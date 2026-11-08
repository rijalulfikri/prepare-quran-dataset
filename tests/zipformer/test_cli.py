# tests/zipformer/test_cli.py
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

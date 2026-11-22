"""A config newer than the code that trains it must fail loudly, not lose its new fields."""

import importlib.util
from pathlib import Path

import pytest
import yaml

from quran_zipformer.config import ZipformerTrainConfig

ZIPFORMER = Path(__file__).resolve().parents[1]
YAML = ZIPFORMER / "configs" / "train_config.yml"


@pytest.fixture(scope="module")
def launcher():
    spec = importlib.util.spec_from_file_location("run_zipformer_training", ZIPFORMER / "vastai" / "run_zipformer_training.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_from_yaml_rejects_keys_the_code_does_not_know(tmp_path):
    data = yaml.safe_load(YAML.read_text(encoding="utf-8"))
    data["pad_pp"] = 0.3  # a typo of pad_p
    path = tmp_path / "typo.yml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="pad_pp"):
        ZipformerTrainConfig.from_yaml(path)


def test_robustness_summary_names_what_is_on_and_what_is_off():
    on = ZipformerTrainConfig.from_yaml(YAML).robustness_summary()
    assert len(on) == 6 and not any("MATI" in line for line in on)
    assert any("pad_p=0.3" in line and "pad_lead_seconds=(0.5, 4.0)" in line for line in on)
    assert any("concat_p=0.3" in line and "concat_max_seconds=45.0" in line for line in on)
    off = ZipformerTrainConfig.from_yaml(YAML.parent / "eval_baseline_v0.yml").robustness_summary()
    assert all("MATI" in line for line in off)  # the v0 pipeline trained against none of them


def test_train_script_validates_the_config_before_fetching_data():
    script = (ZIPFORMER / "train_zipformer.sh").read_text(encoding="utf-8")
    check = script.index("ZipformerTrainConfig.from_yaml(sys.argv[1]).robustness_summary()")
    assert check < script.index("python train_zipformer.py --config")  # before the pipeline (and its data sync)
    import subprocess

    assert subprocess.run(["bash", "-n", str(ZIPFORMER / "train_zipformer.sh")]).returncode == 0


def test_launcher_finds_config_keys_unknown_to_the_code_it_will_train(launcher):
    keys = yaml.safe_load(YAML.read_text(encoding="utf-8"))
    current = (ZIPFORMER / "quran_zipformer" / "config.py").read_text(encoding="utf-8")
    assert launcher.unknown_config_keys(keys, current) == []
    # the same config against code from before the robustness fixes
    old = "\n".join(line for line in current.splitlines()
                    if not line.strip().startswith(("pad_", "tempo_", "clip_", "nonspeech_", "concat_max_seconds")))
    assert launcher.unknown_config_keys(keys, old) == [
        "clip_p", "clip_peak_fraction", "concat_max_seconds", "nonspeech_noise_p", "nonspeech_snr",
        "pad_lead_seconds", "pad_p", "pad_trail_seconds", "tempo_p", "tempo_range",
    ]


def test_launcher_refuses_a_config_ahead_of_the_commit(launcher, monkeypatch, capsys):
    class Shown:
        returncode = 0
        stdout = "class ZipformerTrainConfig(BaseModel):\n    seed: int = 42\n    pad_p: float = 0.0\n"

    monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **k: Shown())
    launcher.check_config_matches_commit({"seed": 1, "pad_p": 0.3}, "abc1234def", "main")  # all known: no exit
    with pytest.raises(SystemExit):
        launcher.check_config_matches_commit({"seed": 1, "tempo_p": 0.3}, "abc1234def", "main")
    assert "tempo_p" in capsys.readouterr().out
    launcher.check_config_matches_commit({"tempo_p": 0.3}, "abc1234def", "main", dry_run=True)  # dry run only reports

    Shown.returncode = 1  # commit not fetched locally: a warning, no exit
    launcher.check_config_matches_commit({"tempo_p": 0.3}, "abc1234def", "main")
    assert "tidak bisa dicek" in capsys.readouterr().out

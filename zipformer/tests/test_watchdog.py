"""Early-stop decision of vastai/zipformer_watchdog.sh, exercised through its DRY_RUN mode."""

import json
import os
import subprocess
from pathlib import Path

import pytest

WATCHDOG = Path(__file__).resolve().parents[1] / "vastai" / "zipformer_watchdog.sh"

# dev PER of the first real run (2026-11-18), epochs 1-19
REAL = [0.065669, 0.040449, 0.027945, 0.033063, 0.029999, 0.016026, 0.014973, 0.010971, 0.010333,
        0.010578, 0.008002, 0.007343, 0.009524, 0.006470, 0.006800, 0.004906, 0.005182, 0.004757, 0.005246]


def _dry_run(tmp_path, pers, num_epochs=30, **env):
    zf = tmp_path / "zipformer"
    (zf / "configs").mkdir(parents=True)
    (zf / "configs" / "train_config.yml").write_text(
        f'num_epochs: {num_epochs}\noutput_dir: "./results"   # comment\n', encoding="utf-8"
    )
    (zf / "results").mkdir()
    (zf / "results" / "dev_metrics.jsonl").write_text(
        "".join(json.dumps({"epoch": i, "per_phonemes": p}) + "\n" for i, p in enumerate(pers, 1)),
        encoding="utf-8",
    )
    full_env = {**os.environ, "DRY_RUN": "1", "REPO_ZIPFORMER_DIR": str(zf), "WLOG": str(tmp_path / "w.log"), **env}
    out = subprocess.run(["bash", str(WATCHDOG)], env=full_env, capture_output=True, text=True, check=True).stdout
    return out.strip().splitlines()[-1].split("dry run: ")[-1]


def test_small_gains_do_not_reset_the_count(tmp_path):
    # epoch 18 is only 3 % better than epoch 16, so epoch 16 stays the best
    line = _dry_run(tmp_path, REAL)
    assert "best 0.004906 at epoch 16, epochs since best 3/4, would_trigger 0" in line


def test_triggers_after_patience_epochs_without_a_5pct_gain(tmp_path):
    line = _dry_run(tmp_path, REAL + [0.0047])
    assert "epochs since best 4/4, would_trigger 1" in line


def test_a_5pct_gain_is_a_new_best(tmp_path):
    line = _dry_run(tmp_path, REAL + [0.0046])
    assert "best 0.004600 at epoch 20, epochs since best 0/4, would_trigger 0" in line


def test_zero_threshold_counts_any_gain(tmp_path):
    line = _dry_run(tmp_path, REAL, MIN_REL_IMPROVEMENT="0")
    assert "best 0.004757 at epoch 18, epochs since best 1/4" in line


def test_never_triggers_once_all_epochs_are_trained(tmp_path):
    # after a stop num_epochs equals the last epoch: averaging / eval / export must not be cut
    line = _dry_run(tmp_path, REAL + [0.0047], num_epochs=20)
    assert "last epoch 20 of 20" in line and "would_trigger 0" in line


def test_reads_the_pipeline_config_from_config_file(tmp_path):
    # run_zipformer_training.py --config: the box trains from /root/zipformer_train_config.yml
    # (CONFIG_FILE); the repo config must be ignored, including its output_dir and num_epochs
    sent = tmp_path / "sent_config.yml"
    sent.write_text(f'num_epochs: 20\noutput_dir: "{tmp_path}/zipformer/results"\n', encoding="utf-8")
    line = _dry_run(tmp_path, REAL + [0.0047], CONFIG_FILE=str(sent))
    assert "last epoch 20 of 20" in line and "would_trigger 0" in line


@pytest.mark.parametrize("patience, expected",[("2", "would_trigger 1"), ("6", "would_trigger 0")])
def test_patience_is_configurable(tmp_path, patience, expected):
    assert expected in _dry_run(tmp_path, REAL, PATIENCE=patience)


def test_no_metrics_yet(tmp_path):
    line = _dry_run(tmp_path, [])
    assert "last epoch 0 of 30" in line and "would_trigger 0" in line

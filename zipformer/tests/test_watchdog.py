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


def _dry_run(tmp_path, pers, num_epochs=30, rows=None, config_extra="", **env):
    """`rows` (full dev_metrics.jsonl dicts) replaces the legacy `{"epoch", "per_phonemes"}` rows."""
    zf = tmp_path / "zipformer"
    (zf / "configs").mkdir(parents=True)
    (zf / "configs" / "train_config.yml").write_text(
        f'num_epochs: {num_epochs}\noutput_dir: "./results"   # comment\n{config_extra}', encoding="utf-8"
    )
    (zf / "results").mkdir()
    if rows is None:
        rows = [{"epoch": i, "per_phonemes": p} for i, p in enumerate(pers, 1)]
    (zf / "results" / "dev_metrics.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8",
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


# ---------------------------------------------------------------- selectable metric

# legacy dev flat after epoch 2 while QDAT keeps improving: the two rules disagree
NEW_ROWS = [
    {"epoch": e, "batch_idx_train": 100 * e, "per_phonemes": dev, "qdat/per_phonemes": q, "dev_unseen/per_phonemes": u}
    for e, dev, q, u in [(1, 0.010, 0.30, 0.02), (2, 0.005, 0.25, 0.010), (3, 0.005, 0.22, 0.010),
                         (4, 0.005, 0.20, 0.010), (5, 0.005, 0.18, 0.010), (6, 0.005, 0.16, 0.010)]
]


def test_default_metric_is_qdat_per(tmp_path):
    line = _dry_run(tmp_path, [], rows=NEW_ROWS)
    assert line.startswith("metric qdat/per_phonemes,")
    assert "best 0.160000 at epoch 6, epochs since best 0/4, would_trigger 0" in line


def test_metric_from_config_early_stop_metric(tmp_path):
    line = _dry_run(tmp_path, [], rows=NEW_ROWS, config_extra='early_stop_metric: "dev_unseen/per_phonemes"  # x\n')
    assert line.startswith("metric dev_unseen/per_phonemes,")
    assert "best 0.010000 at epoch 2, epochs since best 4/4, would_trigger 1" in line


def test_metric_env_overrides_config(tmp_path):
    line = _dry_run(tmp_path, [], rows=NEW_ROWS, config_extra='early_stop_metric: "dev_unseen/per_phonemes"\n',
                    METRIC="per_phonemes")
    assert line.startswith("metric per_phonemes,") and "best 0.005000 at epoch 2" in line


def test_old_dev_metrics_fall_back_to_legacy_per(tmp_path):
    # a v0 file has only per_phonemes: the qdat default must not break it
    line = _dry_run(tmp_path, REAL)
    assert line.startswith("metric per_phonemes,")
    log = (tmp_path / "w.log").read_text()
    assert "metric qdat/per_phonemes belum ada" in log and "memakai per_phonemes" in log


def test_torn_last_line_is_ignored(tmp_path):
    zf_rows = NEW_ROWS[:3]
    _dry_run(tmp_path, [], rows=zf_rows)  # builds the tree
    metrics = tmp_path / "zipformer" / "results" / "dev_metrics.jsonl"
    metrics.write_text(metrics.read_text() + '{"epoch": 4, "qdat/per', encoding="utf-8")
    full_env = {**os.environ, "DRY_RUN": "1", "REPO_ZIPFORMER_DIR": str(tmp_path / "zipformer"), "WLOG": str(tmp_path / "w2.log")}
    out = subprocess.run(["bash", str(WATCHDOG)], env=full_env, capture_output=True, text=True, check=True).stdout
    assert "last epoch 3 of 30, best 0.220000 at epoch 3" in out

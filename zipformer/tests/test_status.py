"""PER table and run-state logic of vastai/zipformer_status.py."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vastai"))
import zipformer_status as zs  # noqa: E402

from .test_watchdog import REAL  # noqa: E402  same history the watchdog tests use


def _metrics(pers):
    return [json.dumps({"epoch": i, "per_phonemes": p, "batch_idx_train": i * 100}) for i, p in enumerate(pers, 1)]


def test_per_table_follows_the_watchdog_rule():
    table, best, since = zs.per_table(_metrics(REAL), 0.05)
    assert (best, since) == (16, 3)
    assert [ep for ep, *_, new_best in table if new_best][-3:] == [12, 14, 16]


def test_per_table_without_threshold_takes_any_gain():
    assert zs.per_table(_metrics(REAL), 0.0)[1:] == (18, 1)


def test_epoch_hours_averages_the_last_three_gaps():
    lines = [f"2026-11-18 0{h}:00:00,000 INFO quran_zipformer.train_loop: epoch {h} finished; backup: ok"
             for h in range(1, 6)]
    assert zs.epoch_hours(lines) == pytest.approx(1.0)
    assert zs.epoch_hours(lines[:1]) is None


def _wait_with(monkeypatch, states):
    """Run wait_for_change over a scripted sequence of collect() results."""
    seq = iter([{"state": s, "last": e} for s, e in states])
    monkeypatch.setattr(zs, "collect", lambda args: next(seq))
    monkeypatch.setattr(zs.time, "sleep", lambda s: None)
    args = type("A", (), {"wait_timeout": 10**6, "poll": 1})()
    return zs.wait_for_change(args)


def test_wait_ignores_a_short_no_instance_blip(monkeypatch):
    # 2026-11-19: the host's public IP changed and one API poll missed the instance
    st = _wait_with(monkeypatch, [("TRAINING", 22), ("NO_INSTANCE", 0), ("API_ERROR", 0),
                                  ("TRAINING", 22), ("TRAINING", 23)])
    assert (st["state"], st["last"]) == ("TRAINING", 23)


def test_wait_reports_no_instance_once_confirmed(monkeypatch):
    st = _wait_with(monkeypatch, [("UPLOADING", 24)] + [("NO_INSTANCE", 0)] * zs.NO_INSTANCE_CONFIRMATIONS)
    assert st["state"] == "NO_INSTANCE"


@pytest.mark.parametrize(
    "sections, num_epochs, last, state",
    [
        ({"TRAINER": ["42"], "SHELL": ["41"]}, 30, 19, "TRAINING"),
        ({"TRAINER": ["42"], "SHELL": ["41"]}, 20, 20, "FINISHING"),
        ({"SHELL": ["41"]}, 30, 0, "PREPARING"),
        ({"UPLOADER": ["43"], "DONE": []}, 20, 20, "UPLOADING"),
        ({"WATCHDOG": ["session", "running", "2026 TRIGGER: 4 epoch tanpa rekor baru"]}, 30, 20, "WATCHDOG_STOPPING"),
        ({"DONE": ["yes"], "ENDED": ["[train_zipformer.sh selesai, exit code 1]"]}, 30, 3, "STOPPED"),
        ({"DONE": ["yes"], "ENDED": ["[train_zipformer.sh selesai, exit code 0]"]}, 20, 20, "ENDED"),
    ],
)
def test_classify(sections, num_epochs, last, state):
    assert zs.classify(sections, num_epochs, last) == state

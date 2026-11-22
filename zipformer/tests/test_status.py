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
    table, best, since, used = zs.per_table(_metrics(REAL), 0.05)
    assert (best, since, used) == (16, 3, "per_phonemes")
    assert [ep for ep, *_, new_best in table if new_best][-3:] == [12, 14, 16]


def test_per_table_without_threshold_takes_any_gain():
    assert zs.per_table(_metrics(REAL), 0.0)[1:3] == (18, 1)


def test_per_table_old_metrics_fall_back_to_legacy_per():
    # a v0 dev_metrics.jsonl with the new default metric: same result as before
    table, best, since, used = zs.per_table(_metrics(REAL), 0.05, "qdat/per_phonemes")
    assert (best, since, used) == (16, 3, "per_phonemes")


def test_per_table_follows_the_chosen_metric():
    from .test_watchdog import NEW_ROWS

    lines = [json.dumps(r) for r in NEW_ROWS] + ['{"epoch": 7, "qdat/per']  # torn last line
    table, best, since, used = zs.per_table(lines, 0.05, "qdat/per_phonemes")
    assert (best, since, used) == (6, 0, "qdat/per_phonemes")
    ep, batch, value, sil, dev_per, new_best = table[-1]
    assert (ep, batch, value, dev_per, new_best) == (6, 600, 0.16, 0.005, True)
    assert zs.per_table(lines, 0.05, "dev_unseen/per_phonemes")[1:] == (2, 4, "dev_unseen/per_phonemes")


@pytest.mark.parametrize(
    "params, rows, expect",
    [
        ("patience 4, min relative improvement 0.05, metric dev_unseen/per_phonemes", "new",
         "Metrik early stop dev_unseen/per_phonemes (rekor = < rekor x 0.95)"),
        ("patience 4, min relative improvement 0.05", "new", "Metrik early stop qdat/per_phonemes"),
        ("patience 4, min relative improvement 0.05", "old",
         "Metrik early stop per_phonemes (rekor = < rekor x 0.95; qdat/per_phonemes belum ada, memakai PER dev lama)"),
    ],
)
def test_print_status_shows_the_metric_in_use(monkeypatch, capsys, params, rows, expect):
    from .test_watchdog import NEW_ROWS

    lines = [json.dumps(r) for r in NEW_ROWS] if rows == "new" else _metrics(REAL)
    sections = {"NOW": ["2026-11-20 10:00:00"], "TRAINER": ["42"], "NUM_EPOCHS": ["30"], "METRICS": lines,
                "WATCHDOG_PARAMS": [params], "WATCHDOG": ["session", "running"]}
    monkeypatch.setattr(zs, "find_box", lambda args: ("1.2.3.4", "22", None))
    monkeypatch.setattr(zs, "probe", lambda args, host, port: sections)
    monkeypatch.setattr(zs, "vast_credit", lambda: None)
    args = type("A", (), {"patience": 4, "min_improvement": 0.05, "metric": None, "rows": 8, "drive": False})()
    zs.print_status(args, zs.collect(args))
    out = capsys.readouterr().out
    assert expect in out
    if rows == "new":
        assert "PER eval set epoch 6: dev 0.500%, qdat/per_phonemes" not in out
        assert "PER eval set epoch 6: dev 0.500%, qdat 16.000%, dev_unseen 1.000%" in out


def test_eval_set_pers_names_every_set():
    from .test_watchdog import NEW_ROWS

    assert zs.eval_set_pers(NEW_ROWS[-1]) == {"dev": 0.005, "qdat": 0.16, "dev_unseen": 0.01}
    assert zs.eval_set_pers({"epoch": 1, "per_phonemes": 0.1}) == {"dev": 0.1}


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


def test_to_wib_shifts_box_utc_stamps_and_rolls_the_date():
    assert zs.to_wib("2026-11-22 17:31:40") == "2026-11-23 00:31:40 WIB"
    assert zs.to_wib("2026-11-22 17:29:39,334 INFO train_zipformer: ===== stage: prepare =====") == (
        "2026-11-23 00:29:39,334 WIB INFO train_zipformer: ===== stage: prepare ====="
    )
    assert zs.to_wib("2026-11-22 10:00:00 watchdog mulai") == "2026-11-22 17:00:00 WIB watchdog mulai"
    assert zs.to_wib("tanpa timestamp") == "tanpa timestamp"


def test_to_wib_leaves_stamps_alone_on_a_box_already_in_wib():
    off = zs.box_offset("2026-11-23 00:34:17 +0700")
    assert zs.to_wib("2026-11-23 00:29:39,334 INFO x", off) == "2026-11-23 00:29:39,334 WIB INFO x"


def test_box_offset_reads_the_utc_offset_and_defaults_to_utc():
    from datetime import timedelta

    assert zs.box_offset("2026-11-22 17:34:17 +0000") == timedelta(0)
    assert zs.box_offset("2026-11-23 00:34:17 +0700") == timedelta(hours=7)
    assert zs.box_offset("2026-11-22 17:34:17") == timedelta(0)  # box probed by an older status script


def test_print_status_header_in_wib_for_a_wib_box(monkeypatch, capsys):
    sections = {"NOW": ["2026-11-23 00:34:17 +0700"], "TRAINER": ["42"], "NUM_EPOCHS": ["30"], "METRICS": [],
                "WATCHDOG": ["session", "running", "2026-11-23 00:27:10 watchdog mulai"]}
    monkeypatch.setattr(zs, "find_box", lambda args: ("1.2.3.4", "22", None))
    monkeypatch.setattr(zs, "probe", lambda args, host, port: sections)
    monkeypatch.setattr(zs, "vast_credit", lambda: None)
    args = type("A", (), {"patience": 4, "min_improvement": 0.05, "metric": None, "rows": 8, "drive": False})()
    zs.print_status(args, zs.collect(args))
    out = capsys.readouterr().out
    assert "waktu box 2026-11-23 00:34:17 WIB," in out
    assert "  2026-11-23 00:27:10 WIB watchdog mulai" in out

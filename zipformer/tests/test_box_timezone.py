"""Launches set the box's timezone to WIB for the training and watchdog sessions."""

import subprocess
import sys
from pathlib import Path

VASTAI = Path(__file__).resolve().parents[1] / "vastai"
sys.path.insert(0, str(VASTAI))
import run_zipformer_training as rzt  # noqa: E402


def _args(**kw):
    base = dict(config=None, auto_restart=False, train_args=[], destroy_when_done=True, results_remote=None,
                early_stop_patience=4, early_stop_min_improvement=0.05, early_stop_metric=None)
    base.update(kw)
    return type("A", (), base)()


def test_train_session_runs_in_wib():
    assert rzt.build_train_command(_args()).startswith(f"export TZ={rzt.BOX_TZ}; ")


def test_watchdog_session_runs_in_wib():
    assert f"TZ={rzt.BOX_TZ} " in rzt.build_watchdog_command(_args())


def test_box_tz_needs_no_tzdata():
    # POSIX TZ string: glibc resolves it without /usr/share/zoneinfo (absent in the pytorch image)
    out = subprocess.run(["date", "+%z"], env={"TZ": rzt.BOX_TZ}, capture_output=True, text=True).stdout
    assert out.strip() == "+0700"


def test_watchdog_log_follows_the_session_timezone():
    say = next(l for l in (VASTAI / "zipformer_watchdog.sh").read_text().splitlines() if l.startswith("say()"))
    assert "date -u" not in say


def test_status_probe_reads_the_clock_in_the_session_timezone():
    # The ssh shell itself stays UTC; the probe must use the sessions' TZ or WIB log stamps get shifted twice
    import zipformer_status as zs

    assert rzt.REMOTE_TZ_FILE == zs.REMOTE_TZ_FILE
    assert f'TZ="$(cat {zs.REMOTE_TZ_FILE} 2>/dev/null)" date' in zs.REMOTE_PROBE


def test_probe_clock_with_and_without_the_tz_file(tmp_path):
    import zipformer_status as zs

    line = next(l for l in zs.REMOTE_PROBE.splitlines() if l.startswith("echo @@NOW"))
    cmd = line.replace("%%", "%").replace(zs.REMOTE_TZ_FILE, str(tmp_path / "tz"))
    run = lambda: subprocess.run(["bash", "-c", cmd], capture_output=True, text=True).stdout.split()[-1]
    assert run() == "+0000"  # older box without the file: UTC, as before
    (tmp_path / "tz").write_text(rzt.BOX_TZ + "\n")
    assert run() == "+0700"

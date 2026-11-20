"""The test suite must never reach W&B, even with a WANDB_API_KEY in zipformer/.env.

2026-11-20: test_signal_to_process_group_stops_gracefully started
`python -m quran_zipformer.train_loop` with WANDB_API_KEY removed from the
environment; `main()` -> `load_dotenv()` read the real key from zipformer/.env,
and two test runs were created and synced to W&B (zipformer/wandb/).

Here the package is copied next to a fake `.env` (python-dotenv finds `.env` by
walking up from the calling module's file, not from the cwd) and a fake `wandb`
module records any `wandb.init` call in a marker file.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("k2")

from .conftest import REPO_ROOT, ZIPFORMER_ROOT  # noqa: E402

FAKE_WANDB = '''
import os
from pathlib import Path


class _Run:
    id = "fake-run"

    def log(self, *a, **k):
        pass

    def finish(self):
        pass


def init(*args, **kwargs):
    Path(os.environ["WANDB_PROBE_MARKER"]).write_text(os.environ.get("WANDB_API_KEY", ""))
    return _Run()
'''


@pytest.fixture
def box(tmp_path, tiny_cfg):
    """`(env, cmd_prefix, cwd, marker)`: a package copy with a fake .env holding a W&B key."""
    pkg_root = tmp_path / "zf"
    shutil.copytree(ZIPFORMER_ROOT / "quran_zipformer", pkg_root / "quran_zipformer",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (pkg_root / ".env").write_text("WANDB_API_KEY=fake-key-from-dotenv\n", encoding="utf-8")
    fake = tmp_path / "fake_wandb"
    (fake / "wandb").mkdir(parents=True)
    (fake / "wandb" / "__init__.py").write_text(FAKE_WANDB, encoding="utf-8")
    cwd = tmp_path / "run"
    cwd.mkdir()
    marker = tmp_path / "wandb_init_called"
    tiny_cfg.num_epochs = 1
    tiny_cfg.to_yaml(tmp_path / "cfg.yml")
    env = dict(os.environ)  # carries the conftest guard
    env["PYTHONPATH"] = os.pathsep.join([str(fake), str(pkg_root), env.get("PYTHONPATH", "")])
    env["WANDB_PROBE_MARKER"] = str(marker)
    cmd = [sys.executable, "-m", "quran_zipformer.train_loop", "--config", str(tmp_path / "cfg.yml"),
           "--vocab", str(REPO_ROOT / "vocab_streaming" / "vocab.json"), "--device", "cpu"]
    return env, cmd, cwd, marker


def _run(cmd, env, cwd):
    return subprocess.run(cmd, env=env, cwd=cwd, capture_output=True, text=True, timeout=600)


def test_conftest_guard_is_in_place():
    assert os.environ["WANDB_API_KEY"] == "" and os.environ["WANDB_MODE"] == "disabled"


def test_load_dotenv_keeps_the_empty_key(tmp_path, monkeypatch):
    from dotenv import load_dotenv

    (tmp_path / ".env").write_text("WANDB_API_KEY=fake-key-from-dotenv\nOTHER_DOTENV_PROBE=1\n")
    monkeypatch.delenv("OTHER_DOTENV_PROBE", raising=False)
    load_dotenv(tmp_path / ".env")
    assert os.environ["WANDB_API_KEY"] == ""  # existing value wins (override=False)
    assert os.environ["OTHER_DOTENV_PROBE"] == "1"  # the file itself was read


def test_test_resume_subprocess_never_calls_wandb(box, tiny_data_dir_big):
    env, cmd, cwd, marker = box
    r = _run([*cmd, "--data-dir", str(tiny_data_dir_big), "--test-resume", "--n-batches", "2"], env, cwd)
    assert r.returncode == 0, r.stderr[-3000:]
    assert not marker.exists()
    assert not (cwd / "wandb").exists()


def test_training_subprocess_never_calls_wandb(box, tiny_data_dir):
    # the full main() reaches make_logger, the path that created the leaked runs
    env, cmd, cwd, marker = box
    r = _run([*cmd, "--data-dir", str(tiny_data_dir)], env, cwd)
    assert r.returncode == 0, r.stderr[-3000:]
    assert not marker.exists(), "wandb.init was called"
    assert not (cwd / "wandb").exists()
    assert "WANDB_API_KEY is not set; logging to TensorBoard only" in r.stderr


def test_without_the_guard_the_dotenv_key_reaches_wandb(box, tiny_data_dir):
    # control: the old pattern (key removed from the env) lets load_dotenv fill it in,
    # proving this setup would catch a leak
    env, cmd, cwd, marker = box
    env = {k: v for k, v in env.items() if k not in ("WANDB_API_KEY", "WANDB_MODE")}
    r = _run([*cmd, "--data-dir", str(tiny_data_dir)], env, cwd)
    assert r.returncode == 0, r.stderr[-3000:]
    assert marker.read_text() == "fake-key-from-dotenv"

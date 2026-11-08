import sys
from pathlib import Path
from prepare_quran_dataset.zipformer import icefall_import as ii

def test_icefall_root_defaults_to_external(monkeypatch):
    monkeypatch.delenv("ICEFALL_ROOT", raising=False)
    assert ii.icefall_root().name == "icefall"
    assert ii.icefall_root().parent.name == "external"

def test_setup_paths_is_idempotent(monkeypatch, tmp_path):
    (tmp_path / "egs/librispeech/ASR/zipformer").mkdir(parents=True)
    monkeypatch.setenv("ICEFALL_ROOT", str(tmp_path))
    before = len(sys.path)
    ii.setup_icefall_paths(); ii.setup_icefall_paths()
    assert len(sys.path) == before + 2
    assert str(tmp_path / "egs/librispeech/ASR/zipformer") in sys.path

def test_load_recipe_module(monkeypatch, tmp_path):
    rec = tmp_path / "egs/librispeech/ASR/zipformer"; rec.mkdir(parents=True)
    (rec / "a-b.py").write_text("X = 42\n")
    monkeypatch.setenv("ICEFALL_ROOT", str(tmp_path))
    assert ii.load_recipe_module("a-b.py").X == 42

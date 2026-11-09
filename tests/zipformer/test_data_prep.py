import io
import json

import numpy as np
import pytest
import soundfile as sf
from lhotse import CutSet

from prepare_quran_dataset.zipformer import data_prep as dp


def _wav_bytes(seconds=1.0, sr=16000):
    buf = io.BytesIO()
    sf.write(buf, np.sin(np.arange(int(sr * seconds)) / 50).astype("float32"), sr, format="WAV")
    return buf.getvalue()


def _speech_row(seg, moshaf_id="6.0", uthmani="بِسْمِ"):
    return {"audio": {"bytes": _wav_bytes(), "path": None}, "uthmani": uthmani,
            "moshaf_id": moshaf_id, "segment_index": seg}


def _noise_rows():
    return [{"audio": {"bytes": _wav_bytes(0.3), "path": None}},
            {"audio": {"bytes": _wav_bytes(5.0), "path": None}}] * 5


@pytest.fixture(autouse=True)
def cache_cleanups(monkeypatch):
    """Never touch the real HF cache from tests; record the cleanup calls."""
    calls = []
    monkeypatch.setattr(dp, "cleanup_hf_cache", lambda name, config: calls.append((name, config)))
    return calls


@pytest.fixture
def prepared_partials(tmp_path, cfg, monkeypatch, base_attrs, special_attrs, tokens):
    """Write the speech (train 6.0, test 19.0) and noise partials into tmp_path."""
    rows = {
        "moshaf_6.0": [_speech_row("001.0001"), _speech_row("010.0158")],
        "moshaf_19.0": [_speech_row("001.0001", moshaf_id="19.0")],
        None: _noise_rows(),
    }
    monkeypatch.setattr(dp, "load_hf_rows", lambda name, config, token: rows[config])
    base = base_attrs | {"19.0": base_attrs["6.0"]}
    for moshaf_id in ["6.0", "19.0"]:
        dp.process_speech_config(moshaf_id, tmp_path, cfg, base, special_attrs, tokens, None)
    dp.process_noise(tmp_path, cfg, None)
    return tmp_path


def test_write_flac_returns_duration(tmp_path):
    d = dp.write_flac(_wav_bytes(1.5), tmp_path / "a.flac")
    assert abs(d - 1.5) < 0.01 and (tmp_path / "a.flac").exists()
    assert sf.info(tmp_path / "a.flac").samplerate == 16000


def test_cut_paths_are_relative(tmp_path):
    dp.write_flac(_wav_bytes(), tmp_path / "audio/x.flac")
    cut = dp.make_cut("audio/x.flac", 1.0, "ب", "c1", {"input_type": "speech"}, tmp_path)
    assert cut.recording.sources[0].source == "audio/x.flac"
    assert cut.supervisions[0].text == "ب"
    assert cut.supervisions[0].custom == {"input_type": "speech"}


def test_speech_config_is_idempotent_and_skips_when_done(
    tmp_path, cfg, monkeypatch, base_attrs, special_attrs, tokens, cache_cleanups
):
    rows = [_speech_row("001.0001"), _speech_row("010.0158")]
    calls = {"n": 0}

    def fake_rows(name, config, token):
        calls["n"] += 1
        assert (name, config) == (dp.SPEECH_DATASET, "moshaf_6.0")
        return rows

    monkeypatch.setattr(dp, "load_hf_rows", fake_rows)
    dp.process_speech_config("6.0", tmp_path, cfg, base_attrs, special_attrs, tokens, None)
    dp.process_speech_config("6.0", tmp_path, cfg, base_attrs, special_attrs, tokens, None)
    assert calls["n"] == 1
    assert cache_cleanups == [(dp.SPEECH_DATASET, "moshaf_6.0")]
    assert (tmp_path / "manifests/partial/speech_moshaf_6.0.done").exists()
    cuts = CutSet.from_file(tmp_path / "manifests/partial/speech_moshaf_6.0.jsonl.gz")
    assert len(cuts) == 2
    assert sorted(cuts.ids) == ["moshaf_6.0__001.0001", "moshaf_6.0__010.0158"]
    for c in cuts:
        custom = c.supervisions[0].custom
        assert custom["input_type"] == "speech" and custom["moshaf_id"] == "6.0"
        text = c.supervisions[0].text
        assert text and custom["num_labels"] == len(text)
        assert c.recording.sources[0].source == f"audio/moshaf_6.0/{custom['segment_index']}.flac"
        assert (tmp_path / c.recording.sources[0].source).exists()


def test_speech_max_audio_seconds_filter(tmp_path, cfg, monkeypatch, base_attrs, special_attrs, tokens):
    cfg.max_audio_seconds = 0.5
    monkeypatch.setattr(dp, "load_hf_rows", lambda *a: [_speech_row("001.0001")])
    dp.process_speech_config("6.0", tmp_path, cfg, base_attrs, special_attrs, tokens, None)
    # lhotse reads an empty manifest back as None
    assert not CutSet.from_file(tmp_path / "manifests/partial/speech_moshaf_6.0.jsonl.gz")
    assert not (tmp_path / "audio/moshaf_6.0/001.0001.flac").exists()
    assert (tmp_path / "manifests/partial/speech_moshaf_6.0.done").exists()


def test_speech_unknown_phoneme_raises(tmp_path, cfg, monkeypatch, base_attrs, special_attrs):
    from prepare_quran_dataset.zipformer.tokens import PhonemeTokens

    no_ba = PhonemeTokens({"[PAD]": 0})
    monkeypatch.setattr(dp, "load_hf_rows", lambda *a: [_speech_row("001.0001")])
    with pytest.raises(KeyError):
        dp.process_speech_config("6.0", tmp_path, cfg, base_attrs, special_attrs, no_ba, None)
    assert not (tmp_path / "manifests/partial/speech_moshaf_6.0.done").exists()


def test_noise_clip_and_pad(tmp_path, cfg, monkeypatch):
    monkeypatch.setattr(dp, "load_hf_rows", lambda *a: _noise_rows())
    dp.process_noise(tmp_path, cfg, None)
    assert (tmp_path / "manifests/partial/noise.done").exists()
    cuts = CutSet.from_file(tmp_path / "manifests/partial/noise_input.jsonl.gz")
    aug = CutSet.from_file(tmp_path / "manifests/partial/noise_aug.jsonl.gz")
    assert len(aug) == 3 and len(cuts) == 7
    assert set(aug.ids).isdisjoint(cuts.ids)
    durs = sorted(c.duration for c in cuts)
    assert durs[0] >= 1.0 and durs[-1] <= 2.0 + 1e-3
    assert all(c.supervisions[0].text == "" for c in cuts)
    assert all(c.supervisions[0].custom["input_type"] == "silence" for c in cuts + aug)
    for c in cuts:
        assert abs(sf.info(tmp_path / c.recording.sources[0].source).duration - c.duration) < 1e-3


def test_merge_splits_and_does_not_duplicate(prepared_partials, cfg):
    root = prepared_partials
    summary1 = dp.merge_manifests(root, cfg)
    summary2 = dp.merge_manifests(root, cfg)
    assert summary1["num_cuts"] == summary2["num_cuts"]
    assert summary1["num_cuts"] == {"train": 8, "dev": 1, "test": 1, "noise_aug": 3}
    assert summary1["vocab_size"] == 44
    assert (root / "tokens.txt").exists()
    train = CutSet.from_file(root / "cuts_train.jsonl.gz")
    dev = CutSet.from_file(root / "cuts_dev.jsonl.gz")
    test = CutSet.from_file(root / "cuts_test.jsonl.gz")
    assert set(train.ids).isdisjoint(set(dev.ids))
    assert any(c.supervisions[0].custom["input_type"] == "silence" for c in train + dev)
    assert list(test.ids) == ["moshaf_19.0__001.0001"]


def test_merge_writes_frame_budget(prepared_partials, cfg):
    root = prepared_partials
    summary = dp.merge_manifests(root, cfg)
    manifest = json.loads((root / "data_manifest.json").read_text())
    assert manifest == summary
    fb = manifest["frame_budget"]
    assert set(fb) == {"f1_s1.0", "f1_s1.1", "f2_s1.0", "f2_s1.1"}
    assert fb["f1_s1.0"]["min"] > fb["f2_s1.0"]["min"]


def test_merge_raises_on_missing_partial(tmp_path, cfg):
    with pytest.raises(FileNotFoundError, match="moshaf_6.0"):
        dp.merge_manifests(tmp_path, cfg)


def test_run_merges_only_when_all_configs_done(tmp_path, cfg, monkeypatch, moshaf_6_0_dict):
    rows = {
        "moshaf_6.0": [_speech_row("001.0001")],
        "moshaf_19.0": [_speech_row("001.0001", moshaf_id="19.0")],
        None: _noise_rows(),
    }
    monkeypatch.setattr(dp, "load_hf_rows", lambda name, config, token: rows[config])
    monkeypatch.setattr(
        dp, "load_moshaf_metadata",
        lambda token: {"6.0": moshaf_6_0_dict, "19.0": moshaf_6_0_dict | {"id": "19.0"}},
    )
    cfg_path = tmp_path / "cfg.yml"
    cfg.to_yaml(cfg_path)
    root = tmp_path / "drive"

    assert dp.run(cfg_path, root, None, only=["moshaf_6.0", "noise"], skip_space_check=True) is None
    assert not (root / "cuts_train.jsonl.gz").exists()
    summary = dp.run(cfg_path, root, None, only=None, skip_space_check=True)
    assert summary["configs"] == ["moshaf_6.0", "moshaf_19.0", "noise"]
    assert summary["num_cuts"]["test"] == 1
    with pytest.raises(ValueError, match="moshaf_99.0"):
        dp.run(cfg_path, root, None, only=["moshaf_99.0"], skip_space_check=True)


def test_check_free_space_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(dp.shutil, "disk_usage", lambda p: type("U", (), {"free": 1 << 30})())
    with pytest.raises(RuntimeError, match="GB"):
        dp.check_free_space(tmp_path, min_gb=70)

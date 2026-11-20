"""Per-epoch eval sets: dev_unseen sample, qdat cuts, stress copies."""

import numpy as np
import pytest
from lhotse import CutSet

from quran_zipformer import eval_sets as es
from quran_zipformer.datamodule import load_cuts

from .conftest import TEST_MOSHAF_SIZES


def _unseen(tiny_cfg_factory, data_dir, n, seed=42):
    return es.dev_unseen_cuts(tiny_cfg_factory(dev_unseen_cuts_per_moshaf=n, seed=seed), data_dir)


def test_dev_unseen_is_balanced_per_moshaf(tiny_cfg_factory, tiny_data_dir_eval):
    cuts = _unseen(tiny_cfg_factory, tiny_data_dir_eval, 4)
    per_moshaf = {}
    for c in cuts:
        m = c.supervisions[0].custom["moshaf_id"]
        per_moshaf[m] = per_moshaf.get(m, 0) + 1
    # 4 of each, all 3 of the smaller moshaf 29.0
    assert per_moshaf == {"19.0": 4, "29.0": 3, "30.0": 4}
    assert len({c.id for c in cuts}) == len(cuts)
    test_ids = {c.id for c in load_cuts(tiny_data_dir_eval, "test")}
    assert {c.id for c in cuts} <= test_ids


def test_dev_unseen_is_deterministic_and_seeded(tiny_cfg_factory, tiny_data_dir_eval):
    a = [c.id for c in _unseen(tiny_cfg_factory, tiny_data_dir_eval, 4)]
    b = [c.id for c in _unseen(tiny_cfg_factory, tiny_data_dir_eval, 4)]
    assert a == b
    other = {tuple(c.id for c in _unseen(tiny_cfg_factory, tiny_data_dir_eval, 4, seed=s)) for s in range(1, 8)}
    assert len(other) > 1  # the seed picks the sample


def test_dev_unseen_does_not_depend_on_manifest_order(tiny_cfg_factory, tiny_data_dir_eval, tmp_path):
    import shutil

    from quran_zipformer.data_prep import _write_cuts

    shuffled = tmp_path / "data"
    shutil.copytree(tiny_data_dir_eval, shuffled)
    cuts = list(load_cuts(tiny_data_dir_eval, "test"))
    reordered = [c.with_recording_path_prefix(str(shuffled)) for c in reversed(cuts)]
    # write back relative paths, as the manifest stores them
    for c in reordered:
        c.recording.sources[0].source = c.recording.sources[0].source.removeprefix(f"{shuffled}/")
    _write_cuts(reordered, shuffled / "cuts_test.jsonl.gz")
    assert [c.id for c in _unseen(tiny_cfg_factory, shuffled, 4)] == [
        c.id for c in _unseen(tiny_cfg_factory, tiny_data_dir_eval, 4)
    ]


def test_dev_unseen_skips_silence_cuts(tiny_cfg_factory, tiny_data_dir):
    # tiny_data_dir has no test split: an empty sample, not a crash
    from quran_zipformer.data_prep import _write_cuts

    _write_cuts([], tiny_data_dir / "cuts_test.jsonl.gz")
    assert len(_unseen(tiny_cfg_factory, tiny_data_dir, 5)) == 0


def test_qdat_cuts_labels_audio_and_cache(fake_qdat, tmp_path, monkeypatch):
    cuts = es.qdat_cuts(tmp_path)
    assert [c.id for c in cuts] == ["qdat__q0", "qdat__q1"]
    sup = cuts[0].supervisions[0]
    assert sup.text == "ءَلِف"  # whitespace removed, as qdat_bench scores
    assert sup.custom == {"input_type": "speech", "qdat_id": "q0", "gender": "female"}
    audio = cuts[0].load_audio()
    assert cuts[0].sampling_rate == 16000 and audio.shape[-1] == 32000  # 2 s resampled from 22.05 kHz
    assert (tmp_path / es.QDAT_DIR / es.QDAT_MANIFEST).exists()

    monkeypatch.setattr(es, "_load_qdat_rows", lambda: pytest.fail("cached set must not be rebuilt"))
    assert [c.id for c in es.qdat_cuts(tmp_path)] == ["qdat__q0", "qdat__q1"]


def test_stress_sets_are_deterministic(tiny_cfg_factory, tiny_data_dir_eval):
    base = _unseen(tiny_cfg_factory, tiny_data_dir_eval, 2)
    clean = base[0].load_audio()
    vol = es.stress_cuts(base, "vol-20db", 42)
    np.testing.assert_allclose(vol[0].load_audio(), clean * 0.1, rtol=1e-5, atol=1e-7)
    fast = es.stress_cuts(base, "speed1.25", 42)
    assert fast[0].duration == pytest.approx(base[0].duration / 1.25, abs=1e-3)
    rev1 = es.stress_cuts(base, "reverb", 42)[0].load_audio()
    rev2 = es.stress_cuts(base, "reverb", 42)[0].load_audio()
    np.testing.assert_array_equal(rev1, rev2)  # same room every epoch
    assert not np.allclose(rev1, clean)
    assert [c.supervisions[0].text for c in vol] == [c.supervisions[0].text for c in base]


def test_build_eval_sets_follows_the_config(tiny_cfg_factory, tiny_data_dir_eval, tokens, fake_qdat, tmp_path):
    import shutil

    data = tmp_path / "data"
    shutil.copytree(tiny_data_dir_eval, data)  # qdat is cached inside data_dir
    dev = load_cuts(data, "dev")
    assert list(es.build_eval_sets(tiny_cfg_factory(), data, dev, tokens)) == ["dev"]  # defaults: legacy only
    cfg = tiny_cfg_factory(dev_unseen_cuts_per_moshaf=2, eval_qdat=True, stress_sets=["vol-20db", "reverb", "speed1.25"])
    sets = es.build_eval_sets(cfg, data, dev, tokens)
    assert list(sets) == ["dev", "dev_unseen", "dev_unseen_vol-20db", "dev_unseen_reverb", "dev_unseen_speed1.25", "qdat"]
    assert sets["dev"] is dev and len(sets["dev_unseen"]) == 6 and len(sets["qdat"]) == 2
    assert all(len(sets[f"dev_unseen_{n}"]) == 6 for n in ("vol-20db", "reverb", "speed1.25"))


def test_build_eval_sets_rejects_unknown_phonemes(tiny_cfg_factory, tmp_path, tokens, fake_qdat):
    fake_qdat[0]["phonetic_transcript"] = "abc"
    with pytest.raises(KeyError):
        es.build_eval_sets(tiny_cfg_factory(eval_qdat=True), tmp_path, CutSet(), tokens)


def test_metric_keys_prefix_all_but_legacy_dev():
    m = {"per_phonemes": 0.1, "silence_per": None, "num_speech": 3, "num_silence": 0}
    assert es.metric_keys("dev", m) == m
    assert es.metric_keys("qdat", m) == {"qdat/per_phonemes": 0.1, "qdat/num_speech": 3}


def test_test_split_fixture_sizes(tiny_data_dir_eval):
    assert len(load_cuts(tiny_data_dir_eval, "test")) == sum(TEST_MOSHAF_SIZES.values())

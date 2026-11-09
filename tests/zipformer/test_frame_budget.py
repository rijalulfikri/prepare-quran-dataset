import math

from lhotse import CutSet

from prepare_quran_dataset.zipformer import frame_budget as fb


def test_repeats_and_required():
    assert fb.count_repeats("رَحِۦۦۦۦم") == 3
    assert fb.required_frames("رَحِۦۦۦۦم") == 9 + 3  # 9 characters incl. diacritics
    assert fb.required_frames("") == 0


def test_fbank_frames():
    assert fb.fbank_frames(2.0) == 200
    assert fb.fbank_frames(0.3, speed=1.1) == 27


def test_output_frames_formulas():
    assert fb.output_frames(107, 1) == 50 and fb.output_frames(107, 2) == 25
    assert fb.output_frames(108, 1) == 50 and fb.output_frames(109, 2) == 26
    assert fb.output_frames(3, 1) == 0 and fb.output_frames(3, 2) == 0


def test_ratio_and_filter(make_cut_in_memory):
    # 10 labels + 9 repeats = 19 > 10 frames (0.3 s / 1.1 at 20 ms)
    short = make_cut_in_memory("a", duration=0.3, text="ۦۦۦۦۦۦۦۦۦۦ")
    ok = make_cut_in_memory("b", duration=2.0, text="بِسمِ")
    sil = make_cut_in_memory("c", duration=1.0, text="", input_type="silence")
    kept, dropped = fb.filter_feasible(CutSet.from_cuts([short, ok, sil]), factor=1, speed=1.1)
    assert dropped == ["a"] and set(kept.ids) == {"b", "c"}
    assert fb.ratio(2.0, "بِسمِ", 1, 1.0) > fb.ratio(2.0, "بِسمِ", 2, 1.0)
    assert math.isinf(fb.ratio(1.0, "", 1, 1.0))


def test_report_keys_and_skips_silence(make_cut_in_memory):
    short = make_cut_in_memory("a", duration=0.3, text="ۦۦۦۦۦۦۦۦۦۦ")
    ok = make_cut_in_memory("b", duration=2.0, text="بِسمِ")
    sil = make_cut_in_memory("c", duration=0.1, text="", input_type="silence")
    rep = fb.report([short, ok, sil])
    assert set(rep) == {"f1_s1.0", "f1_s1.1", "f2_s1.0", "f2_s1.1"}
    for stats in rep.values():
        assert set(stats) == {"min", "p1", "p5", "p50", "num_below_1", "ids_below_1"}
        assert stats["ids_below_1"] == ["a"] and stats["num_below_1"] == 1
    assert rep["f1_s1.0"]["min"] == fb.ratio(0.3, "ۦۦۦۦۦۦۦۦۦۦ", 1, 1.0)


def test_report_empty():
    rep = fb.report([])
    assert rep["f1_s1.0"] == {
        "min": 0.0, "p1": 0.0, "p5": 0.0, "p50": 0.0, "num_below_1": 0, "ids_below_1": []
    }

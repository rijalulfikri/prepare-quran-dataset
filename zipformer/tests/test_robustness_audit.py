"""robustness_audit.py: the signal helpers, the choice of battery utterances and the report
(the decoding itself needs sherpa-onnx and real models)."""

import json

import numpy as np
import pytest

import robustness_audit as ra

SR = ra.SR


def _speech(seconds=2.0, amp=0.1, seed=0):
    return (amp * np.random.default_rng(seed).standard_normal(round(seconds * SR))).astype(np.float32)


def _db(x):
    return 20 * np.log10(ra.rms(x))


def test_speech_rms_ignores_pauses():
    speech = _speech()
    with_pause = np.concatenate([speech, np.zeros(len(speech), np.float32)])
    assert ra.speech_rms(with_pause) == pytest.approx(ra.rms(speech), rel=0.05)
    assert ra.rms(with_pause) < 0.75 * ra.rms(speech)


@pytest.mark.parametrize("noise", [ra.white, ra.pink, ra.room])
def test_noise_makers_have_unit_rms(noise):
    assert ra.rms(noise(SR, np.random.default_rng(0))) == pytest.approx(1.0, rel=1e-3)


def test_lead_and_trail_put_noise_outside_the_speech_at_the_given_snr():
    speech, rng = _speech(), np.random.default_rng(1)
    led = ra.CONDITIONS["lead_room_2s_snr20"](speech, rng, [])
    assert len(led) == len(speech) + 2 * SR
    np.testing.assert_array_equal(led[2 * SR:], speech)
    assert _db(led[: 2 * SR]) - _db(speech) == pytest.approx(-20, abs=0.5)
    trailed = ra.CONDITIONS["trail_room_3s_snr20"](speech, rng, [])
    np.testing.assert_array_equal(trailed[: len(speech)], speech)
    assert _db(trailed[len(speech):]) - _db(speech) == pytest.approx(-20, abs=0.5)
    assert not ra.CONDITIONS["lead_zeros_2s"](speech, rng, [])[: 2 * SR].any()


def test_gain_noise_and_clip_conditions():
    speech, rng = _speech(), np.random.default_rng(2)
    assert _db(ra.CONDITIONS["gain-30db"](speech, rng, [])) - _db(speech) == pytest.approx(-30, abs=1e-3)
    noisy = ra.CONDITIONS["white_snr10"](speech, rng, [])
    assert _db(noisy - speech) - _db(speech) == pytest.approx(-10, abs=0.5)
    clipped = ra.CONDITIONS["clip_10pct_peak"](speech, rng, [])
    assert np.abs(clipped).max() == pytest.approx(0.1 * np.abs(speech).max())
    babble = ra.CONDITIONS["babble_snr5"](speech, rng, [_speech(seed=s) for s in (3, 4, 5, 6)])
    assert _db(babble - speech) - _db(speech) == pytest.approx(-5, abs=0.5)


def test_conditions_without_their_tool_are_left_out(monkeypatch):
    monkeypatch.setattr(ra.shutil, "which", lambda tool: None)
    names = ra.available_conditions()
    assert "clean" in names and "gain-30db" in names and "pair_gap_room_2s_snr20" in names
    assert not {"reverb70", "tempo1.5", "mp3_16k", "far_phone"} & set(names)
    monkeypatch.setattr(ra.shutil, "which", lambda tool: "/usr/bin/" + tool)
    assert {"reverb70", "tempo1.5", "mp3_16k", "far_phone"} <= set(ra.available_conditions())


def _row(reciter, dur, errs, ref="abcdefghij" * 10):
    return {"reciter": reciter, "dur": dur, "file": f"{reciter}/{dur}.mp3",
            **{m: {"hyp": ref, "err": e, "ref": ref} for m, e in errs.items()}}


def test_battery_base_keeps_clean_utterances_at_most_two_per_reciter():
    rows = [_row("a", 5 + i, {"new": 0, "ref": 0}) for i in range(5)]
    rows += [_row("b", 6, {"new": 0, "ref": 1}), _row("b", 7, {"new": 9, "ref": 0}),  # 9 % wrong for one model
             _row("c", 3, {"new": 0, "ref": 0}), _row("c", 20, {"new": 0, "ref": 0})]  # too short / too long
    base = ra.battery_base(rows, ["new", "ref"], size=48)
    assert sorted(r["reciter"] for r in base) == ["a", "a", "b"]
    assert all(4 <= r["dur"] <= 14 for r in base)
    assert len(ra.battery_base(rows, ["new", "ref"], size=1)) == 1


def test_report_marks_what_is_behind_the_reference(tmp_path):
    def battery(cond, new, ref):
        return {"cond": cond, "file": "f", "reciter": "a", "dur": 5.0,
                "new": {"hyp": "", "err": new, "len": 100}, "ref": {"hyp": "", "err": ref, "len": 100}}

    rows = [battery("clean", 0, 0), battery("gain-40db", 54, 1), battery("babble_snr5", 30, 30)]
    (tmp_path / "battery.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    stream = {"reciter": "a", "sura": 1, "n": 7, "dur": 40.0,
              "new": {"separate_err": 0, "stream_err": 14, "len": 200, "starts_lost": 6},
              "ref": {"separate_err": 0, "stream_err": 0, "len": 200, "starts_lost": 0}}
    (tmp_path / "stream.jsonl").write_text(json.dumps(stream), encoding="utf-8")
    text = ra.report(tmp_path, ["new", "ref"])
    lines = {line.split("|")[1].strip(): line for line in text.splitlines() if line.startswith("| ")}
    assert "54.0%" in lines["gain-40db"] and "tertinggal" in lines["gain-40db"]
    assert "tertinggal" not in lines["clean"] and "tertinggal" not in lines["babble_snr5"]  # both models fail alike
    assert "0.0% -> 7.0% (6)" in lines["a"] and "tertinggal" in lines["a"]
    assert (tmp_path / "report.md").read_text(encoding="utf-8") == text


def test_report_sets_aside_what_the_reference_cannot_read_either(tmp_path):
    def panel(reciter, new, ref):
        return {"reciter": reciter, "file": "001001", "dur": 5.0,
                "new": {"hyp": "", "err": new, "ref": "a" * 100}, "ref": {"hyp": "", "err": ref, "ref": "a" * 100}}

    rows = [panel("baik", 6, 0), panel("baik", 4, 0), panel("salah_berkas", 100, 99)]
    (tmp_path / "panel.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    battery = {"cond": "babble_snr5", "file": "f", "reciter": "baik", "dur": 5.0,
               "new": {"hyp": "", "err": 33, "len": 100}, "ref": {"hyp": "", "err": 30, "len": 100}}
    (tmp_path / "battery.jsonl").write_text(json.dumps(battery), encoding="utf-8")
    text = ra.report(tmp_path, ["new", "ref"])
    lines = {line.split("|")[1].strip(): line for line in text.splitlines() if line.startswith("| ")}
    # recordings the reference misreads almost entirely do not match their ayah numbers: not a finding
    assert "| 2 | 5.0% | 0.0% |" in lines["semua"]
    assert "salah_berkas" not in lines and "salah_berkas (1 berkas" in text
    assert "acuan juga gagal" in lines["babble_snr5"] and "tertinggal" in lines["babble_snr5"]
    assert "acuan juga gagal" not in lines["baik"]


def test_report_shows_the_panel_without_words_the_reciter_repeated(tmp_path):
    ref = "abcdefghijklmnopqrstuvwxyz" * 4  # 104 phonemes
    heard = ref[:60] + ref[52:60] + ref[60:]  # the reciter says 8 phonemes again after a breath
    assert ra.repeated(ref, heard) == 8
    assert ra.repeated(ref, ref[:10] + "zz" + ref[10:]) == 0  # a short insertion is a slip, not a repetition
    assert ra.repeated(ref, ref) == 0

    # "new" writes what it hears and misses one phoneme; "ref" leaves the repetition out
    row = {"reciter": "a", "file": "001001", "dur": 9.0,
           "new": {"hyp": heard[1:], "err": 9, "ref": ref}, "ref": {"hyp": ref, "err": 0, "ref": ref}}
    (tmp_path / "panel.jsonl").write_text(json.dumps(row), encoding="utf-8")
    lines = {line.split("|")[1].strip(): line for line in ra.report(tmp_path, ["new", "ref"]).splitlines()
             if line.startswith("| ")}
    assert "| 1 | 8.7% | 0.0% |" in lines["semua"]
    assert "| 1 | 1.0% | 0.0% |" in lines["semua, tanpa kata yang diulang qari"]


def test_per_caps_errors_at_the_reference_length():
    rows = [{"m": {"err": 30, "ref": "a" * 10}}, {"m": {"err": 0, "ref": "a" * 10}}]
    assert ra.per(rows, "m") == 0.5
    assert ra.per([], "m") == 0.0

import json

import pytest

from prepare_quran_dataset.zipformer import qdat


def test_prediction_record_schema(tokens):
    ids = tokens.encode("ب")
    r = qdat.prediction_record("x", ids, tokens)
    assert r == {"id": "x", "levels_labels": {"phonemes": ids}, "level_to_scripts": {"phonemes": "ب"}}


def test_write_predictions_jsonl(tmp_path, tokens):
    p = qdat.write_predictions([qdat.prediction_record("a", [], tokens)], tmp_path / "p.jsonl")
    assert json.loads(p.read_text(encoding="utf-8").splitlines()[0])["id"] == "a"


def test_pad_levels_for_eval_adds_empty_sifat_levels():
    records = [{"id": "a", "levels_labels": {"phonemes": [1, 2]}, "level_to_scripts": {"phonemes": "xy"}}]
    out = qdat.pad_levels_for_eval(records, ["phonemes", "hams", "shidda"])
    assert out[0]["levels_labels"] == {"phonemes": [1, 2], "hams": [], "shidda": []}
    assert out[0]["level_to_scripts"] == {"phonemes": "xy", "hams": [], "shidda": []}
    assert records[0]["levels_labels"] == {"phonemes": [1, 2]}  # input untouched


def test_single_level_metrics_nulls_inapplicable_values():
    metrics = {
        "speech_metrics": {"per_phonemes": 0.1, "per_hams": 0.5, "average_per": 0.3},
        "qdat_metrics": {"f1": 0.9},
        "qdat_avg_metrics": {"avg_per": 0.3, "per_phonemes": 0.1, "avg_f1": 0.8},
    }
    out = qdat.single_level_metrics(metrics)
    assert out["speech_metrics"] == {"per_phonemes": 0.1, "per_hams": None, "average_per": None}
    assert out["qdat_metrics"] == {"f1": 0.9}
    assert out["qdat_avg_metrics"] == {"avg_per": None, "per_phonemes": 0.1, "avg_f1": 0.8}
    assert "not applicable" in out["note"]
    assert metrics["speech_metrics"]["per_hams"] == 0.5  # input untouched


class _Decoder:
    def __init__(self):
        self.calls = 0

    def decode(self, wav, sr=16000):
        self.calls += 1
        return []


@pytest.fixture
def seams(monkeypatch):
    calls = {"load": 0, "eval": 0}

    def fake_load():
        calls["load"] += 1
        return []

    def fake_eval(pred_path, output_dir, tokenizer=None):
        calls["eval"] += 1
        return {"evaluated": True}

    monkeypatch.setattr(qdat, "_load_qdat_dataset", fake_load)
    monkeypatch.setattr(qdat, "evaluate_predictions", fake_eval)
    return calls


def test_run_skips_when_results_exist(tmp_path, tokens, seams):
    (tmp_path / "qdat_bench_test_results.json").write_text('{"cached": 1}', encoding="utf-8")
    dec = _Decoder()
    assert qdat.run_qdat_bench(dec, tokens, tmp_path) == {"cached": 1}
    assert dec.calls == 0 and seams == {"load": 0, "eval": 0}


def test_run_evaluates_only_when_predictions_exist(tmp_path, tokens, seams):
    (tmp_path / "qdat_bench_predictions.jsonl").write_text("", encoding="utf-8")
    dec = _Decoder()
    assert qdat.run_qdat_bench(dec, tokens, tmp_path) == {"evaluated": True}
    assert dec.calls == 0 and seams == {"load": 0, "eval": 1}


def test_run_full_decodes_and_evaluates(tmp_path, tokens, seams, monkeypatch):
    monkeypatch.setattr(qdat, "_load_qdat_dataset", lambda: [{"id": "r1", "audio": {}}])
    monkeypatch.setattr(qdat, "decode_audio", lambda a: type("O", (), {"wav": b"", "sr": 16000})())
    dec = _Decoder()
    qdat.run_qdat_bench(dec, tokens, tmp_path)
    assert dec.calls == 1 and seams["eval"] == 1
    rec = json.loads((tmp_path / "qdat_bench_predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert rec["id"] == "r1"


def test_rerun_ignores_existing_results(tmp_path, tokens, seams):
    (tmp_path / "qdat_bench_test_results.json").write_text('{"cached": 1}', encoding="utf-8")
    (tmp_path / "qdat_bench_predictions.jsonl").write_text("", encoding="utf-8")
    qdat.run_qdat_bench(_Decoder(), tokens, tmp_path, rerun=True)
    assert seams["load"] == 1 and seams["eval"] == 1

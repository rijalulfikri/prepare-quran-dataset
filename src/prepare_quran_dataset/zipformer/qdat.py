"""qdat_bench evaluation through the ONNX streaming decoder.

Findings about `qdat_bench.eval_results.eval_qdat_bench` and a single-level
(phonemes) model:
- It reads `item[level]` from `levels_labels` for EVERY level in the
  `MultiLevelTokenizer` vocab, so a phonemes-only prediction set raises
  KeyError on the first sifat level.
- The tajweed metrics (`qdat_metrics`) use only `level_to_scripts["phonemes"]`,
  so they are valid for a phonemes-only model.
- `speech_metrics` has one `per_<level>` per level plus `average_per`, and
  `qdat_avg_metrics` copies `average_per` into `avg_per`; only
  `per_phonemes` is meaningful here.

File shape:
- `qdat_bench_predictions.jsonl` keeps phonemes only:
  `{"id", "levels_labels": {"phonemes": ids}, "level_to_scripts": {"phonemes": str}}`.
- Before evaluating, the in-memory dataset is padded with empty sifat levels
  (`pad_levels_for_eval`).
- `qdat_bench_test_results.json` holds the metrics with every sifat `per_*`,
  `average_per` and `avg_per` set to null (`single_level_metrics`), plus a `note`.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

PREDICTIONS_NAME = "qdat_bench_predictions.jsonl"
RESULTS_NAME = "qdat_bench_test_results.json"
NOTE = "single-level (phonemes) model: sifat PER and average PER are not applicable"

log = logging.getLogger(__name__)


def prediction_record(cut_id: str, ids: list[int], tokens) -> dict:
    return {
        "id": cut_id,
        "levels_labels": {"phonemes": ids},
        "level_to_scripts": {"phonemes": tokens.decode(ids)},
    }


def write_predictions(records: list[dict], path: Path) -> Path:
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def pad_levels_for_eval(records: list[dict], levels: list[str]) -> list[dict]:
    """Add empty entries for every level other than `phonemes` (eval reads all levels)."""
    padded = copy.deepcopy(records)
    for r in padded:
        for level in levels:
            if level != "phonemes":
                r["levels_labels"][level] = []
                r["level_to_scripts"][level] = []
    return padded


def single_level_metrics(metrics: dict) -> dict:
    out = copy.deepcopy(metrics)
    speech = out["speech_metrics"]
    for key in speech:
        if key == "average_per" or (key.startswith("per_") and key != "per_phonemes"):
            speech[key] = None
    out["qdat_avg_metrics"]["avg_per"] = None
    out["note"] = NOTE
    return out


def _load_qdat_dataset():
    from datasets import Audio, load_dataset

    ds = load_dataset("obadx/qdat_bench", split="train")
    return ds.cast_column("audio", Audio(decode=False))


def decode_audio(audio_dict):
    from qdat_bench.audio_utils import decode_audio as _decode

    return _decode(audio_dict)


def evaluate_predictions(pred_path: Path, output_dir: Path, tokenizer=None) -> dict:
    from datasets import Dataset
    from qdat_bench.eval_results import eval_qdat_bench

    from prepare_quran_dataset.modeling_streaming_rnn.multi_level_tokenizer import MultiLevelTokenizer

    tokenizer = tokenizer or MultiLevelTokenizer("./vocab_streaming")
    records = [json.loads(line) for line in Path(pred_path).read_text(encoding="utf-8").splitlines() if line]
    pred_ds = Dataset.from_list(pad_levels_for_eval(records, list(tokenizer.vocab)))
    metrics = single_level_metrics(eval_qdat_bench(pred_trans_ds=pred_ds, multi_level_tokenizer=tokenizer, bootstrap=False))
    results_path = Path(output_dir) / RESULTS_NAME
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    log.info("qdat_bench results saved to %s", results_path)
    return metrics


def run_qdat_bench(decoder, tokens, output_dir: Path, rerun: bool = False) -> dict:
    output_dir = Path(output_dir)
    pred_path = output_dir / PREDICTIONS_NAME
    results_path = output_dir / RESULTS_NAME
    if results_path.exists() and not rerun:
        log.info("qdat_bench results exist at %s; skipping", results_path)
        return json.loads(results_path.read_text(encoding="utf-8"))
    if pred_path.exists() and not rerun:
        log.info("qdat_bench predictions exist at %s; evaluating only", pred_path)
        return evaluate_predictions(pred_path, output_dir)
    log.info("running qdat_bench decoding into %s", pred_path)
    records = []
    for row in _load_qdat_dataset():
        out = decode_audio(row["audio"])
        records.append(prediction_record(row["id"], decoder.decode(out.wav, out.sr), tokens))
    write_predictions(records, pred_path)
    return evaluate_predictions(pred_path, output_dir)

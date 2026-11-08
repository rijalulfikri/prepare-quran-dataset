"""Threshold logic of the ONNX-vs-PyTorch parity check, on synthetic log-probs."""

import logging

import pytest
import torch

from prepare_quran_dataset.zipformer import export_onnx as ex

V = 5  # blank 0 + tokens 1..4


def _lp(path: list[int]) -> torch.Tensor:
    """(1, T, V) log-probs whose argmax follows `path`."""
    logits = torch.full((len(path), V), -5.0)
    logits[torch.arange(len(path)), torch.tensor(path)] = 5.0
    return logits.log_softmax(-1)[None]


def _path(tokens: list[int], frames: int) -> list[int]:
    """Each token on one frame, separated by blanks, padded with blanks to `frames`."""
    path = []
    for tok in tokens:
        path += [tok, 0, 0, 0]
    return path + [0] * (frames - len(path))


def test_identical_log_probs_agree_fully(caplog):
    lp = _lp(_path([1, 2, 3, 4], 200))
    with caplog.at_level(logging.INFO, logger=ex.__name__):
        stats = ex.compare_onnx_torch([lp], [lp.clone()], blank_id=0)
    assert stats == {"frame_agreement": 1.0, "token_per": 0.0}
    assert "frame agreement" in caplog.text and "token PER" in caplog.text


def test_small_frame_drift_within_tolerance_passes():
    ref = _path([1, 2, 3, 4] * 25, 1000)  # 100 tokens
    got = list(ref)
    got[1] = 1  # token 1 lasts one frame longer: same decode, 1 frame of 1000 differs
    stats = ex.compare_onnx_torch([_lp(got)], [_lp(ref)], blank_id=0)
    assert stats["frame_agreement"] == pytest.approx(0.999)
    assert stats["token_per"] == 0.0


def test_low_frame_agreement_raises_even_with_identical_decodes():
    ref = _path([1, 2], 100)
    got = list(ref)
    got[1], got[5] = 1, 2  # both tokens one frame longer: 2 % of frames differ, same decode
    with pytest.raises(AssertionError, match=r"frame agreement 0\.9800.*token PER 0\.0000"):
        ex.compare_onnx_torch([_lp(got)], [_lp(ref)], blank_id=0)


def test_token_per_above_threshold_raises_with_first_differing_pair():
    ref = _path([1, 2, 3, 4] * 5, 1000)  # 20 tokens over 1000 frames
    good = _lp(ref)
    got = list(ref)
    got[500] = 3  # one inserted token: agreement 0.999, PER 1/20 = 0.05
    with pytest.raises(AssertionError) as info:
        ex.compare_onnx_torch([good, _lp(got)], [good.clone(), _lp(ref)], blank_id=0)
    msg = str(info.value)
    assert "token PER 0.0250" in msg  # 1 error over 40 reference tokens
    assert "wav 1" in msg and "onnx:" in msg and "torch:" in msg


def test_thresholds_are_configurable():
    ref = _path([1, 2, 3, 4] * 5, 1000)
    got = list(ref)
    got[500] = 3
    stats = ex.compare_onnx_torch([_lp(got)], [_lp(ref)], blank_id=0, max_token_per=0.1)
    assert stats == {"frame_agreement": pytest.approx(0.999), "token_per": pytest.approx(0.05)}

import torch

from prepare_quran_dataset.zipformer import decode as dc
from prepare_quran_dataset.zipformer.datamodule import eval_dataloader, load_cuts


def test_greedy_collapses_and_drops_blank():
    lp = torch.full((1, 6, 3), -10.0)
    for t, c in enumerate([1, 1, 0, 2, 2, 0]):
        lp[0, t, c] = 0.0
    assert dc.greedy_ctc(lp, torch.tensor([6]), blank_id=0) == [[1, 2]]
    assert dc.greedy_ctc(lp, torch.tensor([2]), blank_id=0) == [[1]]


def test_per_capped_and_averaged():
    assert dc.per([[1, 2]], [[1, 2]]) == 0.0
    assert dc.per([[1, 2, 3, 4, 5]], [[9]]) == 1.0  # capped at ref len
    assert abs(dc.per([[1], [1, 2]], [[1, 2], [1, 2]]) - 0.25) < 1e-9


def test_per_empty_refs():
    assert dc.per([[1]], [[]]) == 0.0
    assert dc.per([], []) == 0.0


def test_frame_shift_and_timestamps():
    assert dc.frame_shift_seconds(1) == 0.02 and dc.frame_shift_seconds(2) == 0.04
    lp = torch.full((1, 6, 3), -10.0)
    for t, c in enumerate([0, 1, 1, 0, 2, 0]):
        lp[0, t, c] = 0.0
    fr = dc.greedy_ctc_with_frames(lp, torch.tensor([6]), 0)[0]
    assert fr == [(1, 1), (2, 4)]
    assert dc.token_times(fr, dc.frame_shift_seconds(1)) == [(1, 0.02), (2, 0.08)]


def test_silence_frame_error():
    lp = torch.full((1, 4, 2), -10.0)
    lp[0, :, 0] = 0.0
    lp[0, 2, 1] = 1.0
    assert dc.silence_frame_error(lp, torch.tensor([4]), 0) == (1, 4)
    # padding frames beyond lens are ignored
    assert dc.silence_frame_error(lp, torch.tensor([2]), 0) == (0, 2)


def test_evaluate_split_end_to_end(tiny_model, cfg, tokens, tiny_data_dir):
    cfg.num_workers = 0
    loader = eval_dataloader(cfg, load_cuts(tiny_data_dir, "dev"))
    enc = tiny_model.encoder
    before = (enc.chunk_size, enc.left_context_frames)

    res = dc.evaluate_split(
        tiny_model, loader, tokens, "cpu", chunk_size=8, left_context=32
    )
    assert res["num_speech"] + res["num_silence"] == 4
    assert 0.0 <= res["per_phonemes"] <= 1.0
    assert res["silence_per"] is None or 0.0 <= res["silence_per"] <= 1.0
    assert (enc.chunk_size, enc.left_context_frames) == before


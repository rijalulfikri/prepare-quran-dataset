"""Greedy CTC decoding, PER / silence-PER metrics and split evaluation.

This module is the single place that knows the model's output frame shift.
"""

from collections.abc import Sequence

import Levenshtein
import torch
from torch import Tensor

from prepare_quran_dataset.zipformer.datamodule import batch_to_tensors
from prepare_quran_dataset.zipformer.tokens import PhonemeTokens

_BASE_FRAME_SHIFT_S = 0.02  # encoder output at downsampling factor 1


def frame_shift_seconds(output_downsampling_factor: int) -> float:
    """Seconds between consecutive model output (log-prob) frames."""
    return _BASE_FRAME_SHIFT_S * output_downsampling_factor


def greedy_ctc_with_frames(
    log_probs: Tensor, lens: Tensor, blank_id: int
) -> list[list[tuple[int, int]]]:
    """Per utterance `(token, first_frame_index)` of each emitted (collapsed,
    non-blank) token, from the argmax path over the first `lens[i]` frames."""
    best = log_probs.detach().argmax(dim=-1).cpu()
    out = []
    for path, n in zip(best, lens.cpu().tolist()):
        emitted, prev = [], blank_id
        for t, tok in enumerate(path[:n].tolist()):
            if tok != prev and tok != blank_id:
                emitted.append((tok, t))
            prev = tok
        out.append(emitted)
    return out


def greedy_ctc(log_probs: Tensor, lens: Tensor, blank_id: int) -> list[list[int]]:
    """Per utterance token ids of the greedy CTC path."""
    return [
        [tok for tok, _ in utt]
        for utt in greedy_ctc_with_frames(log_probs, lens, blank_id)
    ]


def token_times(
    frames: list[tuple[int, int]], frame_shift_s: float
) -> list[tuple[int, float]]:
    """`(token, start_time_seconds)` from `(token, first_frame_index)`."""
    return [(tok, frame * frame_shift_s) for tok, frame in frames]


def per(hyps: Sequence[Sequence[int]], refs: Sequence[Sequence[int]]) -> float:
    """Summed Levenshtein distance (each capped at its ref length) over total
    ref length; 0.0 when there are no reference tokens."""
    errors = total = 0
    for hyp, ref in zip(hyps, refs, strict=True):
        if not ref:
            continue
        errors += min(Levenshtein.distance(list(hyp), list(ref)), len(ref))
        total += len(ref)
    return errors / total if total else 0.0


def silence_frame_error(
    log_probs: Tensor, lens: Tensor, blank_id: int
) -> tuple[int, int]:
    """`(non_blank_frames, total_frames)` of the argmax path over valid frames."""
    best = log_probs.detach().argmax(dim=-1).cpu()
    valid = torch.arange(best.shape[1])[None, :] < lens.cpu()[:, None]
    return int(((best != blank_id) & valid).sum()), int(valid.sum())


def set_streaming_context(model, chunk_size: int, left_context: int) -> None:
    """Make the (non-streaming) forward simulate streaming with these sizes."""
    model.encoder.chunk_size = (chunk_size,)
    model.encoder.left_context_frames = (left_context,)


@torch.no_grad()
def evaluate_split(
    model,
    loader,
    tokens: PhonemeTokens,
    device,
    chunk_size: int,
    left_context: int,
    max_cuts: int | None = None,
) -> dict:
    """Greedy-decode a split. Speech cuts give `per_phonemes`; silence cuts give
    `silence_per` (fraction of non-blank output frames, None without silence)."""
    encoder = model.encoder
    saved = (encoder.chunk_size, encoder.left_context_frames)
    was_training = model.training
    model.eval()
    set_streaming_context(model, chunk_size, left_context)
    blank = tokens.blank_id
    hyps, refs = [], []
    sil_err = sil_total = num_speech = num_silence = 0
    try:
        for batch in loader:
            if max_cuts is not None and num_speech + num_silence >= max_cuts:
                break
            feats, feat_lens, texts, cuts = batch_to_tensors(batch)
            enc_out, enc_lens = model.forward_encoder(
                feats.to(device), feat_lens.to(device)
            )
            log_probs = model.ctc_output(enc_out)
            for i, cut in enumerate(cuts):
                if max_cuts is not None and num_speech + num_silence >= max_cuts:
                    break
                lp, ln = log_probs[i : i + 1], enc_lens[i : i + 1]
                if cut.supervisions[0].custom["input_type"] == "silence":
                    err, total = silence_frame_error(lp, ln, blank)
                    sil_err, sil_total = sil_err + err, sil_total + total
                    num_silence += 1
                else:
                    hyps.append(greedy_ctc(lp, ln, blank)[0])
                    refs.append(tokens.encode(texts[i]))
                    num_speech += 1
    finally:
        encoder.chunk_size, encoder.left_context_frames = saved
        model.train(was_training)
    return {
        "per_phonemes": per(hyps, refs),
        "silence_per": sil_err / sil_total if sil_total else None,
        "num_speech": num_speech,
        "num_silence": num_silence,
    }

"""CTC frame budget: does a cut have enough encoder output frames for its label?

CTC needs at least `L + R` output frames for a label of length `L` with `R`
identical consecutive pairs (each repeat needs a blank in between). Pure
Python (no torch); used by data prep (report) and training (filter).
"""

from collections.abc import Iterable

import numpy as np
from lhotse import CutSet, MonoCut
from lhotse.utils import compute_num_frames

FRAME_SHIFT = 0.01
SAMPLING_RATE = 16000
REPORT_FACTORS = (1, 2)
REPORT_SPEEDS = (1.0, 1.1)
MAX_REPORTED_IDS = 50


def count_repeats(text: str) -> int:
    """Number of positions `i` with `text[i] == text[i + 1]`."""
    return sum(a == b for a, b in zip(text, text[1:]))


def required_frames(text: str) -> int:
    return len(text) + count_repeats(text)


def fbank_frames(duration: float, speed: float = 1.0) -> int:
    """Fbank frames (10 ms shift) of a cut after speed perturbation."""
    return compute_num_frames(
        duration / speed, frame_shift=FRAME_SHIFT, sampling_rate=SAMPLING_RATE
    )


def output_frames(t_fbank: int, output_downsampling_factor: int) -> int:
    """Encoder output frames: icefall `Conv2dSubsampling` ((T - 7) // 2) then
    Zipformer2's optional output downsampling ((T + 1) // 2)."""
    t = (t_fbank - 7) // 2
    if output_downsampling_factor == 2:
        t = (t + 1) // 2
    elif output_downsampling_factor != 1:
        raise ValueError(f"output_downsampling_factor must be 1 or 2, got {output_downsampling_factor}")
    return max(t, 0)


def ratio(cut_duration: float, text: str, factor: int, speed: float) -> float:
    """Output frames / required frames; `inf` for empty text."""
    needed = required_frames(text)
    if needed == 0:
        return float("inf")
    return output_frames(fbank_frames(cut_duration, speed), factor) / needed


def _is_silence(cut: MonoCut) -> bool:
    custom = cut.supervisions[0].custom or {}
    return custom.get("input_type") == "silence"


def _text(cut: MonoCut) -> str:
    return cut.supervisions[0].text or ""


def _stats(ids: list[str], ratios: list[float]) -> dict:
    if not ratios:
        return {"min": 0.0, "p1": 0.0, "p5": 0.0, "p50": 0.0, "num_below_1": 0, "ids_below_1": []}
    arr = np.asarray(ratios, dtype=float)
    p1, p5, p50 = np.percentile(arr, [1, 5, 50])
    below = sorted((r, i) for i, r in zip(ids, ratios) if r < 1)
    return {
        "min": float(arr.min()),
        "p1": float(p1),
        "p5": float(p5),
        "p50": float(p50),
        "num_below_1": len(below),
        "ids_below_1": [i for _, i in below[:MAX_REPORTED_IDS]],
    }


def report(cuts: Iterable[MonoCut]) -> dict:
    """Ratio statistics of the speech cuts at both frame rates and speeds."""
    speech = [(c.id, c.duration, _text(c)) for c in cuts if not _is_silence(c)]
    ids = [i for i, _, _ in speech]
    return {
        f"f{factor}_s{speed}": _stats(
            ids, [ratio(d, t, factor, speed) for _, d, t in speech]
        )
        for factor in REPORT_FACTORS
        for speed in REPORT_SPEEDS
    }


def filter_feasible(cuts: CutSet, factor: int, speed: float) -> tuple[CutSet, list[str]]:
    """Keep silence cuts and speech cuts with ratio >= 1; return the dropped ids."""
    kept, dropped = [], []
    for cut in cuts:
        if _is_silence(cut) or ratio(cut.duration, _text(cut), factor, speed) >= 1:
            kept.append(cut)
        else:
            dropped.append(cut.id)
    return CutSet.from_cuts(kept), dropped

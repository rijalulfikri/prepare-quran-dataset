"""Per-epoch evaluation sets beyond the legacy dev split.

The legacy dev split (`cuts_dev`, 10 % of the train pool) shares reciters with
train and stopped moving after ~epoch 18 of v0, so it says nothing about
generalisation. The sets here are built at training time from files already in
`data_dir` (the train/dev split and the data shards are untouched):

- `dev_unseen`: a fixed, seeded sample of `cuts_test` (the held-out reciters),
  `dev_unseen_cuts_per_moshaf` cuts per moshaf.
- `qdat`: the 159 clips of `obadx/qdat_bench` (phone recordings, mostly women),
  labels = `phonetic_transcript` without spaces, as qdat_bench's own scorer.
  Evaluation only; never trained on. Cached under `data_dir/eval_qdat/`.
- `dev_unseen_<stress>`: the same `dev_unseen` sample with one deterministic
  perturbation each (`STRESS`).

Set names become key prefixes in dev_metrics.jsonl (`qdat/per_phonemes`);
the legacy dev metrics keep their unprefixed keys.
"""

from __future__ import annotations

import io
import logging
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from lhotse import CutSet, MonoCut, Recording

from quran_zipformer import frame_budget as fb
from quran_zipformer.config import ZipformerTrainConfig
from quran_zipformer.datamodule import load_cuts
from quran_zipformer.tokens import PhonemeTokens

log = logging.getLogger(__name__)

LEGACY_DEV = "dev"
DEV_UNSEEN = "dev_unseen"
QDAT = "qdat"
QDAT_DATASET = "obadx/qdat_bench"
QDAT_DIR = "eval_qdat"
QDAT_MANIFEST = "cuts_qdat.jsonl.gz"
STRESS_VOLUME_DB = -20.0
STRESS_SPEED = 1.25


STRESS_RIR_BANK = 64


def rir_bank(seed: int, n: int = STRESS_RIR_BANK) -> list[Recording]:
    """`n` simulated room impulse responses, identical for the same seed.

    Lhotse's FastRandomRIRGenerator takes room / source seeds but also draws
    reflections with `torch.multinomial` from torch's global RNG, so seeds alone
    do not fix the RIR; it is generated once here under a forked, seeded torch
    RNG and kept as a small in-memory recording (float WAV).
    """
    from lhotse.augmentation.utils import FastRandomRIRGenerator

    bank = []
    threads = torch.get_num_threads()
    # one thread: the generator's `rir[idx] += decay` with repeated indices keeps
    # whichever write lands last, which varies between multithreaded runs
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            for k in range(n):
                rir = FastRandomRIRGenerator(room_seed=seed + 2 * k, source_seed=seed + 2 * k + 1)(nsource=1)
                buf = io.BytesIO()
                sf.write(buf, np.asarray(rir, dtype="float32").T, 16000, format="WAV", subtype="FLOAT")
                bank.append(Recording.from_bytes(buf.getvalue(), recording_id=f"stress_rir_{seed}_{k}"))
    finally:
        torch.set_num_threads(threads)
    return bank


def _stress_volume(cut: MonoCut, i: int, rirs: list[Recording]) -> MonoCut:
    return cut.perturb_volume(factor=10 ** (STRESS_VOLUME_DB / 20))


def _stress_reverb(cut: MonoCut, i: int, rirs: list[Recording]) -> MonoCut:
    # a fixed RIR per cut: the same reverb in every epoch and run
    return cut.reverb_rir(rir_recording=rirs[i % len(rirs)])


def _stress_speed(cut: MonoCut, i: int, rirs: list[Recording]) -> MonoCut:
    return cut.perturb_speed(STRESS_SPEED)


STRESS = {"vol-20db": _stress_volume, "reverb": _stress_reverb, "speed1.25": _stress_speed}


def _moshaf(cut: MonoCut) -> str:
    custom = cut.supervisions[0].custom or {}
    return str(custom.get("moshaf_id") or cut.id.split("__")[0])


def dev_unseen_cuts(cfg: ZipformerTrainConfig, data_dir: Path) -> CutSet:
    """`dev_unseen_cuts_per_moshaf` speech cuts of each test moshaf (all of a smaller one).

    Sampled per moshaf with a string seed of (`cfg.seed`, moshaf) from the cuts
    sorted by id, so the sample does not depend on manifest order or on the
    other moshafs; output sorted by moshaf, then id.
    """
    by_moshaf: dict[str, list[MonoCut]] = defaultdict(list)
    for cut in load_cuts(data_dir, "test"):
        if not fb._is_silence(cut):
            by_moshaf[_moshaf(cut)].append(cut)
    picked = []
    for moshaf in sorted(by_moshaf):
        cuts = sorted(by_moshaf[moshaf], key=lambda c: c.id)
        k = min(cfg.dev_unseen_cuts_per_moshaf, len(cuts))
        sample = random.Random(f"{cfg.seed}:{DEV_UNSEEN}:{moshaf}").sample(cuts, k)
        picked += sorted(sample, key=lambda c: c.id)
        log.info("%s: %d of %d cuts of moshaf %s", DEV_UNSEEN, k, len(cuts), moshaf)
    return CutSet.from_cuts(picked)


def stress_cuts(cuts: CutSet, name: str, seed: int) -> CutSet:
    """`cuts` with the `STRESS[name]` perturbation (ids get Lhotse's suffix)."""
    rirs = rir_bank(seed) if name == "reverb" else []
    return CutSet.from_cuts(STRESS[name](cut, i, rirs) for i, cut in enumerate(cuts))


def _load_qdat_rows():
    from datasets import Audio, load_dataset

    ds = load_dataset(QDAT_DATASET, split="train")
    return ds.cast_column("audio", Audio(decode=False))


def qdat_label(phonetic_transcript: str) -> str:
    """qdat_bench scores against the transcript with all whitespace removed."""
    return re.sub(r"\s+", "", phonetic_transcript)


def qdat_cuts(data_dir: Path) -> CutSet:
    """QDAT clips as 16 kHz FLAC cuts under `data_dir/eval_qdat/` (built once, then cached).

    Audio is decoded as qdat_bench's ONNX path does (librosa, 16 kHz mono).
    """
    from quran_zipformer.data_prep import _write_cuts, _write_wav, make_cut
    from quran_zipformer.qdat import decode_audio

    root = Path(data_dir) / QDAT_DIR
    manifest = root / QDAT_MANIFEST
    if not manifest.exists():
        log.info("building the qdat eval set from %s into %s", QDAT_DATASET, root)
        cuts = []
        for row in _load_qdat_rows():
            rel = f"audio/{row['id']}.flac"
            duration = _write_wav(decode_audio(row["audio"]).wav, root / rel)
            custom = {"input_type": "speech", "qdat_id": row["id"], "gender": row.get("gender")}
            cuts.append(make_cut(rel, duration, qdat_label(row["phonetic_transcript"]), f"qdat__{row['id']}", custom, root))
        _write_cuts(cuts, manifest)
    return CutSet.from_file(manifest).with_recording_path_prefix(str(root))


def build_eval_sets(
    cfg: ZipformerTrainConfig, data_dir: Path, dev_cuts: CutSet, tokens: PhonemeTokens
) -> dict[str, CutSet]:
    """`{"dev": dev_cuts, ...}` plus every set the config enables, in evaluation order.

    Fails here, before training, if a label holds a phoneme outside the vocab
    (evaluation would otherwise crash at the end of the first epoch).
    """
    sets = {LEGACY_DEV: dev_cuts}
    if cfg.dev_unseen_cuts_per_moshaf > 0:
        unseen = dev_unseen_cuts(cfg, data_dir)
        sets[DEV_UNSEEN] = unseen
        for name in cfg.stress_sets:
            sets[f"{DEV_UNSEEN}_{name}"] = stress_cuts(unseen, name, cfg.seed)
    if cfg.eval_qdat:
        sets[QDAT] = qdat_cuts(data_dir)
    for name, cuts in sets.items():
        if name == LEGACY_DEV:
            continue
        for cut in cuts:
            tokens.encode(fb._text(cut))  # KeyError names the phoneme
        log.info("eval set %s: %d cuts, %.2f h", name, len(cuts), sum(c.duration for c in cuts) / 3600)
    return sets


def metric_keys(set_name: str, metrics: dict) -> dict:
    """Row entries of one set: the legacy dev keeps its keys and all of them;
    other sets keep PER and cut count, prefixed with the set name."""
    if set_name == LEGACY_DEV:
        return dict(metrics)
    return {f"{set_name}/{k}": metrics[k] for k in ("per_phonemes", "num_speech")}

"""Lhotse DataLoaders for Zipformer2-CTC: on-the-fly fbank, static speed-perturbed
copies, optional reverb, CutMix noise, optional dB gain, SpecAugment, and a
bucketing sampler that resumes at the exact consumed batch.

Manifests are Task 4's `cuts_<split>.jsonl.gz` under `data_dir`, whose recording
paths are relative to `data_dir`.
"""

import copy
import functools
import logging
import random
import signal
import warnings
from pathlib import Path
from typing import Literal

import lhotse
import torch
from lhotse import CutSet, Fbank, FbankConfig, MonoCut
from lhotse.dataset import (
    CutMix,
    DynamicBucketingSampler,
    K2SpeechRecognitionDataset,
    OnTheFlyFeatures,
    SpecAugment,
)
from lhotse.dataset.dataloading import resolve_seed
from lhotse.dataset.dataloading import worker_init_fn as lhotse_worker_init_fn
from torch import Tensor
from torch.utils.data import DataLoader

from quran_zipformer import frame_budget as fb
from quran_zipformer.config import SPEED_STEP, ZipformerTrainConfig

Split = Literal["train", "dev", "test", "noise_aug"]

log = logging.getLogger(__name__)


def load_cuts(data_dir: Path, split: Split) -> CutSet:
    """Eager CutSet of `cuts_<split>.jsonl.gz` in manifest order, paths resolved."""
    data_dir = Path(data_dir)
    # lhotse reads an empty manifest back as None
    cuts = CutSet.from_file(data_dir / f"cuts_{split}.jsonl.gz") or CutSet()
    return cuts.with_recording_path_prefix(str(data_dir))


def speed_ranges(cfg: ZipformerTrainConfig) -> list[tuple[float, float]]:
    """One `(low, high)` per static copy of a train cut; `[(1, 1)]` without speed perturbation."""
    return [tuple(r) for r in cfg.speed_factor_ranges] if cfg.speed_perturb else [(1.0, 1.0)]


def _grid(low: float, high: float) -> list[float]:
    n = round((high - low) / SPEED_STEP)
    return [round(low + i * SPEED_STEP, 2) for i in range(n + 1)]


def copy_factors(cfg: ZipformerTrainConfig, cut_id: str) -> list[float]:
    """Speed factor of each copy of `cut_id`, one per range.

    Drawn from a string seed of (`cfg.seed`, cut id, range index), so it is the
    same in every process, epoch and resume, and does not depend on cut order.
    """
    out = []
    for i, (low, high) in enumerate(speed_ranges(cfg)):
        grid = _grid(low, high)
        out.append(grid[0] if len(grid) == 1 else random.Random(f"{cfg.seed}:{cut_id}:{i}").choice(grid))
    return out


def speed_copies(cfg: ZipformerTrainConfig, cuts: CutSet) -> tuple[list[MonoCut], list[str]]:
    """`(kept copies, dropped copy ids)` in block order (all copies of range 0, then 1, ...).

    The CTC frame budget is checked per copy at the factor actually applied to it,
    so a cut too short only at its fastest factor keeps its slower copies.
    Perturbed ids get Lhotse's `_sp<factor>` suffix; disjoint ranges keep them unique.
    """
    factor = cfg.output_downsampling_factor
    factors = {cut.id: copy_factors(cfg, cut.id) for cut in cuts}
    kept, dropped = [], []
    for i in range(len(speed_ranges(cfg))):
        for cut in cuts:
            speed = factors[cut.id][i]
            if not fb._is_silence(cut) and fb.ratio(cut.duration, fb._text(cut), factor, speed) < 1:
                dropped.append(cut.id if speed == 1.0 else f"{cut.id}_sp{speed}")
                continue
            kept.append(cut if speed == 1.0 else cut.perturb_speed(speed))
    return kept, dropped


def _segment_key(cut) -> tuple[str, int, int] | None:
    """`(moshaf, sura, position)` from the data-prep custom fields (`segment_index`
    is `"SSS.NNNN"`); None for silence cuts and cuts without a parseable index."""
    custom = cut.supervisions[0].custom or {}
    if custom.get("input_type") != "speech":
        return None
    try:
        sura, pos = str(custom["segment_index"]).split(".")
        return str(custom["moshaf_id"]), int(sura), int(pos)
    except (KeyError, ValueError):
        return None


def _join(group: list, gaps: list[float]):
    """One cut playing `group` back to back (plus `gaps` of silence) with a single
    supervision whose text is the labels joined without separator.

    Each label already ends in its waqf form, which is what the reciter said:
    the segments were cut at his pauses.
    """
    from lhotse.utils import fastcopy

    joined = group[0]
    for gap, nxt in zip(gaps, group[1:]):
        if gap > 0:
            joined = joined.pad(duration=joined.duration + gap, direction="right")
        joined = joined.append(fastcopy(nxt, supervisions=[]))
    text = "".join(fb._text(c) for c in group)
    first = joined.tracks[0]
    sup = group[0].supervisions[0]
    custom = {
        **(sup.custom or {}),
        "num_labels": len(text),
        "num_repeats": fb.count_repeats(text),
        "concat": len(group),
    }
    new_id = f"{group[0].id}_cat{len(group)}"
    # one supervision on the first track spanning the whole joined cut
    # (lhotse's merge_supervisions would give it a synthetic recording id and channel)
    sup = fastcopy(sup, id=new_id, start=0.0, duration=joined.duration, text=text, custom=custom)
    tracks = [fastcopy(first, cut=fastcopy(first.cut, supervisions=[sup])), *joined.tracks[1:]]
    return fastcopy(joined, id=new_id, tracks=tracks)


def concat_consecutive(cfg: ZipformerTrainConfig, cuts: CutSet) -> CutSet:
    """`cuts` with runs of consecutive segments (same moshaf and sura, adjacent
    positions) joined into one cut; identity when `concat_p` is 0.

    Walks the speech cuts in (moshaf, sura, position) order with one RNG seeded
    from `cfg.seed`, so the groups do not depend on manifest order and are the
    same in every epoch, process and resume. A run starts at a cut with
    probability `concat_p` and is as long as the next adjacent segments allow,
    up to a length drawn from `concat_segments`; joined segments replace the
    originals, so an epoch keeps (nearly) the same audio. Output: the joined
    and untouched cuts in manifest order of each group's first cut.
    """
    if cfg.concat_p <= 0:
        return cuts
    rng = random.Random(f"{cfg.seed}:concat")
    cut_list = list(cuts)  # iterate once: a lazy CutSet need not repeat its order
    keyed = sorted((k, i) for i, c in enumerate(cut_list) if (k := _segment_key(c)) is not None)
    groups: dict[int, list[int]] = {}  # first index -> member indices
    consumed: set[int] = set()
    j = 0
    while j < len(keyed):
        (moshaf, sura, pos), i = keyed[j]
        run = [i]
        if rng.random() < cfg.concat_p:
            want = rng.randint(*cfg.concat_segments)
            while len(run) < want and j + len(run) < len(keyed):
                (m2, s2, p2), i2 = keyed[j + len(run)]
                if (m2, s2, p2) != (moshaf, sura, pos + len(run)):
                    break
                run.append(i2)
        if len(run) > 1:
            groups[i] = run
            consumed.update(run)
        j += len(run)
    out = []
    for i, cut in enumerate(cut_list):
        if i in groups:
            members = [cut_list[m] for m in groups[i]]
            gap_rng = random.Random(f"{cfg.seed}:concat_gap:{cut.id}")
            gaps = [round(gap_rng.uniform(*cfg.concat_gap_seconds), 2) for _ in members[1:]]
            out.append(_join(members, gaps))
        elif i not in consumed:
            out.append(cut)
    return CutSet.from_cuts(out)


def _train_cuts(
    cfg: ZipformerTrainConfig, data_dir: Path, cuts: CutSet | None = None
) -> CutSet:
    """Training cuts (`cuts` or the train manifest), consecutive segments joined
    (`concat_consecutive`), then one static copy per speed range.

    The CutSet is fixed before batching (factors come from `copy_factors`), so
    batch durations respect `max_duration` and the sampler order, hence resume,
    stays exact. Copies failing the frame budget are left out (`speed_copies`).
    With speed perturbation the concatenation is shuffled once with `cfg.seed`:
    DynamicBucketingSampler only shuffles within a streaming buffer and fits
    bucket edges to the first cuts, so block order would put all slow audio
    first in every epoch.
    """
    base = cuts if cuts is not None else load_cuts(data_dir, "train")
    kept, _ = speed_copies(cfg, concat_consecutive(cfg, base))
    combined = CutSet.from_cuts(kept)
    if not cfg.speed_perturb:
        return combined
    return combined.shuffle(rng=random.Random(cfg.seed))


def _num_buckets(cuts: CutSet, max_duration: float) -> int:
    # Up to 30 buckets, but each should hold >= one full batch: with drop_last a
    # bucket's partial batch is dropped, so a tiny set would yield no batches.
    # Lhotse requires 2 <= num_buckets <= len(cuts).
    total = sum(c.duration for c in cuts)
    return min(30, len(cuts), max(2, int(total // max_duration)))


def _sampler(cuts: CutSet, max_duration: float, **kwargs) -> DynamicBucketingSampler:
    # Cuts are eager on purpose (len for bucketing, fixed manifest order for a
    # deterministic resume); silence Lhotse's "eagerly read CutSet" hint.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*eagerly read CutSet.*")
        return DynamicBucketingSampler(
            cuts,
            max_duration=max_duration,
            num_buckets=_num_buckets(cuts, max_duration),
            **kwargs,
        )


class _RandomCutTransform:
    """Applies `_apply(cut)` to each cut with probability `p`.

    The RNG is created lazily from `seed`; "randomized" gives every DataLoader
    worker its own stream (needs Lhotse's `worker_init_fn`, as CutMix does).
    Ids are preserved: the per-batch cut ids must not depend on augmentation.
    """

    def __init__(self, p: float, seed: int | str = "randomized") -> None:
        self.p = p
        self.seed = seed
        self.rng: random.Random | None = None

    def __call__(self, cuts: CutSet) -> CutSet:
        if self.rng is None:
            self.rng = random.Random(resolve_seed(self.seed))
        return CutSet.from_cuts(self._apply(c) if self.rng.random() < self.p else c for c in cuts)

    def _apply(self, cut):
        raise NotImplementedError


class RandomReverb(_RandomCutTransform):
    """Reverb with a simulated room impulse response (Lhotse's FastRandomRIRGenerator,
    no RIR download); the room and source seeds come from this transform's RNG.

    The generator also draws from torch's global RNG, so the exact RIRs are not
    reproducible run to run; resume only depends on the sampler order, not on them.
    """

    def _apply(self, cut):
        return cut.reverb_rir(
            affix_id=False,
            room_rng_seed=self.rng.randrange(2**31),
            source_rng_seed=self.rng.randrange(2**31),
        )


class RandomGainDb(_RandomCutTransform):
    """Gain drawn uniformly in dB over `db_range`.

    Lhotse's PerturbVolume draws a linear factor (0.125-2.0 by default), so the
    quiet end (-18 dB) is hardly ever reached; quiet microphones are the case
    to cover. Applied to a MixedCut it scales speech and noise together.
    """

    def __init__(self, p: float, db_range: tuple[float, float], seed: int | str = "randomized") -> None:
        super().__init__(p, seed)
        self.db_range = db_range

    def draw_db(self) -> float:
        return self.rng.uniform(*self.db_range)

    def _apply(self, cut):
        return cut.perturb_volume(factor=10 ** (self.draw_db() / 20), affix_id=False)


def _dataset(
    cfg: ZipformerTrainConfig, noise_aug: CutSet | None, train: bool
) -> K2SpeechRecognitionDataset:
    """Fbank dataset; `train` adds the cut augmentations and SpecAugment (eval
    datasets get no augmentation).

    Cut transforms run in this order: reverb (on the speech), noise CutMix, then
    gain, so a quiet clip is quiet speech plus quiet noise, as from a far microphone.
    """
    transforms: list = []
    if train and cfg.reverb_p > 0:
        transforms.append(RandomReverb(cfg.reverb_p))
    if noise_aug is not None and len(noise_aug) > 0:
        # noise cuts carry an empty "silence" supervision; mixed in, it would add a
        # second supervision (and a bogus "" target) to the speech cut
        noise = noise_aug.drop_supervisions()
        transforms.append(
            CutMix(
                cuts=noise,
                p=cfg.cutmix_p,
                snr=cfg.cutmix_snr,
                preserve_id=True,
                # per-worker RNG: with a fixed seed every persistent worker would
                # draw the same noise / SNR sequence
                seed="randomized",
            )
        )
    if train and cfg.volume_p > 0:
        transforms.append(RandomGainDb(cfg.volume_p, cfg.volume_db))
    input_transforms = []
    # with CR-CTC the loss masks two views of each batch itself (cr_spec_augment)
    if train and not cfg.use_cr_ctc:
        input_transforms.append(  # icefall's defaults
            SpecAugment(
                time_warp_factor=80,
                num_frame_masks=10,
                features_mask_size=27,
                num_feature_masks=2,
                frames_mask_size=100,
            )
        )
    return K2SpeechRecognitionDataset(
        input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=cfg.feature_dim))),
        cut_transforms=transforms,
        input_transforms=input_transforms,
        return_cuts=True,
    )


def cr_spec_augment(cfg: ZipformerTrainConfig) -> SpecAugment:
    """SpecAugment for the two CR-CTC views (icefall `get_spec_augment`): no time
    warp (applied once before the batch is duplicated) and `cr_time_mask_ratio`
    times the time masking."""
    return SpecAugment(
        time_warp_factor=0,
        num_frame_masks=int(10 * cfg.cr_time_mask_ratio),
        features_mask_size=27,
        num_feature_masks=2,
        frames_mask_size=100,
        max_frames_mask_fraction=0.15 * cfg.cr_time_mask_ratio,
    )


def supervision_segments(batch: dict) -> Tensor:
    """`(N, 3)` int tensor `[sequence_idx, start_frame, num_frames]` per supervision
    (the layout icefall's `time_warp` expects)."""
    sups = batch["supervisions"]
    return torch.stack([sups["sequence_idx"], sups["start_frame"], sups["num_frames"]], dim=1)


def _ignore_stop_signals(worker_id: int) -> None:
    """Workers ignore SIGINT/SIGTERM.

    A Ctrl-C or preemption SIGTERM goes to the whole process group; spawned
    workers do not inherit the trainer's handlers and would die, crashing the
    main loop before it can save. The main process owns the graceful stop and
    shuts the workers down itself.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)


def _init_worker(worker_id: int, seed: int) -> None:
    """DataLoader `worker_init_fn`: ignore stop signals and give each worker its
    own RNG seed (Lhotse's `worker_init_fn`, required by `CutMix(seed="randomized")`)."""
    _ignore_stop_signals(worker_id)
    lhotse_worker_init_fn(worker_id, seed=seed)


def _loader(cfg: ZipformerTrainConfig, ds, sampler, seed: int) -> DataLoader:
    """`seed` feeds the per-worker augmentation RNG (Lhotse adds `100 * worker_id`);
    the loader is rebuilt every epoch, so callers vary it per epoch or the
    CutMix/SpecAugment draws replay identically each epoch."""
    return DataLoader(
        ds,
        sampler=sampler,
        batch_size=None,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        worker_init_fn=functools.partial(_init_worker, seed=seed) if cfg.num_workers > 0 else None,
    )


def restore_sampler(
    sampler: DynamicBucketingSampler, state: dict, batches_consumed: int
) -> None:
    """Load `state`, rewinding it to the `batches_consumed` batches actually trained.

    With `num_workers > 0` the DataLoader draws batches ahead of the training
    loop, so the saved state over-counts. Lhotse 1.33's
    `DynamicBucketingSampler.load_state_dict` (lhotse/dataset/sampling/
    dynamic_bucketing.py, `_fast_forward`) re-draws
    `diagnostics.current_epoch_stats.total_batches` batches, where
    `total_batches = kept_batches + discarded_batches` of
    `state["diagnostics"]["stats_per_epoch"][current_epoch]`
    (`EpochDiagnostics` in lhotse/dataset/sampling/base.py). We override
    `kept_batches = batches_consumed` and `discarded_batches = 0` (no discards
    happen mid-epoch). `load_state_dict` mutates its input, hence the deepcopy.
    """
    state = copy.deepcopy(state)
    try:
        diag = state["diagnostics"]
        epoch = int(diag["current_epoch"])
        # a JSON round-trip turns the int epoch keys into strings; Lhotse looks them up by int
        per_epoch = {int(k): v for k, v in diag["stats_per_epoch"].items()}
        stats = per_epoch[epoch]
        missing = {"kept_batches", "discarded_batches"} - set(stats)
    except (KeyError, TypeError, ValueError, AttributeError):
        missing = {"diagnostics.current_epoch / stats_per_epoch[current_epoch]"}
    if missing:
        raise RuntimeError(
            "Lhotse sampler state layout changed; resume positioning needs updating "
            f"(lhotse {lhotse.__version__}; missing {sorted(missing)})"
        )
    diag["current_epoch"] = epoch
    diag["stats_per_epoch"] = per_epoch
    stats["kept_batches"] = batches_consumed
    stats["discarded_batches"] = 0
    sampler.load_state_dict(state)


def train_dataloader(
    cfg: ZipformerTrainConfig,
    data_dir: Path,
    epoch: int = 0,
    sampler_state: dict | None = None,
    batches_consumed: int = 0,
    cuts: CutSet | None = None,
) -> DataLoader:
    """Shuffled, bucketed training loader; the sampler is `loader.sampler`.

    `cuts` replaces the train manifest (e.g. frame-budget-filtered cuts); speed
    perturbation is applied on top. Pass `sampler_state` (a
    `loader.sampler.state_dict()`) and the number of batches the training loop
    consumed to resume mid-epoch.
    """
    log.info("rebuilding training cuts for epoch %d (may take a few minutes)…", epoch)
    train_cuts = _train_cuts(cfg, data_dir, cuts)
    sampler = _sampler(
        train_cuts, cfg.max_duration, shuffle=True, drop_last=True, seed=cfg.seed
    )
    sampler.set_epoch(epoch)
    if sampler_state is not None:
        log.info("fast-forwarding sampler by %d batches…", batches_consumed)
        restore_sampler(sampler, sampler_state, batches_consumed)
    # augmentation RNG differs per epoch; the sampler order (and so resume) does not depend on it
    return _loader(
        cfg,
        _dataset(cfg, load_cuts(data_dir, "noise_aug"), train=True),
        sampler,
        seed=cfg.seed + 1000 * epoch,
    )


def eval_dataloader(
    cfg: ZipformerTrainConfig, cuts: CutSet, max_duration: float | None = None
) -> DataLoader:
    """Unshuffled loader over every cut of `cuts`, no augmentation."""
    max_duration = max_duration if max_duration is not None else cfg.max_duration
    sampler = _sampler(cuts, max_duration, shuffle=False, drop_last=False)
    return _loader(cfg, _dataset(cfg, None, train=False), sampler, seed=cfg.seed)


def batch_to_tensors(batch: dict) -> tuple[Tensor, Tensor, list[str], list[MonoCut]]:
    """`(features (N, T, F), feature_lens (N,), texts, cuts)` of a dataset batch."""
    sups = batch["supervisions"]
    features, texts = batch["inputs"], sups["text"]
    # one supervision per cut, so supervisions align with the batch rows
    assert len(texts) == features.shape[0], (len(texts), features.shape)
    return features, sups["num_frames"], texts, sups["cut"]

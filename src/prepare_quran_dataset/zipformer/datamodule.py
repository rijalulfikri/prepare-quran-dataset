"""Lhotse DataLoaders for Zipformer2-CTC: on-the-fly fbank, CutMix noise, 3x speed
perturbation and a bucketing sampler that resumes at the exact consumed batch.

Manifests are Task 4's `cuts_<split>.jsonl.gz` under `data_dir`, whose recording
paths are relative to `data_dir`.
"""

import copy
import warnings
from pathlib import Path
from typing import Literal

from lhotse import CutSet, Fbank, FbankConfig, MonoCut
from lhotse.dataset import (
    CutMix,
    DynamicBucketingSampler,
    K2SpeechRecognitionDataset,
    OnTheFlyFeatures,
)
from torch import Tensor
from torch.utils.data import DataLoader

from prepare_quran_dataset.zipformer.config import ZipformerTrainConfig

Split = Literal["train", "dev", "test", "noise_aug"]


def load_cuts(data_dir: Path, split: Split) -> CutSet:
    """Eager CutSet of `cuts_<split>.jsonl.gz` in manifest order, paths resolved."""
    data_dir = Path(data_dir)
    # lhotse reads an empty manifest back as None
    cuts = CutSet.from_file(data_dir / f"cuts_{split}.jsonl.gz") or CutSet()
    return cuts.with_recording_path_prefix(str(data_dir))


def _train_cuts(
    cfg: ZipformerTrainConfig, data_dir: Path, cuts: CutSet | None = None
) -> CutSet:
    """Training cuts (`cuts` or the train manifest), x3 with speed perturbation.

    Perturbed ids get Lhotse's `_sp0.9` / `_sp1.1` suffix, so ids stay unique.
    """
    base = cuts if cuts is not None else load_cuts(data_dir, "train")
    if not cfg.speed_perturb:
        return base
    return CutSet.from_cuts([*base.perturb_speed(0.9), *base, *base.perturb_speed(1.1)])


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


def _dataset(cfg: ZipformerTrainConfig, noise_aug: CutSet | None) -> K2SpeechRecognitionDataset:
    transforms = []
    if noise_aug is not None and len(noise_aug) > 0:
        # noise cuts carry an empty "silence" supervision; mixed in, it would add a
        # second supervision (and a bogus "" target) to the speech cut
        noise = noise_aug.drop_supervisions()
        transforms.append(
            CutMix(cuts=noise, p=cfg.cutmix_p, snr=cfg.cutmix_snr, preserve_id=True)
        )
    return K2SpeechRecognitionDataset(
        input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=cfg.feature_dim))),
        cut_transforms=transforms,
        return_cuts=True,
    )


def _loader(cfg: ZipformerTrainConfig, ds, sampler) -> DataLoader:
    return DataLoader(
        ds,
        sampler=sampler,
        batch_size=None,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
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
    diag = state["diagnostics"]
    epoch = int(diag["current_epoch"])
    diag["current_epoch"] = epoch
    # a JSON round-trip turns the int epoch keys into strings; Lhotse looks them up by int
    diag["stats_per_epoch"] = {int(k): v for k, v in diag["stats_per_epoch"].items()}
    stats = diag["stats_per_epoch"][epoch]
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
    train_cuts = _train_cuts(cfg, data_dir, cuts)
    sampler = _sampler(
        train_cuts, cfg.max_duration, shuffle=True, drop_last=True, seed=cfg.seed
    )
    sampler.set_epoch(epoch)
    if sampler_state is not None:
        restore_sampler(sampler, sampler_state, batches_consumed)
    return _loader(cfg, _dataset(cfg, load_cuts(data_dir, "noise_aug")), sampler)


def eval_dataloader(
    cfg: ZipformerTrainConfig, cuts: CutSet, max_duration: float | None = None
) -> DataLoader:
    """Unshuffled loader over every cut of `cuts`, no augmentation."""
    max_duration = max_duration if max_duration is not None else cfg.max_duration
    sampler = _sampler(cuts, max_duration, shuffle=False, drop_last=False)
    return _loader(cfg, _dataset(cfg, None), sampler)


def batch_to_tensors(batch: dict) -> tuple[Tensor, Tensor, list[str], list[MonoCut]]:
    """`(features (N, T, F), feature_lens (N,), texts, cuts)` of a dataset batch."""
    sups = batch["supervisions"]
    features, texts = batch["inputs"], sups["text"]
    # one supervision per cut, so supervisions align with the batch rows
    assert len(texts) == features.shape[0], (len(texts), features.shape)
    return features, sups["num_frames"], texts, sups["cut"]

"""Zipformer2-CTC training loop: ScaledAdam + Eden, mid-epoch checkpoints, exact
resume (local or Drive), per-epoch dev PER, checkpoint averaging, W&B / TensorBoard.

Built for boxes that may die at any moment: every `save_every_n_batches` global
batches a `checkpoint-{batch_idx_train}.pt` is written atomically (with the
sampler position) and mirrored to Drive; on start the newest complete checkpoint
(local or Drive) is restored and the sampler resumes at the exact next batch.

CLI::

    python -m prepare_quran_dataset.zipformer.train_loop --config <yml> --data-dir <dir>
    python -m prepare_quran_dataset.zipformer.train_loop --test-resume --config <yml> \\
        --data-dir <dir> --n-batches N     # prints a JSON summary as the last stdout line
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import torch
from lhotse import CutSet
from torch import Tensor

from prepare_quran_dataset.zipformer import checkpointing as ck
from prepare_quran_dataset.zipformer import frame_budget as fb
from prepare_quran_dataset.zipformer.config import ZipformerTrainConfig
from prepare_quran_dataset.zipformer.datamodule import (
    batch_to_tensors,
    eval_dataloader,
    load_cuts,
    train_dataloader,
)
from prepare_quran_dataset.zipformer.decode import evaluate_split
from prepare_quran_dataset.zipformer.drive_backup import DriveBackup
from prepare_quran_dataset.zipformer.icefall_import import setup_icefall_paths
from prepare_quran_dataset.zipformer.model import build_model
from prepare_quran_dataset.zipformer.tokens import PhonemeTokens

setup_icefall_paths()

from icefall.checkpoint import average_checkpoints  # noqa: E402
from optim import Eden, ScaledAdam  # noqa: E402  (icefall recipe)
from train import get_parameter_groups_with_lrs, set_batch_count  # noqa: E402  (icefall recipe)

log = logging.getLogger(__name__)

REF_DURATION = 600.0  # recipe `params.ref_duration`
DEFAULT_VOCAB = Path(__file__).resolve().parents[3] / "vocab_streaming" / "vocab.json"
DEV_METRICS = "dev_metrics.jsonl"


class IncompatibleCheckpoint(RuntimeError):
    pass


class FrameBudgetError(RuntimeError):
    pass


class TrainingInterrupted(RuntimeError):
    pass


class ResumeError(RuntimeError):
    """Drive holds a newer checkpoint that cannot be fetched; resuming older would lose work."""


class DivergedTraining(RuntimeError):
    """Weights or losses went non-finite; stop before checkpointing the damage."""


MAX_NONFINITE_BATCHES = 20  # consecutive non-finite losses tolerated


@dataclass
class TrainState:
    epoch: int = 1
    batch_idx_train: int = 0
    batches_consumed_in_epoch: int = 0  # batches the loop finished in `epoch`
    sampler_state: dict | None = None  # `loader.sampler.state_dict()` at the last save
    wandb_run_id: str | None = None
    nonfinite_streak: int = 0  # consecutive non-finite losses (not checkpointed)


# ------------------------------------------------------------------ loss


def encode_targets(texts: list[str], tokens: PhonemeTokens, device) -> tuple[Tensor, Tensor]:
    """`(targets 1-D concatenated, target_lengths (N,))`, both int64; "" -> length 0."""
    ids = [tokens.encode(t or "") for t in texts]
    targets = torch.tensor([i for seq in ids for i in seq], dtype=torch.int64, device=device)
    lengths = torch.tensor([len(seq) for seq in ids], dtype=torch.int64, device=device)
    return targets, lengths


def compute_ctc_loss(model, feats: Tensor, feat_lens: Tensor, targets: Tensor, target_lengths: Tensor) -> tuple[Tensor, Tensor]:
    """`(loss_sum, num_frames)`; divide by `num_frames` for logging only (icefall)."""
    enc, enc_lens = model.forward_encoder(feats, feat_lens)
    loss = model.forward_ctc(enc, enc_lens, targets, target_lengths)
    return loss, feat_lens.sum()


# ------------------------------------------------------------------ optimizer / state


def build_optimizer(model, cfg: ZipformerTrainConfig) -> tuple[ScaledAdam, Eden]:
    optimizer = ScaledAdam(
        get_parameter_groups_with_lrs(model, lr=cfg.base_lr, include_names=True),
        lr=cfg.base_lr,
        clipping_scale=2.0,
    )
    scheduler = Eden(optimizer, cfg.lr_batches, cfg.lr_epochs, warmup_batches=cfg.warmup_batches, warmup_start=0.1)
    return optimizer, scheduler


def adjusted_batch_count(batch_idx_train: int, max_duration: float) -> float:
    """Recipe `get_adjusted_batch_count` (world_size 1)."""
    return batch_idx_train * max_duration / REF_DURATION


def make_backup(cfg: ZipformerTrainConfig) -> DriveBackup | None:
    if not cfg.rclone_checkpoint_path:
        return None
    return DriveBackup(cfg.rclone_checkpoint_path, cfg.keep_last_k, cfg.upload_timeout_seconds)


def make_checkpoint(model, opt, sched, st: TrainState, sampler, cfg: ZipformerTrainConfig) -> dict:
    """Checkpoint dict; `sampler.state_dict()` is taken now (None without a sampler)."""
    return {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "batch_idx_train": st.batch_idx_train,
        "epoch": st.epoch,
        "batches_consumed_in_epoch": st.batches_consumed_in_epoch,
        "sampler": sampler.state_dict() if sampler is not None else None,
        "wandb_run_id": st.wandb_run_id,
        "config": cfg.model_dump(mode="json"),
    }


def save_and_backup(ckpt: dict, kind: Literal["batch", "epoch"], cfg: ZipformerTrainConfig, backup: DriveBackup | None) -> Path:
    """Save atomically, queue the Drive upload, prune (never a pending upload)."""
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt["kind"] = kind
    path = out / ck.ckpt_name(kind, ckpt["batch_idx_train"], ckpt["epoch"])
    ck.save_checkpoint_atomic(
        ckpt,
        path,
        {
            "kind": kind,
            "batch_idx_train": ckpt["batch_idx_train"],
            "epoch": ckpt["epoch"],
            "batches_consumed_in_epoch": ckpt["batches_consumed_in_epoch"],
        },
    )
    if backup is not None:
        backup.submit(path)
    ck.prune(out, cfg.keep_last_k, protect=backup.pending_paths() if backup is not None else ())
    return path


def restore(model, opt, sched, ckpt: dict, cfg: ZipformerTrainConfig | None = None) -> TrainState:
    """Load model/optimizer/scheduler state and return where training continues.

    Refuses (before loading anything) a checkpoint of another output frame rate.
    `cfg` (current config) only supplies `max_duration` for the batch count;
    without it the checkpoint's own config is used.
    """
    saved_cfg = ckpt.get("config") or {}
    saved_factor = saved_cfg.get("output_downsampling_factor")
    current_factor = getattr(model.encoder, "out_ds", None)
    if saved_factor != current_factor:
        raise IncompatibleCheckpoint(
            f"checkpoint output_downsampling_factor={saved_factor} != model's {current_factor}"
        )
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optimizer"])
    sched.load_state_dict(ckpt["scheduler"])
    batch_idx_train = int(ckpt["batch_idx_train"])
    max_duration = cfg.max_duration if cfg is not None else saved_cfg["max_duration"]
    set_batch_count(model, adjusted_batch_count(batch_idx_train, max_duration))
    if ckpt.get("kind") == "epoch":
        epoch, consumed, sampler_state = int(ckpt["epoch"]) + 1, 0, None
    else:
        epoch, consumed, sampler_state = int(ckpt["epoch"]), int(ckpt["batches_consumed_in_epoch"]), ckpt.get("sampler")
    return TrainState(
        epoch=epoch,
        batch_idx_train=batch_idx_train,
        batches_consumed_in_epoch=consumed,
        sampler_state=sampler_state,
        wandb_run_id=ckpt.get("wandb_run_id"),
    )


def _load(pt: Path) -> dict:
    return torch.load(pt, map_location="cpu", weights_only=False)


# ------------------------------------------------------------------ resume source


def _newest_valid_local(out: Path) -> tuple[Path, ck.CkptMeta] | None:
    for pt, meta in reversed(ck.list_complete(out)):
        if ck.verify_sha256(pt, meta):
            return pt, meta
        log.warning("local checkpoint %s fails sha256 verification; ignoring it", pt.name)
    return None


def _download_first_usable(backup: DriveBackup, candidates: list[ck.CkptMeta], out: Path) -> tuple[Path, ck.CkptMeta]:
    """Download the newest candidate that verifies (each tried twice), else raise."""
    for meta in candidates:
        for attempt in (1, 2):
            try:
                return backup.download(meta, out), meta
            except Exception as exc:  # noqa: BLE001
                log.warning("Drive download of %s failed (attempt %d/2): %s", meta.name, attempt, exc)
    raise ResumeError(
        f"Drive has newer checkpoints ({', '.join(m.name for m in candidates)}) than {out} "
        "but none could be downloaded and verified; refusing to resume from an older state"
    )


def resolve_resume(cfg: ZipformerTrainConfig, backup: DriveBackup | None) -> Path | None:
    """Newest complete checkpoint of local `output_dir` and Drive (downloaded if newer)."""
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    local = _newest_valid_local(out)
    remote_metas: list[ck.CkptMeta] = []
    if backup is not None:
        try:
            remote_metas = backup.remote_complete_metas()
        except Exception as exc:  # noqa: BLE001 - a listing failure must not block a local resume
            log.warning("cannot list Drive checkpoints (%s); using local ones only", exc)
    newer = [m for m in remote_metas if local is None or m.sort_key() > local[1].sort_key()]
    chosen, source = (local, "local") if local is not None else (None, "none")
    if newer:
        # Drive is known to be ahead: never fall back to an older local state (or
        # scratch), which would later overwrite the real run's checkpoints on Drive.
        chosen, source = (_download_first_usable(backup, newer, out), "drive")
    if backup is not None:
        try:
            backup.download_missing_epochs(out)
        except Exception as exc:  # noqa: BLE001
            log.warning("cannot download missing epoch checkpoints (%s)", exc)
    if chosen is None:
        log.info("resume: source=none checkpoint=- epoch=1 batch_idx_train=0")
        return None
    pt, meta = chosen
    log.info("resume: source=%s checkpoint=%s epoch=%d batch_idx_train=%d", source, pt.name, meta.epoch, meta.batch_idx_train)
    return pt


# ------------------------------------------------------------------ frame budget


def apply_frame_budget(cfg: ZipformerTrainConfig, data_dir: Path) -> tuple[CutSet, CutSet]:
    """Drop CTC-infeasible speech cuts from train and dev; abort if too many.

    Deterministic for the same data and config, so a resume rebuilds the same cuts.
    """
    speed = 1.1 if cfg.speed_perturb else 1.0
    factor = cfg.output_downsampling_factor
    raw = {split: load_cuts(data_dir, split) for split in ("train", "dev")}
    kept, dropped = {}, {}
    for split, cuts in raw.items():
        kept[split], dropped[split] = fb.filter_feasible(cuts, factor, speed)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dropped_cuts.json").write_text(
        json.dumps({"factor": factor, "speed": speed, "train": dropped["train"], "dev": dropped["dev"]}, indent=2)
    )
    num_speech = sum(not fb._is_silence(c) for cuts in raw.values() for c in cuts)
    num_dropped = len(dropped["train"]) + len(dropped["dev"])
    fraction = num_dropped / num_speech if num_speech else 0.0
    log.info("frame budget (factor %d, speed %.1f): dropped %d train + %d dev of %d speech cuts",
             factor, speed, len(dropped["train"]), len(dropped["dev"]), num_speech)
    if fraction > cfg.max_dropped_fraction:
        raise FrameBudgetError(
            f"{num_dropped}/{num_speech} speech cuts ({fraction:.2%}) are too short for their labels at "
            f"output_downsampling_factor={factor}, speed {speed} (max_dropped_fraction="
            f"{cfg.max_dropped_fraction}); see {out / 'dropped_cuts.json'}"
        )
    return kept["train"], kept["dev"]


# ------------------------------------------------------------------ logging


class Logger(Protocol):
    def log(self, metrics: dict[str, float], step: int) -> None: ...

    def close(self) -> None: ...


class NullLogger:
    def log(self, metrics: dict[str, float], step: int) -> None:
        pass

    def close(self) -> None:
        pass


class TensorBoardLogger:
    def __init__(self, log_dir: Path) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(log_dir=str(log_dir))

    def log(self, metrics: dict[str, float], step: int) -> None:
        for key, value in metrics.items():
            if value is not None:
                self._writer.add_scalar(key, value, step)

    def close(self) -> None:
        self._writer.close()


class WandbLogger:
    def __init__(self, project: str, run_id: str | None, config: dict | None = None) -> None:
        import wandb

        self._run = wandb.init(project=project, id=run_id, resume="allow", config=config)

    @property
    def run_id(self) -> str:
        return self._run.id

    def log(self, metrics: dict[str, float], step: int) -> None:
        self._run.log({k: v for k, v in metrics.items() if v is not None}, step=step)

    def close(self) -> None:
        self._run.finish()


class _MultiLogger:
    def __init__(self, loggers: list) -> None:
        self._loggers = loggers

    def log(self, metrics: dict[str, float], step: int) -> None:
        for lg in self._loggers:
            lg.log(metrics, step)

    def close(self) -> None:
        for lg in self._loggers:
            lg.close()


def make_logger(cfg: ZipformerTrainConfig, run_id: str | None) -> tuple[Logger, str | None]:
    """TensorBoard always; plus W&B (resuming `run_id`) when `WANDB_API_KEY` is set."""
    tb = TensorBoardLogger(Path(cfg.output_dir) / "tb")
    if not os.environ.get("WANDB_API_KEY"):
        log.warning("WANDB_API_KEY is not set; logging to TensorBoard only (%s)", Path(cfg.output_dir) / "tb")
        return tb, run_id
    wb = WandbLogger(cfg.wandb_project_name, run_id, cfg.model_dump(mode="json"))
    return _MultiLogger([tb, wb]), wb.run_id


# ------------------------------------------------------------------ signals


def install_signal_handlers(stop_flag: threading.Event) -> None:
    """SIGTERM / first SIGINT request a graceful stop; a second SIGINT interrupts."""

    def on_term(signum, frame):
        log.warning("SIGTERM: stopping after the current batch")
        stop_flag.set()

    def on_int(signum, frame):
        log.warning("SIGINT: stopping after the current batch (press Ctrl-C again to abort)")
        stop_flag.set()
        signal.signal(signal.SIGINT, signal.default_int_handler)

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_int)


# ------------------------------------------------------------------ loop


def _close_loader(loader, it=None) -> None:
    """Stop DataLoader workers (persistent workers outlive an exhausted iterator)."""
    for i in (it, getattr(loader, "_iterator", None)):
        if i is not None and hasattr(i, "_shutdown_workers"):
            i._shutdown_workers()
    if getattr(loader, "_iterator", None) is not None:
        loader._iterator = None


def _train_batches(
    model, opt, sched, loader, st: TrainState, cfg, tokens, device, logger, backup, stop_flag,
    max_batches: int | None = None,
    on_batch: Callable[[list[str]], None] | None = None,
    on_save: Callable[[dict], None] | None = None,
) -> bool:
    """Train on `loader` (at most `max_batches`); True when the loader was exhausted."""
    model.train()
    autocast = cfg.bf16 and device.type == "cuda"

    def save_batch_ckpt() -> None:
        _check_finite_weights(model, st)
        ckpt = make_checkpoint(model, opt, sched, st, loader.sampler, cfg)
        st.sampler_state = ckpt["sampler"]
        save_and_backup(ckpt, "batch", cfg, backup)
        if on_save is not None:
            on_save(ckpt)

    it = iter(loader)
    done = 0
    try:
        for batch in it:
            if st.batch_idx_train % 10 == 0:
                set_batch_count(model, adjusted_batch_count(st.batch_idx_train, cfg.max_duration))
            st.batch_idx_train += 1
            feats, feat_lens, texts, cuts = batch_to_tensors(batch)
            feats, feat_lens = feats.to(device), feat_lens.to(device)
            targets, target_lengths = encode_targets(texts, tokens, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast):
                loss, num_frames = compute_ctc_loss(model, feats, feat_lens, targets, target_lengths)
            finite = bool(torch.isfinite(loss))
            if finite:
                loss.backward()
                st.nonfinite_streak = 0
            else:
                st.nonfinite_streak += 1
                log.warning("non-finite loss at batch %d (%d in a row); skipping step (cuts: %s)",
                            st.batch_idx_train, st.nonfinite_streak, [c.id for c in cuts])
            sched.step_batch(st.batch_idx_train)
            if finite:
                opt.step()
            opt.zero_grad()
            st.batches_consumed_in_epoch += 1
            done += 1
            if on_batch is not None:
                on_batch([c.id for c in cuts])
            if finite and st.batch_idx_train % cfg.log_every == 0:
                logger.log(
                    {
                        "train/loss": loss.item() / max(int(num_frames), 1),
                        "train/lr": opt.param_groups[0]["lr"],
                        "train/epoch": st.epoch,
                    },
                    st.batch_idx_train,
                )
            if st.nonfinite_streak >= MAX_NONFINITE_BATCHES:
                raise DivergedTraining(
                    f"{st.nonfinite_streak} consecutive non-finite losses at batch_idx_train="
                    f"{st.batch_idx_train}; not saving, resume from the last checkpoint"
                )
            saved = st.batch_idx_train % cfg.save_every_n_batches == 0
            if saved:
                save_batch_ckpt()
            if stop_flag is not None and stop_flag.is_set():
                if not saved:
                    save_batch_ckpt()
                if backup is not None and not backup.flush(cfg.shutdown_flush_seconds):
                    log.warning("Drive upload not finished within %ss", cfg.shutdown_flush_seconds)
                raise TrainingInterrupted(f"stopped at batch_idx_train={st.batch_idx_train}")
            if max_batches is not None and done >= max_batches:
                return False
        return True
    finally:
        _close_loader(loader, it)


def _check_finite_weights(model, st: TrainState) -> None:
    """Refuse to checkpoint NaN/inf weights (pruning would then drop the good ones)."""
    bad = [name for name, p in model.named_parameters() if not torch.isfinite(p).all()]
    if bad:
        raise DivergedTraining(f"non-finite weights at batch_idx_train={st.batch_idx_train}: {bad[:5]}")


def train_one_epoch(model, opt, sched, loader, st: TrainState, cfg, tokens, device, logger, backup, stop_flag) -> TrainState:
    _train_batches(model, opt, sched, loader, st, cfg, tokens, device, logger, backup, stop_flag)
    return st


# ------------------------------------------------------------------ dev eval / averaging


def _evaluated_epochs(out: Path) -> set[int]:
    path = out / DEV_METRICS
    if not path.exists():
        return set()
    epochs = set()
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            epochs.add(int(json.loads(line)["epoch"]))
        except (ValueError, KeyError, TypeError):
            log.warning("skipping unparsable line %d of %s: %.80r", n, path, line)
    return epochs


def _dev_eval(model, cfg, dev_cuts: CutSet, tokens, device, epoch: int, batch_idx_train: int, logger) -> dict:
    loader = eval_dataloader(cfg, dev_cuts)
    try:
        metrics = evaluate_split(model, loader, tokens, device, cfg.export_chunk_size, cfg.export_left_context, cfg.dev_max_cuts)
    finally:
        _close_loader(loader)
    row = {"epoch": epoch, "batch_idx_train": batch_idx_train, **metrics}
    path = Path(cfg.output_dir) / DEV_METRICS
    # a crash mid-append can leave a line without "\n"; never glue a new row onto it
    torn = path.exists() and path.stat().st_size > 0 and not path.read_bytes().endswith(b"\n")
    with open(path, "a", encoding="utf-8") as f:
        f.write(("\n" if torn else "") + json.dumps(row) + "\n")
    log.info("dev epoch %d: %s", epoch, metrics)
    logger.log({f"dev/{k}": v for k, v in metrics.items()}, batch_idx_train)
    return row


def _dev_eval_missing(cfg, dev_cuts: CutSet, tokens, device, before_epoch: int, logger) -> None:
    """Dev-evaluate finished epochs (< `before_epoch`) absent from dev_metrics.jsonl."""
    out = Path(cfg.output_dir)
    done = _evaluated_epochs(out)
    for epoch in range(1, before_epoch):
        if epoch in done:
            continue
        pt = out / ck.ckpt_name("epoch", 0, epoch)
        if ck.read_meta(pt) is None:
            log.info("no epoch-%d checkpoint left for its missing dev eval; skipping", epoch)
            continue
        ckpt = _load(pt)
        model = build_model(cfg, tokens.vocab_size)
        model.load_state_dict(ckpt["model"])
        _dev_eval(model.to(device), cfg, dev_cuts, tokens, device, epoch, int(ckpt["batch_idx_train"]), logger)


def _finish_epoch(model, opt, sched, st: TrainState, cfg, backup, dev_cuts, tokens, device, logger) -> None:
    """Save the epoch checkpoint, dev-evaluate, advance `st` to the next epoch."""
    _check_finite_weights(model, st)
    save_and_backup(make_checkpoint(model, opt, sched, st, None, cfg), "epoch", cfg, backup)
    _dev_eval(model, cfg, dev_cuts, tokens, device, st.epoch, st.batch_idx_train, logger)
    st.epoch += 1
    st.batches_consumed_in_epoch = 0
    st.sampler_state = None


def average_last_epochs(output_dir: Path, k: int) -> Path:
    """`output_dir/averaged-{k}.pt` from the newest `k` complete `epoch-*.pt`."""
    out = Path(output_dir)
    epochs = [pt for pt, meta in ck.list_complete(out) if meta.kind == "epoch"][-k:]
    if not epochs:
        raise FileNotFoundError(f"no complete epoch checkpoints in {out}")
    avg = average_checkpoints(epochs, device=torch.device("cpu"))
    config = _load(epochs[-1]).get("config")
    path = out / f"averaged-{k}.pt"
    tmp = path.with_name(path.name + ".tmp")
    torch.save({"model": avg, "config": config, "averaged_from": [p.name for p in epochs]}, tmp)
    os.replace(tmp, path)
    log.info("averaged %s -> %s", [p.name for p in epochs], path)
    return path


# ------------------------------------------------------------------ drivers


def _setup(cfg: ZipformerTrainConfig, data_dir: Path, tokens: PhonemeTokens, device):
    """Shared start: config.yml, backup, resume source, frame budget, model, restore."""
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(out / "config.yml")
    backup = make_backup(cfg)
    resume_pt = resolve_resume(cfg, backup)
    train_cuts, dev_cuts = apply_frame_budget(cfg, data_dir)
    model = build_model(cfg, tokens.vocab_size).to(device)
    opt, sched = build_optimizer(model, cfg)
    st = TrainState()
    if resume_pt is not None:
        st = restore(model, opt, sched, _load(resume_pt), cfg=cfg)
    return backup, train_cuts, dev_cuts, model, opt, sched, st


def run_training(cfg: ZipformerTrainConfig, data_dir: Path, tokens: PhonemeTokens, device) -> Path:
    """Train (resuming if possible) for `cfg.num_epochs`; return the averaged checkpoint."""
    backup, train_cuts, dev_cuts, model, opt, sched, st = _setup(cfg, data_dir, tokens, device)
    out = Path(cfg.output_dir)
    logger, st.wandb_run_id = make_logger(cfg, st.wandb_run_id)
    try:
        _dev_eval_missing(cfg, dev_cuts, tokens, device, min(st.epoch, cfg.num_epochs + 1), logger)
        if st.epoch <= cfg.num_epochs:
            stop_flag = threading.Event()
            install_signal_handlers(stop_flag)
            while st.epoch <= cfg.num_epochs:
                sched.step_epoch(st.epoch - 1)
                loader = train_dataloader(
                    cfg, data_dir, st.epoch, st.sampler_state, st.batches_consumed_in_epoch, cuts=train_cuts
                )
                st = train_one_epoch(model, opt, sched, loader, st, cfg, tokens, device, logger, backup, stop_flag)
                del loader
                _finish_epoch(model, opt, sched, st, cfg, backup, dev_cuts, tokens, device, logger)
                if stop_flag.is_set():  # e.g. SIGTERM during dev eval; epoch ckpt already saved
                    if backup is not None and not backup.flush(cfg.shutdown_flush_seconds):
                        log.warning("Drive upload not finished within %ss", cfg.shutdown_flush_seconds)
                    raise TrainingInterrupted(f"stopped after epoch {st.epoch - 1}")
        else:
            log.info("all %d epochs already complete; averaging only", cfg.num_epochs)
    finally:
        logger.close()
    if backup is not None and not backup.flush(cfg.upload_timeout_seconds):
        log.warning("Drive upload not finished within %ss (last error: %s)", cfg.upload_timeout_seconds, backup.last_error)
    return average_last_epochs(out, cfg.avg_last_k)


def _jsonable(obj):
    return json.loads(json.dumps(obj, default=str, sort_keys=True))


def _lrs(opt) -> list[float]:
    return [g["lr"] for g in opt.param_groups]


def run_steps_for_test(cfg: ZipformerTrainConfig, data_dir: Path, tokens: PhonemeTokens, n_batches: int) -> dict:
    """Resume, train exactly `n_batches` more batches on CPU, flush the backup.

    Returns the state as of the last save (`batch_idx_train`, `lrs`,
    `optimizer_sha256`, JSON-normalised `sampler_state`) plus `cut_ids_per_batch`,
    `start_batch_idx_train` and `first_lrs` (lrs before the first step).
    No signal handlers, no W&B / TensorBoard.
    """
    device = torch.device("cpu")
    backup, train_cuts, dev_cuts, model, opt, sched, st = _setup(cfg, data_dir, tokens, device)
    snap = {
        "batch_idx_train": st.batch_idx_train,
        "lrs": _lrs(opt),
        "optimizer_sha256": ck._optimizer_sha256(opt.state_dict()),
        "sampler_state": _jsonable(st.sampler_state),
    }

    def on_save(ckpt: dict) -> None:
        snap.update(
            batch_idx_train=ckpt["batch_idx_train"],
            lrs=[g["lr"] for g in ckpt["optimizer"]["param_groups"]],
            optimizer_sha256=ck._optimizer_sha256(ckpt["optimizer"]),
            sampler_state=_jsonable(ckpt["sampler"]),
        )

    start, first_lrs, cut_ids = st.batch_idx_train, None, []
    logger = NullLogger()
    while len(cut_ids) < n_batches and st.epoch <= cfg.num_epochs:
        sched.step_epoch(st.epoch - 1)
        if first_lrs is None:
            first_lrs = _lrs(opt)
        loader = train_dataloader(cfg, data_dir, st.epoch, st.sampler_state, st.batches_consumed_in_epoch, cuts=train_cuts)
        finished = _train_batches(
            model, opt, sched, loader, st, cfg, tokens, device, logger, backup, None,
            max_batches=n_batches - len(cut_ids), on_batch=cut_ids.append, on_save=on_save,
        )
        del loader
        if finished:
            _finish_epoch(model, opt, sched, st, cfg, backup, dev_cuts, tokens, device, logger)
    if backup is not None:
        backup.flush(cfg.upload_timeout_seconds)
    return {
        **snap,
        "cut_ids_per_batch": cut_ids,
        "start_batch_idx_train": start,
        "first_lrs": first_lrs if first_lrs is not None else _lrs(opt),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m prepare_quran_dataset.zipformer.train_loop")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None, help="default: config data_dir")
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--device", default=None, help="default: cuda if available else cpu")
    parser.add_argument("--test-resume", action="store_true", help="run --n-batches and print a JSON summary")
    parser.add_argument("--n-batches", type=int, default=3)
    args = parser.parse_args(argv)
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = ZipformerTrainConfig.from_yaml(args.config)
    data_dir = args.data_dir or Path(cfg.data_dir)
    tokens = PhonemeTokens.from_vocab_json(args.vocab)
    if args.test_resume:
        result = run_steps_for_test(cfg, data_dir, tokens, args.n_batches)
        sys.stdout.write(json.dumps(result, default=str) + "\n")
        sys.stdout.flush()
        return 0
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    try:
        path = run_training(cfg, data_dir, tokens, device)
    except TrainingInterrupted as exc:
        log.warning("training interrupted: %s", exc)
        return 130
    log.info("done: %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

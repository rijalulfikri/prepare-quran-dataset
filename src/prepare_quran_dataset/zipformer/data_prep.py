"""Colab data preparation: HF datasets -> 16 kHz FLAC + Lhotse manifests on Drive.

Layout under `drive_root` (all manifest audio paths are relative to it):

    audio/moshaf_<id>/<segment_index>.flac, audio/noise/<i:06d>.flac
    manifests/partial/speech_moshaf_<id>.jsonl.gz (+ .done)
    manifests/partial/noise_{aug,input}.jsonl.gz (+ noise.done)
    cuts_{train,dev,test,noise_aug}.jsonl.gz, tokens.txt, data_manifest.json

Every step is idempotent so a dead Colab session can simply be re-run: a config
with a `.done` marker is skipped, FLACs already on disk are reused, files are
written via temp-file + rename, and the merge always rebuilds from partials.
"""

import io
import json
import logging
import os
import random
import shutil
from collections.abc import Iterable
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from lhotse import CutSet, MonoCut, Recording, SupervisionSegment
from quran_transcript import MoshafAttributes, quran_phonetizer
from tqdm import tqdm

from prepare_quran_dataset.zipformer import frame_budget
from prepare_quran_dataset.zipformer.config import ZipformerTrainConfig
from prepare_quran_dataset.zipformer.moshaf_ways import (
    build_special_moshaf_attrs,
    resolve_moshaf_attr,
)
from prepare_quran_dataset.zipformer.tokens import PhonemeTokens

logger = logging.getLogger(__name__)

SPEECH_DATASET = "obadx/muaalem-annotated-v3"
NOISE_DATASET = "obadx/freesound-commercial-50k-noise-only"
METADATA_DATASET = "obadx/mualem-recitations-annotated"
METADATA_CONFIG = "moshaf_metadata"
SAMPLE_RATE = 16000
VOCAB_PATH = Path(__file__).resolve().parents[3] / "vocab_streaming" / "vocab.json"
NOISE_CONFIG = "noise"


# ----------------------------------------------------------------- HF access
def load_hf_rows(name: str, config: str | None, token: str | None) -> Iterable[dict]:
    """Stream the `train` split of an HF dataset with undecoded audio bytes."""
    from datasets import Audio, load_dataset

    ds = load_dataset(name, name=config, split="train", token=token, streaming=True)
    return ds.cast_column("audio", Audio(decode=False))


def load_moshaf_metadata(token: str | None) -> dict[str, dict]:
    """Moshaf id -> `MoshafAttributes` kwargs (rows of the metadata config)."""
    from datasets import load_dataset

    ds = load_dataset(METADATA_DATASET, name=METADATA_CONFIG, split="train", token=token)
    return {str(row["id"]): dict(row) for row in ds}


def cleanup_hf_cache(name: str, config: str | None) -> None:
    """Best-effort removal of a dataset config's HF cache dir (Colab disk is small)."""
    import datasets.config

    cache = Path(datasets.config.HF_DATASETS_CACHE) / name.replace("/", "___")
    shutil.rmtree(cache / config if config else cache, ignore_errors=True)


# ----------------------------------------------------------------- audio / cuts
def _decode(audio_bytes: bytes) -> np.ndarray:
    wav, _ = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE, mono=True)
    return wav


def _write_wav(wav: np.ndarray, out_path: Path) -> float:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(f".tmp_{out_path.name}")
    sf.write(tmp, wav, SAMPLE_RATE, format="FLAC")
    os.replace(tmp, out_path)
    return len(wav) / SAMPLE_RATE


def write_flac(audio_bytes: bytes, out_path: Path) -> float:
    """Decode to 16 kHz mono, write FLAC, return the duration in seconds."""
    return _write_wav(_decode(audio_bytes), Path(out_path))


def make_cut(
    rec_rel_path: str,
    duration: float,
    text: str,
    cut_id: str,
    custom: dict,
    data_root: Path,
) -> MonoCut:
    """Cut over `data_root/rec_rel_path` whose stored source path is relative."""
    recording = Recording.from_file(Path(data_root) / rec_rel_path, recording_id=cut_id)
    recording.sources[0].source = rec_rel_path
    supervision = SupervisionSegment(
        id=cut_id,
        recording_id=cut_id,
        start=0.0,
        duration=duration,
        text=text,
        custom=custom,
    )
    return MonoCut(
        id=cut_id,
        start=0.0,
        duration=duration,
        channel=0,
        recording=recording,
        supervisions=[supervision],
    )


def _write_cuts(cuts: list[MonoCut], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".tmp_{path.name}")
    CutSet.from_cuts(cuts).to_file(tmp)
    os.replace(tmp, path)


def _partial_dir(drive_root: Path) -> Path:
    return Path(drive_root) / "manifests" / "partial"


def _done_marker(drive_root: Path, config_name: str) -> Path:
    """`.done` marker of a config name (`moshaf_<id>` or `noise`)."""
    stem = config_name if config_name == NOISE_CONFIG else f"speech_{config_name}"
    return _partial_dir(drive_root) / f"{stem}.done"


# ----------------------------------------------------------------- speech
def process_speech_config(
    moshaf_id: str,
    drive_root: Path,
    cfg: ZipformerTrainConfig,
    base_attrs: dict[str, MoshafAttributes],
    special_attrs: dict[str, dict[str, MoshafAttributes]],
    tokens: PhonemeTokens,
    hf_token: str | None,
) -> Path:
    """Write one moshaf's FLACs + partial manifest; skip if already done."""
    drive_root = Path(drive_root)
    config = f"moshaf_{moshaf_id}"
    manifest = _partial_dir(drive_root) / f"speech_{config}.jsonl.gz"
    done = _done_marker(drive_root, config)
    if done.exists():
        logger.info("%s already done, skipping", config)
        return manifest

    cuts, num_too_long = [], 0
    for row in tqdm(load_hf_rows(SPEECH_DATASET, config, hf_token), desc=config):
        row_moshaf, seg = str(row["moshaf_id"]), str(row["segment_index"])
        attr = resolve_moshaf_attr(row_moshaf, seg, base_attrs, special_attrs)
        text = quran_phonetizer(row["uthmani"], attr, remove_spaces=True).phonemes
        tokens.encode(text)  # KeyError on an unknown phoneme: never drop silently

        rel = f"audio/{config}/{seg}.flac"
        out = drive_root / rel
        if out.exists():
            duration = sf.info(out).duration
            wav = None
        else:
            wav = _decode(row["audio"]["bytes"])
            duration = len(wav) / SAMPLE_RATE
        if cfg.max_audio_seconds is not None and duration > cfg.max_audio_seconds:
            num_too_long += 1
            continue
        if wav is not None:
            _write_wav(wav, out)

        custom = {
            "moshaf_id": row_moshaf,
            "segment_index": seg,
            "input_type": "speech",
            "num_labels": len(text),
            "num_repeats": frame_budget.count_repeats(text),
        }
        cuts.append(make_cut(rel, duration, text, f"{config}__{seg}", custom, drive_root))

    if num_too_long:
        logger.info("%s: dropped %d cuts > %s s", config, num_too_long, cfg.max_audio_seconds)
    _write_cuts(cuts, manifest)
    done.touch()
    cleanup_hf_cache(SPEECH_DATASET, config)
    return manifest


# ----------------------------------------------------------------- noise
def process_noise(drive_root: Path, cfg: ZipformerTrainConfig, hf_token: str | None) -> Path:
    """Write noise FLACs (clipped/padded) and the aug/input partial manifests."""
    drive_root = Path(drive_root)
    partial = _partial_dir(drive_root)
    done = _done_marker(drive_root, NOISE_CONFIG)
    if done.exists():
        logger.info("noise already done, skipping")
        return partial

    max_samples = int(cfg.max_noise_input_seconds * SAMPLE_RATE)
    min_samples = int(cfg.min_noise_input_seconds * SAMPLE_RATE)
    cuts = []
    for i, row in enumerate(tqdm(load_hf_rows(NOISE_DATASET, None, hf_token), desc="noise")):
        rel = f"audio/noise/{i:06d}.flac"
        out = drive_root / rel
        if out.exists():
            duration = sf.info(out).duration
        else:
            wav = _decode(row["audio"]["bytes"])[:max_samples]
            if len(wav) < min_samples:
                wav = np.pad(wav, (0, min_samples - len(wav)))
            duration = _write_wav(wav, out)
        custom = {"input_type": "silence"}
        cuts.append(make_cut(rel, duration, "", f"noise__{i:06d}", custom, drive_root))

    perm = np.random.default_rng(cfg.seed).permutation(len(cuts))
    aug_idx = set(perm[: round(cfg.noise_aug_ratio * len(cuts))].tolist())
    _write_cuts([c for i, c in enumerate(cuts) if i in aug_idx], partial / "noise_aug.jsonl.gz")
    _write_cuts(
        [c for i, c in enumerate(cuts) if i not in aug_idx], partial / "noise_input.jsonl.gz"
    )
    done.touch()
    cleanup_hf_cache(NOISE_DATASET, None)
    return partial


# ----------------------------------------------------------------- merge
def _all_configs(cfg: ZipformerTrainConfig) -> list[str]:
    speech = [f"moshaf_{m}" for m in [*cfg.train_moshaf_ids, *cfg.test_moshaf_ids]]
    return [*speech, NOISE_CONFIG]


def _load_partial(drive_root: Path, config_name: str, file_name: str) -> list[MonoCut]:
    done = _done_marker(drive_root, config_name)
    if not done.exists():
        raise FileNotFoundError(f"partial {config_name} is not finished (missing {done})")
    # lhotse reads an empty manifest back as None
    return list(CutSet.from_file(_partial_dir(drive_root) / file_name) or [])


def merge_manifests(drive_root: Path, cfg: ZipformerTrainConfig, vocab_path: Path = VOCAB_PATH) -> dict:
    """Rebuild the final split manifests, tokens.txt and data_manifest.json from partials."""
    drive_root = Path(drive_root)

    def speech(ids: list[str]) -> list[MonoCut]:
        return [
            c
            for m in ids
            for c in _load_partial(drive_root, f"moshaf_{m}", f"speech_moshaf_{m}.jsonl.gz")
        ]

    train_speech = speech(cfg.train_moshaf_ids)
    test = speech(cfg.test_moshaf_ids)
    noise_input = _load_partial(drive_root, NOISE_CONFIG, "noise_input.jsonl.gz")
    noise_aug = _load_partial(drive_root, NOISE_CONFIG, "noise_aug.jsonl.gz")

    pool = train_speech + noise_input
    random.Random(cfg.seed).shuffle(pool)
    num_dev = round(cfg.devset_ratio * len(pool))
    splits = {"train": pool[num_dev:], "dev": pool[:num_dev], "test": test, "noise_aug": noise_aug}
    for split, cuts in splits.items():
        _write_cuts(cuts, drive_root / f"cuts_{split}.jsonl.gz")

    tokens = PhonemeTokens.from_vocab_json(vocab_path)
    tokens.write_tokens_txt(drive_root / "tokens.txt")

    summary = {
        "hours": {s: round(sum(c.duration for c in cuts) / 3600, 4) for s, cuts in splits.items()},
        "num_cuts": {s: len(cuts) for s, cuts in splits.items()},
        "vocab_size": tokens.vocab_size,
        "configs": _all_configs(cfg),
        # train_speech covers the speech cuts of both train and dev
        "frame_budget": frame_budget.report(train_speech + test),
    }
    (drive_root / "data_manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


# ----------------------------------------------------------------- orchestration
def check_free_space(path: Path, min_gb: float = 70) -> None:
    free_gb = shutil.disk_usage(path).free / 1024**3
    if free_gb < min_gb:
        raise RuntimeError(f"only {free_gb:.1f} GB free at {path}, need >= {min_gb} GB")


def run(
    cfg_path: str | Path,
    drive_root: str | Path,
    hf_token: str | None,
    only: list[str] | None = None,
    skip_space_check: bool = False,
) -> dict | None:
    """Space check -> each train+test moshaf -> noise -> merge.

    `only` restricts processing to the given config names (`moshaf_<id>`,
    `noise`). The merge runs only once every config is done; otherwise the
    pending configs are logged and `None` is returned.
    """
    cfg = ZipformerTrainConfig.from_yaml(cfg_path)
    drive_root = Path(drive_root)
    drive_root.mkdir(parents=True, exist_ok=True)
    if not skip_space_check:
        check_free_space(drive_root)

    all_configs = _all_configs(cfg)
    selected = all_configs if not only else list(only)
    unknown = sorted(set(selected) - set(all_configs))
    if unknown:
        raise ValueError(f"unknown configs {unknown}; choose from {all_configs}")

    speech_ids = [
        m
        for m in [*cfg.train_moshaf_ids, *cfg.test_moshaf_ids]
        if f"moshaf_{m}" in selected and not _done_marker(drive_root, f"moshaf_{m}").exists()
    ]
    if speech_ids:
        metadata = load_moshaf_metadata(hf_token)
        base_attrs = {m: MoshafAttributes(**metadata[m]) for m in speech_ids}
        special_attrs = build_special_moshaf_attrs(metadata)
        tokens = PhonemeTokens.from_vocab_json(VOCAB_PATH)
        for m in speech_ids:
            process_speech_config(m, drive_root, cfg, base_attrs, special_attrs, tokens, hf_token)
    if NOISE_CONFIG in selected:
        process_noise(drive_root, cfg, hf_token)

    pending = [c for c in all_configs if not _done_marker(drive_root, c).exists()]
    if pending:
        logger.warning("not merging: configs still pending %s", pending)
        return None
    return merge_manifests(drive_root, cfg)

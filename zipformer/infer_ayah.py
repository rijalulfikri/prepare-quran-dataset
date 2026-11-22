"""Transcribe one ayah recording to phonemes with a Zipformer2-CTC checkpoint.

Greedy CTC decoding with simulated streaming (the same path as the dev / test
evaluation), so the PER printed here is comparable to dev_metrics.jsonl.

Needs torch, k2, lhotse, quran-transcript and levenshtein (the training venv,
or e.g. a conda env with icefall's requirements) and an icefall checkout whose
librispeech zipformer recipe matches the pinned submodule:

    python infer_ayah.py --checkpoint results-zipformer-v0/epoch-18.pt \\
        --audio data-zipformer/audio/moshaf_19.0/001.0002.flac --sura 1 --aya 2 \\
        --icefall-root /path/to/icefall   # or set ICEFALL_ROOT; default: external/icefall

With `--sura/--aya` the reference phonemes come from `quran_phonetizer` (Hafs,
madd lengths from the options below) and the script prints the PER. Reciters
differ (e.g. moshaf 0.0 reads madd 'arid as 2, the default reference uses 4),
so for a dataset recording pass `--moshaf <id>`: the reference then uses that
moshaf's attributes from the dataset metadata, exactly like the training
labels (needs network on first use and HUGGINGFACE_TOKEN, read from .env). Note that a
dataset file name like `002.0255.flac` is `<surah>.<segment>`, not always an
ayah number; for such files pass the cut's label with `--reference`. Works with
`epoch-*.pt`, `checkpoint-*.pt` and `averaged-*.pt` (all store `model` and
`config`).

A streaming `.onnx` / `.ort` checkpoint (this repo's export, or utluma_app's
utluma3) is decoded with sherpa-onnx instead, fed like the app's MesinSherpa;
it needs only sherpa-onnx, lhotse and the reference tools, not icefall:

    python infer_ayah.py --audio ayah.flac --sura 1 --aya 7 \\
        --checkpoint .../utluma_cek_tilawah/model/zipformer_p_arabic_v3_c16_utluma3.int8.ort \\
        --tokens .../utluma_cek_tilawah/assets/model/tokens.txt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

ZIPFORMER_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = ZIPFORMER_DIR / "configs" / "train_config.yml"
# prepare_quran_dataset (moshaf_ways) lives in the repo's src/
sys.path.insert(0, str(ZIPFORMER_DIR.parent / "src"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True,
                   help="epoch-N.pt / checkpoint-N.pt / averaged-K.pt, or a streaming .onnx / .ort "
                        "(e.g. utluma_app's utluma3), decoded with sherpa-onnx")
    p.add_argument("--tokens", help="tokens.txt for an .onnx/.ort checkpoint (default: next to the model)")
    p.add_argument("--audio", required=True, help="audio file (any rate/channels; resampled to 16 kHz mono)")
    p.add_argument("--sura", type=int, help="surah number, for the reference phonemes and PER")
    p.add_argument("--aya", type=int, help="ayah number, for the reference phonemes and PER")
    p.add_argument("--reference", help="reference phonemes for the PER, instead of --sura/--aya "
                                       "(e.g. a cut's supervision text from cuts_test.jsonl.gz)")
    p.add_argument("--chunk-size", type=int, help="streaming chunk in frames (default: export_chunk_size)")
    p.add_argument("--left-context", type=int, help="left context in frames (default: export_left_context)")
    p.add_argument("--moshaf", help="build the reference with this moshaf's recitation attributes (e.g. 0.0), "
                                    "exactly like the training labels; overrides the --madd-* options")
    p.add_argument("--segment", help="segment index for --moshaf's per-segment overrides "
                                     "(default: the audio file name, e.g. 114.0002)")
    p.add_argument("--madd-monfasel", type=int, default=4, help="reference madd munfasil length (default: 4)")
    p.add_argument("--madd-mottasel", type=int, default=4, help="reference madd muttasil length (default: 4)")
    p.add_argument("--madd-aared", type=int, default=4, help="reference madd 'arid lissukun length (default: 4)")
    p.add_argument("--times", action="store_true", help="also print each phoneme's start time")
    p.add_argument("--device", default=None, help="default: cuda if available else cpu")
    p.add_argument("--icefall-root", default=os.environ.get("ICEFALL_ROOT"),
                   help="icefall checkout (default: $ICEFALL_ROOT, else the external/icefall submodule)")
    return p.parse_args(argv)


def load_model(checkpoint: Path, device):
    """Model + config from a training checkpoint (falls back to the repo YAML)."""
    from quran_zipformer.config import ZipformerTrainConfig
    from quran_zipformer.model import build_model
    from quran_zipformer.tokens import PhonemeTokens
    from quran_zipformer.train_loop import DEFAULT_VOCAB

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("config")
    cfg = ZipformerTrainConfig(**cfg_dict) if cfg_dict else ZipformerTrainConfig.from_yaml(DEFAULT_CONFIG)
    tokens = PhonemeTokens.from_vocab_json(DEFAULT_VOCAB)
    model = build_model(cfg, tokens.vocab_size)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), cfg, tokens


def fbank(audio: Path, num_mel_bins: int) -> torch.Tensor:
    """`(1, T, num_mel_bins)` fbank of 16 kHz mono audio, as in training."""
    from lhotse import Fbank, FbankConfig, Recording

    rec = Recording.from_file(str(audio))
    if rec.sampling_rate != 16000:
        rec = rec.resample(16000)
    samples = rec.load_audio().mean(axis=0, keepdims=True)  # downmix to mono
    feats = Fbank(FbankConfig(num_mel_bins=num_mel_bins)).extract(samples, 16000)
    return torch.from_numpy(feats).unsqueeze(0)


MOSHAF_REPO = "obadx/mualem-recitations-annotated"  # data_prep.METADATA_DATASET
MOSHAF_FILE = "moshaf_pool.parquet"  # data file of its "moshaf_metadata" config


def _hf_token() -> str | None:
    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if token:
        return token
    from dotenv import dotenv_values

    return dotenv_values(ZIPFORMER_DIR / ".env").get("HUGGINGFACE_TOKEN")


def moshaf_attributes(moshaf_id: str, segment: str | None):
    """`MoshafAttributes` the training labels used for this moshaf (and segment).

    Reads the same metadata rows as `data_prep.load_moshaf_metadata` (straight
    from the parquet file, cached by huggingface_hub, so the `datasets` package
    is not needed) and applies the same per-segment overrides.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from quran_transcript import MoshafAttributes

    from prepare_quran_dataset.moshaf_ways import build_special_moshaf_attrs, resolve_moshaf_attr

    path = hf_hub_download(MOSHAF_REPO, MOSHAF_FILE, repo_type="dataset", token=_hf_token())
    metadata = {str(row["id"]): row for row in pq.read_table(path).to_pylist()}
    if moshaf_id not in metadata:
        raise SystemExit(f"moshaf {moshaf_id} not in {MOSHAF_REPO}; known: {', '.join(sorted(metadata))}")
    base = {moshaf_id: MoshafAttributes(**metadata[moshaf_id])}
    return resolve_moshaf_attr(moshaf_id, segment or "", base, build_special_moshaf_attrs(metadata))


def reference_phonemes(sura: int, aya: int, args: argparse.Namespace) -> tuple[str, str]:
    """`(uthmani, phonemes)` of an ayah, phonetized like the training labels."""
    from quran_transcript import Aya, MoshafAttributes, quran_phonetizer

    uthmani = Aya(sura, aya).get().uthmani
    if getattr(args, "moshaf", None):
        segment = args.segment or Path(args.audio).stem
        attr = moshaf_attributes(args.moshaf, segment)
        print(f"moshaf     : {args.moshaf} segment {segment} (madd monfasel {attr.madd_monfasel_len}, "
              f"mottasel {attr.madd_mottasel_len}, aared {attr.madd_aared_len})")
    else:
        attr = MoshafAttributes(
            rewaya="hafs",
            madd_monfasel_len=args.madd_monfasel,
            madd_mottasel_len=args.madd_mottasel,
            madd_mottasel_waqf=args.madd_mottasel,
            madd_aared_len=args.madd_aared,
        )
    return uthmani, quran_phonetizer(uthmani, attr, remove_spaces=True).phonemes


@torch.no_grad()
def transcribe(model, cfg, tokens, feats: torch.Tensor, chunk_size: int, left_context: int, device):
    """`(phonemes, [(phoneme, start_seconds), ...])` by greedy CTC."""
    from quran_zipformer.decode import (
        frame_shift_seconds,
        greedy_ctc_with_frames,
        set_streaming_context,
        token_times,
    )

    set_streaming_context(model, chunk_size, left_context)
    feats = feats.to(device)
    feat_lens = torch.tensor([feats.shape[1]], device=device)
    enc_out, enc_lens = model.forward_encoder(feats, feat_lens)
    log_probs = model.ctc_output(enc_out)
    frames = greedy_ctc_with_frames(log_probs, enc_lens, tokens.blank_id)[0]
    ids = [tok for tok, _ in frames]
    times = token_times(frames, frame_shift_seconds(cfg.output_downsampling_factor))
    return tokens.decode(ids), [(tokens.decode([tok]), t) for tok, t in times]


def load_wav_16k(audio: Path):
    """Float32 mono 16 kHz samples of any audio file."""
    from lhotse import Recording

    rec = Recording.from_file(str(audio))
    if rec.sampling_rate != 16000:
        rec = rec.resample(16000)
    return rec.load_audio().mean(axis=0)


def transcribe_sherpa(model: Path, tokens: Path, wav, num_threads: int = 2):
    """`(phonemes, [(unit, start_seconds), ...])` from a sherpa-onnx zipformer2 CTC model.

    Fed like utluma_app's MesinSherpa: 100 ms blocks, then 660 ms of zeros and
    `input_finished`, greedy, no endpointing.
    """
    import numpy as np
    import sherpa_onnx

    rec = sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(
        tokens=str(tokens), model=str(model), num_threads=num_threads,
        sample_rate=16000, feature_dim=80, decoding_method="greedy_search",
        enable_endpoint_detection=False,
    )
    stream = rec.create_stream()
    for i in range(0, len(wav), 1600):
        stream.accept_waveform(16000, wav[i : i + 1600])
        while rec.is_ready(stream):
            rec.decode_stream(stream)
    stream.accept_waveform(16000, np.zeros(int(0.66 * 16000), np.float32))
    stream.input_finished()
    while rec.is_ready(stream):
        rec.decode_stream(stream)
    result = rec.get_result_all(stream)
    return result.text, list(zip(result.tokens, result.timestamps))


SHERPA_FRAME_SHIFT_MS = 40  # sherpa-onnx's online CTC timestamps assume 40 ms output frames


def sherpa_time_scale(model: Path) -> float:
    """Factor that turns sherpa-onnx's timestamps of `model` into seconds.

    sherpa-onnx counts 40 ms per output frame (as zipformer_p-arabic-v3); this
    repo's export may emit 20 ms frames (`output_downsampling_factor: 1`) and
    says so in its `output_frame_shift_ms` metadata, so its timestamps come out
    twice too large. 1.0 for models without that metadata.
    """
    import onnxruntime as ort

    meta = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"]).get_modelmeta().custom_metadata_map
    shift = meta.get("output_frame_shift_ms")
    return int(shift) / SHERPA_FRAME_SHIFT_MS if shift else 1.0


def phoneme_error_rate(hyp: str, ref: str) -> float:
    """Character-level PER, capped at the reference length like `decode.per`.

    One phoneme is one character in quran-transcript's script, so this equals
    `decode.per` for this repo's vocab and also works for models with
    multi-character units (utluma3).
    """
    import Levenshtein

    return min(Levenshtein.distance(hyp, ref), len(ref)) / len(ref) if ref else 0.0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoint = Path(args.checkpoint)
    if checkpoint.suffix in (".onnx", ".ort"):
        tokens_txt = Path(args.tokens) if args.tokens else checkpoint.parent / "tokens.txt"
        wav = load_wav_16k(Path(args.audio))
        hyp, times = transcribe_sherpa(checkpoint, tokens_txt, wav)
        if args.times:
            scale = sherpa_time_scale(checkpoint)
            times = [(ph, t * scale) for ph, t in times]
        duration = len(wav) / 16000
        mode = f"sherpa-onnx (chunking baked into the model), tokens {tokens_txt}"
    else:
        if args.icefall_root:
            os.environ["ICEFALL_ROOT"] = args.icefall_root  # read by quran_zipformer.icefall_import
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model, cfg, tokens = load_model(checkpoint, device)
        chunk = args.chunk_size or cfg.export_chunk_size
        left = args.left_context if args.left_context is not None else cfg.export_left_context
        feats = fbank(Path(args.audio), cfg.feature_dim)
        hyp, times = transcribe(model, cfg, tokens, feats, chunk, left, device)
        duration = feats.shape[1] / 100
        mode = f"chunk={chunk} left_context={left} frames"

    print(f"checkpoint : {args.checkpoint}")
    print(f"audio      : {args.audio} ({duration:.2f} s)")
    print(f"streaming  : {mode}")
    print(f"hypothesis : {hyp}")
    ref = args.reference
    if ref is None and args.sura is not None and args.aya is not None:
        uthmani, ref = reference_phonemes(args.sura, args.aya, args)
        print(f"ayah       : {args.sura}:{args.aya} {uthmani}")
    if ref is not None:
        print(f"reference  : {ref}")
        print(f"PER        : {phoneme_error_rate(hyp, ref):.2%}")
    if args.times:
        for ph, t in times:
            print(f"  {t:6.2f}s  {ph}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

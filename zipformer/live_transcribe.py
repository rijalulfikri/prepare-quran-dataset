"""Live phoneme transcription from the microphone with a streaming Zipformer2-CTC ONNX.

Uses sherpa-onnx's OnlineRecognizer, the same engine as utluma_app's MesinSherpa,
so it runs any sherpa zipformer2 CTC model: this repo's export
(`export_streaming_onnx`, `results-zipformer-v0/export*/`) or the app's utluma3
`.ort` with its own tokens.txt. Both emit quran-transcript phonemes (utluma3 in
multi-character units), so the joined text and the PER are comparable. The
hypothesis updates as you recite; after a pause the line is finished and a new
one starts. Ctrl+C stops.

    python live_transcribe.py --model-dir results-zipformer-v0/export-epoch-18
    python live_transcribe.py --model-dir ... --sura 1 --aya 6   # PER per line
    python live_transcribe.py --model-dir ... --wav ayah.flac    # file, in real time
    python live_transcribe.py --model-dir ... --save-wav sesi.wav  # keep what the model heard
    python live_transcribe.py \\
        --model .../utluma_cek_tilawah/model/zipformer_p_arabic_v3_c16_utluma3.int8.ort \\
        --tokens .../utluma_cek_tilawah/assets/model/tokens.txt

Needs sherpa-onnx and sounddevice (mic) or soundfile (--wav); macOS asks for
microphone permission for the terminal app on first use.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
BLOCK_S = 0.1  # audio fed to the recognizer per step


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--model-dir", help="folder with model(.int8).onnx and tokens.txt (this repo's export)")
    src.add_argument("--model", help="any streaming zipformer2 CTC model file (.onnx or .ort), e.g. utluma3")
    p.add_argument("--tokens", help="tokens.txt for --model (default: next to the model)")
    p.add_argument("--fp32", action="store_true", help="with --model-dir: model.onnx instead of model.int8.onnx")
    p.add_argument("--num-threads", type=int, default=2)
    p.add_argument("--device", help="input device index or name (default: system mic); see --list-devices")
    p.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    p.add_argument("--wav", help="stream this audio file in real time instead of the mic")
    p.add_argument("--save-wav", help="also write the audio fed to the model (16 kHz mono) to this file, "
                   "e.g. to replay a mic session with --wav")
    p.add_argument("--sura", type=int, help="with --aya: print each finished line's PER against this ayah")
    p.add_argument("--aya", type=int)
    p.add_argument("--pause", type=float, default=1.2,
                   help="seconds of silence after speech that end a line (default: 1.2)")
    p.add_argument("--silence-db", type=float, default=-48.0,
                   help="highest background level in dBFS; speech is >= 10 dB above the learned "
                        "floor (default: -48; raise it in a noisy room, lower it for a quiet mic)")
    return p.parse_args(argv)


def model_files(args: argparse.Namespace) -> tuple[Path, Path]:
    """`(model, tokens)` from --model/--tokens, falling back to --model-dir."""
    if args.model:
        model = Path(args.model)
    else:
        model = Path(args.model_dir) / ("model.onnx" if args.fp32 else "model.int8.onnx")
    tokens = Path(args.tokens) if args.tokens else model.parent / "tokens.txt"
    for f in (model, tokens):
        if not f.is_file():
            raise SystemExit(f"not found: {f}")
    return model, tokens


def make_recognizer(args: argparse.Namespace):
    import sherpa_onnx

    model, tokens = model_files(args)
    print(f"model  {model}\ntokens {tokens}")
    return sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(
        tokens=str(tokens),
        model=str(model),
        num_threads=args.num_threads,
        sample_rate=SAMPLE_RATE,
        feature_dim=80,
        # sherpa's endpointing counts trailing CTC blanks as silence, and a held
        # madd (up to 6 harakat) is mostly blanks, so it would cut ayat in the
        # middle; lines are ended by the audio-energy PauseDetector instead
        enable_endpoint_detection=False,
    )


class PauseDetector:
    """Ends a line after `pause` seconds below the speech level, once speech was heard.

    The noise floor is learned from the quietest blocks, so it adapts to the
    room and the mic gain, but never rises above `max_floor_db` dBFS, so a
    recitation that starts without leading silence is not taken for noise; a
    block counts as speech when its RMS is `margin_db` above that floor.
    """

    def __init__(self, pause: float, max_floor_db: float = -48.0, margin_db: float = 10.0):
        self.pause_blocks = max(1, round(pause / BLOCK_S))
        self.margin = 10 ** (margin_db / 20)
        self.max_floor = 10 ** (max_floor_db / 20)
        self.floor = self.max_floor
        self.heard_speech = False
        self.quiet_blocks = 0

    def update(self, block: np.ndarray) -> bool:
        """Feed one block; True when a line (speech followed by a pause) has ended."""
        rms = float(np.sqrt(np.mean(np.square(block)))) + 1e-6
        if rms < self.floor:
            self.floor = rms
        elif rms < self.floor * self.margin:
            # track a slowly rising floor (fan, room noise), capped below speech
            self.floor = min(0.95 * self.floor + 0.05 * rms, self.max_floor)
        if rms >= self.floor * self.margin:
            self.heard_speech, self.quiet_blocks = True, 0
            return False
        self.quiet_blocks += 1
        if self.heard_speech and self.quiet_blocks >= self.pause_blocks:
            self.heard_speech, self.quiet_blocks = False, 0
            return True
        return False


def reference_scorer(args: argparse.Namespace):
    """`text -> 'PER x%'` against the ayah's phonemes, or None without --sura/--aya."""
    if args.sura is None or args.aya is None:
        return None
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from infer_ayah import parse_args as infer_args
    from infer_ayah import reference_phonemes
    import Levenshtein

    uthmani, ref = reference_phonemes(args.sura, args.aya, infer_args(["--checkpoint", "-", "--audio", "-"]))
    print(f"ayah {args.sura}:{args.aya}  {uthmani}\nreference  {ref}\n")

    def score(text: str) -> str:
        hyp = text.replace(" ", "")
        return f"PER {min(Levenshtein.distance(hyp, ref), len(ref)) / len(ref):.1%}"

    return score


def audio_blocks(args: argparse.Namespace):
    """Yield float32 mono 16 kHz blocks of BLOCK_S seconds from the mic or --wav."""
    n = int(SAMPLE_RATE * BLOCK_S)
    if args.wav:
        import soundfile as sf

        wav, sr = sf.read(args.wav, dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
        if sr != SAMPLE_RATE:
            import torch
            import torchaudio

            wav = torchaudio.functional.resample(torch.from_numpy(wav), sr, SAMPLE_RATE).numpy()
        wav = np.concatenate([wav, np.zeros(int(SAMPLE_RATE * (args.pause + 0.5)), np.float32)])
        for i in range(0, len(wav), n):
            time.sleep(BLOCK_S)  # real time, like a mic
            yield wav[i : i + n]
        return

    import sounddevice as sd

    device = int(args.device) if args.device and args.device.isdigit() else args.device
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=n, device=device) as s:
        while True:
            block, overflowed = s.read(n)
            if overflowed:
                print("\n[warning: audio input overflowed; decoding is too slow]", file=sys.stderr)
            yield block[:, 0]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return 0

    recognizer = make_recognizer(args)
    score = reference_scorer(args)
    stream = recognizer.create_stream()
    source = args.wav or "microphone"
    print(f"Listening ({source}); recite, pause {args.pause:g} s to end a line, Ctrl+C to stop.\n")

    pauses = PauseDetector(args.pause, args.silence_db)
    shown = ""
    line_no = 1

    def decode_and_show() -> None:
        nonlocal shown
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
        text = recognizer.get_result(stream)
        if text != shown:
            print(f"\r{line_no:3d}: {text}", end="", flush=True)
            shown = text

    def finish_line() -> None:
        nonlocal stream, shown, line_no
        # flush the last chunk: the model needs right-padding to emit its frames
        stream.accept_waveform(SAMPLE_RATE, np.zeros(int(0.8 * SAMPLE_RATE), np.float32))
        stream.input_finished()
        decode_and_show()
        if shown:
            print(f"   [{score(shown)}]" if score else "")
            line_no += 1
        stream = recognizer.create_stream()
        shown = ""

    saved = None
    if args.save_wav:
        import soundfile as sf

        # written block by block, so Ctrl+C still leaves a playable file
        saved = sf.SoundFile(args.save_wav, "w", samplerate=SAMPLE_RATE, channels=1, subtype="PCM_16")
    try:
        try:
            for block in audio_blocks(args):
                if saved is not None:
                    saved.write(block)
                stream.accept_waveform(SAMPLE_RATE, block)
                decode_and_show()
                if pauses.update(block):
                    finish_line()
        except KeyboardInterrupt:
            pass
        if shown:
            finish_line()
    finally:
        if saved is not None:
            saved.close()
            print(f"saved {args.save_wav}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

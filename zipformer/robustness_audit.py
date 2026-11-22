"""Robustness audit: a streaming phoneme model against a reference model.

Finds where a model is weaker than a model known to be robust (for v0 the
reference was zipformer_p-arabic v3.1), on the same sherpa-onnx runtime as
utluma_app. Every stage decodes the same audio with every model and writes one
JSON line per item to `<out>/<stage>.jsonl`:

  panel    a few ayat of every reciter under --audio-root (`<reciter>/SSSAAA.mp3`),
           PER against the phonetizer, tolerant of the optional madd lengths
  qdat     obadx/qdat_bench (phone recordings, mostly women), PER against its transcript
  battery  panel ayat that every model reads right, under one controlled change each:
           non-speech before / after, level, clipping, noise, reverb, channel, tempo, pitch
  stream   consecutive ayat of one reciter decoded one by one and as a single stream
           (continuous recitation, as the app hears it)
  noise    non-speech input: tokens emitted per minute

    python robustness_audit.py --out audit-v1 \\
        --model v1 results-zipformer-v1/export/model.int8.onnx results-zipformer-v1/export/tokens.txt \\
        --model v3.1 results-zipformer/zipformer_p_arabic_v3.1.int8.onnx results-zipformer/tokens.txt \\
        --audio-root /path/to/everyayah            # all stages, then the report
    python robustness_audit.py --out audit-v1 --report   # the report again, from the .jsonl files

The first --model is the one under test, the last the reference; the report
(`<out>/report.md`) marks every row where the tested model is more than one PER
point behind. Needs sherpa-onnx, lhotse, quran-transcript, Levenshtein; `sox` for
reverb / tempo / pitch / filters and `ffmpeg` for the codecs (conditions whose
tool is missing are skipped).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ZIPFORMER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ZIPFORMER_DIR))

SR = 16000
NAME_RE = re.compile(r"^(\d{3})(\d{3})$")
PROBE_AYAT = ("028032", "107002")  # read by every reciter of the panel when present
QDAT_GLOB = "~/.cache/huggingface/hub/datasets--obadx--qdat_bench/snapshots/*/data/*.parquet"
DEFAULT_ATTRS = dict(rewaya="hafs", madd_monfasel_len=4, madd_mottasel_len=4, madd_mottasel_waqf=4,
                     madd_aared_len=4, madd_alleen_len=4)
STREAMS = [(1, 1, 7), (36, 1, 30), (18, 1, 20), (2, 1, 20)]  # (sura, first ayah, count)
BEHIND = 0.01  # PER points behind the reference that the report marks
REFERENCE_FAILS = 0.10  # above this PER of the reference, a row says more about the test than about the model
MISMATCHED = 0.50  # above this PER of the reference, a reciter's recordings are taken not to match their ayah numbers
REPEAT_RUN = 6  # inserted phonemes in a row that are taken as words the reciter said again, not as a slip

MODELS: dict[str, tuple[str, str]] = {}  # name -> (model, tokens); set in every worker
_recognizers: dict = {}


def _init_worker(models: dict, tmp: str) -> None:
    MODELS.clear()
    MODELS.update(models)
    os.environ["AUDIT_TMP"] = tmp


def decode(name: str, wav: np.ndarray) -> str:
    """Greedy text of `wav`, fed like utluma_app's MesinSherpa (100 ms blocks, 0.66 s of zeros at the end)."""
    if name not in _recognizers:
        import sherpa_onnx

        model, tokens = MODELS[name]
        _recognizers[name] = sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(
            tokens=tokens, model=model, num_threads=1, sample_rate=SR, feature_dim=80,
            decoding_method="greedy_search", enable_endpoint_detection=False)
    rec = _recognizers[name]
    stream = rec.create_stream()
    wav = np.ascontiguousarray(wav, dtype=np.float32)
    for i in range(0, len(wav), 1600):
        stream.accept_waveform(SR, wav[i:i + 1600])
        while rec.is_ready(stream):
            rec.decode_stream(stream)
    stream.accept_waveform(SR, np.zeros(int(0.66 * SR), np.float32))
    stream.input_finished()
    while rec.is_ready(stream):
        rec.decode_stream(stream)
    return rec.get_result(stream).replace(" ", "")


def load(path) -> np.ndarray:
    from infer_ayah import load_wav_16k

    return np.asarray(load_wav_16k(Path(path)), dtype=np.float32)


def dist(a: str, b: str) -> int:
    import Levenshtein

    return Levenshtein.distance(a, b)


def best_ref(hyp: str, refs: list[str]) -> tuple[int, str]:
    """`(errors, reference)` of the reference closest to `hyp`."""
    return min(((dist(hyp, r), r) for r in refs), key=lambda t: (t[0], t[1]))


def tolerant_refs(sura: int, aya: int) -> list[str]:
    """The ayah's phonemes with every allowed length of the optional madds."""
    from compare_models import references

    strict, tolerant = references(sura, aya, dict(DEFAULT_ATTRS))
    return [strict, *sorted(tolerant - {strict})]


# ------------------------------------------------------------------ signal helpers


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def speech_rms(x: np.ndarray) -> float:
    """RMS of the louder half of the 25 ms frames, so pauses do not lower it."""
    n = len(x) // 400
    if n < 4:
        return rms(x)
    energy = np.sqrt(np.mean(np.square(x[: n * 400].reshape(n, 400)), axis=1))
    return float(np.sqrt(np.mean(np.square(np.sort(energy)[n // 2:])))) + 1e-9


def from_db(db: float) -> float:
    return 10 ** (db / 20)


def white(n: int, rng) -> np.ndarray:
    x = rng.standard_normal(n)
    return (x / rms(x)).astype(np.float32)


def pink(n: int, rng) -> np.ndarray:
    spectrum = np.fft.rfft(rng.standard_normal(n))
    spectrum[1:] /= np.sqrt(np.arange(1, len(spectrum)))
    x = np.fft.irfft(spectrum, n=n)
    return (x / rms(x)).astype(np.float32)


_room_spectrum = None


def room(n: int, rng) -> np.ndarray:
    """Stationary noise with the spectrum of the --room-noise recording (pink noise without one), unit RMS."""
    global _room_spectrum
    path = os.environ.get("AUDIT_ROOM_NOISE")
    if not path:
        return pink(n, rng)
    if _room_spectrum is None:
        clip = load(path)
        frames = np.lib.stride_tricks.sliding_window_view(clip, 1024)[::256] * np.hanning(1024)
        _room_spectrum = np.sqrt(np.mean(np.abs(np.fft.rfft(frames, axis=1)) ** 2, axis=0))
    spectrum = np.fft.rfft(rng.standard_normal(n + 2048))
    shape = np.interp(np.linspace(0, 1, len(spectrum)), np.linspace(0, 1, len(_room_spectrum)), _room_spectrum)
    x = np.fft.irfft(spectrum * shape, n=n + 2048)[1024: 1024 + n]
    return (x / rms(x)).astype(np.float32)


def silence(n: int, rng) -> np.ndarray:
    return np.zeros(n, np.float32)


def fit(x: np.ndarray, n: int) -> np.ndarray:
    return np.tile(x, n // len(x) + 1)[:n]


_tmp_dir = None


def run_tool(wav: np.ndarray, cmd: list[str], ext: str = "wav") -> np.ndarray:
    """Output audio (16 kHz) of `cmd`, whose `{i}` / `{o}` are a temp input wav and output file."""
    import soundfile as sf

    global _tmp_dir
    if _tmp_dir is None:
        _tmp_dir = tempfile.mkdtemp(prefix="audit_", dir=os.environ.get("AUDIT_TMP"))
    src, out = os.path.join(_tmp_dir, f"i{os.getpid()}.wav"), os.path.join(_tmp_dir, f"o{os.getpid()}.{ext}")
    sf.write(src, wav, SR, subtype="FLOAT")
    quiet = dict(check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run([c.format(i=src, o=out) for c in cmd], **quiet)
    if ext != "wav":
        decoded = os.path.join(_tmp_dir, f"d{os.getpid()}.wav")
        subprocess.run(["ffmpeg", "-y", "-i", out, "-ac", "1", "-ar", str(SR), "-c:a", "pcm_f32le", decoded], **quiet)
        out = decoded
    return sf.read(out, dtype="float32")[0]


# ------------------------------------------------------------------ battery conditions
# every condition is f(wav, rng, others) -> wav; `others` are other clean utterances (babble)


def lead(noise, seconds, snr_db):
    return lambda w, rng, others: np.concatenate([noise(int(seconds * SR), rng) * speech_rms(w) / from_db(snr_db), w])


def trail(noise, seconds, snr_db):
    return lambda w, rng, others: np.concatenate([w, noise(int(seconds * SR), rng) * speech_rms(w) / from_db(snr_db)])


def add(noise, snr_db, lead_s=0.0, tail_s=0.0):
    def f(w, rng, others):
        level = speech_rms(w) / from_db(snr_db)
        w = np.concatenate([np.zeros(int(lead_s * SR), np.float32), w, np.zeros(int(tail_s * SR), np.float32)])
        return w + noise(len(w), rng) * level
    return f


def babble(snr_db):
    def f(w, rng, others):
        mix = sum(fit(o / speech_rms(o), len(w)) for o in others[:4])
        return w + mix / rms(mix) * speech_rms(w) / from_db(snr_db)
    return f


def gain(db: float):
    return lambda w, rng, others: w * from_db(db)


def clip(peak_fraction: float):
    def f(w, rng, others):
        limit = peak_fraction * float(np.abs(w).max())
        return np.clip(w, -limit, limit)
    return f


def sox(*fx):
    f = lambda w, rng, others: run_tool(w, ["sox", "-D", "{i}", "-e", "floating-point", "-b", "32", "{o}", *fx])  # noqa: E731
    f.tool = "sox"
    return f


def codec(ext, *enc):
    f = lambda w, rng, others: run_tool(w, ["ffmpeg", "-y", "-i", "{i}", "-ac", "1", *enc, "{o}"], ext)  # noqa: E731
    f.tool = "ffmpeg"
    return f


def chain(*fs):
    def f(w, rng, others):
        for g in fs:
            w = g(w, rng, others)
        return w
    f.tool = next((getattr(g, "tool", None) for g in fs if getattr(g, "tool", None)), None)
    return f


CONDITIONS = {
    "clean": lambda w, rng, others: w,
    # --- non-speech before / after the recitation
    "lead_zeros_2s": lead(silence, 2, 0),
    "lead_room_1s_snr20": lead(room, 1, 20),
    "lead_room_2s_snr20": lead(room, 2, 20),
    "lead_room_5s_snr20": lead(room, 5, 20),
    "lead_room_2s_snr35": lead(room, 2, 35),
    "lead_white_2s_snr30": lead(white, 2, 30),
    "trail_room_3s_snr20": trail(room, 3, 20),
    "live_room_snr20": add(room, 20, lead_s=2, tail_s=1),  # noise before, under and after
    # --- level and clipping
    "gain-10db": gain(-10),
    "gain-20db": gain(-20),
    "gain-30db": gain(-30),
    "gain-40db": gain(-40),
    "gain+12db": gain(12),
    "clip_25pct_peak": clip(0.25),
    "clip_10pct_peak": clip(0.1),
    # --- noise under the speech
    "white_snr20": add(white, 20),
    "white_snr10": add(white, 10),
    "pink_snr10": add(pink, 10),
    "room_snr10": add(room, 10),
    "babble_snr15": babble(15),
    "babble_snr5": babble(5),
    # --- reverb
    "reverb40": sox("gain", "-6", "reverb", "40"),
    "reverb70": sox("gain", "-6", "reverb", "70"),
    "reverb90": sox("gain", "-8", "reverb", "90"),
    # --- channel
    "telephone_300-3400": sox("highpass", "300", "lowpass", "3400"),
    "lowpass_2k": sox("lowpass", "2000"),
    "8k_roundtrip": sox("rate", "8000", "rate", "16000"),
    "mp3_16k": codec("mp3", "-ar", "16000", "-c:a", "libmp3lame", "-b:a", "16k"),
    "opus_8k": codec("opus", "-c:a", "libopus", "-b:a", "8k"),
    # --- tempo, pitch, speed
    "tempo0.8": sox("tempo", "-s", "0.8"),
    "tempo1.25": sox("tempo", "-s", "1.25"),
    "tempo1.5": sox("tempo", "-s", "1.5"),
    "pitch-4st": sox("pitch", "-400", "rate", "16000"),
    "pitch+4st": sox("pitch", "400", "rate", "16000"),
    "speed0.85": sox("speed", "0.85", "rate", "16000"),
    "speed1.2": sox("speed", "1.2", "rate", "16000"),
    # --- combinations met in practice
    "mosque_reverb70_tempo1.25": sox("gain", "-6", "tempo", "-s", "1.25", "reverb", "70"),
    "far_phone": chain(gain(-30), add(white, 15), sox("highpass", "300", "lowpass", "3400")),
    "quiet_live_mic": chain(add(room, 20, lead_s=2, tail_s=1), gain(-20)),
}
# two utterances joined across a gap: (noise, seconds, snr)
PAIR_GAPS = {
    "pair_gap_zeros_2s": (silence, 2, 0),
    "pair_gap_room_2s_snr20": (room, 2, 20),
    "pair_gap_room_0.5s_snr20": (room, 0.5, 20),
}


def available_conditions() -> list[str]:
    """Condition names whose external tool (sox / ffmpeg) is installed."""
    return [name for name, f in CONDITIONS.items()
            if getattr(f, "tool", None) is None or shutil.which(f.tool)] + list(PAIR_GAPS)


# ------------------------------------------------------------------ stages


def _panel_task(task):
    path, sura, aya = task
    try:
        wav = load(path)
        refs = tolerant_refs(sura, aya)
    except Exception as exc:  # unreadable file or an ayah the phonetizer does not know
        return {"file": str(path), "error": repr(exc)}
    row = {"file": str(path), "reciter": Path(path).parent.name, "sura": sura, "aya": aya,
           "dur": round(len(wav) / SR, 2), "level_db": round(20 * np.log10(speech_rms(wav)), 1)}
    for m in MODELS:
        hyp = decode(m, wav)
        err, ref = best_ref(hyp, refs)
        row[m] = {"hyp": hyp, "err": err, "ref": ref}
    return row


def panel_tasks(audio_root: Path, per_reciter: int) -> list:
    rng = random.Random(0)
    tasks = []
    for d in sorted(p for p in audio_root.iterdir() if p.is_dir()):
        files = sorted(p for p in d.iterdir() if NAME_RE.match(p.stem))
        if len(files) < 2:
            continue
        probes = [p for p in files if p.stem in PROBE_AYAT]
        rest = [p for p in files if p not in probes and p.stat().st_size < 400_000]  # short ayat
        tasks += [(str(p), int(p.stem[:3]), int(p.stem[3:])) for p in probes + rng.sample(rest, min(per_reciter, len(rest)))]
    return tasks


def _qdat_task(task):
    import librosa
    import soundfile as sf

    row, audio = task
    wav, sr = sf.read(io.BytesIO(audio), dtype="float32", always_2d=True)
    wav = wav.mean(1)
    if sr != SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    row = dict(row, dur=round(len(wav) / SR, 2), level_db=round(20 * np.log10(speech_rms(wav)), 1))
    for m in MODELS:
        hyp = decode(m, wav)
        row[m] = {"hyp": hyp, "err": dist(hyp, row["ref"]), "ref": row["ref"]}
    return row


def qdat_tasks(parquet: str) -> list:
    import pandas as pd

    df = pd.read_parquet(parquet)
    return [({"id": r["id"], "gender": r["gender"], "age": int(r["age"]),
              "ref": re.sub(r"\s+", "", r["phonetic_transcript"])}, r["audio"]["bytes"]) for _, r in df.iterrows()]


def read_rows(out: Path, stage: str) -> list[dict]:
    path = out / f"{stage}.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if "error" not in r]


def battery_base(panel: list[dict], models: list[str], size: int) -> list[dict]:
    """Panel utterances of 4-14 s that every model reads (nearly) right, at most two per reciter."""
    rows = [r for r in panel if 4 <= r["dur"] <= 14 and all(r[m]["err"] <= 0.03 * len(r[m]["ref"]) for m in models)]
    random.Random(1).shuffle(rows)
    per_reciter: dict[str, int] = defaultdict(int)
    base = []
    for r in rows:
        if per_reciter[r["reciter"]] < 2:
            per_reciter[r["reciter"]] += 1
            base.append(r)
    return base[:size]


_wavs: dict = {}


def _wav(path) -> np.ndarray:
    if path not in _wavs:
        _wavs[path] = load(path)
    return _wavs[path]


def _battery_task(task):
    cond, item, others, seed = task
    rng = np.random.default_rng(seed)
    if cond in PAIR_GAPS:
        noise, seconds, snr = PAIR_GAPS[cond]
        a, b = _wav(item["file"]), _wav(others[0]["file"])
        wav = np.concatenate([a, noise(int(seconds * SR), rng) * speech_rms(a) / from_db(snr), b])
        refs = {m: item[m]["ref"] + others[0][m]["ref"] for m in MODELS}
    else:
        wav = CONDITIONS[cond](_wav(item["file"]), rng, [_wav(o["file"]) for o in others])
        refs = {m: item[m]["ref"] for m in MODELS}
    row = {"cond": cond, "file": item["file"], "reciter": item["reciter"], "dur": round(len(wav) / SR, 2)}
    for m in MODELS:
        hyp = decode(m, np.asarray(wav, dtype=np.float32))
        row[m] = {"hyp": hyp, "err": dist(hyp, refs[m]), "len": len(refs[m])}
    return row


def battery_tasks(base: list[dict], conditions: list[str]) -> list:
    tasks = []
    for ci, cond in enumerate(conditions):
        for i, item in enumerate(base):
            others = [base[(i + k) % len(base)] for k in (7, 13, 19, 29)]
            tasks.append((cond, item, others, 1000 * ci + i))
    return tasks


def _stream_task(task):
    reciter_dir, sura, first, count = task
    files = [next(iter(sorted(Path(reciter_dir).glob(f"{sura:03d}{aya:03d}.*"))), None)
             for aya in range(first, first + count)]
    if any(f is None for f in files):
        return {"error": f"missing ayat of {sura} in {reciter_dir}"}
    wavs = [load(f) for f in files]
    row = {"reciter": Path(reciter_dir).name, "sura": sura, "n": count, "dur": round(sum(map(len, wavs)) / SR, 1)}
    for m in MODELS:
        import Levenshtein

        refs, separate = [], 0
        for aya, wav in zip(range(first, first + count), wavs):
            err, ref = best_ref(decode(m, wav), tolerant_refs(sura, aya))
            refs.append(ref)
            separate += err
        ref = "".join(refs)
        ops = Levenshtein.editops(ref, decode(m, np.concatenate(wavs)))
        starts = set(np.cumsum([0, *map(len, refs)])[1:-1].tolist())  # first phoneme of ayat 2..n
        lost = sum(1 for op, i, _ in ops if op != "insert" and i in starts)
        row[m] = {"separate_err": separate, "stream_err": len(ops), "len": len(ref), "starts_lost": lost}
    return row


def _noise_task(task):
    name, seed = task
    rng = np.random.default_rng(seed)
    kind, level = name.rsplit("@", 1)
    wav = {"white": white, "pink": pink, "room": room, "zeros": silence}[kind](30 * SR, rng) * from_db(float(level))
    row = {"cond": name}
    for m in MODELS:
        row[m] = {"tokens_per_min": len(decode(m, wav.astype(np.float32))) * 2}
    return row


def run(stage: str, tasks: list, fn, args, tmp: str) -> None:
    out = Path(args.out) / f"{stage}.jsonl"
    done = 0
    with open(out, "w", encoding="utf-8") as f, \
            ProcessPoolExecutor(args.jobs, initializer=_init_worker, initargs=(dict(MODELS), tmp)) as pool:
        for row in pool.map(fn, tasks, chunksize=1):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            done += 1
            if done % 100 == 0:
                print(f"  {stage}: {done}/{len(tasks)}", flush=True)
    print(f"wrote {out} ({done} rows)")


# ------------------------------------------------------------------ report


def per(rows: list[dict], model: str) -> float:
    """Errors over reference length (errors capped at the length, as `decode.per`)."""
    length = lambda r: r[model].get("len", len(r[model].get("ref", "")))  # noqa: E731
    total = sum(length(r) for r in rows)
    return sum(min(r[model]["err"], length(r)) for r in rows) / total if total else 0.0


def repeated(ref: str, hyp: str) -> int:
    """Errors of `hyp` inside runs of at least REPEAT_RUN inserted phonemes.

    A reciter who stops for breath often says the last words again. The reference text has them once,
    so a model that writes what it hears is charged for them and one that leaves them out is not.
    """
    import Levenshtein

    ops = Levenshtein.editops(ref, hyp)
    total, k = 0, 0
    while k < len(ops):
        end = k
        if ops[k][0] == "insert":
            while end + 1 < len(ops) and ops[end + 1][0] == "insert" and ops[end + 1][2] == ops[end][2] + 1:
                end += 1
            if end - k + 1 >= REPEAT_RUN:
                total += end - k + 1
        k = end + 1
    return total


def _without_repeats(rows: list[dict], models: list[str]) -> list[dict]:
    return [{**r, **{m: {**r[m], "err": r[m]["err"] - repeated(r[m]["ref"], r[m]["hyp"])} for m in models}} for r in rows]


def _mark(tested: float, reference: float) -> str:
    both = " (acuan juga gagal)" if reference > REFERENCE_FAILS else ""
    return both + ("  <-- tertinggal" if tested - reference > BEHIND else "")


def _table(title: str, groups: dict[str, list[dict]], models: list[str], extra=None) -> list[str]:
    lines = [f"## {title}", "", "| | n | " + " | ".join(models) + " | |", "|---|---|" + "---|" * (len(models) + 1)]
    for name, rows in groups.items():
        if not rows:
            continue
        values = [per(rows, m) for m in models]
        note = (extra(rows) if extra else "") + _mark(values[0], values[-1])
        lines.append(f"| {name} | {len(rows)} | " + " | ".join(f"{v:.1%}" for v in values) + f" | {note} |")
    return lines + [""]


def report(out: Path, models: list[str]) -> str:
    """Markdown report of every stage found in `out` (also written to `out/report.md`)."""
    tested, lines = models[0], [f"# Audit ketahanan: {models[0]} terhadap {models[-1]}", ""]
    panel = read_rows(out, "panel")
    if panel:
        by_reciter: dict[str, list[dict]] = defaultdict(list)
        for r in panel:
            by_reciter[r["reciter"]].append(r)
        # a reciter the reference reads almost entirely wrong has recordings that do not match their ayah numbers
        mismatched = {name: rows for name, rows in by_reciter.items() if per(rows, models[-1]) > MISMATCHED}
        usable = {name: rows for name, rows in by_reciter.items() if name not in mismatched}
        ordered = dict(sorted(usable.items(), key=lambda kv: -(per(kv[1], tested) - per(kv[1], models[-1]))))
        everything = [r for rows in usable.values() for r in rows]
        groups = {"semua": everything,
                  "semua, tanpa kata yang diulang qari": _without_repeats(everything, models), **ordered}
        lines += _table("Panel qari (PER, toleran panjang madd)", groups, models)
        if mismatched:
            names = ", ".join(f"{name} ({len(rows)} berkas, {models[-1]} {per(rows, models[-1]):.0%})"
                              for name, rows in sorted(mismatched.items()))
            lines += [f"Tidak dihitung, karena model acuan pun salah di atas {MISMATCHED:.0%} "
                      f"(rekaman tidak cocok dengan nomor ayatnya): {names}.", ""]
    qdat = read_rows(out, "qdat")
    if qdat:
        groups = {"semua": qdat, **{g: [r for r in qdat if r["gender"] == g] for g in ("female", "male")}}
        broken = lambda rows: f"klip PER>30%: {sum(r[tested]['err'] > 0.3 * len(r['ref']) for r in rows)}"  # noqa: E731
        lines += _table("QDAT (rekaman HP)", groups, models, broken)
    battery = read_rows(out, "battery")
    if battery:
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in battery:
            groups[r["cond"]].append(r)
        lines += _table("Baterai degradasi (ucapan bersih yang dibaca benar oleh semua model)", groups, models)
    stream = read_rows(out, "stream")
    if stream:
        lines += ["## Bacaan bersambung (ayat berurutan: per ayat -> satu stream; awal ayat yang hilang)", "",
                  "| qari | surah | ayat | " + " | ".join(models) + " | |", "|---|---|---|" + "---|" * (len(models) + 1)]
        for r in stream:
            cells = [f"{r[m]['separate_err'] / r[m]['len']:.1%} -> {r[m]['stream_err'] / r[m]['len']:.1%}"
                     f" ({r[m]['starts_lost']})" for m in models]
            gap = (r[tested]["stream_err"] - r[models[-1]]["stream_err"]) / r[tested]["len"]
            lines.append(f"| {r['reciter']} | {r['sura']} | {r['n']} | " + " | ".join(cells) + f" | {_mark(gap, 0)} |")
        lines.append("")
    noise = read_rows(out, "noise")
    if noise:
        lines += ["## Masukan tanpa ucapan (token per menit)", "", "| | " + " | ".join(models) + " |",
                  "|---|" + "---|" * len(models)]
        lines += [f"| {r['cond']} | " + " | ".join(str(r[m]["tokens_per_min"]) for m in models) + " |" for r in noise]
        lines.append("")
    text = "\n".join(lines)
    (out / "report.md").write_text(text, encoding="utf-8")
    return text


# ------------------------------------------------------------------ cli


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", nargs=3, action="append", metavar=("NAME", "MODEL", "TOKENS"),
                   help="model to audit (repeat); the first is the one under test, the last the reference")
    p.add_argument("--out", required=True, help="folder of the .jsonl files and report.md")
    p.add_argument("--audio-root", help="folder of <reciter>/SSSAAA.<ext> recordings (panel, battery, stream)")
    p.add_argument("--qdat", help="qdat_bench parquet (default: the Hugging Face cache, stage skipped if absent)")
    p.add_argument("--room-noise", help="recording of room noise whose spectrum the 'room' noise copies (default: pink)")
    p.add_argument("--stages", nargs="+", default=["panel", "qdat", "battery", "stream", "noise"],
                   choices=["panel", "qdat", "battery", "stream", "noise"])
    p.add_argument("--conditions", nargs="+", help="battery conditions (default: all whose tool is installed)")
    p.add_argument("--per-reciter", type=int, default=6, help="ayat per reciter in the panel, besides the probes")
    p.add_argument("--battery-size", type=int, default=48)
    p.add_argument("--stream-reciters", nargs="+", help="reciter folders of the stream stage (default: the 7 largest)")
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    p.add_argument("--report", action="store_true", help="only rebuild the report from existing .jsonl files")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import glob

    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names_file = out / "models.json"
    if args.report:
        print(report(out, json.loads(names_file.read_text(encoding="utf-8"))))
        return 0
    if not args.model or len(args.model) < 2:
        raise SystemExit("give the model under test and a reference: --model NAME MODEL TOKENS, twice")
    MODELS.update({name: (model, tokens) for name, model, tokens in args.model})
    for model, tokens in MODELS.values():
        for f in (model, tokens):
            if not Path(f).is_file():
                raise SystemExit(f"not found: {f}")
    names = list(MODELS)
    names_file.write_text(json.dumps(names), encoding="utf-8")
    if args.room_noise:
        os.environ["AUDIT_ROOM_NOISE"] = args.room_noise
    root = Path(args.audio_root) if args.audio_root else None
    with tempfile.TemporaryDirectory(prefix="audit_tmp_") as tmp:
        for stage in args.stages:
            if stage == "noise":
                tasks = [(n, i) for i, n in enumerate(
                    ["zeros@0", *(f"{k}@{lv}" for k in ("white", "pink", "room") for lv in (-60, -40, -20))])]
                run(stage, tasks, _noise_task, args, tmp)
            elif stage == "qdat":
                found = [args.qdat] if args.qdat else glob.glob(os.path.expanduser(QDAT_GLOB))
                if not found:
                    print("qdat: no parquet given or cached, skipped")
                    continue
                run(stage, qdat_tasks(found[0]), _qdat_task, args, tmp)
            elif root is None:
                print(f"{stage}: needs --audio-root, skipped")
            elif stage == "panel":
                run(stage, panel_tasks(root, args.per_reciter), _panel_task, args, tmp)
            elif stage == "battery":
                base = battery_base(read_rows(out, "panel"), names, args.battery_size)
                if not base:
                    print("battery: run the panel stage first (no clean utterances found), skipped")
                    continue
                print(f"battery: {len(base)} utterances of {len({b['reciter'] for b in base})} reciters")
                run(stage, battery_tasks(base, args.conditions or available_conditions()), _battery_task, args, tmp)
            elif stage == "stream":
                dirs = [root / r for r in args.stream_reciters] if args.stream_reciters else sorted(
                    (p for p in root.iterdir() if p.is_dir()), key=lambda p: -sum(1 for _ in p.iterdir()))[:7]
                run(stage, [(str(d), *s) for d in dirs for s in STREAMS], _stream_task, args, tmp)
    print(report(out, names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

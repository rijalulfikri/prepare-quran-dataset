"""Compare streaming phoneme models on a folder of per-ayah recordings.

Each file must be named `SSSAAA.<ext>` (surah, ayah; e.g. `078001.flac`).
Every model is decoded with sherpa-onnx exactly like `infer_ayah.py` /
utluma_app's MesinSherpa, so .onnx exports of this repo and the app's .ort
(utluma3) are compared on the same runtime.

Two PERs per ayah (character level, one char = one phoneme):

* strict: against the reference phonetized with the reciter's moshaf
  attributes (`--moshaf`, from obadx/mualem-recitations-annotated
  moshaf_metadata), i.e. the same convention as this repo's training labels;
* tolerant: the lowest PER over every allowed length of the optional madds
  ('aarid 2/4/6, leen 2/4/6, munfasil 2/4/5, muttasil 4/5), so a model is not
  penalised for a different but valid madd length convention.

    python compare_models.py --audio-dir .../Abdul_Basit_Murattal_30 --moshaf 2.1 \\
        --model epoch18 results-zipformer-v0/export-epoch-18/model.int8.onnx \\
                results-zipformer-v0/export-epoch-18/tokens.txt \\
        --model utluma3 .../zipformer_p_arabic_v3_c16_utluma3.int8.ort .../tokens.txt \\
        --out compare.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ZIPFORMER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ZIPFORMER_DIR))

META_URL = (
    "https://datasets-server.huggingface.co/rows?dataset=obadx/mualem-recitations-annotated"
    "&config=moshaf_metadata&split=train&offset=0&length=100"
)
NAME_RE = re.compile(r"^(\d{3})(\d{3})$")
TOLERANT_GRID = {
    "madd_aared_len": (2, 4, 6),
    "madd_alleen_len": (2, 4, 6),
    "madd_monfasel_len": (2, 4, 5),
    "madd_mottasel_len": (4, 5),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--audio-dir", required=True)
    p.add_argument("--model", nargs=3, action="append", required=True, metavar=("NAME", "MODEL", "TOKENS"),
                   help="model to compare (repeat); .onnx or .ort plus its tokens.txt")
    p.add_argument("--moshaf", required=True, help="moshaf id of the reciter in moshaf_metadata, e.g. 2.1")
    p.add_argument("--out", default="compare_models.csv", help="per-ayah CSV (default: compare_models.csv)")
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    p.add_argument("--limit", type=int, help="only the first N files (quick check)")
    p.add_argument("--tail-pad", type=float, default=0.66,
                   help="seconds of zeros fed after each file before input_finished (default: 0.66, as the app)")
    return p.parse_args(argv)


def hf_token() -> str | None:
    tok = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    env = ZIPFORMER_DIR / ".env"
    if not tok and env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            if key.strip().removeprefix("export ") in ("HUGGINGFACE_TOKEN", "HF_TOKEN"):
                tok = value.strip().strip("'\"")
    return tok


def moshaf_attrs(moshaf_id: str) -> dict:
    """MoshafAttributes kwargs of `moshaf_id` from the HF moshaf_metadata config."""
    from quran_transcript import MoshafAttributes

    req = urllib.request.Request(META_URL)
    if tok := hf_token():
        req.add_header("Authorization", f"Bearer {tok}")
    rows = json.load(urllib.request.urlopen(req, timeout=60))["rows"]
    row = next((r["row"] for r in rows if r["row"]["id"] == moshaf_id), None)
    if row is None:
        raise SystemExit(f"moshaf {moshaf_id} not in moshaf_metadata")
    print(f"moshaf {moshaf_id}: {row['reciter_english_name']} - {row['name']}")
    return {k: v for k, v in row.items() if k in MoshafAttributes.model_fields}


def references(sura: int, aya: int, attrs: dict) -> tuple[str, set[str]]:
    """`(strict, tolerant_set)` phoneme references of one ayah."""
    from quran_transcript import Aya, MoshafAttributes, quran_phonetizer

    uthmani = Aya(sura, aya).get().uthmani

    def phon(a: dict) -> str:
        return quran_phonetizer(uthmani, MoshafAttributes(**a), remove_spaces=True).phonemes

    strict = phon(attrs)
    tolerant = {strict}
    keys = list(TOLERANT_GRID)
    for values in itertools.product(*TOLERANT_GRID.values()):
        a = dict(attrs, **dict(zip(keys, values)))
        a["madd_mottasel_waqf"] = max(a["madd_mottasel_len"], attrs["madd_mottasel_waqf"])
        if a["madd_alleen_len"] > a["madd_aared_len"]:
            continue  # not a valid reading (MoshafAttributes rejects it)
        tolerant.add(phon(a))
    return strict, tolerant


def errors(hyp: str, ref: str) -> int:
    import Levenshtein

    return min(Levenshtein.distance(hyp, ref), len(ref))


_recognizers: dict = {}


def _decode(task: tuple[str, str, str, str, float]) -> tuple[str, str, str]:
    """(name, path, hypothesis) for one model/file; one recognizer per worker and model."""
    from infer_ayah import load_wav_16k

    name, model, tokens, path, tail_pad = task
    import numpy as np
    import sherpa_onnx

    if name not in _recognizers:
        _recognizers[name] = sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(
            tokens=tokens, model=model, num_threads=1, sample_rate=16000, feature_dim=80,
            decoding_method="greedy_search", enable_endpoint_detection=False,
        )
    rec = _recognizers[name]
    wav = load_wav_16k(Path(path))
    stream = rec.create_stream()
    for i in range(0, len(wav), 1600):
        stream.accept_waveform(16000, wav[i : i + 1600])
        while rec.is_ready(stream):
            rec.decode_stream(stream)
    stream.accept_waveform(16000, np.zeros(int(tail_pad * 16000), np.float32))
    stream.input_finished()
    while rec.is_ready(stream):
        rec.decode_stream(stream)
    return name, path, rec.get_result(stream)


def confusions(hyp: str, ref: str) -> list[str]:
    """Edit operations as readable strings: 'غ→ر', '+ا', '-ه'."""
    import Levenshtein

    out = []
    for op, i, j in Levenshtein.editops(hyp, ref):  # transforms hyp into ref
        if op == "replace":
            out.append(f"{ref[j]}→{hyp[i]}")
        elif op == "insert":
            out.append(f"-{ref[j]}")  # ref char the model dropped
        else:
            out.append(f"+{hyp[i]}")  # extra char the model emitted
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    files = sorted(p for p in Path(args.audio_dir).iterdir() if NAME_RE.match(p.stem))
    if args.limit:
        files = files[: args.limit]
    attrs = moshaf_attrs(args.moshaf)

    print(f"{len(files)} files; building references...")
    refs = {}
    for f in files:
        s, a = NAME_RE.match(f.stem).groups()
        refs[str(f)] = (int(s), int(a), *references(int(s), int(a), attrs))

    names = [m[0] for m in args.model]
    tasks = [(n, m, t, str(f), args.tail_pad) for n, m, t in args.model for f in files]
    hyps: dict[tuple[str, str], str] = {}
    print(f"decoding {len(tasks)} (model, file) pairs with {args.jobs} workers...")
    with ProcessPoolExecutor(args.jobs) as pool:
        for k, (name, path, hyp) in enumerate(pool.map(_decode, tasks, chunksize=4), 1):
            hyps[(name, path)] = hyp
            if k % 100 == 0 or k == len(tasks):
                print(f"  {k}/{len(tasks)}", flush=True)

    totals = {n: Counter() for n in names}
    per_sura = {n: defaultdict(Counter) for n in names}
    conf = {n: Counter() for n in names}
    rows = []
    for f in files:
        sura, aya, strict, tolerant = refs[str(f)]
        row = {"file": f.name, "sura": sura, "aya": aya, "reference": strict}
        for n in names:
            hyp = hyps[(n, str(f))]
            e_strict = errors(hyp, strict)
            best = min(tolerant, key=lambda r: errors(hyp, r) / len(r))
            e_tol = errors(hyp, best)
            for c, key in ((totals[n], None), (per_sura[n][sura], None)):
                c["ref"] += len(strict)
                c["err"] += e_strict
                c["ref_tol"] += len(best)
                c["err_tol"] += e_tol
                c["ayat"] += 1
                c["exact"] += e_tol == 0
            conf[n].update(confusions(hyp, best))
            row |= {f"{n}_hyp": hyp, f"{n}_per": round(e_strict / len(strict), 4),
                    f"{n}_per_tol": round(e_tol / len(best), 4)}
        rows.append(row)

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\nper-ayah results: {args.out}\n")
    print(f"{'model':10s} {'PER strict':>11s} {'PER tolerant':>13s} {'exact ayat':>12s}")
    for n in names:
        t = totals[n]
        print(f"{n:10s} {t['err'] / t['ref']:11.2%} {t['err_tol'] / t['ref_tol']:13.2%} "
              f"{t['exact']:5d}/{t['ayat']:<5d}")

    print("\ntolerant PER per surah")
    print("surah " + " ".join(f"{n:>9s}" for n in names))
    for sura in sorted(per_sura[names[0]]):
        cells = []
        for n in names:
            c = per_sura[n][sura]
            cells.append(f"{c['err_tol'] / c['ref_tol']:9.2%}")
        print(f"{sura:5d} " + " ".join(cells))

    for n in names:
        print(f"\n{n}: most frequent errors (ref→hyp substitution, -dropped, +inserted)")
        print("  " + ", ".join(f"{k} ×{v}" for k, v in conf[n].most_common(15)))
        worst = sorted(rows, key=lambda r: -r[f"{n}_per_tol"])[:8]
        print(f"{n}: worst ayat (tolerant PER)")
        for r in worst:
            print(f"  {r['sura']}:{r['aya']:<4d} {r[f'{n}_per_tol']:.1%}  hyp {r[f'{n}_hyp']}\n"
                  f"  {'':10s}ref {r['reference']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

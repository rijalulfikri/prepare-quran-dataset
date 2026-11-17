# Zipformer2 CTC Streaming Training Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Colab data-prep script that writes FLAC + Lhotse manifests to Google Drive, and a `train_zipformer.py` pipeline that syncs them with rclone, trains an icefall Zipformer2 CTC phoneme model from scratch, evaluates it (dev/test PER, qdat_bench) and exports sherpa-onnx streaming ONNX.

**Architecture:** New package `src/prepare_quran_dataset/zipformer/` holds small focused modules (tokens, config, data prep, datamodule, model, checkpoint store, Drive backup, train loop, decode, export, qdat). icefall lives in `external/icefall` (git submodule) and is imported via `sys.path` setup, never copied. Root-level `train_zipformer.py` is a thin stage runner; `prepare_zipformer_data_colab.py` is the Colab entry point.

**Tech Stack:** Python 3.14 (fallback 3.12), torch 2.9.1 + k2 wheel, Lhotse, icefall (zipformer recipe), onnxruntime, rclone, pydantic, pytest. Repo deps: `quran-transcript==0.1.2`, `qdat-bench` (git).

**Spec:** `docs/superpowers/specs/2026-11-08-zipformer2-ctc-training-design.md`

## Global Constraints

- Single output level: `phonemes` from `vocab_streaming/vocab.json`; `[PAD]=0` is CTC blank; `[EOS]=1` is never in labels.
- Model dims (icefall "medium", causal): `num_encoder_layers=2,2,3,4,3,2`, `downsampling_factor=1,2,4,8,4,2`, `feedforward_dim=512,768,1024,1536,1024,768`, `num_heads=4,4,4,8,4,4`, `encoder_dim=192,256,384,512,384,256`, `query_head_dim=32`, `value_head_dim=12`, `pos_head_dim=4`, `pos_dim=48`, `encoder_unmasked_dim=192,192,256,256,256,192`, `cnn_module_kernel=31,31,15,15,15,31`, `causal=True`, `chunk_size=8,16,24`, `left_context_frames=64,128,256`.
- Features: 80-dim kaldi fbank via Lhotse `Fbank(FbankConfig(num_mel_bins=80))`, computed on the fly.
- **Output frame rate (spec § Model):** `output_downsampling_factor` default **1** → 20 ms output frames, `T_out = (T_fbank - 7) // 2`; factor 2 → 40 ms, `T_out = ((T_fbank - 7) // 2 + 1) // 2`. Frame shift in seconds is `0.02 * output_downsampling_factor` everywhere — define it once as `frame_shift_seconds(factor)` (Task 8) and never hard-code 0.04/40 ms/`// 4`. icefall's `Zipformer2.forward`/`streaming_forward` assert factor 2 and `train.get_encoder_model` hard-codes it (verified at icefall `9c96676`), so the encoder is built by our own subclass in Task 6, not via `get_model`.
- Labels recomputed with `quran_phonetizer(uthmani, moshaf_attr, remove_spaces=True).phonemes`; moshaf `6.0` segment overrides identical to `train_streaming.py`.
- Noise split 30 % (CutMix aug) / 70 % (silence inputs, empty text), seeded. Silence cuts clipped to 40 s, padded to ≥ 1.0 s.
- Manifests store recording paths **relative** to the data root; never absolute Colab paths.
- Export: chunk 24 / left 256 → `model.onnx`, `model.int8.onnx`, `tokens.txt`.
- Never edit `requires-python` in `pyproject.toml`. Never vendor icefall files.
- Plan decision (deviates from spec wording): no online `model_avg`; the final model is the **offline** average of the last `avg_last_k=3` epoch checkpoints via `icefall.checkpoint.average_checkpoints`. Same end result the spec asks for, less state to checkpoint.
- Fault tolerance (spec § *Fault tolerance*): mid-epoch `checkpoint-{batch_idx_train}.pt` every `save_every_n_batches=2000` + `epoch-N.pt`; each `.pt` is complete only when its `<name>.meta.json` exists; prune per kind to `keep_last_k`; resume positions the sampler by **consumed** batches, not drawn batches; call `set_batch_count` on resume; Drive backup via one serial background rclone worker, meta uploaded last, Drive pruning only after the new meta is on Drive; upload failure = warning only.
- Tests must pass on a machine without k2/CUDA: guard icefall-dependent tests with `pytest.importorskip("k2")`. Tests that need the `rclone` binary use `pytest.mark.skipif(shutil.which("rclone") is None, ...)` and a **local** rclone remote defined via env (`RCLONE_CONFIG_TESTLOCAL_TYPE=local`, remote `testlocal:<tmp_path>`), never the network.
- Commit messages: repo style `add: ...` / `fix: ...`, ending with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## Review Focus

1. A phoneme character missing from `vocab.json` must raise, not be dropped silently (pinned in Task 2).
2. Moshaf `6.0` segment `010.0158` must be phonetised with `tasheel_or_madd="tasheel"`, others with the base attributes (pinned in Task 1).
3. Re-running Colab prep after a timeout must skip finished configs and the merge must not duplicate cuts (pinned in Task 4).
4. A silence cut with empty text must yield `target_lengths == 0` and a finite CTC loss, not NaN (pinned in Task 7).
5. The ONNX streaming decoder and PyTorch simulated-streaming decode must emit the same token sequence on the same audio (pinned in Task 9).
6. A box killed while a checkpoint is being written or uploaded must never leave a "latest" checkpoint that is truncated: resume must pick the newest checkpoint **with** a valid meta + matching sha256, locally and on Drive (pinned in Task 7A).
7. Resume with `num_workers > 0` must not skip the batches that were prefetched but never trained on (pinned in Task 5 and Task 7).
8. A cut whose label needs more CTC frames than the audio provides (`L + R > T_out`, worst case at speed 1.1) must be dropped and reported before training, never reach `ctc_loss` and produce `inf` (pinned in Task 4 and Task 7).
9. Changing `output_downsampling_factor` between runs must not silently resume an incompatible checkpoint (pinned in Task 7).

---

### Task 0: Scaffolding — icefall submodule, env files, package skeleton

**Files:**
- Create: `external/icefall` (git submodule → `https://github.com/k2-fsa/icefall`, pinned to the commit checked out at implementation time)
- Create: `requirements-zipformer.txt`
- Create: `src/prepare_quran_dataset/zipformer/__init__.py` (empty)
- Create: `src/prepare_quran_dataset/zipformer/icefall_import.py`
- Create: `tests/zipformer/__init__.py` (empty), `tests/zipformer/test_icefall_import.py`
- Modify: `.gitignore` (append `data-zipformer/`, `results-zipformer*/`, `rclone.conf`)

**Interfaces:**
- Produces: `icefall_import.icefall_root() -> Path` (env `ICEFALL_ROOT` or `<repo>/external/icefall`); `icefall_import.setup_icefall_paths() -> Path` (inserts `<root>` and `<root>/egs/librispeech/ASR/zipformer` into `sys.path`, idempotent, returns recipe dir); `icefall_import.load_recipe_module(filename: str) -> ModuleType` (loads a hyphenated recipe file such as `export-onnx-streaming-ctc.py` with `importlib.util`).

- [ ] **Step 1: Add submodule and requirements**

```bash
git submodule add https://github.com/k2-fsa/icefall external/icefall
```

`requirements-zipformer.txt` content (pins are the builder's to verify against the k2 wheel index; record the verified torch/k2 pair in a comment at the top):

```
# verified pair: torch==2.9.1 + k2==<wheel version> (cp314, cuda 12.x)  <- fill in
torch==2.9.1
torchaudio==2.9.1
lhotse>=1.30
onnx
onnxruntime
onnxsim
kaldifeat ; platform_system == "Linux"
sentencepiece
tensorboard
wandb
python-dotenv
Levenshtein
pyyaml
pydantic
quran-transcript==0.1.2
qdat-bench @ git+https://github.com/obadx/qdat_bench.git
```

- [ ] **Step 2: Write the failing test**

```python
# tests/zipformer/test_icefall_import.py
import sys
from pathlib import Path
from prepare_quran_dataset.zipformer import icefall_import as ii

def test_icefall_root_defaults_to_external(monkeypatch):
    monkeypatch.delenv("ICEFALL_ROOT", raising=False)
    assert ii.icefall_root().name == "icefall"
    assert ii.icefall_root().parent.name == "external"

def test_setup_paths_is_idempotent(monkeypatch, tmp_path):
    (tmp_path / "egs/librispeech/ASR/zipformer").mkdir(parents=True)
    monkeypatch.setenv("ICEFALL_ROOT", str(tmp_path))
    before = len(sys.path)
    ii.setup_icefall_paths(); ii.setup_icefall_paths()
    assert len(sys.path) == before + 2
    assert str(tmp_path / "egs/librispeech/ASR/zipformer") in sys.path

def test_load_recipe_module(monkeypatch, tmp_path):
    rec = tmp_path / "egs/librispeech/ASR/zipformer"; rec.mkdir(parents=True)
    (rec / "a-b.py").write_text("X = 42\n")
    monkeypatch.setenv("ICEFALL_ROOT", str(tmp_path))
    assert ii.load_recipe_module("a-b.py").X == 42
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/zipformer/test_icefall_import.py -v` → FAIL (module not found)

- [ ] **Step 4: Implement `icefall_import.py` with the three functions above**

- [ ] **Step 5: Run tests → PASS, then commit**

```bash
git add .gitmodules external/icefall requirements-zipformer.txt src/prepare_quran_dataset/zipformer tests/zipformer .gitignore
git commit -m "add: zipformer package skeleton, icefall submodule and env pins"
```

---

### Task 1: Shared moshaf overrides (fixes the global-variable bug)

**Files:**
- Create: `src/prepare_quran_dataset/zipformer/moshaf_ways.py`
- Test: `tests/zipformer/test_moshaf_ways.py`
- Modify: `train_streaming.py:720-746` (replace `prepare_special_moshaf_ways` with an import) and `train_streaming.py:964-966` (call site)

**Interfaces:**
- Produces: `SPECIAL_SEGMENT_OVERRIDES: dict[str, dict[str, dict]]` = `{"6.0": {"010.0158": {"tasheel_or_madd": "tasheel"}, "027.0107": {...same}, "010.0140": {...}, "010.0233": {...}, "012.0023": {"noon_tamnna": "rawm"}, "026.0076": {"raa_firq": "tarqeeq"}}}`;
  `build_special_moshaf_attrs(moshaf_id_to_dict: dict[str, dict]) -> dict[str, dict[str, MoshafAttributes]]`;
  `resolve_moshaf_attr(moshaf_id: str, segment_index: str, base: dict[str, MoshafAttributes], special: dict[str, dict[str, MoshafAttributes]]) -> MoshafAttributes`.

- [ ] **Step 1: Write the failing test**

```python
# tests/zipformer/test_moshaf_ways.py
from prepare_quran_dataset.zipformer import moshaf_ways as mw

def _base_dict():
    # minimal valid MoshafAttributes kwargs; take a real row from
    # obadx/mualem-recitations-annotated moshaf_metadata and hard-code it here.
    ...

def test_override_applies_only_to_listed_segment():
    base_d = {"6.0": _base_dict()}
    base = {k: mw.MoshafAttributes(**v) for k, v in base_d.items()}
    special = mw.build_special_moshaf_attrs(base_d)
    assert mw.resolve_moshaf_attr("6.0", "010.0158", base, special).tasheel_or_madd == "tasheel"
    assert mw.resolve_moshaf_attr("6.0", "001.0001", base, special) == base["6.0"]

def test_unknown_moshaf_falls_back_to_base():
    base_d = {"1.0": _base_dict()}
    base = {k: mw.MoshafAttributes(**v) for k, v in base_d.items()}
    special = mw.build_special_moshaf_attrs(base_d)   # has no "1.0" entry
    assert mw.resolve_moshaf_attr("1.0", "010.0158", base, special) == base["1.0"]
```

- [ ] **Step 2: Run → FAIL**
- [ ] **Step 3: Implement `moshaf_ways.py`**; `build_special_moshaf_attrs` must use its argument (not a global) and skip moshaf ids absent from the input dict.
- [ ] **Step 4: Replace `prepare_special_moshaf_ways` in `train_streaming.py`** with `from prepare_quran_dataset.zipformer.moshaf_ways import build_special_moshaf_attrs` and call `build_special_moshaf_attrs(moshaf_id_to_moshaf_dict)` at line ~964. `python -c "import ast,sys; ast.parse(open('train_streaming.py').read())"` must succeed.
- [ ] **Step 5: Run tests → PASS, commit** `fix: share moshaf segment overrides and drop global lookup`

---

### Task 2: Phoneme tokens (vocab.json ↔ ids ↔ tokens.txt)

**Files:**
- Create: `src/prepare_quran_dataset/zipformer/tokens.py`
- Test: `tests/zipformer/test_tokens.py`

**Interfaces:**
- Produces: `class PhonemeTokens` with `@classmethod from_vocab_json(path: str | Path) -> PhonemeTokens`; properties `blank_id: int` (== 0), `vocab_size: int` (== len of `phonemes` dict, 44); `encode(phonemes: str) -> list[int]` (raises `KeyError` with the offending char on unknown symbol; `""` → `[]`); `decode(ids: Sequence[int]) -> str` (skips blank and `[EOS]`); `write_tokens_txt(path) -> Path` (first line `<blk> 0`, then `<sym> <id>` for every non-`[PAD]` entry sorted by id, including `[EOS] 1`); `@classmethod from_tokens_txt(path) -> PhonemeTokens`.

- [ ] **Step 1: Write the failing test**

```python
# tests/zipformer/test_tokens.py
import pytest
from prepare_quran_dataset.zipformer.tokens import PhonemeTokens

VOCAB = "vocab_streaming/vocab.json"

def test_blank_and_size():
    t = PhonemeTokens.from_vocab_json(VOCAB)
    assert t.blank_id == 0 and t.vocab_size == 44

def test_encode_decode_roundtrip():
    t = PhonemeTokens.from_vocab_json(VOCAB)
    s = "بِسْ"[:2]  # two symbols that exist in the vocab
    assert t.decode(t.encode(s)) == s
    assert t.encode("") == []

def test_unknown_symbol_raises():
    t = PhonemeTokens.from_vocab_json(VOCAB)
    with pytest.raises(KeyError, match="Z"):
        t.encode("Z")

def test_tokens_txt_roundtrip(tmp_path):
    t = PhonemeTokens.from_vocab_json(VOCAB)
    p = t.write_tokens_txt(tmp_path / "tokens.txt")
    lines = p.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "<blk> 0" and lines[1] == "[EOS] 1" and len(lines) == 44
    t2 = PhonemeTokens.from_tokens_txt(p)
    assert t2.encode("ب") == t.encode("ب")
```

(Pick the two-symbol string from `vocab.json` keys when writing the test; the vocab is per character, so any two existing keys concatenated work.)

- [ ] **Step 2: Run → FAIL**
- [ ] **Step 3: Implement `tokens.py`**
- [ ] **Step 4: Run → PASS, commit** `add: phoneme token table for zipformer ctc`

---

### Task 3: Training config

**Files:**
- Create: `src/prepare_quran_dataset/zipformer/config.py`
- Create: `configs/train/zipformer/train_config.yml`
- Test: `tests/zipformer/test_config.py`

**Interfaces:**
- Produces: `class ZipformerTrainConfig(BaseModel)` with `from_yaml(path)` / `to_yaml(path)` (same pattern as `TrainConfig` in `train_streaming.py`). Fields and defaults:
  - data: `train_moshaf_ids: list[str]`, `test_moshaf_ids: list[str]`, `devset_ratio: float = 0.1`, `seed: int = 42`, `max_audio_seconds: float | None = None`, `noise_aug_ratio: float = 0.3`, `max_noise_input_seconds: float = 40.0`, `min_noise_input_seconds: float = 1.0`, `data_dir: str = "./data-zipformer"`, `rclone_remote_path: str = "gdrive:muaalem-zipformer-data"`
  - model: the eleven dim strings from Global Constraints as `str` fields with those defaults, `output_downsampling_factor: Literal[1, 2] = 1`, `causal: bool = True`, `chunk_size: str = "8,16,24"`, `left_context_frames: str = "64,128,256"`, `feature_dim: int = 80`
  - training: `max_duration: float = 600`, `num_workers: int = 8`, `num_epochs: int = 30`, `base_lr: float = 0.045`, `lr_batches: float = 7500`, `lr_epochs: float = 3.5`, `warmup_batches: int = 500`, `keep_last_k: int = 3`, `avg_last_k: int = 3`, `bf16: bool = True`, `log_every: int = 50`, `dev_max_cuts: int | None = 2000`, `cutmix_p: float = 0.5`, `cutmix_snr: tuple[float, float] = (10.0, 20.0)`, `speed_perturb: bool = True`, `output_dir: str = "./results-zipformer-v0"`
  - eval/export: `export_chunk_size: int = 24`, `export_left_context: int = 256`, `hub_model_id: str = "obadx/muaalem-zipformer-ctc-v0"`, `wandb_project_name: str = "muaalem-zipformer-ctc-v0"`
  - frame budget: `max_dropped_fraction: float = 0.01` (abort `--stage train` if more speech cuts than this fail the CTC frame check).
  - fault tolerance: `save_every_n_batches: int = 2000`, `rclone_checkpoint_path: str | None = None` (YAML sets `"gdrive:muaalem-zipformer-data/checkpoints/muaalem-zipformer-ctc-v0"`; `None` disables Drive backup), `upload_timeout_seconds: int = 1800`, `shutdown_flush_seconds: int = 120`. `keep_last_k` (above) applies separately to `checkpoint-*.pt` and to `epoch-*.pt`.
  - validator: `avg_last_k <= keep_last_k`; `export_chunk_size` must be one of `chunk_size` values; `save_every_n_batches >= 1`; `rclone_checkpoint_path`, when set, must contain `:` (an rclone `remote:path`).

- [ ] **Step 1: Write the failing test**

```python
# tests/zipformer/test_config.py
import pytest
from prepare_quran_dataset.zipformer.config import ZipformerTrainConfig

def test_default_yaml_loads_and_matches_v3_dims():
    c = ZipformerTrainConfig.from_yaml("configs/train/zipformer/train_config.yml")
    assert c.encoder_dim == "192,256,384,512,384,256"
    assert c.chunk_size == "8,16,24" and c.left_context_frames == "64,128,256"
    assert c.test_moshaf_ids == ["19.0", "29.0", "30.0"]
    assert len(c.train_moshaf_ids) == 22

def test_export_chunk_must_be_trained():
    with pytest.raises(ValueError):
        ZipformerTrainConfig(train_moshaf_ids=["0.0"], test_moshaf_ids=[], export_chunk_size=32)

def test_roundtrip(tmp_path):
    c = ZipformerTrainConfig(train_moshaf_ids=["0.0"], test_moshaf_ids=["1.0"])
    c.to_yaml(tmp_path / "c.yml")
    assert ZipformerTrainConfig.from_yaml(tmp_path / "c.yml") == c

def test_fault_tolerance_defaults():
    c = ZipformerTrainConfig.from_yaml("configs/train/zipformer/train_config.yml")
    assert c.save_every_n_batches == 2000 and c.keep_last_k == 3
    assert c.rclone_checkpoint_path.startswith("gdrive:")
    assert ZipformerTrainConfig(train_moshaf_ids=["0.0"], test_moshaf_ids=[]).rclone_checkpoint_path is None

def test_output_downsampling_default_and_choices():
    assert ZipformerTrainConfig.from_yaml("configs/train/zipformer/train_config.yml").output_downsampling_factor == 1
    ZipformerTrainConfig(train_moshaf_ids=["0.0"], test_moshaf_ids=[], output_downsampling_factor=2)

@pytest.mark.parametrize("bad", [{"save_every_n_batches": 0}, {"rclone_checkpoint_path": "/no/remote"}, {"output_downsampling_factor": 4}])
def test_fault_tolerance_validation(bad):
    with pytest.raises(ValueError):
        ZipformerTrainConfig(train_moshaf_ids=["0.0"], test_moshaf_ids=[], **bad)
```

- [ ] **Step 2: Run → FAIL**
- [ ] **Step 3: Implement `config.py` and write `train_config.yml`** with the same 22 train / 3 test moshaf ids as `configs/train/streaming/train_config.yml` and all defaults spelled out with short comments.
- [ ] **Step 4: Run → PASS, commit** `add: zipformer training config`

---

### Task 4: Colab data preparation (HF → FLAC + Lhotse manifests on Drive)

**Files:**
- Create: `src/prepare_quran_dataset/zipformer/data_prep.py`
- Create: `src/prepare_quran_dataset/zipformer/frame_budget.py`
- Create: `prepare_zipformer_data_colab.py` (root CLI)
- Create: `notebooks/prepare_zipformer_data.ipynb` (3 cells: mount Drive + set `DRIVE_ROOT`, `pip install` repo + requirements, run CLI)
- Test: `tests/zipformer/test_data_prep.py`, `tests/zipformer/test_frame_budget.py`

**Interfaces:**
- Consumes: Task 1 `resolve_moshaf_attr`, `build_special_moshaf_attrs`; Task 2 `PhonemeTokens` (only to validate that every phoneme string encodes); Task 3 config.
- Produces (all in `data_prep.py`):
  - `write_flac(audio_bytes: bytes, out_path: Path) -> float` (decode with librosa to 16 kHz mono, write FLAC with soundfile, return duration seconds)
  - `make_cut(rec_rel_path: str, duration: float, text: str, cut_id: str, custom: dict, data_root: Path) -> MonoCut` (Lhotse `Recording.from_file(data_root/rec_rel_path)` then `recording.sources[0].source = rec_rel_path` so the stored path is relative; one `SupervisionSegment` spanning the cut, `text=text`, `custom=custom`)
  - `process_speech_config(moshaf_id: str, drive_root: Path, cfg: ZipformerTrainConfig, base_attrs, special_attrs, tokens, hf_token: str | None) -> Path` → writes `drive_root/audio/moshaf_<id>/<segment_index>.flac` and `drive_root/manifests/partial/speech_moshaf_<id>.jsonl.gz`, then `drive_root/manifests/partial/speech_moshaf_<id>.done`; returns early if `.done` exists; `custom={"moshaf_id":..., "segment_index":..., "input_type":"speech"}`; applies `max_audio_seconds` filter; **raises** if a phoneme string fails `tokens.encode`.
  - `process_noise(drive_root, cfg, hf_token) -> Path` → `audio/noise/<i:06d>.flac`, clip to `max_noise_input_seconds`, pad to `min_noise_input_seconds`; split 30/70 with `np.random.default_rng(cfg.seed)`; writes `partial/noise_aug.jsonl.gz` and `partial/noise_input.jsonl.gz` (text `""`, `input_type="silence"`), plus `noise.done`.
  - `merge_manifests(drive_root: Path, cfg) -> dict` → reads all partials, builds `cuts_train.jsonl.gz` / `cuts_dev.jsonl.gz` (speech train configs + noise_input, shuffled with seed, split by `devset_ratio`), `cuts_test.jsonl.gz` (test configs only), `cuts_noise_aug.jsonl.gz`; writes `tokens.txt` and `data_manifest.json` `{"hours": {...per split}, "num_cuts": {...}, "vocab_size": 44, "configs": [...], "frame_budget": frame_budget.report(all speech cuts)}`; returns that dict. Idempotent: always rebuilds from partials, so re-running never duplicates.
  - speech cuts additionally store `custom["num_labels"] = L` and `custom["num_repeats"] = R` (see `frame_budget`).
  - `check_free_space(path: Path, min_gb: float = 70) -> None` (raises `RuntimeError` with the measured free GB).
  - `run(cfg_path, drive_root, hf_token, only: list[str] | None)` orchestrates: check space → each train+test moshaf → noise → merge.
- Produces in `frame_budget.py` (pure Python, no torch; used by Task 4 and Task 7):
  - `count_repeats(text: str) -> int` (number of `i` with `text[i] == text[i+1]`).
  - `required_frames(text: str) -> int` = `len(text) + count_repeats(text)`.
  - `fbank_frames(duration: float, speed: float = 1.0) -> int` = Lhotse's `compute_num_frames(duration / speed, frame_shift=0.01, sampling_rate=16000)`.
  - `output_frames(t_fbank: int, output_downsampling_factor: int) -> int` = `(t_fbank - 7) // 2` for 1, `((t_fbank - 7) // 2 + 1) // 2` for 2 (icefall `Conv2dSubsampling` + `Zipformer2` rounding; must agree with Task 6's real model, which its test checks).
  - `ratio(cut_duration, text, factor, speed) -> float` = `output_frames(...) / required_frames(text)`; `inf` for empty text.
  - `report(cuts: Iterable[MonoCut]) -> dict` → `{"f1_s1.0": {...}, "f1_s1.1": {...}, "f2_s1.0": {...}, "f2_s1.1": {...}}`, each `{"min", "p1", "p5", "p50", "num_below_1", "ids_below_1" (first 50)}`; silence cuts skipped.
  - `filter_feasible(cuts: CutSet, factor: int, speed: float) -> tuple[CutSet, list[str]]` (keeps speech cuts with ratio ≥ 1 and all silence cuts; returns dropped ids).

- [ ] **Step 1: Write the failing tests** (no network: build fake HF rows with a 1 s sine encoded as WAV bytes via `soundfile`; monkeypatch the loader function `data_prep.load_hf_rows(name, config, token) -> Iterable[dict]` to return them)

```python
# tests/zipformer/test_data_prep.py
import io, json, numpy as np, soundfile as sf, pytest
from lhotse import CutSet
from prepare_quran_dataset.zipformer import data_prep as dp
from prepare_quran_dataset.zipformer.config import ZipformerTrainConfig
from prepare_quran_dataset.zipformer.tokens import PhonemeTokens

def _wav_bytes(seconds=1.0, sr=16000):
    buf = io.BytesIO(); sf.write(buf, np.sin(np.arange(int(sr*seconds))/50).astype("float32"), sr, format="WAV"); return buf.getvalue()

def _speech_row(seg, uthmani="بِسْمِ"):
    return {"audio": {"bytes": _wav_bytes(), "path": None}, "uthmani": uthmani,
            "moshaf_id": "6.0", "segment_index": seg}

@pytest.fixture
def cfg(): return ZipformerTrainConfig(train_moshaf_ids=["6.0"], test_moshaf_ids=["19.0"], min_noise_input_seconds=1.0, max_noise_input_seconds=2.0)

def test_write_flac_returns_duration(tmp_path):
    d = dp.write_flac(_wav_bytes(1.5), tmp_path / "a.flac")
    assert abs(d - 1.5) < 0.01 and (tmp_path / "a.flac").exists()

def test_cut_paths_are_relative(tmp_path):
    dp.write_flac(_wav_bytes(), tmp_path / "audio/x.flac")
    cut = dp.make_cut("audio/x.flac", 1.0, "ب", "c1", {"input_type": "speech"}, tmp_path)
    assert cut.recording.sources[0].source == "audio/x.flac"
    assert cut.supervisions[0].text == "ب"

def test_speech_config_is_idempotent_and_skips_when_done(tmp_path, cfg, monkeypatch, base_attrs, special_attrs):
    rows = [_speech_row("001.0001"), _speech_row("010.0158")]
    calls = {"n": 0}
    def fake_rows(name, config, token): calls["n"] += 1; return rows
    monkeypatch.setattr(dp, "load_hf_rows", fake_rows)
    tokens = PhonemeTokens.from_vocab_json("vocab_streaming/vocab.json")
    dp.process_speech_config("6.0", tmp_path, cfg, base_attrs, special_attrs, tokens, None)
    dp.process_speech_config("6.0", tmp_path, cfg, base_attrs, special_attrs, tokens, None)
    assert calls["n"] == 1
    cuts = CutSet.from_file(tmp_path / "manifests/partial/speech_moshaf_6.0.jsonl.gz")
    assert len(cuts) == 2 and all(c.supervisions[0].custom["input_type"] == "speech" for c in cuts)

def test_noise_clip_and_pad(tmp_path, cfg, monkeypatch):
    rows = [{"audio": {"bytes": _wav_bytes(0.3), "path": None}}, {"audio": {"bytes": _wav_bytes(5.0), "path": None}}] * 5
    monkeypatch.setattr(dp, "load_hf_rows", lambda *a: rows)
    dp.process_noise(tmp_path, cfg, None)
    cuts = CutSet.from_file(tmp_path / "manifests/partial/noise_input.jsonl.gz")
    durs = sorted(c.duration for c in cuts)
    assert durs[0] >= 1.0 and durs[-1] <= 2.0 + 1e-3
    assert all(c.supervisions[0].text == "" for c in cuts)

def test_merge_splits_and_does_not_duplicate(tmp_path, cfg, monkeypatch, base_attrs, special_attrs):
    # arrange partials via the two process_* functions above, then:
    summary1 = dp.merge_manifests(tmp_path, cfg)
    summary2 = dp.merge_manifests(tmp_path, cfg)
    assert summary1["num_cuts"] == summary2["num_cuts"]
    assert (tmp_path / "tokens.txt").exists()
    train = CutSet.from_file(tmp_path / "cuts_train.jsonl.gz"); dev = CutSet.from_file(tmp_path / "cuts_dev.jsonl.gz")
    assert set(train.ids).isdisjoint(set(dev.ids))
    assert any(c.supervisions[0].custom["input_type"] == "silence" for c in train + dev)

def test_merge_writes_frame_budget(tmp_path, cfg, monkeypatch, base_attrs, special_attrs):
    # after the arrange of test_merge_..., the manifest carries both frame rates and both speeds
    summary = dp.merge_manifests(tmp_path, cfg)
    fb = json.loads((tmp_path / "data_manifest.json").read_text())["frame_budget"]
    assert set(fb) == {"f1_s1.0", "f1_s1.1", "f2_s1.0", "f2_s1.1"}
    assert fb["f1_s1.0"]["min"] > fb["f2_s1.0"]["min"]

def test_check_free_space_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(dp.shutil, "disk_usage", lambda p: type("U", (), {"free": 1 << 30})())
    with pytest.raises(RuntimeError, match="GB"):
        dp.check_free_space(tmp_path, min_gb=70)
```

```python
# tests/zipformer/test_frame_budget.py
from lhotse import CutSet
from prepare_quran_dataset.zipformer import frame_budget as fb

def test_repeats_and_required():
    assert fb.count_repeats("رَحِۦۦۦۦم") == 3
    assert fb.required_frames("رَحِۦۦۦۦم") == 9 + 3      # 9 characters incl. diacritics
    assert fb.required_frames("") == 0

def test_output_frames_formulas():
    assert fb.output_frames(107, 1) == 50 and fb.output_frames(107, 2) == 25
    assert fb.output_frames(108, 1) == 50 and fb.output_frames(109, 2) == 26

def test_ratio_and_filter(make_cut_in_memory):
    short = make_cut_in_memory("a", duration=0.3, text="ۦۦۦۦۦۦۦۦۦۦ")   # 10 labels + 9 repeats = 19 > 10 frames (0.3 s / 1.1 at 20 ms)
    ok = make_cut_in_memory("b", duration=2.0, text="بِسمِ")
    sil = make_cut_in_memory("c", duration=1.0, text="", input_type="silence")
    kept, dropped = fb.filter_feasible(CutSet.from_cuts([short, ok, sil]), factor=1, speed=1.1)
    assert dropped == ["a"] and set(kept.ids) == {"b", "c"}
    assert fb.ratio(2.0, "بِسمِ", 1, 1.0) > fb.ratio(2.0, "بِسمِ", 2, 1.0)
```

`make_cut_in_memory(id, duration, text, input_type="speech")` builds a `MonoCut` over a `Recording` of zeros (`Recording.from_... ` in-memory, no file) with one supervision; put it in `tests/zipformer/conftest.py`.

Fixtures `base_attrs` / `special_attrs` come from the hard-coded moshaf dict of Task 1's test (put it in `tests/zipformer/conftest.py`).

- [ ] **Step 2: Run → FAIL**
- [ ] **Step 3: Implement `data_prep.py`**; `load_hf_rows` uses `datasets.load_dataset(name, name=config, split="train", token=token, streaming=True)` with `cast_column("audio", Audio(decode=False))`, and `process_*` call `datasets` cache cleanup (`shutil.rmtree` of that config's cache dir) after writing the `.done` marker.
- [ ] **Step 4: Write `prepare_zipformer_data_colab.py`** (argparse: `--config`, `--drive-root`, `--only` repeatable, `--hf-token` default `os.environ.get("HF_TOKEN")`) and the notebook.
- [ ] **Step 5: Run → PASS, commit** `add: colab data prep for zipformer (flac + lhotse manifests on drive, ctc frame budget report)`

---

### Task 5: Lhotse datamodule (on-the-fly fbank, CutMix, speed perturb)

**Files:**
- Create: `src/prepare_quran_dataset/zipformer/datamodule.py`
- Test: `tests/zipformer/test_datamodule.py`

**Interfaces:**
- Consumes: Task 3 config; manifests written by Task 4.
- Produces: `load_cuts(data_dir: Path, split: Literal["train","dev","test","noise_aug"]) -> CutSet` (applies `CutSet.with_recording_path_prefix(str(data_dir))`); `train_dataloader(cfg, data_dir, epoch: int, sampler_state: dict | None = None, batches_consumed: int = 0, cuts: CutSet | None = None) -> DataLoader` (`cuts`, when given, replaces `load_cuts(data_dir, "train")` — used by Task 7 to pass frame-budget-filtered cuts; speed perturb is applied after) (calls `sampler.set_epoch(epoch)`; when `sampler_state` is given, restores it via `restore_sampler`); `restore_sampler(sampler: DynamicBucketingSampler, state: dict, batches_consumed: int) -> None` (loads `state` with its drawn-batch counter **replaced by `batches_consumed`** before `load_state_dict`, so prefetched-but-untrained batches are re-served; find the counter field in the pinned Lhotse's `lhotse/dataset/sampling/base.py` (`diagnostics` / epoch stats) and name it in a comment; if the pinned version cannot be corrected this way, fall back to: fresh sampler with same seed, `set_epoch(epoch)`, iterate and discard `batches_consumed` batches — manifests only, no audio); `train_dataloader` also exposes the sampler as `loader.sampler`. Dataset: `K2SpeechRecognitionDataset(input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=cfg.feature_dim))), cut_transforms=[CutMix(cuts=noise_aug, p=cfg.cutmix_p, snr=cfg.cutmix_snr, preserve_id=True)], return_cuts=True)`; speed perturb as `cuts.perturb_speed(0.9) + cuts + cuts.perturb_speed(1.1)` when `cfg.speed_perturb`; `DynamicBucketingSampler(cuts, max_duration=cfg.max_duration, shuffle=True, num_buckets=30, drop_last=True, seed=cfg.seed)`; `DataLoader(ds, sampler=sampler, batch_size=None, num_workers=cfg.num_workers, persistent_workers=cfg.num_workers>0)`); `eval_dataloader(cfg, cuts: CutSet, max_duration: float | None = None) -> DataLoader` (no transforms, `shuffle=False`); `batch_to_tensors(batch) -> tuple[Tensor, Tensor, list[str], list[MonoCut]]` returning `(features (N,T,80), feature_lens (N,), texts, cuts)`.

- [ ] **Step 1: Write the failing test** (build a 4-cut data dir with Task 4's `write_flac`/`make_cut`, save as `cuts_train/dev/noise_aug.jsonl.gz`, `num_workers=0`, `max_duration=10`)

```python
# tests/zipformer/test_datamodule.py
def test_relative_paths_resolve_after_prefix(tiny_data_dir):
    cuts = dm.load_cuts(tiny_data_dir, "dev")
    assert cuts[0].load_audio().shape[-1] > 0

def test_train_batch_shapes(tiny_data_dir, cfg):
    cfg.num_workers = 0; cfg.max_duration = 10
    batch = next(iter(dm.train_dataloader(cfg, tiny_data_dir)))
    feats, lens, texts, cuts = dm.batch_to_tensors(batch)
    assert feats.ndim == 3 and feats.shape[-1] == 80
    assert lens.shape[0] == feats.shape[0] == len(texts) == len(cuts)
    assert feats.shape[1] == lens.max()

def test_speed_perturb_triples_cut_count(tiny_data_dir, cfg):
    cfg.num_workers = 0
    n = len(dm.load_cuts(tiny_data_dir, "train"))
    assert len(dm._train_cuts(cfg, tiny_data_dir)) == 3 * n

def _ids(batch): return [c.id for c in batch["supervisions"]["cut"]]

@pytest.mark.parametrize("workers", [0, 2])
def test_sampler_resume_continues_at_consumed_position(tiny_data_dir_big, cfg, workers):
    # tiny_data_dir_big: ~60 one-second cuts so an epoch has >= 10 batches at max_duration=4
    cfg.num_workers = workers; cfg.max_duration = 4
    ref = [_ids(b) for b in dm.train_dataloader(cfg, tiny_data_dir_big, epoch=1)]
    loader = dm.train_dataloader(cfg, tiny_data_dir_big, epoch=1)
    it = iter(loader); consumed = [_ids(next(it)) for _ in range(4)]
    state = loader.sampler.state_dict()          # drawn count may be > 4 with workers
    del it, loader
    resumed = dm.train_dataloader(cfg, tiny_data_dir_big, epoch=1, sampler_state=state, batches_consumed=4)
    rest = [_ids(b) for b in resumed]
    assert consumed + rest == ref                # no replay, no skip, same order
```

(`return_cuts=True` places cuts at `batch["supervisions"]["cut"]`; if the pinned Lhotse puts them elsewhere, adapt `_ids` only. `test_sampler_resume...` needs `pytest` imported at the top of the file.)

- [ ] **Step 2: Run → FAIL**
- [ ] **Step 3: Implement `datamodule.py`** (`_train_cuts` is the helper that applies speed perturb; verify in the training env that `cuts.perturb_speed(0.9)[0].load_audio()` works — if it needs sox, document the apt package in Task 11's shell script).
- [ ] **Step 4: Run → PASS, commit** `add: lhotse datamodule for zipformer training`

---

### Task 6: Model construction from config (Zipformer2 with configurable output frame rate)

**Files:**
- Create: `src/prepare_quran_dataset/zipformer/model.py`
- Test: `tests/zipformer/test_model.py` (`pytest.importorskip("k2")`)

**Interfaces:**
- Consumes: Task 0 `setup_icefall_paths`; Task 3 config.
- Produces:
  - `build_params(cfg: ZipformerTrainConfig, vocab_size: int) -> AttributeDict` (starts from recipe `train.get_params()`, overrides every model field from cfg, sets `use_transducer=False`, `use_ctc=True`, `use_attention_decoder=False`, `use_cr_ctc=False`, `causal=cfg.causal`, `feature_dim=cfg.feature_dim`, `vocab_size=vocab_size`, `blank_id=0`, `output_downsampling_factor=cfg.output_downsampling_factor`).
  - `class Zipformer2OutDS(Zipformer2)`: `__init__(self, output_downsampling_factor: int, **zipformer2_kwargs)` calls `super().__init__(output_downsampling_factor=2, **kwargs)` (icefall's forward asserts 2), stores `self.out_ds = output_downsampling_factor`, and when it is 1 sets `self.downsample_output = nn.Identity()`. Overrides `forward(x, x_lens, src_key_padding_mask=None)` and `streaming_forward(x, x_lens, states, src_key_padding_mask)`: call `super()`, and when `self.out_ds == 1` replace the returned lengths with the **input** `x_lens`. The exporter patches `encoder.__class__.forward = encoder.__class__.streaming_forward`, so the override must live on the class (it does).
  - `build_encoder(params) -> Zipformer2OutDS` with exactly the kwargs of recipe `get_encoder_model` (copy the argument list — it is ~15 keyword arguments, not file vendoring) plus `output_downsampling_factor`.
  - `build_model(cfg, vocab_size) -> AsrModel`: `AsrModel(encoder_embed=train.get_encoder_embed(params), encoder=build_encoder(params), decoder=None, joiner=None, attention_decoder=None, encoder_dim=max(_to_int_tuple(params.encoder_dim)), decoder_dim=params.decoder_dim, vocab_size=vocab_size, use_transducer=False, use_ctc=True, use_attention_decoder=False)`; then `for name, m in model.named_modules(): m.name = name` like recipe `run()`.
  - `count_params(model) -> int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/zipformer/test_model.py
import pytest, torch; pytest.importorskip("k2")
from prepare_quran_dataset.zipformer.model import build_model, count_params
from prepare_quran_dataset.zipformer.config import ZipformerTrainConfig

def test_medium_model_matches_v3_size():
    m = build_model(ZipformerTrainConfig(train_moshaf_ids=["0.0"], test_moshaf_ids=[]), vocab_size=44)
    n = count_params(m)
    assert 63_000_000 < n < 68_000_000
    linear = [l for l in m.ctc_output if isinstance(l, torch.nn.Linear)][0]  # icefall: Sequential(Dropout, Linear, LogSoftmax)
    assert m.use_ctc and not m.use_transducer and linear.out_features == 44
```

@pytest.mark.parametrize("factor", [1, 2])
def test_output_frames_match_frame_budget(tiny_cfg_factory, factor):
    from prepare_quran_dataset.zipformer import frame_budget as fb
    cfg = tiny_cfg_factory(output_downsampling_factor=factor)
    m = build_model(cfg, vocab_size=44).eval()
    for t_fbank in (107, 108, 200, 333):
        x = torch.randn(1, t_fbank, 80); lens = torch.tensor([t_fbank])
        out, out_lens = m.forward_encoder(x, lens)
        assert out.shape[1] == out_lens.item() == fb.output_frames(t_fbank, factor)

def test_icefall_still_hardcodes_factor_two():
    # canary: if icefall stops asserting/hard-coding 2, revisit Zipformer2OutDS
    import inspect; from prepare_quran_dataset.zipformer.model import Zipformer2
    assert "assert self.output_downsampling_factor == 2" in inspect.getsource(Zipformer2.forward)
```

`tiny_cfg_factory(**overrides)` lives in `tests/zipformer/conftest.py` and returns the tiny-dims config of Task 7 with overrides applied; `tiny_model` / `make_tiny_model` use it with `output_downsampling_factor=1`.

- [ ] **Step 2: Run → SKIP locally / FAIL on the GPU box**
- [ ] **Step 3: Implement `model.py`**
- [ ] **Step 4: Run on the training env → PASS, commit** `add: build zipformer2 ctc model with configurable output frame rate`

---

### Task 7A: Checkpoint store and Google Drive backup (no k2 needed)

**Files:**
- Create: `src/prepare_quran_dataset/zipformer/checkpointing.py`
- Create: `src/prepare_quran_dataset/zipformer/drive_backup.py`
- Test: `tests/zipformer/test_checkpointing.py`, `tests/zipformer/test_drive_backup.py` (rclone tests skip if `rclone` is absent; use the local remote from Global Constraints)

**Interfaces:**
- Consumes: Task 3 config (`keep_last_k`, `rclone_checkpoint_path`, `upload_timeout_seconds`).
- Produces in `checkpointing.py` (plain torch, no icefall):
  - `@dataclass CkptMeta(kind: Literal["batch","epoch"], batch_idx_train: int, epoch: int, batches_consumed_in_epoch: int, size_bytes: int, sha256: str)`; `meta_path(pt: Path) -> Path` (`<name>.meta.json`).
  - `ckpt_name(kind, batch_idx_train, epoch) -> str` → `checkpoint-{batch_idx_train}.pt` or `epoch-{epoch}.pt`.
  - `save_checkpoint_atomic(state: dict, path: Path, meta_fields: dict) -> CkptMeta` → `torch.save` to `<path>.tmp`, `os.replace`, compute sha256 + size, write meta JSON (via its own tmp + replace) **after** the `.pt`.
  - `read_meta(pt: Path) -> CkptMeta | None` (None if missing/invalid JSON).
  - `list_complete(dir: Path) -> list[tuple[Path, CkptMeta]]` (only `.pt` with a valid meta whose `size_bytes` matches the file; sorted by `batch_idx_train`; epoch checkpoints sort after a batch checkpoint with the same `batch_idx_train`).
  - `latest_complete(dir) -> tuple[Path, CkptMeta] | None`.
  - `prune(dir, keep_last_k) -> list[Path]` (per kind; deletes `.pt` and meta; also deletes stray `*.tmp` and `.pt` files without meta older than the newest complete one; returns deleted).
  - `verify_sha256(pt: Path, meta: CkptMeta) -> bool`.
  - CLI: `python -m prepare_quran_dataset.zipformer.checkpointing dump <pt>` prints a JSON summary used by Task 7's cross-process resume test (keys: `batch_idx_train`, `epoch`, `batches_consumed_in_epoch`, `lrs` (list of param-group lr), `optimizer_sha256` (sha256 over every optimizer state tensor in sorted key order), `sampler_state`, `wandb_run_id`).
- Produces in `drive_backup.py`:
  - `class Rclone(binary: str = "rclone")` with `copyto(src: str, dst: str, timeout: float) -> None`, `lsjson(remote_dir: str) -> list[dict]`, `cat(remote_file: str) -> str`, `deletefile(remote_file: str) -> None`; every call is `subprocess.run([...], check=True, capture_output=True, timeout=...)`; non-zero exit → `RcloneError` with stderr.
  - `class DriveBackup(remote_dir: str, keep_last_k: int, timeout: float, rclone: Rclone | None = None)`:
    - `submit(pt: Path) -> None` (non-blocking; one daemon worker thread; queue semantics: a pending, not-yet-started **batch** upload is replaced by a newer submit; **epoch** uploads are never dropped).
    - worker per item: `copyto(pt, remote/pt.name)` → `copyto(meta, remote/meta.name)` (meta last) → `prune_remote()`; any exception → `logging.warning(...)` and continue.
    - `prune_remote() -> list[str]` (per kind, keep newest `keep_last_k` that have a meta on remote; never touches files newer than the newest complete one).
    - `latest_remote() -> CkptMeta | None` and `download(meta: CkptMeta, local_dir: Path) -> Path` (copies `.pt` then meta; raises if `verify_sha256` fails, deleting the bad local file).
    - `download_missing_epochs(local_dir: Path) -> list[Path]` (all remote epoch checkpoints with meta not present locally).
    - `flush(timeout: float) -> bool` (waits for the queue to drain; `False` on timeout).
    - `last_error: str | None` for logging/tests.

- [ ] **Step 1: Write the failing tests**

```python
# tests/zipformer/test_checkpointing.py
import json, os, torch
from prepare_quran_dataset.zipformer import checkpointing as ck

def _save(d, kind, b, e, n=0):
    return ck.save_checkpoint_atomic({"x": torch.arange(4)}, d / ck.ckpt_name(kind, b, e),
        {"kind": kind, "batch_idx_train": b, "epoch": e, "batches_consumed_in_epoch": n})

def test_meta_written_after_pt_and_sha_matches(tmp_path):
    m = _save(tmp_path, "batch", 2000, 1, 2000)
    pt = tmp_path / "checkpoint-2000.pt"
    assert ck.read_meta(pt) == m and ck.verify_sha256(pt, m)

def test_incomplete_checkpoints_are_ignored(tmp_path):
    _save(tmp_path, "batch", 2000, 1, 2000)
    (tmp_path / "checkpoint-4000.pt").write_bytes(b"truncated")              # no meta
    _save(tmp_path, "batch", 6000, 1, 6000)
    (tmp_path / "checkpoint-6000.pt").write_bytes(b"short")                  # size mismatch
    assert ck.latest_complete(tmp_path)[1].batch_idx_train == 2000

def test_epoch_sorts_after_batch_at_same_step(tmp_path):
    _save(tmp_path, "batch", 8000, 1, 8000); _save(tmp_path, "epoch", 8000, 1, 8000)
    assert ck.latest_complete(tmp_path)[0].name == "epoch-1.pt"

def test_prune_per_kind(tmp_path):
    for b in (2000, 4000, 6000, 8000): _save(tmp_path, "batch", b, 1, b)
    for e in (1, 2, 3, 4): _save(tmp_path, "epoch", 10000 * e, e)
    ck.prune(tmp_path, keep_last_k=3)
    assert sorted(p.name for p in tmp_path.glob("checkpoint-*.pt")) == ["checkpoint-4000.pt", "checkpoint-6000.pt", "checkpoint-8000.pt"]
    assert sorted(p.name for p in tmp_path.glob("epoch-*.pt")) == ["epoch-2.pt", "epoch-3.pt", "epoch-4.pt"]
    assert not list(tmp_path.glob("*.tmp"))
```

```python
# tests/zipformer/test_drive_backup.py
import shutil, time, pytest, torch
from prepare_quran_dataset.zipformer import checkpointing as ck
from prepare_quran_dataset.zipformer.drive_backup import DriveBackup, Rclone, RcloneError
pytestmark = pytest.mark.skipif(shutil.which("rclone") is None, reason="rclone not installed")

@pytest.fixture
def remote(tmp_path, monkeypatch):
    monkeypatch.setenv("RCLONE_CONFIG_TESTLOCAL_TYPE", "local")
    (tmp_path / "remote").mkdir()
    return f"testlocal:{tmp_path / 'remote'}"

def _save(d, kind, b, e):
    return ck.save_checkpoint_atomic({"x": torch.randn(1000)}, d / ck.ckpt_name(kind, b, e),
        {"kind": kind, "batch_idx_train": b, "epoch": e, "batches_consumed_in_epoch": b})

def test_upload_then_prune_remote(tmp_path, remote):
    local = tmp_path / "local"; local.mkdir()
    bk = DriveBackup(remote, keep_last_k=2, timeout=60)
    for b in (2000, 4000, 6000):
        _save(local, "batch", b, 1); bk.submit(local / f"checkpoint-{b}.pt"); assert bk.flush(60)
    names = sorted(e["Name"] for e in Rclone().lsjson(remote))
    assert names == ["checkpoint-4000.meta.json", "checkpoint-4000.pt", "checkpoint-6000.meta.json", "checkpoint-6000.pt"]

def test_download_latest_into_empty_dir_and_verify(tmp_path, remote):
    src = tmp_path / "src"; src.mkdir(); dst = tmp_path / "dst"; dst.mkdir()
    bk = DriveBackup(remote, keep_last_k=3, timeout=60)
    _save(src, "batch", 2000, 1); _save(src, "epoch", 4000, 1)
    for p in ("checkpoint-2000.pt", "epoch-1.pt"): bk.submit(src / p)
    assert bk.flush(60)
    m = bk.latest_remote(); assert m.kind == "epoch" and m.batch_idx_train == 4000
    p = bk.download(m, dst); assert ck.verify_sha256(p, ck.read_meta(p))

def test_remote_pt_without_meta_is_not_latest(tmp_path, remote):
    src = tmp_path / "src"; src.mkdir()
    bk = DriveBackup(remote, keep_last_k=3, timeout=60)
    _save(src, "batch", 2000, 1); bk.submit(src / "checkpoint-2000.pt"); assert bk.flush(60)
    _save(src, "batch", 4000, 1)
    Rclone().copyto(str(src / "checkpoint-4000.pt"), f"{remote}/checkpoint-4000.pt", 60)  # simulate death before meta
    assert bk.latest_remote().batch_idx_train == 2000

def test_upload_failure_is_a_warning(tmp_path, caplog):
    bk = DriveBackup("nonexistent-remote:x", keep_last_k=3, timeout=10)
    _save(tmp_path, "batch", 2000, 1); bk.submit(tmp_path / "checkpoint-2000.pt")
    assert bk.flush(30) and bk.last_error is not None
    assert any(r.levelname == "WARNING" for r in caplog.records)

def test_newer_batch_replaces_pending_batch_but_not_epoch(tmp_path, remote, monkeypatch):
    # Block the worker on the first upload, then submit epoch + two batch checkpoints.
    ...  # monkeypatch Rclone.copyto with a version that waits on a threading.Event for the first call
    # assert uploaded set == {first, epoch-1, newest batch}; the middle batch was dropped
```

(Finish the last test with a `threading.Event` gate; it is the only place the queue-replacement rule is pinned.)

- [ ] **Step 2: Run** `pytest tests/zipformer/test_checkpointing.py tests/zipformer/test_drive_backup.py -v` → FAIL
- [ ] **Step 3: Implement `checkpointing.py` and `drive_backup.py`**
- [ ] **Step 4: Run → PASS (rclone tests run on this Mac: rclone 1.72 is installed), commit** `add: atomic checkpoint store and rclone google drive backup`

---

### Task 7: Training loop driver (ScaledAdam + Eden, mid-epoch checkpoints, resume, averaging, W&B)

**Files:**
- Create: `src/prepare_quran_dataset/zipformer/train_loop.py`
- Test: `tests/zipformer/test_train_loop.py` (`importorskip("k2")`, tiny model override via `cfg` dims, CPU)
- Test: `tests/zipformer/test_resume.py` (`importorskip("k2")`, CPU, uses `tiny_data_dir_big` from Task 5's conftest and the local rclone remote)

**Interfaces:**
- Consumes: Task 5 `train_dataloader(cfg, data_dir, epoch, sampler_state, batches_consumed)`, `eval_dataloader`, `batch_to_tensors`; Task 6 `build_model`; Task 2 `PhonemeTokens`; Task 7A `save_checkpoint_atomic`, `ckpt_name`, `latest_complete`, `list_complete`, `prune`, `DriveBackup`; Task 8 `evaluate_split` (dev PER each epoch — stub-able).
- Produces:
  - `encode_targets(texts: list[str], tokens: PhonemeTokens, device) -> tuple[Tensor, Tensor]` → `(targets 1-D concatenated, target_lengths (N,))`; empty text → length 0.
  - `compute_ctc_loss(model, feats, feat_lens, targets, target_lengths) -> tuple[Tensor, Tensor]` → `(loss_sum, num_frames)` using `model.forward_encoder` then `model.forward_ctc`; loss normalised by `num_frames` for logging only, like icefall.
  - `build_optimizer(model, cfg) -> tuple[ScaledAdam, Eden]` (`get_parameter_groups_with_lrs(model, lr=cfg.base_lr, include_names=True)`, `clipping_scale=2.0`; `Eden(opt, cfg.lr_batches, cfg.lr_epochs, warmup_batches=cfg.warmup_batches)`).
  - `@dataclass TrainState(epoch: int = 1, batch_idx_train: int = 0, batches_consumed_in_epoch: int = 0, sampler_state: dict | None = None, wandb_run_id: str | None = None)`.
  - `make_backup(cfg) -> DriveBackup | None` (`None` when `cfg.rclone_checkpoint_path` is unset).
  - `make_checkpoint(model, opt, sched, st: TrainState, sampler, cfg) -> dict` (`sampler` may be `None` → stored as `None`; keys: `model`, `optimizer`, `scheduler`, `batch_idx_train`, `epoch`, `batches_consumed_in_epoch`, `sampler`, `wandb_run_id`, `config`).
  - `save_and_backup(ckpt: dict, kind: Literal["batch","epoch"], cfg, backup: DriveBackup | None) -> Path` → `save_checkpoint_atomic` to `output_dir/ckpt_name(...)`, `prune(output_dir, cfg.keep_last_k)`, `backup.submit(path)` if backup.
  - `restore(model, opt, sched, ckpt: dict) -> TrainState` (loads the three state dicts; calls recipe `set_batch_count(model, batch_idx_train)` — use the same adjusted count the recipe's `train_one_epoch` uses (`get_adjusted_batch_count`); an `epoch` checkpoint resumes at `epoch+1`, `batches_consumed_in_epoch=0`, `sampler_state=None`; raises `IncompatibleCheckpoint` if `ckpt["config"]["output_downsampling_factor"]` differs from the current config, before loading any state dict).
  - `resolve_resume(cfg, backup: DriveBackup | None) -> Path | None` (newest of `latest_complete(output_dir)` and `backup.latest_remote()` by `(batch_idx_train, kind=="epoch")`; downloads the Drive one if newer; then `backup.download_missing_epochs(output_dir)`; logs which source won; remote listing failure → warning, use local).
  - `train_one_epoch(model, opt, sched, loader, st: TrainState, cfg, tokens, device, logger, backup, stop_flag) -> TrainState` (bf16 autocast when `cfg.bf16`; per batch: `set_batch_count`, forward/backward, `opt.step()`, `sched.step_batch(st.batch_idx_train)`, `st.batch_idx_train += 1`, `st.batches_consumed_in_epoch += 1`; every `cfg.save_every_n_batches` global batches → `save_and_backup(..., "batch")`; when `stop_flag.is_set()` → save a batch checkpoint, `backup.flush(cfg.shutdown_flush_seconds)`, raise `TrainingInterrupted`).
  - `install_signal_handlers(stop_flag: threading.Event) -> None` (SIGTERM, SIGINT → `stop_flag.set()`; second SIGINT → default behaviour).
  - `average_last_epochs(output_dir, k) -> Path` → `output_dir/averaged-{k}.pt` from the newest `k` complete `epoch-*.pt` (`icefall.checkpoint.average_checkpoints`, key `model`).
  - `run_training(cfg, data_dir, tokens, device) -> Path`: `backup = make_backup(cfg)`; `resolve_resume`; **frame budget**: `frame_budget.filter_feasible` on train and dev cuts with `cfg.output_downsampling_factor` and speed `1.1 if cfg.speed_perturb else 1.0`, write `output_dir/dropped_cuts.json` (`{"factor", "speed", "train": [...ids], "dev": [...ids]}`), log the count, raise `FrameBudgetError` if dropped speech cuts / speech cuts > `cfg.max_dropped_fraction`; the filtered CutSets are what `train_dataloader`/dev eval receive (pass them in; Task 5's `train_dataloader` gains an optional `cuts: CutSet | None = None` argument that replaces `load_cuts(...,"train")` when given); build model/opt/sched; `restore` if found; per epoch: `sched.step_epoch(epoch - 1)`, `train_dataloader(cfg, data_dir, epoch, st.sampler_state, st.batches_consumed_in_epoch)`, `train_one_epoch`, save `epoch` checkpoint, dev eval via Task 8 (results appended to `output_dir/dev_metrics.jsonl`; on resume, run dev eval for any finished epoch missing there); after the last epoch `backup.flush(cfg.upload_timeout_seconds)` and return `average_last_epochs(...)`. If all `num_epochs` are already complete on resume, skip straight to averaging.
  - `Logger` protocol with `log(dict, step)`; implementations `TensorBoardLogger(output_dir/"tb")` and optional `WandbLogger(project, run_id)` (`wandb.init(id=run_id, resume="allow")`); `make_logger(cfg, run_id: str | None) -> tuple[Logger, str | None]` picks W&B only when `WANDB_API_KEY` is set, else prints a warning; returns the run id to store in `TrainState`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/zipformer/test_train_loop.py
import pytest, torch; pytest.importorskip("k2")
from prepare_quran_dataset.zipformer import train_loop as tl

def test_encode_targets_handles_empty(tokens):
    tgt, lens = tl.encode_targets(["ب", "", "بب"], tokens, torch.device("cpu"))
    assert lens.tolist() == [1, 0, 2] and tgt.numel() == 3

def test_ctc_loss_finite_with_silence_cut(tiny_model, tokens):
    feats = torch.randn(2, 120, 80); lens = torch.tensor([120, 100])
    tgt, tl_ = tl.encode_targets(["ب", ""], tokens, torch.device("cpu"))
    loss, n = tl.compute_ctc_loss(tiny_model, feats, lens, tgt, tl_)
    assert torch.isfinite(loss) and n > 0

def test_restore_sets_zipformer_batch_count(tiny_model, make_tiny_model, cfg):
    opt, sched = tl.build_optimizer(tiny_model, cfg)
    st = tl.TrainState(epoch=2, batch_idx_train=5000, batches_consumed_in_epoch=10)
    ckpt = tl.make_checkpoint(tiny_model, opt, sched, st, sampler=None, cfg=cfg)
    m2 = make_tiny_model(); o2, s2 = tl.build_optimizer(m2, cfg)
    st2 = tl.restore(m2, o2, s2, ckpt)
    assert st2.batch_idx_train == 5000
    counts = {mod.batch_count for mod in m2.modules() if hasattr(mod, "batch_count")}
    assert counts and all(c > 0 for c in counts)

def test_restore_refuses_other_frame_rate(tiny_model, tiny_cfg_factory):
    cfg2 = tiny_cfg_factory(output_downsampling_factor=2)
    opt, sched = tl.build_optimizer(tiny_model, cfg2)
    ck = tl.make_checkpoint(tiny_model, opt, sched, tl.TrainState(), sampler=None, cfg=tiny_cfg_factory(output_downsampling_factor=1))
    with pytest.raises(tl.IncompatibleCheckpoint):
        tl.restore(build_model(cfg2, 44), opt, sched, ck)

def test_frame_budget_abort(tiny_cfg, tiny_data_dir_infeasible, tokens):
    # tiny_data_dir_infeasible: 10 speech cuts, 2 of them 0.3 s with 10-char repeated labels
    tiny_cfg.max_dropped_fraction = 0.1
    with pytest.raises(tl.FrameBudgetError):
        tl.run_training(tiny_cfg, tiny_data_dir_infeasible, tokens, torch.device("cpu"))

def test_epoch_checkpoint_resumes_next_epoch(tiny_model, cfg):
    opt, sched = tl.build_optimizer(tiny_model, cfg)
    st = tl.TrainState(epoch=3, batch_idx_train=9000, batches_consumed_in_epoch=3000)
    ck = tl.make_checkpoint(tiny_model, opt, sched, st, sampler=None, cfg=cfg); ck["kind"] = "epoch"
    st2 = tl.restore(tiny_model, opt, sched, ck)
    assert (st2.epoch, st2.batches_consumed_in_epoch, st2.sampler_state) == (4, 0, None)
```

```python
# tests/zipformer/test_resume.py  — the "strong resume" test the owner asked for
import json, subprocess, sys, shutil, pytest, torch
pytest.importorskip("k2")
from prepare_quran_dataset.zipformer import train_loop as tl, checkpointing as ck

def _dump(pt):  # fresh interpreter = "new process"
    out = subprocess.run([sys.executable, "-m", "prepare_quran_dataset.zipformer.checkpointing", "dump", str(pt)],
                         check=True, capture_output=True, text=True).stdout
    return json.loads(out)

def test_state_identical_after_reload_in_new_process(tmp_path, tiny_cfg, tiny_data_dir_big, tokens):
    # tiny_cfg: tiny dims, save_every_n_batches=3, num_workers=2, max_duration=4, num_epochs=1, cpu
    tiny_cfg.output_dir = str(tmp_path / "out")
    snap = tl.run_steps_for_test(tiny_cfg, tiny_data_dir_big, tokens, n_batches=5)   # trains 5 batches, saves at 3, returns in-memory snapshot at batch 3
    pt = ck.latest_complete(tmp_path / "out")[0]
    d = _dump(pt)
    assert d["batch_idx_train"] == snap["batch_idx_train"] == 3
    assert d["batches_consumed_in_epoch"] == 3
    assert d["lrs"] == pytest.approx(snap["lrs"], rel=0, abs=0)
    assert d["optimizer_sha256"] == snap["optimizer_sha256"]
    assert d["sampler_state"] == snap["sampler_state"]

def test_resumed_run_sees_exact_next_batches(tmp_path, tiny_cfg, tiny_data_dir_big, tokens):
    tiny_cfg.output_dir = str(tmp_path / "out")
    ref = tl.run_steps_for_test(tiny_cfg, tiny_data_dir_big, tokens, n_batches=6)["cut_ids_per_batch"]
    shutil.rmtree(tmp_path / "out")
    tl.run_steps_for_test(tiny_cfg, tiny_data_dir_big, tokens, n_batches=3)          # "box dies" after checkpoint at 3
    saved_lrs = _dump(ck.latest_complete(tmp_path / "out")[0])["lrs"]
    resumed = subprocess.run([sys.executable, "-m", "prepare_quran_dataset.zipformer.train_loop",
                              "--test-resume", "--config", str(tmp_path / "out/config.yml"),
                              "--data-dir", str(tiny_data_dir_big), "--n-batches", "3"],
                             check=True, capture_output=True, text=True).stdout
    r = json.loads(resumed.splitlines()[-1])
    assert r["start_batch_idx_train"] == 3
    assert r["cut_ids_per_batch"] == ref[3:6]
    assert r["first_lrs"] == saved_lrs           # LR continues, not restarted by warmup

def test_new_box_pulls_newest_checkpoint_from_drive(tmp_path, tiny_cfg, tiny_data_dir_big, tokens, local_remote):
    # local_remote: "testlocal:<tmp>/drive" via RCLONE_CONFIG_TESTLOCAL_TYPE=local
    tiny_cfg.output_dir = str(tmp_path / "boxA"); tiny_cfg.rclone_checkpoint_path = local_remote
    tl.run_steps_for_test(tiny_cfg, tiny_data_dir_big, tokens, n_batches=6)          # saves at 3 and 6, uploads both
    tiny_cfg.output_dir = str(tmp_path / "boxB")                                    # empty dir = new box
    pt = tl.resolve_resume(tiny_cfg, tl.make_backup(tiny_cfg))
    assert pt is not None and ck.read_meta(pt).batch_idx_train == 6

def test_local_newer_than_drive_wins(tmp_path, tiny_cfg, tiny_data_dir_big, tokens, local_remote):
    # upload only the batch-3 checkpoint, keep batch-6 local-only; resolve_resume must return the local batch-6
    ...
```

`run_steps_for_test(cfg, data_dir, tokens, n_batches) -> dict` is a small public helper in `train_loop.py` (runs `run_training`'s inner loop for exactly `n_batches`, honouring resume, flushing backup) returning `{"batch_idx_train", "lrs", "optimizer_sha256", "sampler_state", "cut_ids_per_batch"}` as of the last saved checkpoint plus all batches seen; `train_loop.py` also gets a `__main__` with `--test-resume` that calls it and prints the JSON as its last line. Keep both tiny; they exist to make the resume contract testable across processes.

`tiny_model` fixture: `build_model` with `num_encoder_layers="1,1"`, `downsampling_factor="1,2"`, `feedforward_dim="64,64"`, `num_heads="2,2"`, `encoder_dim="32,32"`, `encoder_unmasked_dim="32,32"`, `cnn_module_kernel="7,7"`, `chunk_size="8"`, `left_context_frames="32"` (config validator must allow 2-stack dims; `export_chunk_size` fixture = 8). `tiny_cfg` = the same dims plus the values in the comment above. Put `tiny_model`, `make_tiny_model` (factory returning a fresh tiny model), `tiny_cfg`, `tokens`, `local_remote` in `tests/zipformer/conftest.py`. The `--test-resume` JSON also carries `start_batch_idx_train` and `first_lrs` (param-group lrs right after `restore`, before the first step).

- [ ] **Step 2: Run → FAIL**
- [ ] **Step 3: Implement `train_loop.py`**
- [ ] **Step 4: Run → PASS on the k2 env (the rclone/Drive tests also need the `rclone` binary), commit** `add: zipformer ctc training loop with mid-epoch checkpoints, drive-backed resume and averaging`

---

### Task 8: Greedy CTC decoding, PER, silence PER, split evaluation

**Files:**
- Create: `src/prepare_quran_dataset/zipformer/decode.py`
- Test: `tests/zipformer/test_decode.py` (pure torch/numpy parts run everywhere)

**Interfaces:**
- Consumes: Task 5 `eval_dataloader`, `batch_to_tensors`; Task 2 tokens.
- Produces: `frame_shift_seconds(output_downsampling_factor: int) -> float` (= `0.02 * factor`; the only place the output frame shift is defined); `greedy_ctc(log_probs: Tensor (N,T,V), lens: Tensor, blank_id: int) -> list[list[int]]` (argmax, collapse repeats, drop blank); `greedy_ctc_with_frames(log_probs, lens, blank_id) -> list[list[tuple[int, int]]]` (`(token, first_frame_index)` per emitted token; `greedy_ctc` is its projection); `token_times(frames: list[tuple[int, int]], frame_shift_s: float) -> list[tuple[int, float]]`; `per(hyps: list[list[int]], refs: list[list[int]]) -> float` (Levenshtein distance summed / total ref length, each distance capped at ref length like `compute_per_level_stats`); `silence_frame_error(log_probs, lens, blank_id) -> tuple[int,int]` (non-blank frames, total frames; counted on model output frames, i.e. 20 ms frames with factor 1 — comparable with muaalem's `silence_per`); `set_streaming_context(model, chunk_size: int, left_context: int) -> None` (sets `model.encoder.chunk_size=(chunk,)`, `left_context_frames=(left,)`); `evaluate_split(model, loader, tokens, device, chunk_size, left_context, max_cuts: int | None) -> dict` → `{"per_phonemes": float, "silence_per": float | None, "num_speech": int, "num_silence": int}` where speech/silence is decided by `cut.supervisions[0].custom["input_type"]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/zipformer/test_decode.py
import torch
from prepare_quran_dataset.zipformer import decode as dc

def test_greedy_collapses_and_drops_blank():
    lp = torch.full((1, 6, 3), -10.0)
    for t, c in enumerate([1, 1, 0, 2, 2, 0]): lp[0, t, c] = 0.0
    assert dc.greedy_ctc(lp, torch.tensor([6]), blank_id=0) == [[1, 2]]
    assert dc.greedy_ctc(lp, torch.tensor([2]), blank_id=0) == [[1]]

def test_per_capped_and_averaged():
    assert dc.per([[1, 2]], [[1, 2]]) == 0.0
    assert dc.per([[1, 2, 3, 4, 5]], [[9]]) == 1.0          # capped at ref len
    assert abs(dc.per([[1], [1, 2]], [[1, 2], [1, 2]]) - 0.25) < 1e-9

def test_frame_shift_and_timestamps():
    assert dc.frame_shift_seconds(1) == 0.02 and dc.frame_shift_seconds(2) == 0.04
    lp = torch.full((1, 6, 3), -10.0)
    for t, c in enumerate([0, 1, 1, 0, 2, 0]): lp[0, t, c] = 0.0
    fr = dc.greedy_ctc_with_frames(lp, torch.tensor([6]), 0)[0]
    assert fr == [(1, 1), (2, 4)]
    assert dc.token_times(fr, dc.frame_shift_seconds(1)) == [(1, 0.02), (2, 0.08)]

def test_silence_frame_error():
    lp = torch.full((1, 4, 2), -10.0); lp[0, :, 0] = 0.0; lp[0, 2, 1] = 1.0
    assert dc.silence_frame_error(lp, torch.tensor([4]), 0) == (1, 4)
```

- [ ] **Step 2: Run → FAIL**
- [ ] **Step 3: Implement `decode.py`**
- [ ] **Step 4: Run → PASS, commit** `add: greedy ctc decoding and per evaluation for zipformer`

---

### Task 9: ONNX streaming export + onnxruntime streaming decoder

**Files:**
- Create: `src/prepare_quran_dataset/zipformer/export_onnx.py`
- Test: `tests/zipformer/test_export_onnx.py` (`importorskip("k2")`, `importorskip("onnxruntime")`, tiny model)

**Interfaces:**
- Consumes: Task 0 `load_recipe_module("export-onnx-streaming-ctc.py")`, `setup_icefall_paths`; Task 6 model; Task 2 tokens; Task 8 `greedy_ctc`, `set_streaming_context`.
- Produces:
  - `export_streaming_onnx(model, out_dir: Path, tokens: PhonemeTokens, chunk_size: int, left_context: int, output_downsampling_factor: int, opset: int = 13) -> dict[str, Path]` → `{"fp32": out_dir/"model.onnx", "int8": out_dir/"model.int8.onnx", "tokens": out_dir/"tokens.txt"}`; steps: `set_streaming_context`, `convert_scaled_to_non_scaled(model, inplace=True)` (from recipe `scaling_converter`), wrap `OnnxModel(encoder=model.encoder, encoder_embed=model.encoder_embed, ctc_output=model.ctc_output)` (icefall's `ctc_output` already ends in `LogSoftmax`; do **not** add another), `export_streaming_ctc_model_onnx(onnx_model, str(fp32), opset_version=opset)`, then add our extra metadata to the fp32 file with the recipe's `add_meta_data`: `output_downsampling_factor`, `subsampling_factor` (`2 * factor`), `output_frame_shift_ms` (`20 * factor`), `vocab_size`, `blank_id`; `quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"])` (verify the int8 file keeps all metadata; re-add if not), write tokens. icefall's exporter computes `T` and `decode_chunk_len` from the input side only, so those keys are correct for both factors without changes.
  - `class OnnxStreamingDecoder` with `__init__(onnx_path: Path, blank_id: int = 0)`, `.chunk_frames: int` (from metadata `T`), `.decode_chunk_len: int`, `.frame_shift_s: float` (from `output_frame_shift_ms`; raise if the key is missing — never assume 40 ms), `.out_frames_per_chunk: int` (`decode_chunk_len // 2 // output_downsampling_factor`; asserted against the first `log_probs` returned), `decode_with_times(wav, sr=16000) -> list[tuple[int, float]]` (Task 8 `token_times` with frame indices offset across chunks), `decode(wav: np.ndarray, sr: int = 16000) -> list[int]` (kaldi fbank 80-dim via `kaldifeat` or `torchaudio.compliance.kaldi.fbank(dither=0, snip_edges=False)`; feed `T` frames per call with hop `decode_chunk_len`, carry all state outputs to the next inputs, zero-init from `get_init_states`-equivalent shapes read from the session inputs; greedy-collapse across chunks, i.e. carry the previous non-blank token between chunks). Mirrors icefall `onnx_pretrained-streaming-ctc.py`.
  - `assert_onnx_matches_torch(model, decoder: OnnxStreamingDecoder, wavs: list[np.ndarray], tokens, chunk_size, left_context) -> None` → raises `AssertionError` with both sequences if any differ.

- [ ] **Step 1: Write the failing test**

```python
# tests/zipformer/test_export_onnx.py
import pytest, numpy as np, torch
pytest.importorskip("k2"); pytest.importorskip("onnxruntime")
from prepare_quran_dataset.zipformer import export_onnx as ex

@pytest.mark.parametrize("factor", [1, 2])
def test_export_and_streaming_decode_matches_torch(tmp_path, tiny_cfg_factory, tokens, factor):
    from prepare_quran_dataset.zipformer.model import build_model
    model = build_model(tiny_cfg_factory(output_downsampling_factor=factor), 44).eval()
    paths = ex.export_streaming_onnx(model, tmp_path, tokens, chunk_size=8, left_context=32, output_downsampling_factor=factor)
    assert all(p.exists() for p in paths.values())
    dec = ex.OnnxStreamingDecoder(paths["fp32"], blank_id=0)
    assert dec.out_frames_per_chunk == 8 // factor
    assert dec.frame_shift_s == pytest.approx(0.02 * factor)
    wav = np.random.randn(16000 * 2).astype("float32") * 0.1
    ex.assert_onnx_matches_torch(model, dec, [wav], tokens, chunk_size=8, left_context=32)
    assert isinstance(dec.decode(wav), list)

def test_metadata_survives_int8(tmp_path, tiny_model, tokens):
    import onnxruntime as ort
    paths = ex.export_streaming_onnx(tiny_model, tmp_path, tokens, chunk_size=8, left_context=32, output_downsampling_factor=1)
    for k in ("fp32", "int8"):
        md = ort.InferenceSession(str(paths[k]), providers=["CPUExecutionProvider"]).get_modelmeta().custom_metadata_map
        assert md["model_type"] == "zipformer2" and md["output_frame_shift_ms"] == "20" and md["subsampling_factor"] == "2"
```

(Untrained tiny model mostly emits blanks; the match assertion still exercises the state plumbing.)

- [ ] **Step 2: Run → FAIL**
- [ ] **Step 3: Implement `export_onnx.py`**
- [ ] **Step 4: Run → PASS, commit** `add: sherpa-onnx streaming ctc export and onnxruntime streaming decoder`

---

### Task 10: qdat_bench via the ONNX streaming decoder

**Files:**
- Create: `src/prepare_quran_dataset/zipformer/qdat.py`
- Test: `tests/zipformer/test_qdat.py` (schema test only, no network; fake decoder)

**Interfaces:**
- Consumes: Task 9 `OnnxStreamingDecoder`; Task 2 tokens; `qdat_bench.audio_utils.decode_audio`, `qdat_bench.eval_results.eval_qdat_bench`; `MultiLevelTokenizer("./vocab_streaming")`.
- Produces: `prediction_record(cut_id: str, ids: list[int], tokens) -> dict` → `{"id": cut_id, "levels_labels": {"phonemes": ids}, "level_to_scripts": {"phonemes": tokens.decode(ids)}}`; `write_predictions(records, path) -> Path` (jsonl, `ensure_ascii=False`); `run_qdat_bench(decoder, tokens, output_dir: Path, rerun: bool = False) -> dict` (same skip logic as `train_streaming.py` lines 1119–1146: results exist → skip, predictions exist → eval only, else decode `obadx/qdat_bench` and eval; writes `qdat_bench_predictions.jsonl` and `qdat_bench_test_results.json`). **Builder check inside this task:** run `eval_qdat_bench` once with a phonemes-only prediction set; if it raises on missing sifat levels, catch, log which qdat metrics are unavailable, and write only `speech_metrics`-style phoneme results. Record the finding in the module docstring.

- [ ] **Step 1: Write the failing test**

```python
# tests/zipformer/test_qdat.py
import json
from prepare_quran_dataset.zipformer import qdat

def test_prediction_record_schema(tokens):
    ids = tokens.encode("ب")
    r = qdat.prediction_record("x", ids, tokens)
    assert r == {"id": "x", "levels_labels": {"phonemes": ids}, "level_to_scripts": {"phonemes": "ب"}}

def test_write_predictions_jsonl(tmp_path, tokens):
    p = qdat.write_predictions([qdat.prediction_record("a", [], tokens)], tmp_path / "p.jsonl")
    assert json.loads(p.read_text(encoding="utf-8").splitlines()[0])["id"] == "a"
```

- [ ] **Step 2: Run → FAIL**; **Step 3: Implement**; **Step 4: Run → PASS**
- [ ] **Step 5: Commit** `add: qdat_bench evaluation through onnx streaming decoder`

---

### Task 11: Stage runner `train_zipformer.py`, `train_zipformer.sh`, README section

**Files:**
- Create: `train_zipformer.py` (root)
- Create: `train_zipformer.sh` (root)
- Modify: `README.md` (new section "Zipformer2 CTC streaming model")
- Test: `tests/zipformer/test_cli.py` (argparse + stage selection only)

**Interfaces:**
- Consumes: everything above.
- Produces: `parse_args(argv) -> Namespace` with `--config` (default `configs/train/zipformer/train_config.yml`), `--stage {prepare,train,eval,export,all}` default `all`, `--push-to-hub`, `--rerun-testset`, `--rerun-qdat-bench`; `stages_to_run(stage: str) -> list[str]` (`all` → `["prepare","train","eval","export"]`); `main(argv=None)`:
  - `prepare`: `rclone sync <cfg.rclone_remote_path> <cfg.data_dir> --progress` via `subprocess.run(check=True)`; fail fast with a clear message if `rclone` is missing or no remote is configured (`rclone listremotes` empty).
  - `train`: `run_training` → averaged `.pt`. Resume is automatic (Task 7 `resolve_resume`); `main` logs one line before training: `resume: source=<local|drive|none> checkpoint=<name> epoch=<e> batch_idx_train=<b>`. If `rclone_checkpoint_path` is set but `rclone lsjson` on it fails, log a warning and continue with local-only checkpoints (do not abort a run that could otherwise train).
  - `TrainingInterrupted` from Task 7 → exit code 143 after the flush, so a supervisor/restart loop can tell a graceful stop from a crash.
  - `eval`: test PER → `output_dir/test_results.json` (skip unless `--rerun-testset`).
  - `export`: `export_streaming_onnx(..., output_downsampling_factor=cfg.output_downsampling_factor)` to `output_dir/export/`, `assert_onnx_matches_torch` on 5 test cuts, then `run_qdat_bench` with the fp32 ONNX decoder.
  - `--push-to-hub`: `HfApi().upload_folder(output_dir/"export")` + averaged `.pt`, config YAML, the two results JSON to `cfg.hub_model_id`.
- `train_zipformer.sh`: mirrors `train_and_export_tflite.sh` sections: (1) `uv venv .venv-zipformer --python 3.14` (fallback `3.12` with `PYTHONPATH=src` if the k2 wheel for cp314 is unavailable), install torch from the pinned index, k2 wheel URL from `K2_WHEEL_URL` env with a documented default, `requirements-zipformer.txt`, `pip install -e .` (3.14) ; `git submodule update --init external/icefall`; (2) install rclone (`curl https://rclone.org/install.sh | bash`) and require `rclone.conf` at `~/.config/rclone/rclone.conf` or `RCLONE_CONFIG` env, abort otherwise; then `rclone lsd <remote>:` as a connectivity check and print the Drive checkpoint folder contents (`rclone lsl <rclone_checkpoint_path>`, read from the YAML with a one-line python) so the operator sees what will be resumed; (3) `python train_zipformer.py --config ... --stage all "$@"`; (4) print export paths and the `scp` hint. Every step is idempotent so **the same command is the restart command** on a new box: the header comment says exactly that. Optional `AUTO_RESTART=1` env wraps step (3) in a loop that re-runs on non-zero exit codes other than 0/143, max 3 times, 60 s apart (covers transient CUDA/NCCL/data-loader crashes; does nothing for a dead box).

- [ ] **Step 1: Write the failing test**

```python
# tests/zipformer/test_cli.py
import train_zipformer as tz

def test_stage_expansion():
    assert tz.stages_to_run("all") == ["prepare", "train", "eval", "export"]
    assert tz.stages_to_run("export") == ["export"]

def test_defaults():
    a = tz.parse_args([])
    assert a.config.endswith("configs/train/zipformer/train_config.yml") and a.stage == "all" and not a.push_to_hub
```

- [ ] **Step 2: Run → FAIL**; **Step 3: Implement `train_zipformer.py`, `train_zipformer.sh`, README section** (README: how to run the Colab notebook, how to create the rclone Google Drive remote, how to launch on Vast.ai, where outputs land). README must also contain a subsection **"Jika box Vast.ai mati di tengah training"** covering:
  - Recovery = rent a new box, copy `rclone.conf` + `.env`, run the same `bash train_zipformer.sh`; what the `resume:` log line should show; expected loss ≤ one `save_every_n_batches` interval (~25 min on a 4090) + setup (~30–60 min: env install + ~60 GB data sync).
  - If the *same* interruptible box comes back, just re-run: local checkpoints are used unless Drive has a newer one.
  - **On-demand vs interruptible**: on-demand is not reclaimed by Vast.ai but hosts can still fail (pick high reliability score); interruptible is cheaper but can be stopped any time when outbid, possibly for hours. Recommendation: interruptible is acceptable only with `rclone_checkpoint_path` set; switch to on-demand if interrupted more than ~once a day or near a deadline. Tell the reader to check current prices on Vast.ai rather than quoting numbers.
  - Drive space needed (data ~60 GB + 2 × `keep_last_k` × ~1 GB checkpoints) and the 750 GB/day Drive upload limit (≈60 GB/day used).
  - TensorBoard history is lost with the box; W&B continues the same run.
- README also gets a subsection **"Frame rate 20 ms dan sherpa-onnx"**: the model emits one frame per 20 ms (v3: 40 ms) and why (spec § Model); recognised **text is unaffected** in sherpa-onnx; sherpa-onnx's online CTC recognizer hard-codes `subsampling_factor = 4` (verified at sherpa-onnx `347ec76`, `online-recognizer-ctc-impl.h` `GetResult`/`IsEndpoint`), so token timestamps come out 2× too large (divide by 2, or read `output_frame_shift_ms` from the model metadata) and endpoint trailing-silence rules fire after half the configured silence (double `rule1/2/3_min_trailing_silence`). Our ONNX carries `output_frame_shift_ms`/`subsampling_factor` metadata for apps that read it.
- [ ] **Step 4: Run → PASS; `bash -n train_zipformer.sh` → no syntax errors; commit** `add: train_zipformer stage runner, vast.ai bootstrap script and docs`

---

### Task 12: Smoke run on the training box (not unit-testable locally)

**Files:** none new; may adjust pins in `requirements-zipformer.txt` / `train_zipformer.sh`.

- [ ] **Step 1:** On a Vast.ai box with the synced data, run `python train_zipformer.py --stage train` with a temporary config override `num_epochs: 1`, `max_duration: 300`, `dev_max_cuts: 200` and confirm: loss decreases over the first 500 batches, GPU utilisation > 80 % (`nvidia-smi dmon`), `epoch-1.pt` written, dev `per_phonemes` logged.
- [ ] **Step 2: Kill test, same box.** With `save_every_n_batches: 200` and `rclone_checkpoint_path` pointing at a scratch Drive folder (`.../checkpoints/smoke-<date>`), start `bash train_zipformer.sh --stage train`; after the log shows `checkpoint-400.pt` saved **and** its upload finished, `kill -9` the python process (hard death, no flush). Re-run the same command. Pass criteria: `resume: source=local checkpoint=checkpoint-400.pt`; first logged step is 401; LR at step 401 equals the LR logged at step 400 (to 6 significant digits); training loss at 401–450 is in the range of 350–400, not back at the from-scratch level.
- [ ] **Step 3: Kill test, new box.** Repeat the kill at `checkpoint-600.pt`, then `rm -rf <output_dir>` (simulates a fresh box; data dir kept to save time) and re-run. Pass: `resume: source=drive checkpoint=checkpoint-600.pt`, sha256 verified, same continuity checks as Step 2.
- [ ] **Step 4: Graceful stop.** Send `kill -TERM`; pass: a new `checkpoint-<b>.pt` appears with meta, its upload completes within `shutdown_flush_seconds`, process exits 143.
- [ ] **Step 5:** Run `--stage export` and confirm `assert_onnx_matches_torch` passes, `model.int8.onnx` loads in `onnxruntime`, and the first `log_probs` per call has 24 frames (factor 1). Record measured checkpoint size and upload time per checkpoint in the README subsection, and the `frame_budget` numbers from `data_manifest.json` plus the count in `dropped_cuts.json`.
- [ ] **Step 5b: sherpa-onnx parity.** `pip install sherpa-onnx` (record the version), build `sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(tokens=..., model=model.int8.onnx, num_threads=1, sample_rate=16000, feature_dim=80, decoding_method="greedy_search")`, decode 20 test cuts (from `cuts_test.jsonl.gz`, pad each with 1 s of silence + `input_finished()`), and decode the same audio with our `OnnxStreamingDecoder` on the int8 file. Pass: recognised text identical on ≥ 19/20 clips (a difference on 1 clip is investigated and documented, not ignored). Also log sherpa's timestamps next to ours for 3 clips and confirm sherpa's are ≈ 2× ours (documents the hard-coded factor 4); if sherpa's are not 2×, update the README subsection with what was observed.
- [ ] **Step 6:** Delete the scratch Drive folder, record the verified torch/k2/Lhotse/icefall-commit versions in `requirements-zipformer.txt` header and commit `fix: record verified zipformer env pins`.

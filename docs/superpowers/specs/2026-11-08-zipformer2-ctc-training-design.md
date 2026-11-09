# Zipformer2 CTC streaming phoneme model — training pipeline design

Date: 2026-11-08
Status: builder check done; awaiting owner approval of this document

## Intent

Produce a streaming Qur'an phoneme recogniser with the same architecture as
`Quran-Lab/zipformer_p-arabic-v3` (icefall Zipformer2, causal, CTC-only head,
~65.5 M params), trained **from scratch** on this repo's data, and exported
as a sherpa-onnx compatible streaming ONNX model for Android.

The existing `train_streaming.py` (w2v-bert + RNN streaming head, multi-level
CTC, HF Trainer) stays as is. This is a sibling pipeline, not a replacement.

Why: w2v-bert is heavy for phones and its streaming export to TFLite has been
painful. Zipformer2 + sherpa-onnx is a proven on-device streaming stack, and
Quran-Lab showed it works for this exact task (1.4–3.7 % PER). Using their
*weights* is off the table (NPL-1.2 no-profit license), so we replicate the
recipe with our own data.

## Goals

- One command, one YAML config → trained model, PER on dev/test, qdat_bench
  results, `model.onnx` + `model.int8.onnx` + `tokens.txt` ready for
  sherpa-onnx.
- Same data sources, moshaf split, label generation and evaluation as
  `train_streaming.py`, so results are comparable with the muaalem models.
- Resumable: re-running the same command, on the same box or on a new box,
  continues from the newest checkpoint (local or Google Drive backup) at the
  exact data position, and skips evaluation/export steps whose outputs
  already exist. See *Fault tolerance*.
- Runs on 1× RTX 4090 (24 GB) on Vast.ai.

## Non-goals

- Multi-level (sifat) output. Single `phonemes` level only (decided: sherpa-onnx
  assumes one `log_probs` output).
- Transducer / RNN-T, LM rescoring, hotwords. CTC greedy only, like v3.
- Android/Kotlin integration. Export stops at the ONNX files.
- TFLite or CoreML export.
- Reproducing Quran-Lab's 250 multi-character unit inventory. We use our own
  per-character phoneme vocabulary.

## Anti-goals (would make this a failure even if it "works")

- Depending on Quran-Lab weights or any NPL-licensed artifact at runtime.
- A pipeline that needs the HF dataset to be converted to a multi-hundred-GB
  feature store before training can start.
- Silent label drift: labels must come from the same `quran_phonetizer` +
  `MoshafAttributes` (+ moshaf 6.0 segment overrides) as `train_streaming.py`.

## Constraints

- **Separate Python environment.** The repo pins Python ≥3.14 / torch 2.12.
  k2 publishes prebuilt wheels only for specific (python, torch, CUDA)
  triples (see the [k2 wheel index](https://k2-fsa.github.io/k2/cuda.html)). At the time of writing
  cp314 wheels exist for torch 2.9.1, so the preferred env is **Python 3.14 +
  torch 2.9.1 + matching k2 wheel + Lhotse + icefall from source**; then
  `pip install -e .` of this repo works unchanged. If no cp314 wheel is
  available when building, fall back to Python 3.12 and make the repo code
  importable via `PYTHONPATH=src` **without** editing `requires-python` in
  `pyproject.toml`. (Design call.) Pin `quran-transcript==0.1.2` as in
  `uv.lock` to avoid label drift. `qdat-bench` is a git dependency
  (`[tool.uv.sources]`), install it the same way.
- icefall is an external checkout at `external/icefall` (git submodule or
  pinned clone), not vendored into the package. Our code lives under
  `src/prepare_quran_dataset/zipformer/` and under the repo root
  (`train_zipformer.py`, `configs/train/zipformer/`).
- icefall's zipformer recipe is in `egs/librispeech/ASR/zipformer/`. We reuse
  its `zipformer.py`, `scaling.py`, `subsampling.py`, `optim.py`
  (ScaledAdam/Eden), `model.py` (CTC path), `export-onnx-streaming-ctc.py`
  and `streaming_decode.py` logic unchanged. We write our own `train.py`-like
  driver and datamodule because the data source is an HF dataset, not
  LibriSpeech manifests.
- GPU budget: 24 GB. `max_duration` (Lhotse dynamic bucketing, seconds of audio
  per batch) is the knob; start around 600 s with fp16/bf16 and reduce if OOM.
- Secrets via `.env` (`HUGGINGFACE_TOKEN`, `WANDB_API_KEY`), same as today.

## Decisions already made

### Model (matches v3, except the output frame rate)

- icefall Zipformer2, `causal=True`.
- **Output frame rate 20 ms, deliberately different from v3 (40 ms).**
  Config field `output_downsampling_factor: 1 | 2`, default **1**.
  `encoder_embed` (Conv2dSubsampling) turns 10 ms fbank frames into 20 ms
  frames: `T_embed = (T_fbank - 7) // 2`. The encoder stacks downsample
  internally and upsample back to that rate; `output_downsampling_factor`
  is an extra downsample at the very end (`SimpleDownsample`). Factor 1
  → `T_out = T_embed` (20 ms, like muaalem/w2v-bert); factor 2 →
  `T_out = (T_embed + 1) // 2` (40 ms, v3).
  - **Why.** v3 uses 250 multi-character units, so a 6-harakat madd or a
    shadda is one or two symbols. Our vocabulary is per character: madd and
    gemination are written as runs of the same character (e.g. Al-Fatiha 1
    → `…رَحِۦۦۦۦم`). CTC must put a blank between identical consecutive
    labels, so a cut needs at least `L + R` output frames (`L` = label
    length, `R` = number of identical consecutive pairs). On sample ayat
    `R` adds 18–25 % on top of `L` (1:1 L=32 R=8; 1:7 L=77 R=14; 2:255
    L=369 R=65). At 40 ms, fast recitation plus speed perturb 1.1 leaves
    little or no blank headroom; 20 ms doubles it and keeps `silence_per`
    and timing directly comparable with the muaalem models.
  - **Cost.** Encoder compute is unchanged (the downsample is after the
    last stack); only the CTC head and greedy decode run on 2× frames.
    Parameter count is unchanged (SimpleDownsample has `factor` params).
  - **How (verified at icefall `9c96676`).** `Zipformer2.__init__` accepts
    `output_downsampling_factor`, but `Zipformer2.forward` and
    `streaming_forward` both `assert self.output_downsampling_factor == 2`
    and hard-code `lengths = (x_lens + 1) // 2`; `train.get_encoder_model`
    hard-codes `output_downsampling_factor=2`. So we do **not** go through
    `get_encoder_model`/`get_model`. Our `model.py` defines a thin subclass
    of `Zipformer2` that is constructed with factor 2 (to satisfy the
    asserts), replaces `downsample_output` with `nn.Identity()` when the
    config says 1, and overrides `forward` / `streaming_forward` to call
    `super()` and return the *input* `x_lens` as the output lengths in that
    case. `AsrModel` is assembled directly from recipe pieces
    (`get_encoder_embed`, our encoder, `AsrModel`). No icefall file is
    copied or edited. The icefall commit is pinned in the submodule; a
    test fails loudly if a future icefall changes these lines.
  - Resume refuses a checkpoint whose stored config has a different
    `output_downsampling_factor` (the state dicts differ and the frame
    rate would silently change).
- Default "medium" dims: `num_encoder_layers=2,2,3,4,3,2`,
  `downsampling_factor=1,2,4,8,4,2`, `feedforward_dim=512,768,1024,1536,1024,768`,
  `num_heads=4,4,4,8,4,4`, `encoder_dim=192,256,384,512,384,256`,
  `query_head_dim=32`, `value_head_dim=12`, `pos_head_dim=4`, `pos_dim=48`,
  `encoder_unmasked_dim=192,192,256,256,256,192`, `cnn_module_kernel=31,31,15,15,15,31`.
  This is ~65.5 M params with a CTC head, i.e. v3's size.
- Training chunk schedule: `chunk_size=8,16,24`, `left_context_frames=64,128,256`
  (frames at 20 ms after `encoder_embed`; 24 frames = 480 ms). Follows v3.
- Features: 80-dim kaldi fbank, 10 ms hop, 25 ms window, 16 kHz. Lhotse
  `Fbank(FbankConfig(num_mel_bins=80))`. Total subsampling 2 → 20 ms/frame out (4 → 40 ms with `output_downsampling_factor=2`).
- Head: single linear + log-softmax over the phoneme vocab. `use_transducer=0`,
  `use_ctc=1`. Loss: CTC (`k2.ctc_loss` via icefall's model, or
  `torch.nn.functional.ctc_loss`; builder's choice, must be `reduction="sum"`
  normalised like icefall to keep ScaledAdam/Eden hyperparameters valid).

### Vocabulary / tokens

- Source of truth: `vocab_streaming/vocab.json["phonemes"]` (44 entries,
  `[PAD]=0` is the CTC blank, `[EOS]=1`).
- `tokens.txt` for sherpa-onnx is generated from it: `<blk> 0`, then each
  phoneme symbol with its id. `[EOS]` keeps id 1 but is **never** in training
  labels (no `add_eos`; CTC does not need it). Unused ids are harmless.
- Labels: `quran_phonetizer(uthmani, moshaf_attr, remove_spaces=True).phonemes`
  → ids via `MultiLevelTokenizer("./vocab_streaming")` phonemes level only, or
  a direct char→id map built from `vocab.json` (builder's choice; must agree
  with `vocab.json`).
- Silence/noise inputs have empty label sequences.

### Data

- Speech: `obadx/muaalem-annotated-v3`, configs `moshaf_{id}` for
  `train_moshaf_ids` and `test_moshaf_ids` exactly as in
  `configs/train/streaming/train_config.yml`. Dev split = 10 % of train,
  seeded. `max_audio_seconds` filter applies (null = no filter; dataset max is
  ~52 s, fine for Lhotse bucketing).
- Moshaf attributes: `obadx/mualem-recitations-annotated` config
  `moshaf_metadata` → `MoshafAttributes`, plus the moshaf `6.0` per-segment
  overrides from `prepare_special_moshaf_ways` in `train_streaming.py`
  (move that table into a shared module so both scripts import it; the
  current function reads a global by accident and must be fixed to use its
  argument).
- Noise: `obadx/freesound-commercial-50k-noise-only`, seeded split
  30 % / 70 %:
  - 30 % → noise cuts used for **CutMix** augmentation (Lhotse `CutMix`,
    SNR range ~[10, 20] dB, `p≈0.5`).
  - 70 % → **silence input** cuts with empty supervision text, concatenated
    into the training CutSet. Clip to `max_noise_input_seconds=40`, pad to at
    least one streaming chunk (24 encoder frames × 20 ms = 0.48 s plus the 13-frame embed padding; 1 s is a safe floor).
- **Data preparation runs on Google Colab, not on the training box.**
  (Decided in conversation.) `prepare_zipformer_data_colab.py` (plus a thin
  `.ipynb` wrapper that mounts Drive and runs it) iterates the HF datasets
  **once**, writes every clip as **16 kHz mono FLAC** under a user-supplied
  Google Drive path (`DRIVE_ROOT`, e.g.
  `/content/drive/MyDrive/muaalem-zipformer-data/`), and emits Lhotse
  `CutSet` jsonl.gz files there: `cuts_train.jsonl.gz`, `cuts_dev.jsonl.gz`,
  `cuts_test.jsonl.gz`, `cuts_noise_aug.jsonl.gz`. Phoneme text is stored in
  `supervision.text` already phonetised, so training never needs
  `quran_transcript`. Store `moshaf_id`, `segment_index`, `input_type` in
  `supervision.custom`.
  - Colab constraints the script must respect: local disk ≈ 100 GB and
    sessions die after hours, so process **one HF config (moshaf / noise
    shard) at a time**: download → convert → write FLAC+partial manifest to
    Drive → delete the local HF cache for that config. **Idempotent**:
    a config whose `done` marker exists on Drive is skipped, so the notebook
    can simply be re-run after a timeout. Final step merges partial
    manifests into the four CutSets above and writes `tokens.txt` and
    `data_manifest.json` (counts, hours, checksum of vocab).
  - Expected Drive footprint ≈ 55–65 GB (FLAC ≈ half of WAV). The script
    checks free Drive space up front and aborts with a clear message.
  - Recording paths in manifests are **relative to `DRIVE_ROOT`**; the
    training side rebases them (`CutSet.with_recording_path_prefix` or
    equivalent) to wherever rclone put the data. Never store absolute
    Colab paths.
- **Training box pulls the data with rclone.** `train_zipformer.sh` runs
  `rclone sync <remote>:<path> <local_data_dir>` before training. The rclone
  Google Drive remote comes from an `RCLONE_CONFIG_*` env block or a
  mounted `rclone.conf` (user supplies it; the script documents both
  options and refuses to start without one). `--stage prepare` on the
  training box therefore means "sync from Drive", not "rebuild".
- Features computed **on the fly** (`OnTheFlyFeatures(Fbank)`), no
  precomputed feature store. Speed perturbation 0.9/1.0/1.1 on the fly
  (Lhotse `CutSet.perturb_speed` applied lazily or icefall's
  `--enable-musan`-style option re-pointed at our noise set). SpecAugment on
  with icefall defaults.
- Augmentation set is **icefall-style only** (speed perturb + CutMix noise +
  SpecAugment). The audiomentations chain of `train_streaming.py` (RIR,
  MP3, filters, pitch, …) is intentionally **not** ported in this version.
  (Decided in conversation.) If robustness on phone audio turns out worse
  than the muaalem models, adding Lhotse `ReverbWithImpulseResponse` is the
  first follow-up, not part of this build.
- Sampler: `DynamicBucketingSampler` with `max_duration` from config,
  `shuffle=True`, `drop_last=True`, `num_buckets=30`.

### Training

- Driver `train_zipformer.py` (repo root) reads `configs/train/zipformer/
  train_config.yml` (Pydantic `TrainConfig` like the existing one: moshaf ids,
  split ratio, seed, output_dir, hub ids, wandb project, model dims, chunk
  schedule, `max_duration`, `num_epochs`, `base_lr`, `lr_epochs`,
  `lr_batches`, `keep_last_k`, `avg_last_k`, `save_every_n_batches`,
  `rclone_checkpoint_path`, bf16).
- Optimizer ScaledAdam (`base_lr=0.045`), scheduler Eden
  (`lr_batches=7500`, `lr_epochs=3.5`), icefall defaults; CTC-only recipe
  typically also uses `--ctc-loss-scale 1.0`.
- Checkpoints: mid-epoch `checkpoint-{batch_idx_train}.pt` every
  `save_every_n_batches` and `epoch-N.pt` at the end of each epoch (see
  *Fault tolerance*). Final model = **offline** average of the last
  `avg_last_k=3` epoch checkpoints (`icefall.checkpoint.average_checkpoints`),
  like v3 ("average of the last 3 training epochs"). No online `model_avg`.
- Default `num_epochs=30`. Budget on 1× RTX 4090: one epoch ≈ 2,850 h of
  audio after 3× speed perturb; extrapolating icefall's Zipformer-M
  LibriSpeech throughput gives **3.5–4 h/epoch** at `max_duration≈600`,
  i.e. **4–5 days** for 30 epochs. Accepted by the owner. Config is the only
  place this is set; a faster GPU only changes `max_duration` and wall time.
  The data loader must not be the bottleneck: budget ≥ 8 vCPU for on-the-fly
  fbank + speed perturb + CutMix, and make `num_workers` configurable.
- Logging: TensorBoard (icefall native) + W&B (`WANDB_PROJECT` from config,
  log the same scalars icefall writes). W&B is optional; missing token prints
  a warning and continues. The W&B run id is stored in every checkpoint so a
  resumed run (same or new box) continues the same W&B run
  (`wandb.init(id=..., resume="allow")`).

### Fault tolerance (GPU box dies mid-training)

Requirement from the owner: a Vast.ai box can die at any time; losing more
than one checkpoint interval of work, or silently replaying/skipping data,
is a failure.

- **Mid-epoch checkpoints.** Every `save_every_n_batches` (default 2000)
  training batches, save `checkpoint-{batch_idx_train}.pt` containing:
  model, optimizer (ScaledAdam), scheduler (Eden), `batch_idx_train`
  (global), `epoch`, `batches_consumed_in_epoch`, the Lhotse
  `sampler.state_dict()`, the W&B run id, and the config dump. At the end of
  every epoch also save `epoch-N.pt` with the same contents (marked as
  end-of-epoch). Old files are pruned per kind: keep the newest
  `keep_last_k` `checkpoint-*.pt` and the newest `keep_last_k` `epoch-*.pt`.
  At ~0.7 s/batch on a 4090, 2000 batches ≈ 25 min, so a crash loses at most
  ~25 min of training. Checkpoint size ≈ 1 GB (model + ScaledAdam state);
  measure in the smoke run.
- **Exact data position.** Resume must continue with the very next batch the
  dead run would have trained on: no replay, no skip. Lhotse samplers count
  batches *drawn*, which with DataLoader workers runs ahead of the batches
  *consumed* by up to `num_workers × prefetch_factor`. The checkpoint
  therefore records `batches_consumed_in_epoch` from the training loop, and
  the restored sampler is positioned by that count (same seed, same epoch),
  not by the drawn count. Augmentation randomness (CutMix pick, SNR,
  SpecAugment masks) after resume need not be bit-identical; the cut ids
  and their order must be.
- **Zipformer schedules.** Zipformer's `ScheduledFloat`s (dropout,
  balancer/whitener limits, bypass scales) depend on the model's batch
  count. On resume, call icefall's `set_batch_count(model, ...)` with the
  restored `batch_idx_train` before the first step, so schedules do not
  restart from zero.
- **Atomic, self-describing files.** Write `<name>.pt.tmp` → `os.replace` to
  `<name>.pt`, then write `<name>.meta.json` (`{"kind": "batch"|"epoch",
  "batch_idx_train", "epoch", "batches_consumed_in_epoch", "size_bytes",
  "sha256"}`). A checkpoint without its meta file is incomplete and is
  ignored everywhere.
- **Google Drive backup.** If `rclone_checkpoint_path` is set (e.g.
  `gdrive:muaalem-zipformer-data/checkpoints/<run-name>`), every saved
  checkpoint is copied to Drive by a **single background worker** (serial
  queue, so uploads never overlap). Per file: `rclone copyto <file>
  <remote>/<name>.pt` (rclone verifies size + hash), then upload the meta
  file **last** as the commit marker. Only after the new checkpoint's meta is
  on Drive are older Drive checkpoints of the same kind pruned to
  `keep_last_k`. Training never waits on an upload, except: if the previous
  upload is still running when the next checkpoint is due, the new one
  replaces any not-yet-started upload in the queue (the newest checkpoint
  always wins; epoch checkpoints are never dropped). Upload failures and
  timeouts (`upload_timeout_seconds`, default 1800) log a warning and
  training continues; the next checkpoint retries naturally.
- **Graceful stop.** On SIGTERM/SIGINT, finish the current batch, save a
  checkpoint, and wait up to `shutdown_flush_seconds` (default 120) for the
  upload queue to drain. Best effort: a hard power-off skips this and the
  last periodic checkpoint is used instead. (Design call.)
- **Automatic resume on any box.** At the start of `--stage train`:
  1. Find the newest complete local checkpoint (largest `batch_idx_train`
     among files with a valid meta).
  2. If backup is configured, list the Drive checkpoint meta files the same
     way.
  3. Use whichever is newer; if the Drive one wins, download it (and its
     meta) into `output_dir` and verify sha256 first. Also download any
     `epoch-*.pt` kept on Drive that is missing locally, because final
     averaging needs the last `avg_last_k` epochs.
  4. No checkpoint anywhere → start from scratch.
  The same command (`bash train_zipformer.sh` / `python train_zipformer.py
  --stage all`) is therefore used for a first run, a restart on the same box,
  and a restart on a brand-new box. On a new box, `prepare` re-syncs the
  ~60 GB of data from Drive first (rclone sync is incremental, so it is a
  no-op on the same box).
- **Drive quotas.** One ~1 GB checkpoint per ~25 min ≈ 60 GB/day of uploads,
  well under Google Drive's 750 GB/day per-user upload limit. Drive needs
  room for data (~60 GB) + 2 × `keep_last_k` checkpoints (~6 GB).
- **Instance choice (documented in README).** On-demand instances are not
  reclaimed by Vast.ai but the host can still fail; interruptible instances
  are cheaper but can be stopped at any time when outbid. With this design
  interruptible is viable: each interruption costs at most one checkpoint
  interval plus new-box setup (env install + data sync, ~30–60 min).
  Recommend interruptible only with Drive backup enabled and on hosts with
  high reliability scores; recommend on-demand when interruptions happen
  more than about once a day or close to a deadline.

### Evaluation

- Dev PER every epoch (greedy CTC over the dev CutSet, chunk 24 / left 256),
  logged to TB/W&B. Metric name `per_phonemes` to match the existing scripts.
- After training: test-set PER (3 test moshafs) → `test_results.json`.
  Skipped if file exists unless `--rerun-testset`.
- Silence handling mirrors `train_streaming.py`: noise input cuts are added
  to the training pool **before** the 10 % dev split, so dev contains silence
  cuts and `silence_per` (frame-level non-blank rate on them, counted on
  20 ms output frames, so directly comparable with muaalem's `silence_per`)
  is a **dev** metric only. The test set has no silence cuts and no `silence_per`.
  (Design call.) Dev membership follows the same policy (10 %, same seed),
  not the exact same cuts as the HF pipeline.
- qdat_bench: run the averaged model over `obadx/qdat_bench` with streaming
  greedy decoding, write `qdat_bench_predictions.jsonl` in the same schema
  `run_qdat_bench_test` writes (`levels_labels` and `level_to_scripts` with
  only the `phonemes` key), then `eval_qdat_bench(...)`. **Open check for the
  builder:** confirm `eval_qdat_bench` / `MultiLevelTokenizer` tolerate a
  prediction set with only the phonemes level. If not, restrict the metrics
  it computes to phoneme-based ones and document which qdat metrics are
  unavailable. Skipped if results exist unless `--rerun-qdat-bench`.

### Export

- `export-onnx-streaming-ctc.py` from icefall, with
  `--chunk-size 24 --left-context-frames 256` (decode_chunk_len 48,
  T = 61 input fbank frames per call, as v3). Output `log_probs` per call:
  **24 frames** (one per 20 ms of the 480 ms chunk) with
  `output_downsampling_factor=1`; 12 frames (v3) with 2. Input side (`T`,
  `decode_chunk_len`, states) is identical for both factors.
- Extra ONNX metadata we add (sherpa-onnx ignores unknown keys):
  `output_downsampling_factor`, `subsampling_factor` (= 2 × factor, i.e.
  2 or 4), `output_frame_shift_ms` (20 or 40), `vocab_size`, `blank_id`.
  Our onnxruntime decoder reads `output_frame_shift_ms` for timestamps.
- Outputs in `<output_dir>/export/`: `model.onnx`, `model.int8.onnx`
  (onnxruntime `quantize_dynamic`, MatMul, QInt8), `tokens.txt`.
- Sanity check after export: load both ONNX files with onnxruntime, run one
  chunk with zero-initialised states, assert `log_probs` shape
  `(1, export_chunk_size // output_downsampling_factor, vocab)`, i.e.
  `(1, 24, vocab)` by default; decode 3–5 test cuts with the ONNX streaming loop and
  compare token output with the PyTorch greedy decode (expect identical or
  near-identical).
- `--push-to-hub`: upload `model.onnx`, `model.int8.onnx`, `tokens.txt`,
  the averaged `.pt`, config YAML, `test_results.json`,
  `qdat_bench_test_results.json` to `hub_model_id`.

### Frame-budget check (CTC feasibility)

- For every speech cut, `R` = number of `i` with `text[i] == text[i+1]`,
  `L = len(text)`; required frames `L + R`. Available frames
  `T_out(duration, factor, speed)` with `T_fbank = round(duration / speed
  / 0.01)` (Lhotse frame count), then the formulas in *Model*.
- **Colab prep** stores `num_labels` and `num_repeats` in
  `supervision.custom` and writes to `data_manifest.json` a
  `frame_budget` block: for factor 1 and 2 and speed 1.0 and 1.1, the
  ratio `T_out / (L + R)` as min / p1 / p5 / p50 and the count and ids
  (first 50) of cuts with ratio < 1. Silence cuts are excluded (`L = 0`).
- **Start of `--stage train`** recomputes the same check for the configured
  factor at the *largest* speed in use (1.1 when `speed_perturb`), drops
  train/dev cuts with ratio < 1 (a CTC loss of `inf` otherwise), logs how
  many and which, and writes `output_dir/dropped_cuts.json`. If more than
  1 % of speech cuts would be dropped, abort with a message suggesting
  factor 1 or turning off speed perturb, rather than silently training on
  a biased subset. Test cuts are never dropped; they are reported.

### sherpa-onnx compatibility (Android)

Verified at sherpa-onnx `347ec76`
(`sherpa-onnx/csrc/online-recognizer-ctc-impl.h`,
`online-zipformer2-ctc-model.cc`):

- The model reads only `T`, `decode_chunk_len`, `left_context_len`,
  encoder shape keys and `feature`; it consumes whatever number of output
  frames the model returns per chunk. **Recognised text is therefore
  unaffected by the 20 ms frame rate.**
- `GetResult` hard-codes `frame_shift_ms = 10`, `subsampling_factor = 4`
  for timestamps, and `IsEndpoint` multiplies trailing blank frames by the
  same hard-coded 4. With our 20 ms model: token **timestamps come out
  2× too large** (divide by 2 in the app, or use our metadata), and
  trailing silence is over-counted 2×, so endpoint rules fire after half
  the configured silence (double `rule1/rule2/rule3` min-trailing-silence
  values in the app config to compensate). No metadata key changes this
  in the current sherpa-onnx; the README states all of this.
- Task 12 decodes test clips with the sherpa-onnx Python binding and our
  onnxruntime decoder and requires identical text.

### CLI

`python train_zipformer.py --config configs/train/zipformer/train_config.yml
[--stage prepare|train|eval|export|all] [--push-to-hub] [--rerun-testset]
[--rerun-qdat-bench]`. Default `--stage all` runs everything idempotently.

A `train_zipformer.sh` mirroring `train_and_export_tflite.sh` sets up the
separate venv (uv, Python 3.14 preferred, torch + k2 wheel + lhotse +
icefall), installs rclone, syncs the data from Google Drive and runs the
pipeline on a fresh Vast.ai box. Disk needed on the box ≈ data (≈ 60 GB) +
checkpoints (2 × `keep_last_k` × ~1 GB + averaged model) + ONNX; 100 GB is
comfortable. The script is also the restart command after a box dies.

## Left to the builder

- Exact (python, torch, CUDA, k2) pins; pick the newest combination with a
  published k2 wheel and record it in `train_zipformer.sh` and a
  `requirements-zipformer.txt`.
- Whether to call icefall's `train.py` with a custom `--datamodule` or to
  write a thin driver that imports icefall's `get_model`, `compute_loss`,
  `train_one_epoch`. Prefer the thin driver: the loop is small and icefall's
  `train.py` is tightly coupled to LibriSpeech argparse flags.
- How noise augmentation is sequenced with speed perturb and SpecAugment.
- FLAC directory layout on Drive (per moshaf / per noise shard) and the
  partial-manifest merge format.
- How the Colab script authenticates to HF (Colab secrets `HF_TOKEN` is the
  expected route) and how `DRIVE_ROOT` is passed (notebook variable).
- W&B integration mechanics.
- Unit tests: at minimum the manifest builder (one cut per input type with
  correct text/ids), `tokens.txt` generation (round-trips `vocab.json`), and
  the ONNX-vs-PyTorch decode comparison.

## Open risks

- k2 wheel availability may force an older torch than the rest of the repo;
  acceptable because the env is separate.
- 700–800 h is enough for the medium model but PER will likely land above
  v3's 1.4 %; that is expected and not a blocker. A smaller model is a
  YAML-only change.
- `eval_qdat_bench` single-level compatibility (see Evaluation).
- Lhotse sampler `state_dict` field names differ between versions; the
  consumed-count correction must be written against the pinned Lhotse and
  guarded by the resume test, not assumed.
- Our 20 ms Zipformer2 deviates from every published icefall Zipformer2
  recipe; schedules (dropout, warmup) were tuned for 40 ms. Expected to be
  harmless (the encoder is unchanged), checked by the smoke run's loss
  curve.
- A future sherpa-onnx may start honouring a subsampling metadata key; our
  extra `subsampling_factor` key is set to the true value (2) so it would
  then become correct automatically.
- TensorBoard event files live only on the box; after a box dies, history
  continues in W&B, not in TensorBoard.

# Zipformer2 CTC streaming model

Model streaming icefall Zipformer2 (`causal=True`) dengan satu head CTC (level `phonemes`, vocab `vocab_streaming/vocab.json` di root repo), diekspor ke ONNX streaming untuk sherpa-onnx (Android). Semua parameter ada di `configs/train_config.yml`.

Semua kode training Zipformer berakar di folder `zipformer/` ini; semua path di dokumen ini relatif terhadap `zipformer/` (jalankan perintah dari folder ini). Kode ini tetap memakai paket `prepare_quran_dataset` di root repo (tokenizer qdat_bench, `moshaf_ways`) dan `vocab_streaming/`.

| Path | Isi |
|---|---|
| `quran_zipformer/` | paket Python (data prep, datamodule, model, training loop, decode, export ONNX, qdat_bench) |
| `train_zipformer.py`, `train_zipformer.sh` | CLI per stage dan pipeline end-to-end untuk Vast.ai |
| `prepare_zipformer_data_colab.py`, `notebooks/prepare_zipformer_data.ipynb` | persiapan data di Colab |
| `configs/train_config.yml` | config training |
| `requirements-zipformer.txt` | dependency training |
| `external/icefall` | submodule icefall (`git submodule update --init external/icefall`) |
| `tests/` | test (`pytest` dari folder ini) |

Spec desain dan rencana implementasi tetap di `docs/superpowers/` di root repo (lokasi yang dipakai skill superpowers): `specs/2026-11-08-zipformer2-ctc-training-design.md` dan `plans/2026-11-08-zipformer2-ctc-training.md`. Keduanya dokumen historis; path di dalamnya masih layout lama `src/prepare_quran_dataset/zipformer/`.

Alurnya: data disiapkan sekali di Google Colab dan disimpan di Google Drive → box GPU Vast.ai menyalin data dari Drive dengan `rclone`, training, evaluasi, lalu export.

## 1. Siapkan data di Colab

1. Buka `zipformer/notebooks/prepare_zipformer_data.ipynb` di Google Colab.
2. Simpan `HF_TOKEN` (token HuggingFace dengan akses ke dataset) di **Colab secrets**.
3. Jalankan semua sel. Notebook me-mount Drive, meng-clone repo ini, lalu dari folder `zipformer/` menjalankan:

   ```bash
   python prepare_zipformer_data_colab.py --drive-root /content/drive/MyDrive/RClone/Utluma/zipformer-data
   ```

   Hasilnya (FLAC + manifest Lhotse `cuts_train/dev/test/noise_aug.jsonl.gz`) ditulis ke folder Drive `MyDrive/RClone/Utluma/zipformer-data` (~60 GB).
4. Aman dijalankan ulang jika sesi Colab mati: config yang sudah selesai (marker `.done`) dilewati. Untuk memproses sebagian saja: `--only moshaf_6.0 --only noise`.

## 2. Buat remote rclone Google Drive

Buat remote di komputer lokal (yang punya browser), lalu salin `rclone.conf` ke box:

```bash
rclone config create utluma drive scope=drive   # login lewat browser
rclone lsd utluma:RClone/Utluma/zipformer-data  # harus menampilkan audio/ dan manifests/
```

Nama remote (`utluma`) harus sama dengan prefix di `rclone_remote_path` dan `rclone_checkpoint_path`. Scope harus `drive` (akses penuh): dengan `drive.file` rclone hanya melihat file yang dibuat rclone sendiri, sehingga data buatan Colab tidak terlihat. Jangan set `root_folder_id`; path di config relatif terhadap root My Drive.

File konfigurasinya ada di `~/.config/rclone/rclone.conf` (cek dengan `rclone config file`). Jika box tidak punya browser dan Anda mengonfigurasi langsung di box, jawab `n` pada "Use web browser to automatically authenticate" lalu jalankan `rclone authorize "drive"` di komputer lokal.

`train_zipformer.sh` menerima salah satu dari tiga sumber konfigurasi rclone: file `~/.config/rclone/rclone.conf`, file yang ditunjuk `RCLONE_CONFIG=/path/ke/rclone.conf`, atau remote yang didefinisikan lewat environment (`RCLONE_CONFIG_<REMOTE>_TYPE=drive` beserta `RCLONE_CONFIG_<REMOTE>_TOKEN` dst., mis. `RCLONE_CONFIG_UTLUMA_TYPE=drive`).

Field terkait di config:

| Field | Default | Keterangan |
|---|---|---|
| `hf_data_repo` | `null` (di `train_config.yml`: `rijalulfikri/utluma-zipformer-data`) | sumber pertama `--stage prepare`: repo dataset HF (private) berisi shard tar `data_dir`, diunduh, diverifikasi sha256, lalu diekstrak |
| `drive_data_shards_path` | `null` (di `train_config.yml`: `utluma:RClone/Utluma/zipformer-shards`) | sumber kedua: shard yang sama (byte identik, `index.json` sama) di Drive, dipakai jika HF gagal |
| `rclone_remote_path` | `utluma:RClone/Utluma/zipformer-data` | sumber terakhir: `rclone sync` file mentah ke `data_dir`, jika kedua sumber shard kosong atau gagal |
| `data_dir` | `./data-zipformer` | lokasi data lokal di box |
| `rclone_checkpoint_path` | `null` (di `train_config.yml`: `utluma-edit:RClone/Utluma/zipformer-data/checkpoints/utluma-zipformer-ctc-v1`) | backup checkpoint ke Drive; `null` menonaktifkan backup |

**Waktu unduh data.** Dataset berisi ~306 ribu file FLAC kecil (~65 GiB). Google Drive membatasi jumlah request per detik, sehingga `rclone sync` butuh ±2 jam di box Vast.ai. Karena itu `data_dir` yang sudah jadi juga disimpan sebagai 17 shard tar (±4 GiB) di repo dataset HF private `hf_data_repo`, yang terunduh jauh lebih cepat: semua shard (65 GiB) terunduh dan terverifikasi dalam ±4,5 menit (±250 MiB/s) di box RTX 5090 Vast.ai. Hasil ekstraksinya identik byte demi byte dengan data di Drive, jadi resume checkpoint tetap tepat.

Salinan shard yang sama juga ada di Drive (`drive_data_shards_path`, di luar folder `zipformer-data` supaya tidak ikut `rclone sync`) sebagai cadangan jika HF tidak bisa diakses; semua shard dari Drive terunduh dan terverifikasi dalam ±5,5 menit (±200 MiB/s), tetap jauh lebih cepat daripada `rclone sync` file mentah. Urutan sumber di `--stage prepare`: shard HF → shard Drive → `rclone sync` file mentah. Penanda per shard berisi sha256-nya, yang sama di kedua sumber, jadi shard yang sudah diekstrak dari HF tidak diunduh ulang dari Drive.

Shard dibuat sekali dari box yang sudah punya data lengkap (perlu `HUGGINGFACE_TOKEN` dengan izin tulis, dan remote rclone dengan izin tulis ke Drive). Jalankan ulang keduanya setiap kali data di Drive berubah, karena shard tidak ikut berubah sendiri:

```bash
cd zipformer && set -a && . ./.env && set +a
.venv-zipformer/bin/python vastai/pack_data_to_hf.py --data-dir data-zipformer --repo rijalulfikri/utluma-zipformer-data
.venv-zipformer/bin/python vastai/mirror_shards_to_drive.py --repo rijalulfikri/utluma-zipformer-data \
    --dest utluma-edit:RClone/Utluma/zipformer-shards
```

## 3. Jalankan di Vast.ai

Sewa box dengan 1 GPU NVIDIA (mis. RTX 4090) dan disk yang cukup untuk data (~60 GB), environment, dan checkpoint, lalu:

```bash
git clone https://github.com/rijalulfikri/prepare-quran-dataset.git && cd prepare-quran-dataset/zipformer
mkdir -p ~/.config/rclone
# dari komputer lokal:
#   scp -P <PORT> ~/.config/rclone/rclone.conf root@<IP_VAST_AI>:~/.config/rclone/rclone.conf
#   scp -P <PORT> .env root@<IP_VAST_AI>:~/prepare-quran-dataset/zipformer/.env   # WANDB_API_KEY, HUGGINGFACE_TOKEN (lihat .env.example)
bash train_zipformer.sh
```

Atau jalankan semuanya dari komputer lokal dengan satu perintah (clone repo, kirim `rclone.conf` dan `.env`, lalu `train_zipformer.sh` berjalan di sesi tmux `zipformer` dengan log di `~/train_zipformer.log` di box):

```bash
python vastai/run_zipformer_training.py --offer-id <OFFER_ID> --follow
# --offer-id: buat instance baru (default image pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime,
#   disk 150 GB, label zipformer; ubah dengan --image / --disk / --label), tunggu SSH siap, lalu training
# atau box yang sudah ada: --host <IP_VAST_AI> --port <PORT>
# tanpa --offer-id/--host/--port: instance running pertama dari Vast.ai API (opsional --label)
# --offer-id dan mode API butuh VAST_API_KEY dan paket vastai di lokal
# argumen untuk train_zipformer.sh setelah "--", mis.: ... -- --push-to-hub
```

Box meng-clone branch `--branch` (default `main`) dari GitHub, jadi push dulu commit yang ingin dilatih. `K2_WHEEL_URL` / `K2_WHEEL_URL_CP312` dibaca dari `zipformer/.env`.

Tambahkan `--destroy-when-done` supaya box menghapus dirinya sendiri setelah training sukses (exit 0): `output_dir` (tanpa `checkpoint-*` / `epoch-*`, yang sudah di-backup) di-upload ke `--results-remote` (default `<rclone_checkpoint_path>-results`), diverifikasi dengan `rclone check`, lalu instance dihapus lewat API Vast.ai dengan `CONTAINER_ID` / `CONTAINER_API_KEY` milik instance itu (`vastai/upload_results_and_destroy.sh`). Jika training gagal atau berhenti (exit selain 0), atau upload/verifikasi gagal, instance dibiarkan hidup.

Tambahkan `--early-stop-patience N` untuk berhenti lebih awal saat metrik early stop mendatar: `vastai/zipformer_watchdog.sh` berjalan di box (sesi tmux `watchdog`, log `~/zipformer_watchdog.log`) dan memeriksa `dev_metrics.jsonl` tiap 2 menit. Metriknya adalah kunci `early_stop_metric` di config (bawaan `qdat/per_phonemes`, PER QDAT), atau `--early-stop-metric <kunci>` untuk run itu saja (lihat [Evaluasi tiap epoch](#evaluasi-tiap-epoch-augmentasi-dan-baseline)); `dev_metrics.jsonl` yang belum memuat kunci itu (mis. run lama) memakai `per_phonemes` dev lama, dan log watchdog menyebutkannya. Sebuah epoch dihitung rekor baru hanya jika metriknya lebih rendah setidaknya `--early-stop-min-improvement` (default `0.05`, relatif) dari rekor sebelumnya. Setelah N epoch berturut-turut tanpa rekor baru, watchdog mengirim SIGTERM ke training (berhenti rapi, checkpoint di-flush), mengubah `num_epochs` di config box ke epoch terakhir, lalu menjalankan ulang pipeline yang sama: averaging 3 epoch terakhir, evaluasi test, export, dan dengan `--destroy-when-done` juga upload hasil + destroy. Watchdog tidak pernah memotong averaging / eval / export (hanya bertindak selama epoch terakhir < `num_epochs`).

Status run kapan saja: `python vastai/zipformer_status.py` (instance dicari lewat API Vast.ai, atau `--host/--port`). Baris pertamanya `STATUS: <keadaan>`, lalu tabel metrik early stop per epoch (beserta PER dev lama) dengan hitungan early stop, PER semua eval set di epoch terakhir, menit per epoch, sisa credit dan perkiraan biaya, watchdog, GPU, dan error terakhir; `--drive` mencocokkan SHA-256 checkpoint epoch terbaru di Drive, dan `--wait` menunggu epoch baru atau perubahan status. Setelah instance di-destroy, skrip ini menampilkan isi folder hasil di Drive. Di Claude Code, `/zipformer-monitor` (`.claude/skills/zipformer-monitor`) memakai skrip ini untuk memantau run: melapor tiap epoch, mendiagnosis crash, dan memverifikasi destroy serta hasil di Drive; keputusan berhenti dan destroy tetap di watchdog dan di tangan kamu.

`train_zipformer.sh` melakukan, dan semua langkahnya idempotent:

1. Berhenti sebelum membuat venv atau menginstall torch jika `K2_WHEEL_URL` dan `K2_WHEEL_URL_CP312` masih placeholder. Membuat `.venv-zipformer` (Python 3.14; fallback 3.12 dengan `PYTHONPATH=<repo>/src` jika wheel k2 untuk cp314 tidak ada), menginstall torch 2.9.1, k2 (`K2_WHEEL_URL`, pilih wheel di https://k2-fsa.github.io/k2/cuda.html), `requirements-zipformer.txt`, `pip install -e ..` (paket `prepare_quran_dataset` di root repo; hanya di Python 3.14; fallback 3.12 memakai `PYTHONPATH=<repo>/src` karena `pyproject.toml` mensyaratkan Python >=3.14), cek import modul training/qdat_bench (gagal dalam hitungan menit jika ada dependency yang kurang), dan submodule `external/icefall`.
2. Menginstall rclone jika belum ada, menolak jalan tanpa konfigurasi rclone (`~/.config/rclone/rclone.conf`, `RCLONE_CONFIG`, atau `RCLONE_CONFIG_<REMOTE>_TYPE` di environment), cek koneksi (`rclone lsd`), dan menampilkan isi folder checkpoint Drive (`rclone lsl <rclone_checkpoint_path>`) supaya terlihat apa yang akan di-resume.
3. `python train_zipformer.py --config configs/train_config.yml --stage all "$@"`. Dengan `AUTO_RESTART=1`, crash dengan exit code selain 0/143 diulang maksimal 3 kali dengan jeda 60 detik (untuk error CUDA/NCCL/data-loader sementara; tidak menolong jika box-nya mati).
4. Menampilkan path hasil export dan contoh perintah `scp`.

Argumen tambahan diteruskan ke `train_zipformer.py`, mis. `bash train_zipformer.sh --push-to-hub`.

`train_zipformer.py` bisa juga dijalankan per stage:

| Argumen | Default | Keterangan |
|---|---|---|
| `--config` | `configs/train_config.yml` | YAML config |
| `--stage` | `all` | `prepare`, `train`, `eval`, `export`, atau `all` (= keempatnya berurutan); `evalsets` (= `prepare` lalu skor checkpoint `--checkpoint`, tanpa training; tidak termasuk `all`) |
| `--checkpoint` | - | untuk `--stage evalsets`: path lokal atau rclone `remote:path` sebuah `epoch-*.pt` / `checkpoint-*.pt` / `averaged-*.pt`; boleh diulang |
| `--push-to-hub` | `false` | upload `export/`, `averaged-<k>.pt`, `config.yml`, `test_results.json`, `qdat_bench_test_results.json`, `eval_sets_results.json` ke `hub_model_id` |
| `--rerun-testset` | `false` | evaluasi ulang test set dan eval set walau `test_results.json` / `eval_sets_results.json` sudah ada |
| `--rerun-qdat-bench` | `false` | jalankan ulang qdat_bench walau hasilnya sudah ada |
| `--device` | `cuda` jika ada, selain itu `cpu` | device untuk train/eval |

Stage:

* `prepare`: unduh shard tar (4 paralel) dari `hf_data_repo`, atau dari `drive_data_shards_path` jika HF gagal, cek sha256 terhadap `index.json`, ekstrak ke `data_dir`; shard yang sudah selesai dilewati saat dijalankan ulang. Jika kedua sumber gagal (atau kosong): `rclone sync <rclone_remote_path> <data_dir> --progress --transfers 32 --checkers 32 --fast-list --drive-pacer-min-sleep 10ms --drive-pacer-burst 200 --exclude "/checkpoints/**"`. Gagal dengan pesan jelas jika `rclone` tidak ada atau belum ada remote (`rclone listremotes` kosong).
* `train`: training dengan resume otomatis dari checkpoint lokal atau Drive (mana yang lebih baru), lalu merata-ratakan `avg_last_k` epoch terakhir menjadi `averaged-<k>.pt`. Jika `rclone lsjson` ke Drive gagal, muncul warning dan training lanjut dengan checkpoint lokal saja.
* `eval`: model rata-rata pada seluruh test set (`per_phonemes`) dengan `export_chunk_size` / `export_left_context` → `test_results.json`, dan pada semua eval set tiap epoch (kunci sama dengan `dev_metrics.jsonl`) → `eval_sets_results.json`; masing-masing dilewati jika sudah ada, kecuali `--rerun-testset`.
* `evalsets` (hanya jika diminta): skor checkpoint lama pada eval set config tanpa training → `eval_sets/<nama checkpoint>.json`; lihat [baseline](#baseline-checkpoint-lama).
* `export`: ONNX streaming fp32 + int8 ke `export/`, cek bahwa decode ONNX cocok dengan PyTorch pada 5 cut test (kesesuaian argmax per frame ≥ 99 % dan PER token ONNX vs PyTorch ≤ 1 %; keduanya dicatat di log), lalu qdat_bench dengan model ONNX fp32.

Exit code: `0` sukses, `143` berhenti dengan aman (mis. SIGTERM; checkpoint sudah di-flush), `1` gagal.

## Evaluasi tiap epoch, augmentasi, dan baseline

Model v0 (Drive: `checkpoints/utluma-zipformer-ctc-v0`, rata-rata epoch 28–30) sudah jenuh di format datanya sendiri: test PER 0,26 % (qari yang disisihkan), dev PER 0,29 %, dan kurva dev datar sejak sekitar epoch 18. Pada suara awam ia jauh lebih buruk (QDAT, 159 rekaman HP, 120 di antaranya perempuan: PER 16,0 %) dan peka terhadap volume pelan, gema, dan tempo cepat. Dev lama (10 % acak dari kumpulan latih, qari yang sama) tidak bisa melihat itu. Karena itu tiap epoch sekarang dievaluasi pada beberapa set, dan augmentasi baru menyasar kelemahan yang terukur.

### Eval set tiap epoch

Set dibentuk saat training dimulai dari berkas yang sudah ada di `data_dir`. Pembagian train/dev dan shard data tidak berubah.

| Set (prefix kunci) | Isi | Field config |
|---|---|---|
| (tanpa prefix) | dev lama, `dev_max_cuts` cut pertama, persis seperti v0 | `dev_max_cuts` |
| `dev_unseen` | sampel tetap dari `cuts_test` (moshaf 19.0, 29.0, 30.0), `dev_unseen_cuts_per_moshaf` cut per moshaf, diambil dengan seed `seed` dari cut yang diurutkan per id | `dev_unseen_cuts_per_moshaf` (0 = mati) |
| `qdat` | 159 klip `obadx/qdat_bench`, label `phonetic_transcript` tanpa spasi (sama dengan penilai qdat_bench); audio di-decode seperti jalur ONNX (librosa 16 kHz), disimpan sekali di `data_dir/eval_qdat/`. Hanya untuk evaluasi, tidak pernah dilatih | `eval_qdat` |
| `dev_unseen_vol-20db` | sampel `dev_unseen` dengan volume −20 dB | `stress_sets` |
| `dev_unseen_reverb` | sampel `dev_unseen` dengan gema simulasi tetap (bank 64 RIR dari seed `seed`, cut ke-*i* memakai RIR ke-*i* mod 64) | `stress_sets` |
| `dev_unseen_speed1.25` | sampel `dev_unseen` dengan speed ×1,25 | `stress_sets` |

Semua set di-decode dengan jalur PyTorch yang sama (`evaluate_split`, `export_chunk_size` / `export_left_context`). Satu epoch = satu baris `dev_metrics.jsonl`; kunci dev lama tidak berubah, set lain diberi prefix:

```json
{"epoch": 5, "batch_idx_train": 72100, "per_phonemes": 0.0061, "silence_per": 0.0001, "num_speech": 1819, "num_silence": 181,
 "dev_unseen/per_phonemes": 0.0052, "dev_unseen/num_speech": 3000,
 "dev_unseen_vol-20db/per_phonemes": 0.011, "dev_unseen_vol-20db/num_speech": 3000, "...": "...",
 "qdat/per_phonemes": 0.152, "qdat/num_speech": 159}
```

TensorBoard / W&B: dev lama di `dev/*`, set lain di `eval/<set>/*`. Stage `eval` menulis metrik yang sama untuk model rata-rata ke `eval_sets_results.json`.

`dev_unseen` memakai qari test. Itu disengaja (qari latih hanya 17, jadi tidak ada qari yang disisihkan lagi), dengan konsekuensi: setelah memilih epoch atau config berdasarkan `dev_unseen`, `test_results.json` bukan lagi angka yang sama sekali tak tersentuh.

Biaya: di run v0 (RTX 5090), jeda antara log train terakhir dan baris dev, termasuk simpan checkpoint epoch dan eval dev 2000 cut (±6 jam audio), hanya 12–17 detik per epoch, dari epoch ±57,5 menit. Set baru di `train_config.yml` berisi ±30 jam audio (3000 cut `dev_unseen` × 4 set ditambah QDAT), sehingga perkiraannya +1–1,5 menit per epoch (±2–3 %). Bank RIR dibuat sekali saat start (±1 detik), dan QDAT diunduh sekali per `data_dir`.

### Metrik early stop

`early_stop_metric` (bawaan `qdat/per_phonemes`) adalah kunci `dev_metrics.jsonl` yang diikuti aturan rekor watchdog dan `zipformer_status.py`. Pilihannya `per_phonemes` (dev lama) atau `<set>/per_phonemes` dari tabel di atas. `run_zipformer_training.py --early-stop-metric <kunci>` mengganti nilai config untuk satu run. Jika belum ada baris yang memuat kunci itu (`dev_metrics.jsonl` run lama), watchdog dan status memakai `per_phonemes` dan menyebutkannya, sehingga berkas lama tetap terbaca.

Catatan soal QDAT sebagai metrik henti: QDAT hanya 159 klip pendek (9.241 fonem). Bootstrap per klip atas prediksi v0 memberi PER 16,0 % dengan simpangan baku 1,6 poin (±10 % relatif, CI 95 % 13,0–19,3 %), dan PER per klip sangat berekor (p90 40 %). Ambang 5 % relatif berada di bawah derau pengambilan sampel itu; selisih antar epoch pada klip yang sama lebih kecil, tetapi segelintir klip bisa menggeser rekor. Selain itu, berhenti berdasarkan QDAT berarti QDAT tidak lagi angka akhir yang netral. Alternatif yang lebih stabil adalah `dev_unseen_vol-20db/per_phonemes` atau `dev_unseen_reverb/per_phonemes` (3000 cut), dengan QDAT tetap dicatat.

### Augmentasi

Semua bawaan di `config.py` mematikan augmentasi baru, sehingga config lama menghasilkan pipeline v0. `train_config.yml` menyalakan nilai baru:

| Field | Bawaan (= v0) | `train_config.yml` | Keterangan |
|---|---|---|---|
| `speed_perturb`, `speed_factor_ranges` | `true`, `[[0.9,0.9],[1.0,1.0],[1.1,1.1]]` | `[[0.85,0.95],[1.0,1.0],[1.05,1.25]]` | satu salinan statis per rentang; faktor tiap cut diambil di grid 0,05 dari seed + id cut (`copy_factors`). Rentang harus terurut dan tidak tumpang tindih |
| `cutmix_p`, `cutmix_snr` | `0.5`, `[10, 20]` | `0.5`, `[5, 20]` | derau dicampur pada SNR seragam |
| `reverb_p` | `0.0` | `0.3` | gema RIR simulasi Lhotse (tanpa unduhan), dikenakan pada ucapan sebelum derau |
| `volume_p`, `volume_db` | `0.0`, `[-30, 6]` | `0.5`, `[-30, 6]` | gain seragam dalam dB, paling akhir (sesudah derau), seperti mikrofon pelan. `PerturbVolume` Lhotse mengambil faktor linear 0,125–2,0 sehingga level pelan hampir tak pernah terambil |

Urutan transform: gema → CutMix → gain → fbank → SpecAugment. Dengan nilai di atas, 35 % salinan tidak kena gema maupun gain, dan 17,5 % juga tanpa derau, sehingga PER bersih tetap terlatih.

Speed tetap berupa salinan statis, bukan transform acak sesudah batching. Akibatnya CutSet sudah tetap sebelum sampler, durasi batch tetap menghormati `max_duration` (memori GPU), dan resume tetap tepat. Kelayakan frame CTC (`L + R`) dicek per salinan pada faktor yang benar-benar dikenakan; cut yang hanya tidak muat pada salinan tercepatnya tetap memakai salinan lainnya (`dropped_cuts.json` mencatat id salinan). `max_dropped_fraction` dihitung atas salinan train ditambah cut dev. Di data saat ini rasio frame minimum pada speed 1,1 adalah 2,9, jadi pada 1,25 (±2,6) tidak ada yang terbuang. Rata-rata `1/faktor` turun dari 3,020 ke 2,986 jam audio per jam data, sehingga epoch ±1,1 % lebih pendek (±14,26 ribu batch dari ±14,43 ribu). Jadwal LR Eden (per batch dan per epoch) praktis tidak berubah.

Resume: sampler tetap melanjutkan tepat di batch berikutnya dengan isi cut yang sama. Dengan CutMix aktif, urutan cut *di dalam* sebuah batch bisa berbeda setelah resume (CutMix mem-pad cut yang dicampur, dataset mengurutkan batch per durasi, dan RNG CutMix mulai ulang). Ini sudah berlaku di v0; tes lama memakai cut berdurasi sama sehingga tidak terlihat.

Tidak dikerjakan:

* Tempo: `PerturbTempo` Lhotse memakai `torchaudio.sox_effects`, yang tidak ada lagi di torchaudio 2.9+ (sudah dicek di 2.11). Rentang speed yang lebih lebar menggantikannya.
* Geser pitch: `torchaudio.functional.pitch_shift` butuh ±0,36 detik per klip 12 detik (1 thread). Pada p 0,3 itu ±5 detik per batch per worker, setara dengan anggaran ±5,8 detik per batch per worker (24 worker, ±0,24 detik per langkah GPU), sehingga data loader akan menjadi leher botol. Gema pada p 0,3 hanya menambah ±0,3 detik per batch per worker (±19 ms per klip).

### Baseline checkpoint lama

Angka v0 pada set baru, tanpa training, dengan `configs/eval_baseline_v0.yml` (dimensi model dan eval set sama dengan `train_config.yml`, `output_dir` sendiri):

```bash
# di box GPU (stage evalsets menjalankan prepare dulu; shard yang sudah ada dilewati)
python train_zipformer.py --config configs/eval_baseline_v0.yml --stage evalsets \
    --checkpoint utluma-edit:RClone/Utluma/zipformer-data/checkpoints/utluma-zipformer-ctc-v0-results/averaged-3.pt \
    --checkpoint utluma-edit:RClone/Utluma/zipformer-data/checkpoints/utluma-zipformer-ctc-v0/epoch-30.pt

# atau dari komputer lokal lewat Vast.ai (upload hasil + destroy setelah selesai)
python vastai/run_zipformer_training.py --offer-id <OFFER_ID> --config configs/eval_baseline_v0.yml \
    --destroy-when-done --follow -- --stage evalsets \
    --checkpoint utluma-edit:RClone/Utluma/zipformer-data/checkpoints/utluma-zipformer-ctc-v0-results/averaged-3.pt
```

Hasilnya `results-zipformer-v0-baseline/eval_sets/<checkpoint>.json`, dengan kunci yang sama dengan `dev_metrics.jsonl`; dengan `--destroy-when-done` diunggah ke `.../utluma-zipformer-ctc-v0-baseline-results`. Angka pemeriksa untuk `averaged-3.pt`: dev ±0,29 % (epoch 28–30 tercatat 0,29–0,32 %), dan QDAT ±16,0 %. Jalur ONNX v0 memberi 16,03 %; skor ulang prediksi itu dengan tokenisasi dan label tanpa spasi yang dipakai di sini juga 16,03 %. Jika QDAT jauh dari 16 %, curigai decode audio atau label. Test set penuh (0,26 %) tidak termasuk `evalsets`; jalankan `--stage eval` dengan `averaged-3.pt` di `output_dir` jika perlu.

## 4. Lokasi output

Semua di `output_dir` (di `train_config.yml`: `./results-zipformer-v1`, yaitu `zipformer/results-zipformer-v1`):

| Path | Isi |
|---|---|
| `checkpoint-*.pt`, `epoch-*.pt` (+ `.meta.json`) | checkpoint tiap `save_every_n_batches` batch dan tiap epoch (masing-masing `keep_last_k` terakhir); juga di-backup ke `rclone_checkpoint_path` |
| `averaged-<avg_last_k>.pt` | model akhir (rata-rata epoch terakhir) |
| `config.yml` | config yang dipakai |
| `dev_metrics.jsonl` | satu baris per epoch: metrik dev lama + semua eval set (lihat [Evaluasi tiap epoch](#evaluasi-tiap-epoch-augmentasi-dan-baseline)) |
| `dropped_cuts.json` | salinan speed (train) dan cut dev yang gagal cek frame CTC |
| `tb/` | log TensorBoard (W&B juga dipakai jika `WANDB_API_KEY` di-set) |
| `test_results.json`, `eval_sets_results.json` | hasil stage `eval` (model rata-rata) |
| `eval_sets/<checkpoint>.json` | hasil `--stage evalsets` |
| `qdat_bench_predictions.jsonl`, `qdat_bench_test_results.json` | hasil qdat_bench |
| `export/model.onnx`, `export/model.int8.onnx`, `export/tokens.txt` | model untuk sherpa-onnx |

Download ke komputer lokal:

```bash
scp -P <PORT> -r root@<IP_VAST_AI>:~/prepare-quran-dataset/zipformer/results-zipformer-v1/export ./
```

## Jika box Vast.ai mati di tengah training

**Pemulihan:** sewa box baru, salin `rclone.conf` dan `.env` seperti di atas, lalu jalankan perintah yang sama:

```bash
bash train_zipformer.sh
```

Perintah ini sekaligus perintah restart: data diunduh ulang (shard HF, lalu shard Drive, lalu file mentah Drive sebagai cadangan), checkpoint terbaru diunduh dari `rclone_checkpoint_path`, dan training lanjut dari batch tersebut. Sebelum training, log menampilkan satu baris seperti:

```
resume: source=drive checkpoint=checkpoint-48000.pt epoch=5 batch_idx_train=48000 age=0h 23m
```

`age` adalah umur checkpoint yang dipakai (dari waktu modifikasi file `.pt`; `rclone copyto` mempertahankan waktu modifikasi dari Drive). `source=drive` berarti checkpoint diambil dari Drive (normal di box baru), `source=local` berarti checkpoint lokal dipakai, `source=none checkpoint=- epoch=1 batch_idx_train=0` berarti training mulai dari awal. Jika di box baru muncul `source=none` padahal Drive berisi checkpoint, hentikan dan periksa `rclone_checkpoint_path` / `rclone.conf`.

Perkiraan kerugian: progres maksimal satu interval `save_every_n_batches` (~25 menit di 4090) ditambah waktu setup ~30–60 menit (install environment + sinkronisasi data ~60 GB). Angka setup ini belum terukur: sinkronisasi ~250–300 ribu file kecil dari Drive bisa berjam-jam (lihat "Waktu sinkronisasi data" di atas).

**Jika box interruptible yang sama hidup lagi**, cukup jalankan ulang perintah yang sama: checkpoint lokal dipakai, kecuali Drive punya checkpoint yang lebih baru.

**On-demand vs interruptible:**

* **On-demand** tidak akan diambil kembali oleh Vast.ai, tetapi host tetap bisa gagal (mati, jaringan putus); pilih host dengan reliability score tinggi.
* **Interruptible** lebih murah, tetapi bisa dihentikan kapan saja jika ada yang menawar lebih tinggi, dan bisa berhenti berjam-jam.
* Rekomendasi: interruptible hanya layak jika `rclone_checkpoint_path` di-set (bukan `null`). Pindah ke on-demand jika terhenti lebih dari ~sekali sehari, atau jika mendekati deadline. Cek harga terkini langsung di Vast.ai.

**Ruang Google Drive** yang dibutuhkan: data ~60 GB + 2 × `keep_last_k` × ~1 GB checkpoint (`checkpoint-*.pt` dan `epoch-*.pt` masing-masing `keep_last_k`). Batas upload Google Drive 750 GB/hari; backup checkpoint memakai ≈60 GB/hari, jadi masih jauh di bawah batas.

**Log:** riwayat TensorBoard (`tb/`) hilang bersama box. W&B melanjutkan run yang sama (run id disimpan di checkpoint), jadi kurva lengkap ada di W&B.

## Frame rate 20 ms dan sherpa-onnx

Model ini mengeluarkan **satu frame per 20 ms** (`output_downsampling_factor: 1`), berbeda dengan v3 yang 40 ms (`output_downsampling_factor: 2`). Alasannya: vocab kita per karakter, sehingga mad dan tasydid ditulis sebagai deretan karakter yang sama (mis. `…رَحِۦۦۦۦم`). CTC wajib menyisipkan blank di antara dua label identik yang berurutan, jadi satu cut butuh minimal `L + R` frame output (`L` = panjang label, `R` = jumlah pasangan identik berurutan; `R` menambah 18–25 % pada contoh ayat). Pada 40 ms, bacaan cepat ditambah speed perturb 1.1 hampir tidak menyisakan ruang untuk blank; 20 ms menggandakannya dan membuat `silence_per` serta timing bisa dibandingkan langsung dengan model muaalem. Komputasi encoder tidak berubah (downsample terakhir ada setelah stack terakhir); hanya head CTC dan greedy decode yang berjalan pada 2× frame.

Dampak di sherpa-onnx (diverifikasi pada sherpa-onnx `347ec76`, `sherpa-onnx/csrc/online-recognizer-ctc-impl.h`):

* **Teks hasil pengenalan tidak terpengaruh.** sherpa-onnx memakai berapa pun frame output yang dikembalikan model per chunk.
* `GetResult` meng-hard-code `frame_shift_ms = 10` dan `subsampling_factor = 4`, sehingga **timestamp token 2× terlalu besar**. Bagi 2 di aplikasi, atau baca `output_frame_shift_ms` dari metadata model.
* `IsEndpoint` juga mengalikan trailing blank frame dengan angka 4 yang sama, sehingga trailing silence terhitung 2× dan aturan endpoint terpicu setelah setengah dari silence yang dikonfigurasi. Gandakan nilai `rule1_min_trailing_silence`, `rule2_min_trailing_silence`, dan `rule3_min_trailing_silence` di konfigurasi aplikasi untuk mengompensasi.
* Tidak ada key metadata yang mengubah perilaku ini di sherpa-onnx saat ini. ONNX kita tetap membawa metadata `output_frame_shift_ms` (`20`), `subsampling_factor` (`2`) dan `output_downsampling_factor` untuk aplikasi yang membacanya.

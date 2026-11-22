# Zipformer2 CTC streaming model

Model streaming icefall Zipformer2 (`causal=True`) dengan satu head CTC (level `phonemes`, vocab `vocab_streaming/vocab.json` di root repo), diekspor ke ONNX streaming untuk sherpa-onnx (Android). Semua parameter ada di `configs/train_config.yml`.

Semua kode training Zipformer berakar di folder `zipformer/` ini; semua path di dokumen ini relatif terhadap `zipformer/` (jalankan perintah dari folder ini). Kode ini tetap memakai paket `prepare_quran_dataset` di root repo (tokenizer qdat_bench, `moshaf_ways`) dan `vocab_streaming/`.

| Path | Isi |
|---|---|
| `quran_zipformer/` | paket Python (data prep, datamodule, model, training loop, decode, export ONNX, qdat_bench) |
| `train_zipformer.py`, `train_zipformer.sh` | CLI per stage dan pipeline end-to-end untuk Vast.ai |
| `prepare_zipformer_data_colab.py`, `notebooks/prepare_zipformer_data.ipynb` | persiapan data di Colab |
| `configs/train_config.yml` | config training |
| `infer_ayah.py`, `live_transcribe.py`, `compare_models.py` | transkripsi satu berkas, transkripsi langsung dari mikrofon, dan perbandingan dua model |
| `robustness_audit.py` | audit ketahanan satu model ONNX terhadap model acuan (lihat [Audit ketahanan](#audit-ketahanan-v0-terhadap-model-acuan)) |
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

Sewa box sesuai [patokan box](#patokan-box-dan-run-vastai) (GPU, **CPU, RAM**, disk, reliability, credit), lalu:

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

Tambahkan `--early-stop-patience N` untuk berhenti lebih awal saat metrik early stop mendatar: `vastai/zipformer_watchdog.sh` berjalan di box (sesi tmux `watchdog`, log `~/zipformer_watchdog.log`) dan memeriksa `dev_metrics.jsonl` tiap 2 menit. Metriknya adalah kunci `early_stop_metric` di config (`dev_unseen/per_phonemes` di `configs/train_config.yml`; bawaan kode jika kuncinya tidak ada: `qdat/per_phonemes`), atau `--early-stop-metric <kunci>` untuk run itu saja (lihat [Evaluasi tiap epoch](#evaluasi-tiap-epoch-augmentasi-dan-baseline)); `dev_metrics.jsonl` yang belum memuat kunci itu (mis. run lama) memakai `per_phonemes` dev lama, dan log watchdog menyebutkannya. Sebuah epoch dihitung rekor baru hanya jika metriknya lebih rendah setidaknya `--early-stop-min-improvement` (default `0.05`, relatif) dari rekor sebelumnya. Setelah N epoch berturut-turut tanpa rekor baru, watchdog mengirim SIGTERM ke training (berhenti rapi, checkpoint di-flush), mengubah `num_epochs` di config box ke epoch terakhir, lalu menjalankan ulang pipeline yang sama: averaging 3 epoch terakhir, evaluasi test, export, dan dengan `--destroy-when-done` juga upload hasil + destroy. Watchdog tidak pernah memotong averaging / eval / export (hanya bertindak selama epoch terakhir < `num_epochs`).

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

### Patokan box dan run Vast.ai

Patokan dari run v1 (2026-11-23). Ikuti ini untuk setiap run berikutnya; tiap baris berasal dari kegagalan atau biaya nyata.

| Aspek | Patokan | Kenapa |
|---|---|---|
| GPU | 1× RTX 5090 32 GB (atau VRAM ≥ 32 GB) | `max_duration: 600` memakai ±22–30 GB VRAM |
| **CPU** | `cpu_cores_effective` ≥ `num_workers` di config (24) | fbank, CutMix, speed perturb, reverb, dan noise dihitung on the fly di worker dataloader |
| **RAM** | ≥ 64 GB | box 8 core / 30 GB RAM di-OOM-kill (exit 137, `oom_kill` cgroup) 4 menit setelah epoch 1 dimulai; box 48 core / 108 GB memakai ±34 GB. Launcher menolak box di bawah CPU/RAM ini di langkah [1/6] (`--skip-resource-check` untuk melewatinya) |
| Disk | `--disk 110` cukup | data 65 GiB (tar dihapus setelah diekstrak) + environment + checkpoint (`keep_last_k: 3`): terpakai ±85 GB sampai epoch 4 |
| Tipe sewa | on-demand, bukan interruptible (bid) | instance bid bisa direbut penyewa lain di tengah epoch; `vastai search offers` default-nya on-demand |
| Reliability | ≥ 0.99 | host 0.971 berubah `actual_status: offline` lalu hilang dari API sebelum training mulai |
| Download | `inet_down` ≥ 500 Mbps, perhatikan `inet_down_cost` ($/GB) | 65 GiB: ±15 menit di 590 Mbps, ±4 menit di 7.9 Gbps; dengan $0.039/GB data saja sudah ±$2.7 |
| Credit | cek sebelum launch: ±$0.85/epoch (±73 menit/epoch di RTX 5090 + 48 core, $0.695/jam) → 30 epoch ±$26 + tahap akhir | credit yang habis mematikan box sebelum averaging/test/export; `zipformer_status.py` menandai `CREDIT TIDAK CUKUP` |
| Metrik early stop | `dev_unseen/per_phonemes` (default config) | QDAT (159 klip) terlalu berisik, lihat [Metrik early stop](#metrik-early-stop) |
| Zona waktu | sesi training & watchdog berjalan di WIB (`TZ=WIB-7`, di-set launcher) | log dan `zipformer_status.py` tampil WIB; shell SSH tetap UTC |

Mencari offer yang memenuhi patokan (urut harga):

```bash
vastai search offers 'gpu_name=RTX_5090 num_gpus=1 reliability>0.99 verified=true rentable=true cuda_vers>=12.8 disk_space>=110 cpu_cores_effective>=24 cpu_ram>=64000 inet_down>=500' -o 'dph_total' --storage 110
python vastai/run_zipformer_training.py --offer-id <OFFER_ID> --disk 110 --destroy-when-done --early-stop-patience 4
```

Jika `--offer-id` gagal dengan "Unknown error", jalankan `vastai show instances`: instance bisa sudah terbuat dalam status `stopped` (GPU-nya keburu disewa orang lain) dan tetap ditagih storage; hapus dengan `vastai destroy instance <ID>` sebelum mencoba offer lain. Image pytorch tidak punya `unzip` (dibutuhkan installer rclone); launcher memasangnya di langkah [2/6].

## Evaluasi tiap epoch, augmentasi, dan baseline

Model v0 (Drive: `checkpoints/utluma-zipformer-ctc-v0`, rata-rata epoch 28–30) sudah jenuh di format datanya sendiri: test PER 0,26 % (qari yang disisihkan), dev PER 0,29 %, dan kurva dev datar sejak sekitar epoch 18. Pada suara awam ia jauh lebih buruk (QDAT, 159 rekaman HP, 120 di antaranya perempuan: PER 16,0 %) dan peka terhadap volume pelan, gema, dan tempo cepat. Dev lama (10 % acak dari kumpulan latih, qari yang sama) tidak bisa melihat itu. Karena itu tiap epoch sekarang dievaluasi pada beberapa set, dan augmentasi baru menyasar kelemahan yang terukur.

### Audit ketahanan v0 terhadap model acuan

`robustness_audit.py` membandingkan satu model ONNX dengan model acuan pada audio yang sama. Keduanya di-decode lewat sherpa-onnx streaming dalam blok 100 ms, seperti di aplikasi. Angka di bawah adalah v0 (`export-epoch-20/model.int8.onnx`) terhadap `zipformer_p_arabic_v3.1.int8.onnx`, dijalankan 2026-11-22. PER dihitung terhadap teks Al-Qur'an dan menoleransi panjang madd yang boleh berbeda.

| Kelemahan v0 | v0 | v3.1 | Jawaban di `train_config.yml` | Terlihat tiap epoch di |
|---|---|---|---|---|
| Bacaan bersambung: 539 ayat berurutan dari 7 qari, tiap blok sebagai satu stream | PER 4,7 % per ayat → 6,3 % satu stream; awal 269 dari 539 ayat hilang | 2,8 % → 3,2 %; 31 dari 539 (26 di satu qari) | `concat_p` dengan jeda berisi derau | `dev_unseen_pair-gap` |
| Hal yang sama pada Al-Fatihah saja (7 qari) | 0,5 % → 6,3 %; 33 dari 49 awal ayat hilang | 0,0 % → 0,7 %; 3 dari 49 | sama | sama |
| Derau ruangan 2–5 detik sebelum ucapan | 2,9–3,8 % | 0,3 % | `pad_p`, `nonspeech_noise_p` | `dev_unseen_lead-noise` |
| Jeda berderau 0,5–2 detik di antara dua ucapan | 2,6–3,1 % | 0,3–0,4 % | `concat_gap_seconds` | `dev_unseen_pair-gap` |
| Volume pelan: −20 / −30 / −40 dB | 3,3 / 22,9 / 54,2 % (pada −40 dB, 17 dari 48 keluaran hampir kosong) | 0,3 / 0,5 / 0,7 % | `volume_db: [-40, 6]`, `volume_p: 0.6` | `dev_unseen_vol-20db`, `dev_unseen_vol-35db` |
| HP dari jauh (−30 dB, derau SNR 15, pita 300–3400 Hz) | 63,8 % | 0,8 % | gabungan volume dan CutMix | `dev_unseen_vol-35db` (hanya sisi volumenya) |
| Mikrofon terlalu keras: gelombang terpotong pada 25 % / 10 % puncak | 2,1 / 14,8 % | 0,4 / 1,7 % | `clip_p` | `dev_unseen_clip` |
| Tempo cepat tanpa perubahan pitch: ×1,25 / ×1,5 | 2,7 / 7,6 % | 0,5 / 1,0 % | `tempo_p` | tidak ada; hanya audit ini |
| Speed ×1,2; pitch +4 semiton (suara lebih tinggi) | 2,9 %; 3,5 % | 0,2 %; 0,3 % | `speed_factor_ranges` sampai 1,3 | `dev_unseen_speed1.25` |
| Gema; gema dengan tempo ×1,25 | 0,7–1,2 %; 3,8 % | 0,2–0,3 %; 0,5 % | `reverb_p`, `tempo_p` | `dev_unseen_reverb` |
| Codec Opus 8 kbps | 4,1 % | 0,7 % | belum ada (lihat bawah) | tidak ada; hanya audit ini |
| Rekaman HP orang awam (QDAT, 120 dari 159 perempuan) | 16,1 %; 24 klip di atas 30 % | 7,3 %; 1 klip | butuh data baru (lihat bawah) | `qdat` |
| Panel 56 folder qari, 374 ayat | 5,3 % (3,2 % tanpa kata yang diulang qari); fonem pertama salah atau hilang di 48 ayat | 1,5 % (0,7 %); 14 ayat | butuh data baru | `dev_unseen` (3 moshaf) |

v0 setara dengan acuan pada derau sesudah ucapan, volume +12 dB tanpa terpotong, pita telepon, low-pass 2 kHz, sampling 8 kHz, MP3 16 kbps, derau merah muda dan derau ruangan pada SNR 10, tempo ×0,8, dan speed ×0,85. Pada masukan tanpa ucapan (hening digital, derau putih, merah muda, dan ruangan sampai −20 dBFS) kedua model tidak mengeluarkan token. Ocehan orang lain pada SNR 5 menggagalkan keduanya (32 % dan 30 %), jadi itu bukan kelemahan khusus v0.

Cara membaca angkanya:

* PER di panel dan di bacaan bersambung memuat pengulangan qari. Di ayat panjang qari sering berhenti mengambil napas lalu mengulang beberapa kata, sedangkan teks rujukan memuatnya sekali (contoh: Al-Kahf 14, 16, dan 19 bacaan Husary, 42–86 "kesalahan" per ayat di kedua model). v0 selalu menuliskan pengulangan itu. v3.1 menuliskan sebagian besar, tetapi kadang melewatkannya (contoh: Yasin 14 dan 18 bacaan Hudhaify), sehingga pengulangan ikut memperlebar selisih. Karena itu laporan panel punya baris "tanpa kata yang diulang qari", yang membuang rentetan sisipan 6 fonem atau lebih (`repeated`). Tahap `stream` tidak dikoreksi; di sana yang bermakna adalah kenaikan dari per ayat ke satu stream dan jumlah awal ayat yang hilang.
* Pada lima qari latih v0 (Husary, Minshawi, Abdul Basit, Hudhaify, Ayman Sowaid; 385 ayat, satu berkas per ayat, tanpa kata yang diulang), v0 sudah dekat dengan acuan selama ayat dibaca dalam satu napas: 0,26 % terhadap 0,06 % (263 ayat; 234 dan 254 di antaranya tanpa satu pun kesalahan). Pada ayat yang memuat jeda selisihnya melebar: 2,25 % terhadap 0,65 % (122 ayat). Jika ayat yang sama dipotong di jedanya, seperti format data latih v0, kesalahan v0 turun ke 1,61 %. Bagian itulah yang disasar `concat_p` dan `pad_p`; sisanya tidak. Ini pengukuran sekali jalan, tidak ada di `robustness_audit.py`.
* Baterai degradasi memakai 48 ucapan 4–14 detik yang dibaca hampir benar oleh kedua model saat bersih (paling banyak dua per qari), jadi selisihnya berasal dari degradasinya.
* Satu folder panel (`Abdullaah_3awwaad_Al-Juhaynee_128kbps`) tidak dihitung: model acuan pun salah 99 %, artinya berkasnya tidak cocok dengan nomor ayatnya. Laporan menyisihkan folder seperti itu dan menyebutkannya.

Yang tidak bisa dijawab augmentasi:

* **Ragam penutur.** v0 dilatih pada 23 moshaf dari ±18 qari profesional laki-laki. Pada klip QDAT yang gagal, membuang hening awal, menggeser pitch, dan menormalkan level tidak menolong (PER tetap 45–60 %). Selisih di QDAT dan di qari yang tidak dilatih butuh data baru: suara perempuan, anak, dan rekaman HP. Model acuan dilatih pada lebih dari seribu penutur.
* **Bacaan cepat asli.** `tempo_p` meniru tempo, bukan cara membaca hadr. Moshaf `25.0` (hadr, ±17 jam) belum dipakai; memasukkannya berarti menyiapkan ulang data dan shard, dan pembagian dev ikut berubah.
* **Codec bitrate rendah.** Meniru Opus butuh encoder di data loader. Aplikasi mengambil PCM langsung dari mikrofon, dan MP3 16 kbps serta pita telepon tidak bermasalah, jadi ini ditunda.

Menjalankan audit untuk model baru (butuh `sherpa-onnx`, `librosa`, `sox`, dan `ffmpeg`; tanpa `sox` atau `ffmpeg`, kondisi yang memakainya dilewati):

```bash
python robustness_audit.py --out audit-v1 \
    --model v1 results-zipformer-v1/export/model.int8.onnx results-zipformer-v1/export/tokens.txt \
    --model v3.1 results-zipformer/zipformer_p_arabic_v3.1.int8.onnx results-zipformer/tokens.txt \
    --audio-root <folder berisi <qari>/SSSAAA.mp3> \
    --room-noise <rekaman derau ruangan.wav>

python robustness_audit.py --out audit-v1 --report   # susun ulang laporan dari .jsonl yang ada
```

`--model` pertama adalah model yang diuji dan yang terakhir acuannya. Tahapnya `panel`, `qdat`, `battery`, `stream`, dan `noise` (`--stages`); tiap tahap menulis `<out>/<tahap>.jsonl`, dan `<out>/report.md` menandai baris yang tertinggal lebih dari 1 poin PER dari acuan. `qdat` memakai parquet `obadx/qdat_bench` dari cache Hugging Face atau `--qdat`. Seluruh tahap untuk dua model selesai dalam ±7 menit dengan `--jobs 7` di Mac 18 core.

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
| `dev_unseen_vol-35db` | sampel `dev_unseen` dengan volume −35 dB (mikrofon sangat pelan) | `stress_sets` |
| `dev_unseen_clip` | sampel `dev_unseen` yang gelombangnya dipotong pada 10 % puncaknya (mikrofon terlalu keras). Cut tidak bisa menyimpan operasi ini, jadi loader set ini memakai `wave_transforms` (`eval_sets.wave_transforms`) | `stress_sets` |
| `dev_unseen_lead-noise` | sampel `dev_unseen` dengan 2 detik derau sebelum tiap cut, 30 dB di bawah ucapan (derau ruangan sebelum qari mulai) | `stress_sets` |
| `dev_unseen_pair-gap` | cut bertetangga dari satu moshaf digabung dua-dua melewati jeda 1 detik berisi derau (−30 dB); labelnya gabungan kedua label. Jumlah cut jadi separuh (1500), cut ganjil terakhir tiap moshaf dibuang | `stress_sets` |

Derau untuk `lead-noise` dan `pair-gap` adalah derau merah muda sintetis dari `seed` (`eval_sets.noise_bank`), bukan `cuts_noise_aug`, supaya set stres tidak memakai derau yang dilatihkan.

Semua set di-decode dengan jalur PyTorch yang sama (`evaluate_split`, `export_chunk_size` / `export_left_context`). Satu epoch = satu baris `dev_metrics.jsonl`; kunci dev lama tidak berubah, set lain diberi prefix:

```json
{"epoch": 5, "batch_idx_train": 72100, "per_phonemes": 0.0061, "silence_per": 0.0001, "num_speech": 1819, "num_silence": 181,
 "dev_unseen/per_phonemes": 0.0052, "dev_unseen/num_speech": 3000,
 "dev_unseen_vol-20db/per_phonemes": 0.011, "dev_unseen_vol-20db/num_speech": 3000, "...": "...",
 "qdat/per_phonemes": 0.152, "qdat/num_speech": 159}
```

TensorBoard / W&B: dev lama di `dev/*`, set lain di `eval/<set>/*`. Stage `eval` menulis metrik yang sama untuk model rata-rata ke `eval_sets_results.json`.

`dev_unseen` memakai qari test. Itu disengaja (qari latih hanya 17, jadi tidak ada qari yang disisihkan lagi), dengan konsekuensi: setelah memilih epoch atau config berdasarkan `dev_unseen`, `test_results.json` bukan lagi angka yang sama sekali tak tersentuh.

Biaya: di run v0 (RTX 5090), jeda antara log train terakhir dan baris dev, termasuk simpan checkpoint epoch dan eval dev 2000 cut (±6 jam audio), hanya 12–17 detik per epoch, dari epoch ±57,5 menit. Set di `train_config.yml` berisi ±60 jam audio (3000 cut `dev_unseen` × 8 set, termasuk tujuh set stres, ditambah QDAT), sehingga perkiraannya +2–3 menit per epoch (±4–5 %). Angka ini ekstrapolasi dari kecepatan eval dev di run v0, belum diukur. Bank RIR dibuat sekali saat start (±1 detik), dan QDAT diunduh sekali per `data_dir`.

### Metrik early stop

`early_stop_metric` (`dev_unseen/per_phonemes` di `configs/train_config.yml` sejak 2026-11-23; bawaan kode `qdat/per_phonemes`) adalah kunci `dev_metrics.jsonl` yang diikuti aturan rekor watchdog dan `zipformer_status.py`. Pilihannya `per_phonemes` (dev lama) atau `<set>/per_phonemes` dari tabel di atas. `run_zipformer_training.py --early-stop-metric <kunci>` mengganti nilai config untuk satu run. Jika belum ada baris yang memuat kunci itu (`dev_metrics.jsonl` run lama), watchdog dan status memakai `per_phonemes` dan menyebutkannya, sehingga berkas lama tetap terbaca.

Catatan soal QDAT sebagai metrik henti: QDAT hanya 159 klip pendek (9.241 fonem). Bootstrap per klip atas prediksi v0 memberi PER 16,0 % dengan simpangan baku 1,6 poin (±10 % relatif, CI 95 % 13,0–19,3 %), dan PER per klip sangat berekor (p90 40 %). Ambang 5 % relatif berada di bawah derau pengambilan sampel itu; selisih antar epoch pada klip yang sama lebih kecil, tetapi segelintir klip bisa menggeser rekor. Selain itu, berhenti berdasarkan QDAT berarti QDAT tidak lagi angka akhir yang netral. Alternatif yang lebih stabil adalah `dev_unseen_vol-20db/per_phonemes` atau `dev_unseen_reverb/per_phonemes` (3000 cut), dengan QDAT tetap dicatat. Run v1 membuktikannya: QDAT epoch 1–4 bergerak 22,9 / 22,4 / 23,5 / 19,3 % sementara `dev_unseen` turun tiap epoch (9,7 / 5,7 / 5,0 / 4,1 %); dengan QDAT, watchdog nyaris menghentikan training setelah epoch 5 saat model masih membaik tajam. Karena itu config memakai `dev_unseen/per_phonemes` (3000 cut) dan QDAT hanya dilaporkan.

### Augmentasi

Semua bawaan di `config.py` mematikan augmentasi baru, sehingga config lama menghasilkan pipeline v0. `train_config.yml` menyalakan nilai baru:

| Field | Bawaan (= v0) | `train_config.yml` | Keterangan |
|---|---|---|---|
| `speed_perturb`, `speed_factor_ranges` | `true`, `[[0.9,0.9],[1.0,1.0],[1.1,1.1]]` | `[[0.85,0.95],[1.0,1.0],[1.05,1.3]]` | satu salinan statis per rentang; faktor tiap cut diambil di grid 0,05 dari seed + id cut (`copy_factors`). Rentang harus terurut dan tidak tumpang tindih. Speed menaikkan tempo dan pitch sekaligus |
| `tempo_p`, `tempo_range` | `0.0`, `[1.0, 1.4]` | `0.3`, `[1.0, 1.4]` | tempo tanpa mengubah pitch: frame fbank satu batch di-resample sepanjang waktu (interpolasi linear) dengan satu faktor per batch (`tempo_stretch`, di loop training). Faktor dibatasi supaya label cut terketat di batch tetap muat di frame CTC (`apply_tempo`). Hanya ≥ 1, jadi batch tidak pernah lebih panjang dari `max_duration` |
| `cutmix_p`, `cutmix_snr` | `0.5`, `[10, 20]` | `0.5`, `[5, 20]` | derau dicampur pada SNR seragam |
| `reverb_p` | `0.0` | `0.3` | gema RIR simulasi Lhotse (tanpa unduhan), dikenakan pada ucapan sebelum derau |
| `volume_p`, `volume_db` | `0.0`, `[-30, 6]` | `0.6`, `[-40, 6]` | gain seragam dalam dB, sesudah derau, seperti mikrofon pelan. `PerturbVolume` Lhotse mengambil faktor linear 0,125–2,0 sehingga level pelan hampir tak pernah terambil |
| `clip_p`, `clip_peak_fraction` | `0.0`, `[0.05, 0.5]` | `0.1`, `[0.05, 0.5]` | gelombang akhir dipotong keras pada sebagian dari puncaknya (`RandomClip`, lewat `wave_transforms`), seperti mikrofon yang terlalu keras |
| `pad_p`, `pad_lead_seconds`, `pad_trail_seconds` | `0.0`, `[0.5, 4.0]`, `[0.0, 2.0]` | `0.3`, sama | bagian tanpa ucapan sebelum dan sesudah cut ucapan (`pad_non_speech`). Supervisi cut diperpanjang menutupi bagian itu, jadi frame-nya ikut dilatih (label tidak berubah) |
| `nonspeech_noise_p`, `nonspeech_snr` | `0.75`, `[10, 45]` | sama | isi bagian tanpa ucapan itu dan jeda cut gabungan: potongan cut `cuts_noise_aug` pada SNR terhadap ucapan, selain itu hening digital (`NonSpeech`) |
| `concat_p`, `concat_segments`, `concat_gap_seconds`, `concat_max_seconds` | `0.0`, `[2, 4]`, `[0, 0]`, `null` | `0.3`, `[2, 4]`, `[0, 1.5]`, `45` | segmen berurutan digabung jadi satu cut (lihat di bawah) |

Urutan: gabung segmen → salinan speed → bagian tanpa ucapan (ketiganya menetapkan CutSet sebelum sampler) → gema → CutMix → gain → potong puncak → fbank → SpecAugment → tempo (per batch).

Bagian tanpa ucapan ditetapkan sebelum batching, seperti speed, sehingga durasi batch sudah menghitungnya dan resume tetap tepat. Panjangnya diambil ulang tiap epoch dari seed (`seed`, epoch, id cut): berbeda antar epoch, sama saat epoch yang sama dilanjutkan. Dengan `pad_p` 0,3 dan rata-rata ±3,25 detik per cut yang kena, audio per epoch bertambah paling banyak ±10 % (dihitung untuk cut rata-rata ±9 detik; cut gabungan lebih panjang, jadi persentasenya lebih kecil). Tempo memendekkan 30 % batch rata-rata ±16 %. Perkiraan bersihnya epoch ±5 % lebih lama daripada tanpa keduanya. Ini hitungan di atas kertas, belum diukur di GPU.

Speed tetap berupa salinan statis, bukan transform acak sesudah batching. Akibatnya CutSet sudah tetap sebelum sampler, durasi batch tetap menghormati `max_duration` (memori GPU), dan resume tetap tepat. Kelayakan frame CTC (`L + R`) dicek per salinan pada faktor yang benar-benar dikenakan; cut yang hanya tidak muat pada salinan tercepatnya tetap memakai salinan lainnya (`dropped_cuts.json` mencatat id salinan). `max_dropped_fraction` dihitung atas salinan train ditambah cut dev. Di data saat ini rasio frame minimum pada speed 1,1 adalah 2,9, jadi pada 1,3 (±2,45) tidak ada yang terbuang. Rata-rata `1/faktor` turun dari 3,020 ke 2,969 jam audio per jam data, sehingga dari speed saja epoch ±1,7 % lebih pendek (±14,19 ribu batch dari ±14,43 ribu, sebelum bagian tanpa ucapan dihitung). Jadwal LR Eden (per batch dan per epoch) praktis tidak berubah.

Resume: sampler tetap melanjutkan tepat di batch berikutnya dengan isi cut yang sama. Dengan CutMix aktif, urutan cut *di dalam* sebuah batch bisa berbeda setelah resume (CutMix mem-pad cut yang dicampur, dataset mengurutkan batch per durasi, dan RNG CutMix mulai ulang). Ini sudah berlaku di v0; tes lama memakai cut berdurasi sama sehingga tidak terlihat.

Tidak dikerjakan:

* Tempo di gelombang: `PerturbTempo` Lhotse memakai `torchaudio.sox_effects`, yang tidak ada lagi di torchaudio 2.9+ (sudah dicek di 2.11). Tempo dikerjakan di frame fbank (`tempo_p` di atas), yang tidak butuh sox dan tidak menambah kerja data loader.
* Geser pitch: `torchaudio.functional.pitch_shift` butuh ±0,36 detik per klip 12 detik (1 thread). Pada p 0,3 itu ±5 detik per batch per worker, setara dengan anggaran ±5,8 detik per batch per worker (24 worker, ±0,24 detik per langkah GPU), sehingga data loader akan menjadi leher botol. Gema pada p 0,3 hanya menambah ±0,3 detik per batch per worker (±19 ms per klip).

### Gabungan segmen (nyala di v1) dan CR-CTC (run pembanding)

**Gabungan segmen berurutan** (`concat_p`, `concat_segments`, `concat_gap_seconds`, `concat_max_seconds`) nyala di `train_config.yml`. Semula opsi ini disiapkan untuk run pembanding; audit ketahanan menunjukkan ia menjawab kelemahan terbesar v0 (awal ayat hilang setelah jeda di bacaan bersambung), jadi sekarang bagian dari v1. Untuk mengukur efeknya sendiri, latih salinan config dengan `concat_p: 0.0`.

Aplikasi mendengar bacaan bersambung, sedangkan data latih berisi potongan per jeda. `concat_consecutive` menggabungkan 2–4 segmen yang berurutan (moshaf dan surah sama, `segment_index` bersebelahan) menjadi satu cut. Labelnya adalah label segmen yang disambung tanpa pemisah; tiap label sudah berakhir dalam bentuk waqf, sesuai yang dibaca karena segmen dipotong di jeda qari. Segmen yang digabung menggantikan aslinya, jadi ucapan per epoch tetap sama. Pengelompokan diambil dari `seed` atas urutan (moshaf, surah, posisi), sehingga sama di setiap epoch dan proses, tidak bergantung pada urutan manifest, dan resume tetap tepat. Salinan speed, cek frame CTC, bagian tanpa ucapan, gema, CutMix, dan gain dikenakan pada cut gabungan. Manifest tidak menyimpan stempel waktu, jadi panjang jeda asli tidak bisa dipulihkan. Segmen moshaf 6.0 sudah membawa ±0,5 detik hening di tiap ujung, sedangkan moshaf 19.0 tidak; `concat_gap_seconds` menambah jeda di tiap sambungan (v1: 0–1,5 detik), yang diisi derau atau hening digital seperti bagian tanpa ucapan (`nonspeech_noise_p`, `nonspeech_snr`). `concat_max_seconds` (v1: 45) menghentikan penggabungan sebelum ucapan cut gabungan melewati batas itu; tanpa batas, cut gabungan bisa mencapai empat kali panjang segmen terpanjang. Segmen tunggal yang lebih panjang dari batas itu tetap dipakai apa adanya. Dengan `concat_p` 0,3 kira-kira separuh segmen berada di dalam cut gabungan.

**CR-CTC** mati secara bawaan dan juga di `train_config.yml`. Nyalakan di run terpisah lalu bandingkan `dev_metrics.jsonl`-nya dengan v1.

**CR-CTC** (`use_cr_ctc`, `cr_loss_scale`, `cr_time_mask_ratio`, `cr_warm_steps`), mengikuti `forward_cr_ctc` icefall:

* time warp sekali;
* batch digandakan;
* tiap salinan mendapat SpecAugment sendiri dengan masking waktu ×2,5;
* loss = CTC rata-rata kedua salinan + skala × KL konsistensi antara keduanya.

Skala naik linear dari 0 ke `cr_loss_scale` (0,2) selama `cr_warm_steps` batch. SpecAugment dataset dimatikan selama CR-CTC aktif, karena masking dipasang di loop training. Tidak ada parameter baru, jadi checkpoint, rata-rata, ekspor ONNX, dan eval tidak berubah. W&B/TensorBoard mencatat `train/ctc_loss` dan `train/cr_loss` di samping `train/loss`.

Encoder menghitung dua kali per batch. Resep icefall memakai `max_duration` setengah dari biasanya, jadi waktu per epoch kira-kira 1,6–2× v1.

Resep run pembanding (mis. v1b):

```bash
cp configs/train_config.yml configs/train_config_v1b.yml
# ubah di salinan: semua nama run v1 -> v1b (output_dir, rclone_checkpoint_path,
#   hub_model_id, wandb_project_name), lalu
#   use_cr_ctc: true
#   max_duration: 300
```

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

`eval_baseline_v0.yml` memuat tujuh set stres yang sama dengan `train_config.yml`. Baseline yang dijalankan sebelum `vol-35db`, `clip`, `lead-noise`, dan `pair-gap` ditambahkan tidak punya keempat kunci itu; jalankan ulang `--stage evalsets` supaya v1 punya pembanding v0 di tiap set.

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
* `GetResult` meng-hard-code `frame_shift_ms = 10` dan `subsampling_factor = 4`, sehingga **timestamp token 2× terlalu besar**. Bagi 2 di aplikasi, atau baca `output_frame_shift_ms` dari metadata model. `infer_ayah.py --times` sudah melakukannya (`sherpa_time_scale`): timestamp sherpa dikalikan `output_frame_shift_ms / 40`, dan model tanpa metadata itu (v3) tidak diubah.
* `IsEndpoint` juga mengalikan trailing blank frame dengan angka 4 yang sama, sehingga trailing silence terhitung 2× dan aturan endpoint terpicu setelah setengah dari silence yang dikonfigurasi. Gandakan nilai `rule1_min_trailing_silence`, `rule2_min_trailing_silence`, dan `rule3_min_trailing_silence` di konfigurasi aplikasi untuk mengompensasi.
* Tidak ada key metadata yang mengubah perilaku ini di sherpa-onnx saat ini. ONNX kita tetap membawa metadata `output_frame_shift_ms` (`20`), `subsampling_factor` (`2`) dan `output_downsampling_factor` untuk aplikasi yang membacanya.

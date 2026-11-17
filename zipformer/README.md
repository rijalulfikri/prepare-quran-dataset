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
| `rclone_remote_path` | `utluma:RClone/Utluma/zipformer-data` | sumber data untuk `--stage prepare` (`rclone sync` ke `data_dir`) |
| `data_dir` | `./data-zipformer` | lokasi data lokal di box |
| `rclone_checkpoint_path` | `null` (di `train_config.yml`: `utluma:RClone/Utluma/zipformer-data/checkpoints/utluma-zipformer-ctc-v0`) | backup checkpoint ke Drive; `null` menonaktifkan backup |

**Waktu sinkronisasi data.** Dataset berisi ~250–300 ribu file FLAC kecil, dan Google Drive membatasi jumlah request per detik, sehingga `rclone sync` pertama bisa memakan waktu berjam-jam, bukan 30–60 menit. Ukur waktu sync untuk satu moshaf saat smoke run (Task 12) sebelum memutuskan layout data ini; perkiraan waktu "setup" di bagian [Jika box Vast.ai mati di tengah training](#jika-box-vastai-mati-di-tengah-training) bergantung pada hasil pengukuran tersebut.

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
| `--stage` | `all` | `prepare`, `train`, `eval`, `export`, atau `all` (= keempatnya berurutan) |
| `--push-to-hub` | `false` | upload `export/`, `averaged-<k>.pt`, `config.yml`, `test_results.json`, `qdat_bench_test_results.json` ke `hub_model_id` |
| `--rerun-testset` | `false` | evaluasi ulang test set walau `test_results.json` sudah ada |
| `--rerun-qdat-bench` | `false` | jalankan ulang qdat_bench walau hasilnya sudah ada |
| `--device` | `cuda` jika ada, selain itu `cpu` | device untuk train/eval |

Stage:

* `prepare`: `rclone sync <rclone_remote_path> <data_dir> --progress --transfers 32 --checkers 32 --fast-list --drive-pacer-min-sleep 10ms --drive-pacer-burst 200`. Gagal dengan pesan jelas jika `rclone` tidak ada atau belum ada remote (`rclone listremotes` kosong).
* `train`: training dengan resume otomatis dari checkpoint lokal atau Drive (mana yang lebih baru), lalu merata-ratakan `avg_last_k` epoch terakhir menjadi `averaged-<k>.pt`. Jika `rclone lsjson` ke Drive gagal, muncul warning dan training lanjut dengan checkpoint lokal saja.
* `eval`: PER test set (`per_phonemes`) dengan `export_chunk_size` / `export_left_context` → `test_results.json` (dilewati jika sudah ada, kecuali `--rerun-testset`).
* `export`: ONNX streaming fp32 + int8 ke `export/`, cek bahwa decode ONNX cocok dengan PyTorch pada 5 cut test (kesesuaian argmax per frame ≥ 99 % dan PER token ONNX vs PyTorch ≤ 1 %; keduanya dicatat di log), lalu qdat_bench dengan model ONNX fp32.

Exit code: `0` sukses, `143` berhenti dengan aman (mis. SIGTERM; checkpoint sudah di-flush), `1` gagal.

## 4. Lokasi output

Semua di `output_dir` (default `./results-zipformer-v0`, yaitu `zipformer/results-zipformer-v0`):

| Path | Isi |
|---|---|
| `checkpoint-*.pt`, `epoch-*.pt` (+ `.meta.json`) | checkpoint tiap `save_every_n_batches` batch dan tiap epoch (masing-masing `keep_last_k` terakhir); juga di-backup ke `rclone_checkpoint_path` |
| `averaged-<avg_last_k>.pt` | model akhir (rata-rata epoch terakhir) |
| `config.yml` | config yang dipakai |
| `dev_metrics.jsonl` | metrik dev per epoch |
| `tb/` | log TensorBoard (W&B juga dipakai jika `WANDB_API_KEY` di-set) |
| `test_results.json` | hasil stage `eval` |
| `qdat_bench_predictions.jsonl`, `qdat_bench_test_results.json` | hasil qdat_bench |
| `export/model.onnx`, `export/model.int8.onnx`, `export/tokens.txt` | model untuk sherpa-onnx |

Download ke komputer lokal:

```bash
scp -P <PORT> -r root@<IP_VAST_AI>:~/prepare-quran-dataset/zipformer/results-zipformer-v0/export ./
```

## Jika box Vast.ai mati di tengah training

**Pemulihan:** sewa box baru, salin `rclone.conf` dan `.env` seperti di atas, lalu jalankan perintah yang sama:

```bash
bash train_zipformer.sh
```

Perintah ini sekaligus perintah restart: data disinkronkan ulang dari Drive, checkpoint terbaru diunduh dari `rclone_checkpoint_path`, dan training lanjut dari batch tersebut. Sebelum training, log menampilkan satu baris seperti:

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

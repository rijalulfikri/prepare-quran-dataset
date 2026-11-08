#!/usr/bin/env bash
# ==============================================================================
# Pipeline End-to-End: Training Zipformer2-CTC (streaming) & Export ke sherpa-onnx
# Didesain untuk 1x GPU NVIDIA (mis. RTX 4090) di Vast.ai
#
# PERINTAH YANG SAMA ADALAH PERINTAH RESTART DI BOX BARU.
# Setiap langkah idempotent: jika box mati, sewa box baru, salin rclone.conf
# (+ .env), clone repo ini, lalu jalankan lagi:
#
#     bash train_zipformer.sh
#
# Training otomatis lanjut dari checkpoint terbaru (lokal atau Google Drive).
#
# Variabel environment opsional:
#   CONFIG_FILE      default: configs/train/zipformer/train_config.yml
#   K2_WHEEL_URL     URL wheel k2 untuk cp314 + torch 2.9.1 (lihat di bawah)
#   K2_WHEEL_URL_CP312  URL wheel k2 untuk cp312 (fallback jika cp314 tidak ada)
#   TORCH_INDEX_URL  default: https://download.pytorch.org/whl/cu128
#   AUTO_RESTART=1   ulangi training (maks 3x, jeda 60 dtk) jika crash dengan
#                    exit code selain 0/143 (CUDA/NCCL/data-loader sementara);
#                    tidak menolong jika box-nya sendiri mati. PERHATIAN: kegagalan
#                    deterministik (config salah, parity assertion ONNX) juga
#                    diulang; baca log sebelum mengandalkan AUTO_RESTART.
#   Argumen tambahan diteruskan ke train_zipformer.py, mis.:
#     bash train_zipformer.sh --push-to-hub
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"  # qdat_bench memuat ./vocab_streaming relatif terhadap cwd

CONFIG_FILE="${CONFIG_FILE:-configs/train/zipformer/train_config.yml}"
VENV_DIR=".venv-zipformer"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
# Pilih wheel k2 yang cocok (torch 2.9.1, CUDA 12.x) di https://k2-fsa.github.io/k2/cuda.html
# Default di bawah adalah placeholder; Task 12 mengisi URL yang sudah diverifikasi.
K2_PLACEHOLDER="FILL_IN_FROM_https://k2-fsa.github.io/k2/cuda.html"
K2_WHEEL_URL="${K2_WHEEL_URL:-$K2_PLACEHOLDER}"
K2_WHEEL_URL_CP312="${K2_WHEEL_URL_CP312:-$K2_PLACEHOLDER}"

# Gagal cepat, sebelum membuat venv atau menginstall torch (beberapa GB).
if [[ "$K2_WHEEL_URL" == "$K2_PLACEHOLDER" && "$K2_WHEEL_URL_CP312" == "$K2_PLACEHOLDER" ]]; then
    echo "ERROR: K2_WHEEL_URL dan K2_WHEEL_URL_CP312 belum diisi."
    echo "Pilih wheel k2 untuk torch 2.9.1 + CUDA 12.x di https://k2-fsa.github.io/k2/cuda.html, lalu mis.:"
    echo "  K2_WHEEL_URL=<url wheel cp314> K2_WHEEL_URL_CP312=<url wheel cp312> bash train_zipformer.sh"
    exit 1
fi

echo "============================================================"
echo "  MUAALEM ZIPFORMER2-CTC: PIPELINE TRAINING & EXPORT ONNX"
echo "============================================================"

# ------------------------------------------------------------------------------
# 1. Setup Virtual Environment & Dependencies
# ------------------------------------------------------------------------------
echo -e "\n[1/4] Memeriksa Environment & Dependencies..."
if ! command -v uv &> /dev/null; then
    echo "Menginstall uv package manager..."
    pip install uv
fi

install_k2() {
    # $1 = URL wheel k2
    if python -c "import k2" &> /dev/null; then
        echo "k2 sudah terinstall."
        return 0
    fi
    if [[ "$1" == "$K2_PLACEHOLDER" ]]; then
        echo "K2_WHEEL_URL belum diisi (lihat https://k2-fsa.github.io/k2/cuda.html)."
        return 1
    fi
    uv pip install "$1"
}

make_venv() {
    # $1 = versi Python
    echo "Membuat virtual environment Python $1 di $VENV_DIR..."
    uv venv "$VENV_DIR" --python "$1"
}

if [ ! -d "$VENV_DIR" ]; then
    make_venv 3.14
fi
source "$VENV_DIR/bin/activate"

echo "Menginstall torch 2.9.1 dari $TORCH_INDEX_URL..."
uv pip install "torch==2.9.1" "torchaudio==2.9.1" --index-url "$TORCH_INDEX_URL"

PY_VERSION="$(python -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
if [[ "$PY_VERSION" == "3.14" ]] && ! install_k2 "$K2_WHEEL_URL"; then
    echo "Wheel k2 untuk cp314 tidak tersedia; fallback ke Python 3.12 (PYTHONPATH=src)..."
    deactivate
    rm -rf "$VENV_DIR"
    make_venv 3.12
    source "$VENV_DIR/bin/activate"
    uv pip install "torch==2.9.1" "torchaudio==2.9.1" --index-url "$TORCH_INDEX_URL"
    PY_VERSION="3.12"
fi
if [[ "$PY_VERSION" == "3.12" ]] && ! install_k2 "$K2_WHEEL_URL_CP312"; then
    echo "ERROR: k2 tidak bisa diinstall. Set K2_WHEEL_URL / K2_WHEEL_URL_CP312."
    exit 1
fi

echo "Menginstall requirements-zipformer.txt..."
uv pip install -r requirements-zipformer.txt
if [[ "$PY_VERSION" == "3.14" ]]; then
    uv pip install -e .
else
    # pyproject.toml mensyaratkan Python >=3.14, jadi paket dipakai lewat PYTHONPATH
    uv pip install huggingface_hub
    export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
fi

echo "Mengambil submodule icefall..."
git submodule update --init external/icefall

echo "Cek import modul training & qdat_bench..."
if ! python -c "import prepare_quran_dataset.zipformer.qdat, prepare_quran_dataset.zipformer.train_loop, prepare_quran_dataset.modeling_streaming_rnn.multi_level_tokenizer, qdat_bench.eval_results"; then
    echo "ERROR: import gagal (lihat traceback di atas); ada dependency yang belum terinstall."
    exit 1
fi

# ------------------------------------------------------------------------------
# 2. Rclone (Google Drive): data + backup checkpoint
# ------------------------------------------------------------------------------
echo -e "\n[2/4] Memeriksa rclone & koneksi Google Drive..."
if ! command -v rclone &> /dev/null; then
    echo "Menginstall rclone..."
    curl https://rclone.org/install.sh | bash
fi

# Diterima: file RCLONE_CONFIG, ~/.config/rclone/rclone.conf, atau remote yang
# didefinisikan lewat environment (RCLONE_CONFIG_<REMOTE>_TYPE=drive, dst.).
RCLONE_CONF_FILE="${RCLONE_CONFIG:-$HOME/.config/rclone/rclone.conf}"
if [ -f "$RCLONE_CONF_FILE" ]; then
    echo "rclone config: $RCLONE_CONF_FILE"
elif grep -q '^RCLONE_CONFIG_.*_TYPE=' < <(env); then
    echo "rclone config: remote dari environment (RCLONE_CONFIG_<REMOTE>_TYPE)"
else
    echo "ERROR: konfigurasi rclone tidak ditemukan ($RCLONE_CONF_FILE tidak ada dan tidak ada RCLONE_CONFIG_<REMOTE>_TYPE)."
    echo "Pilih salah satu:"
    echo "  - salin rclone.conf dari komputer lokal, mis.:"
    echo "      scp -P <PORT> ~/.config/rclone/rclone.conf root@<IP_VAST_AI>:~/.config/rclone/rclone.conf"
    echo "  - set RCLONE_CONFIG=/path/ke/rclone.conf"
    echo "  - definisikan remote lewat environment, mis. RCLONE_CONFIG_GDRIVE_TYPE=drive"
    echo "      RCLONE_CONFIG_GDRIVE_SCOPE=drive RCLONE_CONFIG_GDRIVE_TOKEN='<json token>'"
    exit 1
fi
if [ ! -f ".env" ]; then
    echo "PERINGATAN: .env tidak ada (WANDB_API_KEY / HUGGINGFACE_TOKEN). Lihat .env.example."
fi

read_cfg() {
    # $1 = nama field di YAML config; mencetak "None" jika null
    python -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1], encoding='utf-8')).get(sys.argv[2]))" "$CONFIG_FILE" "$1"
}
DATA_REMOTE="$(read_cfg rclone_remote_path)"
CKPT_REMOTE="$(read_cfg rclone_checkpoint_path)"
OUTPUT_DIR="$(read_cfg output_dir)"

echo "Cek koneksi ke remote ${DATA_REMOTE%%:*}: ..."
rclone lsd "${DATA_REMOTE%%:*}:"

if [[ "$CKPT_REMOTE" == "None" || -z "$CKPT_REMOTE" ]]; then
    echo "rclone_checkpoint_path = null: backup checkpoint ke Drive dinonaktifkan."
else
    echo "Isi folder checkpoint di Drive ($CKPT_REMOTE), yang akan dipakai untuk resume:"
    rclone lsl "$CKPT_REMOTE" || echo "(folder belum ada / belum ada checkpoint di Drive)"
fi

# ------------------------------------------------------------------------------
# 3. Jalankan Pipeline (prepare -> train -> eval -> export)
# ------------------------------------------------------------------------------
echo -e "\n[3/4] Menjalankan train_zipformer.py (stage all)..."
echo "Config: $CONFIG_FILE"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'NVIDIA GPU')"

MAX_RETRIES=3
attempt=0
while true; do
    set +e
    python train_zipformer.py --config "$CONFIG_FILE" --stage all "$@"
    EXIT_CODE=$?
    set -e
    if [[ $EXIT_CODE -eq 0 || $EXIT_CODE -eq 143 ]]; then
        break
    fi
    if [[ "${AUTO_RESTART:-0}" != "1" || $attempt -ge $MAX_RETRIES ]]; then
        break
    fi
    attempt=$((attempt + 1))
    echo "Crash (exit code $EXIT_CODE). AUTO_RESTART: percobaan $attempt/$MAX_RETRIES dalam 60 detik..."
    sleep 60
done

if [[ $EXIT_CODE -eq 143 ]]; then
    echo -e "\nTraining dihentikan dengan aman (exit 143); checkpoint sudah disimpan."
    echo "Jalankan lagi 'bash train_zipformer.sh' (di box ini atau box baru) untuk melanjutkan."
    exit 143
elif [[ $EXIT_CODE -ne 0 ]]; then
    echo -e "\nERROR: train_zipformer.py gagal (exit code $EXIT_CODE). Periksa log di atas."
    exit "$EXIT_CODE"
fi

# ------------------------------------------------------------------------------
# 4. Selesai
# ------------------------------------------------------------------------------
EXPORT_DIR="$OUTPUT_DIR/export"
echo -e "\n[4/4] Hasil export di $EXPORT_DIR:"
ls -lh "$EXPORT_DIR"
echo -e "\n============================================================"
echo "  SUKSES! Model streaming sherpa-onnx berhasil dibuat."
echo "  Hasil test : $OUTPUT_DIR/test_results.json"
echo "  qdat_bench : $OUTPUT_DIR/qdat_bench_test_results.json"
echo "============================================================"
echo "Anda dapat mendownload model ini ke komputer lokal menggunakan scp:"
case "$EXPORT_DIR" in
    /*) EXPORT_ABS="$EXPORT_DIR" ;;
    *) EXPORT_ABS="$PWD/${EXPORT_DIR#./}" ;;
esac
echo "scp -P <PORT> -r root@<IP_VAST_AI>:$EXPORT_ABS ./"

#!/usr/bin/env bash
# ==============================================================================
# Pipeline End-to-End: Training Muaalem v3 & Export ke TFLite (Int8)
# Didesain khusus untuk 1x GPU NVIDIA RTX 4090 (24GB VRAM) di Vast.ai
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUTPUT_DIR="./exported_tflite"
RESULTS_DIR="./results-streaming-rnn-v3"
CONFIG_FILE="./configs/train/streaming/train_config_4090.yml"
TARGET_TFLITE="$OUTPUT_DIR/muaalem_v3_int8.tflite"

echo "============================================================"
echo "  MUAALEM V3: PIPELINE TRAINING & EXPORT TFLITE INT8"
echo "============================================================"

# ------------------------------------------------------------------------------
# 1. Setup Virtual Environment & Dependencies
# ------------------------------------------------------------------------------
echo -e "\n[1/5] Memeriksa Environment & Dependencies..."
if ! command -v uv &> /dev/null; then
    echo "Menginstall uv package manager..."
    pip install uv
fi

if [ ! -d ".venv" ]; then
    echo "Membuat virtual environment Python 3.12..."
    uv venv .venv --python 3.12
fi

# Aktifkan venv
source .venv/bin/activate

echo "Menginstall / Memperbarui dependensi yang dibutuhkan..."
uv pip install -e ".[annotate]"
uv pip install "litert-torch" "ai-edge-quantizer" "accelerate" "wandb" "pyyaml"

# ------------------------------------------------------------------------------
# 2. Buat Konfigurasi Khusus RTX 4090 (24GB VRAM)
# ------------------------------------------------------------------------------
echo -e "\n[2/5] Menyiapkan Konfigurasi Training untuk RTX 4090..."
mkdir -p "$(dirname "$CONFIG_FILE")"

# Membuat config turunan dari train_config_v3.yml dengan batch size yang aman dari OOM
EPOCHS="${EPOCHS:-1}" # Default 1 epoch untuk fine-tuning cepat, bisa di-override via env
python3 - << EOF
import os
import yaml

base_config_path = "./configs/train/streaming/train_config_v3.yml"
out_config_path = "./configs/train/streaming/train_config_4090.yml"

with open(base_config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# Penyesuaian khusus untuk 24GB VRAM RTX 4090
cfg["per_device_train_batch_size"] = 16
cfg["pre_device_eval_batch_size"] = 16
cfg["gradient_checkpoiniting"] = True
cfg["output_dir"] = "./results-streaming-rnn-v3"
cfg["num_epochs"] = int("$EPOCHS")

# Pastikan base model valid jika checkpoint lokal belum ada
if not os.path.exists(str(cfg.get("base_model_name_or_path", ""))):
    print("Checkpoint lokal tidak ditemukan, fallback ke base model 'facebook/w2v-bert-2.0'...")
    cfg["base_model_name_or_path"] = "facebook/w2v-bert-2.0"

with open(out_config_path, "w", encoding="utf-8") as f:
    yaml.dump(cfg, f, default_flow_style=False)

print(f"Config berhasil dibuat di: {out_config_path} (Epochs: {cfg['num_epochs']})")
EOF

# ------------------------------------------------------------------------------
# 3. Jalankan Training (Dataset otomatis di-stream/download dari HuggingFace)
# ------------------------------------------------------------------------------
echo -e "\n[3/5] Memulai Pelatihan Model (train_streaming.py)..."
echo "Dataset: obadx/muaalem-annotated-v3"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'NVIDIA GPU')"

# Jalankan via accelerate
accelerate launch train_streaming.py --config "$CONFIG_FILE"

# ------------------------------------------------------------------------------
# 4. Deteksi Checkpoint Terbaik / Terakhir
# ------------------------------------------------------------------------------
echo -e "\n[4/5] Mencari Checkpoint Terbaik..."
if [ ! -d "$RESULTS_DIR" ]; then
    echo "ERROR: Direktori hasil $RESULTS_DIR tidak ditemukan!"
    exit 1
fi

LATEST_CHECKPOINT=$(find "$RESULTS_DIR" -maxdepth 1 -type d -name "checkpoint-*" | sort -V | tail -n 1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "ERROR: Tidak ditemukan checkpoint di $RESULTS_DIR!"
    exit 1
fi

echo "Checkpoint yang akan diekspor: $LATEST_CHECKPOINT"

# ------------------------------------------------------------------------------
# 5. Ekspor ke TFLite (Float32, Int8, Int4)
# ------------------------------------------------------------------------------
echo -e "\n[5/5] Mengekspor Model ke LiteRT (.tflite)..."
mkdir -p "$OUTPUT_DIR"

python3 export_rnn_streaming_to_litert.py \
    --model-id "$LATEST_CHECKPOINT" \
    --output-dir "$OUTPUT_DIR" \
    --prefix "muaalem_v3"

# ------------------------------------------------------------------------------
# Selesai
# ------------------------------------------------------------------------------
if [ -f "$TARGET_TFLITE" ]; then
    FILE_SIZE=$(du -h "$TARGET_TFLITE" | cut -f1)
    echo -e "\n============================================================"
    echo "  SUKSES! Model berhasil dibuat:"
    echo "  File : $TARGET_TFLITE"
    echo "  Size : $FILE_SIZE"
    echo "============================================================"
    echo "Anda dapat mendownload model ini ke komputer lokal menggunakan scp:"
    echo "scp root@<IP_VAST_AI>:$PWD/$TARGET_TFLITE ./"
else
    echo -e "\nPERINGATAN: File $TARGET_TFLITE belum terbentuk. Periksa log di atas."
    exit 1
fi

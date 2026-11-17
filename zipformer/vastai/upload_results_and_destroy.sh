#!/usr/bin/env bash
# ==============================================================================
# Dijalankan DI BOX Vast.ai setelah train_zipformer.sh sukses (exit 0), oleh
# run_zipformer_training.py --destroy-when-done:
#
#   1. Upload output_dir (tanpa checkpoint-*/epoch-*, yang sudah di-backup ke
#      rclone_checkpoint_path) ke Google Drive, lalu verifikasi dengan rclone check.
#   2. Hapus (destroy) instance Vast.ai ini lewat API, memakai CONTAINER_ID dan
#      CONTAINER_API_KEY (key khusus instance ini, hanya bisa start/stop/destroy
#      instance ini). Keduanya tidak selalu ada di sesi SSH/tmux, jadi dibaca juga
#      dari environment PID 1 container.
#
# Instance TIDAK dihapus jika ada langkah yang gagal.
#
# Variabel environment:
#   REPO_ZIPFORMER_DIR  folder zipformer/ di box (wajib)
#   CONFIG_FILE         default: configs/train_config.yml (relatif terhadap zipformer/)
#   RESULTS_REMOTE      tujuan upload; default: <rclone_checkpoint_path>-results
# ==============================================================================
set -euo pipefail
trap 'echo "ERROR: perintah gagal di baris $LINENO; instance TIDAK dihapus (masih ditagih)."' ERR

cd "${REPO_ZIPFORMER_DIR:?REPO_ZIPFORMER_DIR belum di-set}"
CONFIG_FILE="${CONFIG_FILE:-configs/train_config.yml}"

read_cfg() {
    # $1 = nama field di YAML config; mencetak "None" jika null
    .venv-zipformer/bin/python -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1], encoding='utf-8')).get(sys.argv[2]))" "$CONFIG_FILE" "$1"
}

container_var() {
    # $1 = nama variabel; cari di environment sendiri, lalu di environment PID 1
    local value="${!1:-}"
    if [[ -z "$value" && -r /proc/1/environ ]]; then
        value="$(tr '\0' '\n' < /proc/1/environ | sed -n "s/^$1=//p" | head -n 1)"
    fi
    echo "$value"
}

echo "============================================================"
echo "  UPLOAD HASIL & DESTROY INSTANCE"
echo "============================================================"

# Cek kredensial dulu supaya tidak upload percuma jika instance tidak bisa dihapus
CONTAINER_ID="$(container_var CONTAINER_ID)"
CONTAINER_API_KEY="$(container_var CONTAINER_API_KEY)"
if [[ -z "$CONTAINER_ID" || -z "$CONTAINER_API_KEY" ]]; then
    echo "ERROR: CONTAINER_ID / CONTAINER_API_KEY tidak ditemukan; instance TIDAK dihapus."
    exit 1
fi

OUTPUT_DIR="$(read_cfg output_dir)"
if [[ -z "${RESULTS_REMOTE:-}" ]]; then
    CKPT_REMOTE="$(read_cfg rclone_checkpoint_path)"
    if [[ "$CKPT_REMOTE" == "None" || -z "$CKPT_REMOTE" ]]; then
        echo "ERROR: rclone_checkpoint_path = null dan RESULTS_REMOTE kosong; instance TIDAK dihapus."
        exit 1
    fi
    RESULTS_REMOTE="${CKPT_REMOTE%/}-results"
fi

# Checkpoint sudah ada di rclone_checkpoint_path; averaged-*.pt, export/, tb/, dll. ikut
EXCLUDES=(--exclude "checkpoint-*" --exclude "epoch-*")
echo "Upload $OUTPUT_DIR -> $RESULTS_REMOTE ..."
rclone copy "$OUTPUT_DIR" "$RESULTS_REMOTE" "${EXCLUDES[@]}" --transfers 8 --stats 30s --stats-one-line
echo "Verifikasi upload (rclone check)..."
rclone check "$OUTPUT_DIR" "$RESULTS_REMOTE" "${EXCLUDES[@]}" --one-way
echo "Isi $RESULTS_REMOTE:"
rclone lsl "$RESULTS_REMOTE"

echo "Destroy instance $CONTAINER_ID ..."
RESPONSE="$(curl -sS --fail-with-body -X DELETE \
    -H "Authorization: Bearer $CONTAINER_API_KEY" \
    "https://console.vast.ai/api/v0/instances/$CONTAINER_ID/")"
echo "Respons Vast.ai: $RESPONSE"
if ! grep -q '"success": *true' <<< "$RESPONSE"; then
    echo "ERROR: destroy gagal; instance masih berjalan (dan ditagih). Hapus manual di https://cloud.vast.ai/instances/"
    exit 1
fi
echo "Instance $CONTAINER_ID sedang dihapus. Hasil ada di $RESULTS_REMOTE"

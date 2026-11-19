#!/bin/bash
# zipformer_watchdog.sh — early stop for a zipformer run on a Vast.ai box.
#
# Dikirim dan dijalankan oleh run_zipformer_training.py --early-stop-patience N di
# sesi tmux `watchdog`. Tiap POLL detik membaca dev_metrics.jsonl. Sebuah epoch
# dihitung sebagai rekor baru hanya jika PER-nya < rekor x (1 - MIN_REL_IMPROVEMENT).
# Jika PATIENCE epoch berturut-turut tanpa rekor baru (dan training masih berjalan,
# epoch terakhir < num_epochs):
#   1. SIGTERM ke trainer (berhenti rapi, checkpoint di-flush, exit 143),
#   2. num_epochs di config box diubah ke epoch terakhir yang selesai,
#   3. RELAUNCH dijalankan: pipeline yang sama (train -> "averaging only", eval,
#      export, dan dengan --destroy-when-done: upload hasil ke Drive + destroy).
# Watchdog keluar setelah bertindak, atau saat run selesai sendiri.
#
# Variabel environment (semua opsional):
#   PATIENCE             epoch tanpa rekor sebelum berhenti (default 4)
#   MIN_REL_IMPROVEMENT  perbaikan relatif minimal untuk rekor baru (default 0.05)
#   POLL                 interval cek dalam detik (default 120)
#   REPO_ZIPFORMER_DIR   folder zipformer di box (default /root/prepare-quran-dataset/zipformer)
#   CONFIG_FILE          config training yang dipakai pipeline, sama seperti di
#                        train_zipformer.sh (default configs/train_config.yml,
#                        relatif terhadap REPO_ZIPFORMER_DIR). num_epochs diubah di
#                        file ini, jadi harus file yang sama dengan milik pipeline.
#   METRICS              dev_metrics.jsonl (default <output_dir dari CONFIG_FILE>/dev_metrics.jsonl)
#   RELAUNCH             skrip relaunch pipeline (default /root/zipformer_relaunch.sh)
#   WLOG                 log watchdog (default /root/zipformer_watchdog.log)
#   DRY_RUN=1            cetak keputusan sekali lalu keluar, tanpa menyentuh apa pun
set -u
unset TMUX  # RELAUNCH membuat sesi tmux pipeline dari dalam sesi tmux ini
PATIENCE=${PATIENCE:-4}
MIN_REL_IMPROVEMENT=${MIN_REL_IMPROVEMENT:-0.05}
POLL=${POLL:-120}
Z=${REPO_ZIPFORMER_DIR:-/root/prepare-quran-dataset/zipformer}
CONFIG=${CONFIG_FILE:-configs/train_config.yml}
case "$CONFIG" in /*) ;; *) CONFIG="$Z/$CONFIG" ;; esac
RELAUNCH=${RELAUNCH:-/root/zipformer_relaunch.sh}
WLOG=${WLOG:-/root/zipformer_watchdog.log}
LOG=/root/train_zipformer.log
DONE_MARKER=/root/.train_zipformer.done
TRAINER='python [t]rain_zipformer.py --config'

say() { echo "$(date -u '+%F %T') $*" | tee -a "$WLOG"; }

# Nilai skalar `key: value` dari config YAML (tanpa kutip / komentar)
config_value() {
    sed -n "s/^$1:[[:space:]]*//p" "$CONFIG" | sed 's/[[:space:]]*#.*//; s/["'\'']//g' | head -1
}

if [ -z "${METRICS:-}" ]; then
    out_dir=$(config_value output_dir)
    case "$out_dir" in /*) ;; *) out_dir="$Z/$out_dir" ;; esac
    METRICS="$out_dir/dev_metrics.jsonl"
fi

# prints: "<last_epoch> <best_epoch> <best_per> <epochs_since_best>"
decision() {
    python3 - "$METRICS" "$MIN_REL_IMPROVEMENT" <<'EOF'
import json, sys
rows = {}
try:
    for line in open(sys.argv[1]):
        if line.strip():
            r = json.loads(line)
            rows[r["epoch"]] = r["per_phonemes"]
except FileNotFoundError:
    pass
if not rows:
    print("0 0 nan 0")
    raise SystemExit
min_rel = float(sys.argv[2])
best_ep = None
for ep in sorted(rows):
    if best_ep is None or rows[ep] < rows[best_ep] * (1 - min_rel):
        best_ep = ep
last = max(rows)
print(last, best_ep, f"{rows[best_ep]:.6f}", last - best_ep)
EOF
}

trigger() {
    local n=$1 ended_before
    say "TRIGGER: $PATIENCE epoch tanpa rekor baru; berhenti setelah epoch $n"
    ended_before=$(grep -a -c 'selesai, exit code' "$LOG")
    pgrep -af "$TRAINER" | tee -a "$WLOG"
    pkill -TERM -f "$TRAINER" && say "SIGTERM terkirim"
    for _ in $(seq 1 40); do
        sleep 30
        if [ "$(grep -a -c 'selesai, exit code' "$LOG")" -gt "$ended_before" ] && [ -f "$DONE_MARKER" ]; then
            break
        fi
    done
    grep -a 'selesai, exit code' "$LOG" | tail -1 | tee -a "$WLOG"
    if [ ! -f "$DONE_MARKER" ] || pgrep -f "$TRAINER" >/dev/null; then
        say "BATAL: training tidak berhenti dalam 20 menit; tidak ada yang diubah"
        exit 2
    fi
    sed -i "s/^num_epochs:.*/num_epochs: $n/" "$CONFIG"
    say "config: $(grep '^num_epochs' "$CONFIG")"
    bash "$RELAUNCH" 2>&1 | tee -a "$WLOG"
    sleep 5
    say "relaunch: $(tmux ls 2>&1 | tr '\n' ' ')"
    exit 0
}

say "watchdog mulai (patience $PATIENCE, min relative improvement $MIN_REL_IMPROVEMENT, poll ${POLL}s, dry_run ${DRY_RUN:-0})"
seen_trainer=0
while true; do
    read -r last best_ep best_per since <<<"$(decision)"
    num_epochs=$(config_value num_epochs)
    running=0
    pgrep -f "$TRAINER" >/dev/null && running=1 && seen_trainer=1
    # Jangan pernah memotong tahap averaging / eval / export: hanya saat epoch training masih tersisa
    would_trigger=0
    if [ "$last" -gt 0 ] && [ "$since" -ge "$PATIENCE" ] && [ "$last" -lt "${num_epochs:-0}" ]; then
        would_trigger=1
    fi
    if [ "${DRY_RUN:-0}" = 1 ]; then
        say "dry run: last epoch $last of $num_epochs, best $best_per at epoch $best_ep, epochs since best $since/$PATIENCE, would_trigger $would_trigger"
        exit 0
    fi
    if [ "$would_trigger" = 1 ] && [ "$running" = 1 ]; then
        trigger "$last"
    fi
    if [ "$seen_trainer" = 1 ] && [ "$running" = 0 ] && [ -f "$DONE_MARKER" ]; then
        say "run selesai sendiri (epoch terakhir $last, rekor $best_per di epoch $best_ep); watchdog keluar"
        exit 0
    fi
    sleep "$POLL"
done

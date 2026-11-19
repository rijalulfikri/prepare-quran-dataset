#!/usr/bin/env python3

"""
zipformer_status.py

Status satu run training Zipformer di Vast.ai yang dijalankan oleh
run_zipformer_training.py, dari komputer lokal:

- STATUS: satu kata keadaan run (lihat STATES), untuk manusia dan agen.
- Tabel dev PER per epoch dengan rekor dan hitungan early stop memakai aturan
  zipformer_watchdog.sh (patience dan ambang dibaca dari log watchdog di box).
- Kecepatan per epoch, perkiraan selesai, sisa credit Vast.ai dan biaya.
- Watchdog, GPU, error terakhir di log, tahap destroy.
- Tanpa instance: isi folder hasil (<rclone_checkpoint_path>-results) di Drive.

--wait menunggu (poll tiap --poll detik) sampai ada epoch baru atau STATUS
berubah, lalu mencetak status; dipakai agen untuk memantau di latar belakang.
--drive juga mencocokkan SHA-256 epoch-*.pt terbaru di Drive dengan meta-nya.

Contoh:
    python zipformer/vastai/zipformer_status.py
    python zipformer/vastai/zipformer_status.py --host 71.104.167.38 --port 51462 --drive
    python zipformer/vastai/zipformer_status.py --wait
"""

import argparse
import contextlib
import io
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ZIPFORMER_DIR = os.path.dirname(SCRIPT_DIR)

REMOTE_Z = "/root/prepare-quran-dataset/zipformer"
DEFAULT_PATIENCE = 4
DEFAULT_MIN_IMPROVEMENT = 0.05
FINISH_HOURS = 1.0  # averaging + test eval + export + upload, perkiraan kasar

STATES = {
    "PREPARING": "train_zipformer.sh berjalan, trainer belum mulai (setup / data)",
    "TRAINING": "trainer berjalan, epoch training masih tersisa",
    "FINISHING": "semua epoch selesai: averaging / eval test / export",
    "WATCHDOG_STOPPING": "watchdog menghentikan training dan akan menjalankan ulang pipeline",
    "UPLOADING": "training sukses, upload hasil + destroy sedang berjalan",
    "STOPPED": "run berakhir dengan exit code != 0; instance hidup dan ditagih",
    "ENDED": "run berakhir sukses; instance masih hidup (tanpa --destroy-when-done atau destroy gagal)",
    "NO_INSTANCE": "tidak ada instance Vast.ai yang cocok (mis. sudah di-destroy)",
    "API_ERROR": "API Vast.ai tidak bisa dibaca; keadaan instance tidak diketahui",
}
# --wait menerima NO_INSTANCE hanya setelah sekian poll berturut-turut: saat jaringan
# host bermasalah (mis. IP publik mesin berganti) API bisa sesaat tidak melaporkannya
NO_INSTANCE_CONFIRMATIONS = 3

# Satu panggilan SSH: bagian-bagian dipisah baris "@@NAMA"
REMOTE_PROBE = r"""
Z=%(z)s
LOG=/root/train_zipformer.log
TRAINER='python [t]rain_zipformer.py --config'
pid=$(pgrep -f "$TRAINER" | head -1)
cfg=""
[ -n "$pid" ] && cfg=$(tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | sed -n 's/^CONFIG_FILE=//p')
[ -z "$cfg" ] && [ -f /root/zipformer_relaunch.sh ] && cfg=$(grep -o 'CONFIG_FILE=[^ ;]*' /root/zipformer_relaunch.sh | head -1 | cut -d= -f2)
cfg=${cfg:-configs/train_config.yml}
case "$cfg" in /*) ;; *) cfg="$Z/$cfg" ;; esac
val() { sed -n "s/^$1:[[:space:]]*//p" "$cfg" | sed 's/[[:space:]]*#.*//; s/["'\'']//g' | head -1; }
out=$(val output_dir); case "$out" in /*) ;; *) out="$Z/$out" ;; esac
# Log berisi semua run di box ini; bagian per-run dimulai dari header train_zipformer.sh terakhir
start=$(grep -a -n 'PIPELINE TRAINING' "$LOG" 2>/dev/null | tail -1 | cut -d: -f1)
cur() { tail -n +"${start:-1}" "$LOG" 2>/dev/null; }
echo @@NOW; date -u '+%%F %%T'
echo @@CONFIG; echo "$cfg"
echo @@NUM_EPOCHS; val num_epochs
echo @@CKPT_PATH; val rclone_checkpoint_path
echo @@TRAINER; echo "$pid"
# Dijangkarkan ke awal command line: perintah sesi tmux juga memuat nama-nama skrip ini
echo @@SHELL; pgrep -f '^bash train_zipformer\.sh' | head -1
echo @@UPLOADER; pgrep -f '^bash /root/upload_results_and_destroy\.sh' | head -1
echo @@DONE; [ -f /root/.train_zipformer.done ] && echo yes
echo @@METRICS; cat "$out/dev_metrics.jsonl" 2>/dev/null
echo @@EPOCH_END; grep -a 'epoch [0-9]* finished' "$LOG" 2>/dev/null | tail -40
echo @@ENDED; cur | grep -a 'selesai, exit code' | tail -1
echo @@FINISH; cur | grep -a 'Destroy instance\|sedang dihapus\|TIDAK dihapus\|destroy gagal\|dibiarkan hidup' | tail -3
echo @@ERRORS; cur | tail -c 200000 | tr '\r' '\n' | grep -a 'Traceback\|Error\|ERROR\|Killed\|CUDA out of memory' | grep -v 'Whitening\|ScheduledFloat' | tail -4
echo @@STAGE; cur | tail -c 200000 | tr '\r' '\n' | grep -a 'INFO train_zipformer\|averaged\|test results\|all [0-9]* epochs already complete\|rebuilding training cuts' | tail -2
echo @@WATCHDOG; tmux has-session -t watchdog 2>/dev/null && echo session; pgrep -f '^bash /root/zipformer_watchdog\.sh' >/dev/null && echo running; tail -3 /root/zipformer_watchdog.log 2>/dev/null
echo @@WATCHDOG_PARAMS; grep -a -o 'patience [0-9]*, min relative improvement [0-9.]*' /root/zipformer_watchdog.log 2>/dev/null | tail -1
echo @@GPU; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null
"""



def run(cmd, timeout=120, stdin=None):
    """subprocess.run -> (returncode, stdout); timeout dianggap gagal."""
    try:
        p = subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout
    except subprocess.TimeoutExpired:
        return 124, ""


def find_box(args):
    """(host, port, instance_id) dari --host/--port atau Vast.ai API.

    Returns None jika API menjawab tanpa instance yang cocok, "API_ERROR" jika
    API tidak bisa dibaca (mis. jaringan putus): itu bukan bukti instance hilang.
    """
    if args.host and args.port:
        return args.host, str(args.port), None
    sys.path.insert(0, SCRIPT_DIR)
    from vast_api import get_first_started_instance, get_instances

    # get_instances mencetak pesan + traceback saat gagal
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        if get_instances() is None:
            return "API_ERROR"
    info = get_first_started_instance(args.label or os.getenv("VAST_LABEL"), args.host, verbose=False)
    if not info:
        return None
    return info["ip"], str(info["ssh_port"]), info["instance_id"]


def ssh_cmd(args, host, port):
    return ["ssh", "-p", port, "-i", args.identity, "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
            "-o", "StrictHostKeyChecking=accept-new", "-o", "LogLevel=ERROR", f"root@{host}"]


def probe(args, host, port):
    """Jalankan REMOTE_PROBE di box; dict bagian -> list baris, atau None jika SSH gagal."""
    code, out = run(ssh_cmd(args, host, port) + ["bash -s"], stdin=REMOTE_PROBE % {"z": REMOTE_Z})
    if code != 0 and "@@NOW" not in out:
        return None
    sections, current = {}, None
    for line in out.splitlines():
        if line.startswith("@@"):
            current = line[2:]
            sections[current] = []
        elif current and "setlocale" not in line:
            sections[current].append(line)
    return sections


def first(sections, key):
    lines = [l for l in sections.get(key, []) if l.strip()]
    return lines[0].strip() if lines else ""


def per_table(metrics_lines, min_rel):
    """Baris tabel + (best_epoch, since) dengan aturan zipformer_watchdog.sh."""
    rows = {}
    for line in metrics_lines:
        if line.strip():
            r = json.loads(line)
            rows[r["epoch"]] = r
    best, table = None, []
    for ep in sorted(rows):
        per = rows[ep]["per_phonemes"]
        new_best = best is None or per < rows[best]["per_phonemes"] * (1 - min_rel)
        if new_best:
            best = ep
        table.append((ep, rows[ep].get("batch_idx_train"), per, rows[ep].get("silence_per"), new_best))
    since = (max(rows) - best) if rows else 0
    return table, best, since


def epoch_hours(epoch_end_lines):
    """Rata-rata durasi 3 epoch terakhir (jam) dari timestamp 'epoch N finished'."""
    stamps = []
    for line in epoch_end_lines:
        m = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", line)
        if m:
            stamps.append(datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
    gaps = [(b - a).total_seconds() / 3600 for a, b in zip(stamps, stamps[1:])][-3:]
    return sum(gaps) / len(gaps) if gaps else None


def vast_credit():
    """(credit, $/jam instance running) dari REST API Vast.ai; None jika gagal."""
    try:
        sys.path.insert(0, SCRIPT_DIR)
        from vast_api import get_api_key

        key = get_api_key()

        def get(path):
            req = urllib.request.Request(f"https://console.vast.ai/api/v0/{path}",
                                         headers={"Authorization": f"Bearer {key}"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)

        credit = get("users/current/").get("credit")
        insts = get("instances/").get("instances", [])
        burn = sum(i.get("dph_total") or 0 for i in insts if i.get("actual_status") == "running")
        return credit, burn
    except Exception:  # noqa: BLE001 - status tetap dicetak tanpa credit
        return None


def classify(s, num_epochs, last_epoch):
    trainer, shell, uploader = first(s, "TRAINER"), first(s, "SHELL"), first(s, "UPLOADER")
    done = first(s, "DONE") == "yes"
    watchdog = s.get("WATCHDOG", [])
    if uploader:
        return "UPLOADING"
    if trainer:
        return "TRAINING" if num_epochs is None or last_epoch < num_epochs else "FINISHING"
    if shell:
        return "PREPARING"
    # Antara SIGTERM dan relaunch tidak ada trainer / shell; log watchdog sedang di tengah trigger
    wd_log = [l for l in watchdog if l not in ("session", "running")]
    if "running" in watchdog and any("TRIGGER" in l or "SIGTERM" in l for l in wd_log):
        return "WATCHDOG_STOPPING"
    ended = first(s, "ENDED")
    if done and "exit code 0" in ended:
        return "ENDED"
    return "STOPPED"


def collect(args):
    """dict status run; NO_INSTANCE jika tidak ada box."""
    box = find_box(args)
    if box is None:
        return {"state": "NO_INSTANCE"}
    if box == "API_ERROR":
        return {"state": "API_ERROR"}
    host, port, instance_id = box
    s = probe(args, host, port)
    if s is None:
        return {"state": "SSH_FAILED", "host": host, "port": port, "instance_id": instance_id}
    params = first(s, "WATCHDOG_PARAMS")
    m = re.match(r"patience (\d+), min relative improvement ([0-9.]+)", params)
    patience = int(m.group(1)) if m else args.patience
    min_rel = float(m.group(2)) if m else args.min_improvement
    table, best, since = per_table(s.get("METRICS", []), min_rel)
    num_epochs = first(s, "NUM_EPOCHS")
    num_epochs = int(num_epochs) if num_epochs.isdigit() else None
    last = table[-1][0] if table else 0
    return {
        "state": classify(s, num_epochs, last), "host": host, "port": port, "instance_id": instance_id,
        "sections": s, "table": table, "best": best, "since": since, "patience": patience,
        "min_rel": min_rel, "watchdog_params_known": bool(m), "num_epochs": num_epochs, "last": last,
        "epoch_hours": epoch_hours(s.get("EPOCH_END", [])),
    }


def print_no_instance(args):
    print("STATUS: NO_INSTANCE —", STATES["NO_INSTANCE"])
    cfg_path = args.config or os.path.join(ZIPFORMER_DIR, "configs", "train_config.yml")
    ckpt = None
    if os.path.isfile(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            for line in f:
                m = re.match(r"rclone_checkpoint_path:\s*['\"]?([^'\"#\s]+)", line)
                if m:
                    ckpt = m.group(1)
    if not ckpt:
        return
    results = args.results_remote or f"{ckpt}-results"
    print(f"\nHasil di Drive ({results}):")
    code, out = run(["rclone", "lsl", results], timeout=180)
    print(out.rstrip() if code == 0 and out.strip() else "  (kosong / tidak bisa dibaca)")
    code, out = run(["rclone", "lsf", ckpt], timeout=180)
    if code == 0:
        epochs = sorted((l for l in out.split() if re.fullmatch(r"epoch-\d+\.pt", l)), key=lambda n: int(n[6:-3]))
        print(f"Checkpoint epoch di {ckpt}: {', '.join(epochs) or '-'}")


def print_status(args, st):
    if st["state"] == "NO_INSTANCE":
        print_no_instance(args)
        return
    if st["state"] == "SSH_FAILED":
        print(f"STATUS: SSH_FAILED — box {st['host']}:{st['port']} tidak bisa dihubungi lewat SSH")
        return
    if st["state"] == "API_ERROR":
        print(f"STATUS: API_ERROR — {STATES['API_ERROR']}")
        return
    s = st["sections"]
    print(f"STATUS: {st['state']} — {STATES[st['state']]}")
    print(f"Box: {st['host']}:{st['port']}" + (f" (instance {st['instance_id']})" if st["instance_id"] else "")
          + f", waktu box {first(s, 'NOW')} UTC, config {first(s, 'CONFIG')}")

    table, best = st["table"], st["best"]
    rule = f"rekor = < rekor x {1 - st['min_rel']:.2f}" + ("" if st["watchdog_params_known"] else " (default; watchdog tidak ditemukan)")
    print(f"\nDev PER ({rule}):")
    print("  epoch     batch   PER fonem   silence PER")
    for ep, batch, per, sil, new_best in table[-args.rows:]:
        mark = "  <- rekor" + (" berlaku" if ep == best else "") if new_best else ""
        sil_s = f"{sil * 100:.3f}%" if sil is not None else "-"
        print(f"  {ep:5d} {batch or 0:9d}   {per * 100:8.3f}%   {sil_s:>10}{mark}")
    if not table:
        print("  (belum ada epoch selesai)")
    else:
        print(f"Epoch tanpa rekor: {st['since']}/{st['patience']}"
              f" (rekor berlaku epoch {best}, {table[[r[0] for r in table].index(best)][2] * 100:.3f}%)")

    hours, num_epochs, last = st["epoch_hours"], st["num_epochs"], st["last"]
    if hours:
        print(f"\nKecepatan: {hours * 60:.0f} menit/epoch (rata-rata 3 epoch terakhir); epoch {last}/{num_epochs}")
    if st["state"] == "TRAINING" and table:
        stop_at = last + st["patience"] - st["since"]
        print(f"Early stop paling cepat setelah epoch {stop_at} (jika tidak ada rekor baru)")
    credit = vast_credit()
    if credit:
        c, burn = credit
        line = f"Credit Vast.ai: ${c:.2f}, instance ${burn:.3f}/jam"
        if burn and hours and num_epochs:
            worst = ((num_epochs - last) * hours + FINISH_HOURS) * burn
            line += f"; sampai epoch {num_epochs}: ±${worst:.2f}" + (" — CREDIT TIDAK CUKUP" if worst > c else "")
        print(line)

    wd = s.get("WATCHDOG", [])
    wd_state = "berjalan" if "running" in wd else ("sesi ada, proses mati" if "session" in wd else "tidak ada")
    print(f"\nWatchdog: {wd_state}")
    for l in wd:
        if l not in ("session", "running"):
            print(f"  {l}")
    if first(s, "GPU"):
        print(f"GPU: {first(s, 'GPU')}")
    for key, label in (("STAGE", "Tahap terakhir"), ("ENDED", "Akhir run"), ("FINISH", "Upload/destroy"), ("ERRORS", "Error di log")):
        lines = [l for l in s.get(key, []) if l.strip()]
        if lines:
            print(f"{label}:")
            for l in lines:
                print(f"  {l[:220]}")

    if args.drive:
        ckpt = first(s, "CKPT_PATH")
        print(f"Checkpoint Drive ({ckpt}): {verify_drive_epoch(ckpt)}")


def verify_drive_epoch(ckpt):
    """Cocokkan SHA-256 epoch-*.pt terbaru di Drive dengan meta.json-nya (rclone lokal).

    rclone di box hanya memberi MD5 untuk Google Drive, jadi ini dijalankan dari lokal.
    """
    code, out = run(["rclone", "lsf", ckpt], timeout=180)
    epochs = sorted((n for n in out.split() if re.fullmatch(r"epoch-\d+\.pt", n)), key=lambda n: int(n[6:-3]))
    if code != 0 or not epochs:
        return "belum ada epoch-*.pt (atau folder tidak bisa dibaca)"
    name = epochs[-1]
    code, meta = run(["rclone", "cat", f"{ckpt}/{name[:-3]}.meta.json"], timeout=180)
    want = json.loads(meta).get("sha256") if code == 0 and meta.strip() else None
    code, listing = run(["rclone", "lsjson", "--hash", f"{ckpt}/{name}"], timeout=300)
    entries = json.loads(listing) if code == 0 and listing.strip() else []
    got = entries[0].get("Hashes", {}).get("sha256") if entries else None
    if not want or not got:
        return f"{name}: sha256 tidak tersedia (meta={'ada' if want else 'tidak ada'}, file={'ada' if entries else 'tidak ada'})"
    return f"{name}: {'COCOK' if want == got else 'TIDAK COCOK'} (sha256 {got[:12]}…)"


def wait_for_change(args):
    """Poll sampai ada epoch baru, STATUS berubah, atau --wait-timeout habis."""
    start = collect(args)
    deadline = time.monotonic() + args.wait_timeout
    no_instance = 0
    while time.monotonic() < deadline:
        time.sleep(args.poll)
        st = collect(args)
        # Gangguan sesaat bukan perubahan keadaan run
        if st["state"] in ("SSH_FAILED", "API_ERROR"):
            continue
        no_instance = no_instance + 1 if st["state"] == "NO_INSTANCE" else 0
        if 0 < no_instance < NO_INSTANCE_CONFIRMATIONS and start["state"] != "NO_INSTANCE":
            continue
        if st["state"] != start["state"] or st.get("last", 0) != start.get("last", 0):
            print(f"PERUBAHAN: {start['state']} epoch {start.get('last', 0)} -> {st['state']} epoch {st.get('last', 0)}\n")
            return st
    print(f"TIDAK ADA PERUBAHAN dalam {args.wait_timeout} detik\n")
    return collect(args)


def main():
    parser = argparse.ArgumentParser(description="Status run training Zipformer di Vast.ai.")
    parser.add_argument("--host", help="IP / hostname SSH box (default: instance dari Vast.ai API)")
    parser.add_argument("--port", help="port SSH box")
    parser.add_argument("--label", help="filter label instance saat memakai Vast.ai API (default: VAST_LABEL)")
    parser.add_argument("--identity", default=os.path.expanduser("~/.ssh/id_rsa"), help="private key SSH")
    parser.add_argument("--config", help="config lokal untuk mencari hasil di Drive saat instance sudah tidak ada "
                                         "(default: zipformer/configs/train_config.yml)")
    parser.add_argument("--results-remote", help="folder hasil di Drive (default: <rclone_checkpoint_path>-results)")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE,
                        help="dipakai jika log watchdog tidak ada di box (default: 4)")
    parser.add_argument("--min-improvement", type=float, default=DEFAULT_MIN_IMPROVEMENT,
                        help="dipakai jika log watchdog tidak ada di box (default: 0.05)")
    parser.add_argument("--rows", type=int, default=8, help="jumlah epoch terakhir di tabel (default: 8)")
    parser.add_argument("--drive", action="store_true", help="cocokkan SHA-256 epoch-*.pt terbaru di Drive")
    parser.add_argument("--wait", action="store_true", help="tunggu epoch baru / perubahan STATUS, lalu cetak status")
    parser.add_argument("--poll", type=int, default=120, help="interval poll --wait dalam detik (default: 120)")
    parser.add_argument("--wait-timeout", type=int, default=6600, help="batas --wait dalam detik (default: 6600)")
    args = parser.parse_args()

    st = wait_for_change(args) if args.wait else collect(args)
    print_status(args, st)


if __name__ == "__main__":
    main()

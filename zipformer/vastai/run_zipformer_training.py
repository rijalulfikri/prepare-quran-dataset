#!/usr/bin/env python3

"""
run_zipformer_training.py

Satu skrip dari komputer lokal untuk menjalankan langkah 3 README Zipformer
("Jalankan di Vast.ai") di box GPU yang sudah disewa:

1. Cek koneksi SSH dan GPU di box.
2. Install git, tmux, dan curl di box jika belum ada.
3. Clone repo ini di box (atau fetch + checkout branch jika sudah ada).
4. Kirim ~/.config/rclone/rclone.conf ke box.
5. Kirim zipformer/.env ke box (WANDB_API_KEY, HUGGINGFACE_TOKEN, K2_WHEEL_URL,
   K2_WHEEL_URL_CP312).
6. Jalankan `bash train_zipformer.sh` di sesi tmux `zipformer`, dengan log di
   ~/train_zipformer.log, sehingga training tetap berjalan walau SSH putus.

Dengan --destroy-when-done, setelah train_zipformer.sh sukses (exit 0) box
menjalankan upload_results_and_destroy.sh: output_dir di-upload ke Drive
(--results-remote, default <rclone_checkpoint_path>-results), diverifikasi, lalu
instance Vast.ai dihapus. Jika training gagal/berhenti atau upload gagal,
instance dibiarkan hidup.

Skrip ini aman dijalankan ulang, termasuk di box baru setelah box lama mati:
train_zipformer.sh melanjutkan dari checkpoint terbaru di Google Drive.

Box ditentukan dengan salah satu cara:
- --offer-id: buat instance baru dari offer tersebut (image --image, disk --disk,
  label --label), tunggu sampai SSH siap, lalu lanjut seperti biasa.
- --host/--port: box yang sudah ada.
- --host saja: port SSH instance dengan IP tersebut dicari lewat Vast.ai API.
- tanpa keduanya: instance running/loading pertama dari Vast.ai API (bisa
  difilter --label).
Tanpa --port, skrip menunggu (maks --wait-timeout detik) sampai instance yang
cocok muncul dengan info SSH, lalu menunggu SSH siap.
Semua cara selain --host + --port butuh VAST_API_KEY dan paket `vastai`.

Contoh:
    python zipformer/vastai/run_zipformer_training.py --destroy-when-done --follow
    python zipformer/vastai/run_zipformer_training.py --offer-id 12345678 --destroy-when-done --follow
    python zipformer/vastai/run_zipformer_training.py --host 45.81.32.13 --follow
    python zipformer/vastai/run_zipformer_training.py --host 45.81.32.13 --port 23330 --follow
    python zipformer/vastai/run_zipformer_training.py --label zipformer --auto-restart --destroy-when-done
    python zipformer/vastai/run_zipformer_training.py --host ... --port ... -- --push-to-hub

Prasyarat:
- Commit yang ingin dilatih sudah di-push ke GitHub (box meng-clone dari origin).
- rclone sudah dikonfigurasi di lokal (remote `utluma:`, lihat README bagian 2).
- zipformer/.env berisi K2_WHEEL_URL / K2_WHEEL_URL_CP312 (atau set di environment lokal).
"""

import argparse
import os
import re
import shlex
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ZIPFORMER_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(ZIPFORMER_DIR)

DEFAULT_REPO_URL = "https://github.com/rijalulfikri/prepare-quran-dataset.git"
REMOTE_REPO_DIR = "/root/prepare-quran-dataset"
REMOTE_LOG = "/root/train_zipformer.log"
REMOTE_DONE_MARKER = "/root/.train_zipformer.done"
TMUX_SESSION = "zipformer"
FINISH_SCRIPT = "upload_results_and_destroy.sh"
REMOTE_FINISH_SCRIPT = f"/root/{FINISH_SCRIPT}"
K2_VARS = ("K2_WHEEL_URL", "K2_WHEEL_URL_CP312")
DEFAULT_TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"  # sama dengan train_zipformer.sh

# Untuk --offer-id. train_zipformer.sh membuat venv sendiri (uv, torch cu128), jadi
# image cukup punya pip; driver host harus mendukung CUDA 12.8.
DEFAULT_IMAGE = "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime"
# Data ~60 GB + venv ~15 GB + checkpoint (lokal & salinan dari Drive) + hasil
DEFAULT_DISK = 150.0
DEFAULT_LABEL = "zipformer"
# Supaya CONTAINER_ID / CONTAINER_API_KEY terbaca di sesi SSH (dipakai --destroy-when-done)
ONSTART_CMD = "env >> /etc/environment"
SSH_READY_TIMEOUT = 600
INSTANCE_WAIT_TIMEOUT = 1800


def run_command(command, dry_run=False, check=True):
    """Jalankan command lokal; berhenti (exit 1) jika gagal dan check=True.

    Returns:
        int: exit code command (0 saat dry run).
    """
    print(f"$ {shlex.join(command)}")
    if dry_run:
        return 0
    result = subprocess.run(command)
    if check and result.returncode != 0:
        print(f"Error: command gagal (exit code {result.returncode}).")
        sys.exit(1)
    return result.returncode


def read_env_keys(env_path):
    """Kembalikan dict KEY -> value (tanpa kutip) dari file .env sederhana."""
    values = {}
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.removeprefix("export ").strip()
            values[key] = value.strip().strip("'\"")
    return values


def resolve_box(args):
    """Kembalikan (host, port, created_instance_id, from_api) box.

    created_instance_id hanya terisi jika instance baru dibuat dari --offer-id;
    from_api True jika port didapat dari Vast.ai API (SSH mungkin belum siap).
    """
    if args.offer_id is not None and (args.host or args.port):
        print("Error: --offer-id tidak bisa digabung dengan --host/--port.")
        sys.exit(1)
    if args.host and args.port:
        return args.host, str(args.port), None, False
    if args.port:
        print("Error: --port butuh --host.")
        sys.exit(1)

    # Import di sini supaya --host/--port tidak butuh paket vastai / VAST_API_KEY
    sys.path.insert(0, SCRIPT_DIR)
    from vast_api import create_instance_with_ssh_info, get_first_started_instance

    if args.offer_id is not None:
        if args.dry_run:
            print(f"(dry run) instance dari offer {args.offer_id} tidak dibuat; memakai host contoh.")
            return "<IP_INSTANCE_BARU>", "<PORT>", None, False
        info = create_instance_with_ssh_info(
            offer_id=args.offer_id,
            image=args.image,
            onstart_cmd=ONSTART_CMD,
            disk=args.disk,
            label=args.label or DEFAULT_LABEL,
        )
        if not info:
            print(f"Error: gagal membuat instance dari offer {args.offer_id} (atau timeout menunggu running).")
            print("Cek https://cloud.vast.ai/instances/ ; hapus manual jika instance sempat dibuat.")
            sys.exit(1)
        print(f"Instance {info['instance_id']} dibuat: {info['ip']}:{info['ssh_port']}")
        return info["ip"], str(info["ssh_port"]), info["instance_id"], True

    label = args.label or os.getenv("VAST_LABEL")
    info = wait_for_instance(get_first_started_instance, label, args.host, args.wait_timeout, args.dry_run)
    if not info and args.dry_run:
        print("(dry run) instance belum ada; tidak ditunggu, memakai host contoh.")
        return args.host or "<IP_INSTANCE>", "<PORT>", None, False
    if not info:
        print("Error: tidak ada instance Vast.ai running/loading yang cocok. "
              "Isi --host/--port, --offer-id, atau naikkan --wait-timeout.")
        sys.exit(1)
    print(f"Instance {info['instance_id']}: {info['ip']}:{info['ssh_port']}")
    return info["ip"], str(info["ssh_port"]), None, True


def wait_for_instance(lookup, label, ip, timeout, dry_run=False, interval=15):
    """Poll Vast.ai sampai ada instance running/loading (filter label/IP) dengan info SSH.

    Returns:
        dict | None: hasil lookup(), atau None jika timeout (atau tidak ketemu saat dry run).
    """
    desc = " dan ".join(f for f in [label and f"label '{label}'", ip and f"IP {ip}"] if f)
    print(f"Mencari instance Vast.ai{' dengan ' + desc if desc else ''}...")
    deadline = time.monotonic() + timeout
    waited = False
    while True:
        info = lookup(label, ip, verbose=False)
        if info or dry_run:
            if waited:
                print()
            return info
        if time.monotonic() >= deadline:
            print()
            return None
        remaining = int(deadline - time.monotonic())
        print(f"  belum ada instance yang siap; cek lagi dalam {interval} detik (sisa {remaining} detik)...",
              end="\r", flush=True)
        waited = True
        time.sleep(interval)


def wait_for_ssh(ssh, timeout=SSH_READY_TIMEOUT, interval=15):
    """Tunggu sampai box menerima SSH (status 'running' belum berarti sshd siap)."""
    print(f"Menunggu SSH siap (maks {timeout} detik)...")
    deadline = time.monotonic() + timeout
    while True:
        probe = ssh[:1] + ["-o", "ConnectTimeout=10", "-o", "BatchMode=yes"] + ssh[1:] + ["true"]
        if subprocess.run(probe, capture_output=True).returncode == 0:
            print("SSH siap.")
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(interval)


def check_local_prerequisites(args):
    """Validasi rclone.conf, .env, K2_WHEEL_URL, dan commit yang akan di-clone box."""
    if not os.path.isfile(args.rclone_conf):
        print(f"Error: {args.rclone_conf} tidak ditemukan. Konfigurasi rclone dulu (README bagian 2).")
        sys.exit(1)

    env_values = {}
    if os.path.isfile(args.env_file):
        env_values = read_env_keys(args.env_file)
    else:
        print(f"PERINGATAN: {args.env_file} tidak ada; W&B / HuggingFace tidak akan aktif.")

    if not any(os.getenv(k) or env_values.get(k) for k in K2_VARS):
        print("Error: K2_WHEEL_URL / K2_WHEEL_URL_CP312 belum diisi di .env maupun environment lokal.")
        print("Pilih wheel k2 untuk torch 2.9.1 + CUDA 12.8 di https://k2-fsa.github.io/k2/cuda.html")
        sys.exit(1)

    # k2 menolak di-import jika versi CUDA wheel-nya beda dengan torch (TORCH_INDEX_URL)
    torch_index = os.getenv("TORCH_INDEX_URL") or env_values.get("TORCH_INDEX_URL") or DEFAULT_TORCH_INDEX_URL
    torch_cu = re.search(r"/cu(\d+)(\d)/?$", torch_index)
    for k in K2_VARS:
        k2_cu = re.search(r"\+cuda(\d+\.\d+)\.", os.getenv(k) or env_values.get(k) or "")
        if torch_cu and k2_cu and k2_cu.group(1) != f"{torch_cu.group(1)}.{torch_cu.group(2)}":
            print(f"Error: {k} adalah build CUDA {k2_cu.group(1)}, tetapi torch diinstall dari {torch_index} "
                  f"(CUDA {torch_cu.group(1)}.{torch_cu.group(2)}). Samakan versinya.")
            sys.exit(1)

    # Box meng-clone dari GitHub, jadi commit lokal yang belum di-push tidak ikut dilatih
    local_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
    ).stdout.strip()
    remote_head = subprocess.run(
        ["git", "ls-remote", args.repo_url, f"refs/heads/{args.branch}"],
        capture_output=True, text=True,
    ).stdout.split("\t")[0].strip()
    if not remote_head:
        print(f"Error: branch '{args.branch}' tidak ditemukan di {args.repo_url}.")
        sys.exit(1)
    if local_head and local_head != remote_head:
        print(f"PERINGATAN: HEAD lokal ({local_head[:7]}) berbeda dengan origin/{args.branch} "
              f"({remote_head[:7]}). Box akan melatih {remote_head[:7]}; push dulu jika perlu.")
        # Gagal sebelum menyewa box jika commit di GitHub belum punya folder zipformer/
        has_zipformer = subprocess.run(
            ["git", "cat-file", "-e", f"{remote_head}:zipformer/train_zipformer.sh"],
            cwd=REPO_ROOT, capture_output=True,
        ).returncode == 0
        if not has_zipformer and not args.dry_run:
            print(f"Error: commit {remote_head[:7]} di GitHub tidak berisi zipformer/train_zipformer.sh "
                  f"(atau belum di-fetch). Push dulu: git push origin {args.branch}")
            sys.exit(1)


def build_train_command(args):
    """Command bash yang dijalankan di dalam sesi tmux: training (+ upload & destroy)."""
    exports = [f"export {k}={shlex.quote(os.environ[k])}" for k in K2_VARS if os.getenv(k)]
    if args.auto_restart:
        exports.append("export AUTO_RESTART=1")
    train_cmd = shlex.join(["bash", "train_zipformer.sh", *args.train_args])

    # .env di-source supaya K2_WHEEL_URL* terbaca oleh train_zipformer.sh;
    # export dari environment lokal (jika ada) menimpa isi .env.
    inner = " && ".join([
        f"cd {REMOTE_REPO_DIR}/zipformer",
        "if [ -f .env ]; then set -a; . ./.env; set +a; fi",
        *exports,
        f"{train_cmd} 2>&1 | tee -a {REMOTE_LOG}",
    ])
    inner = (
        f"rm -f {REMOTE_DONE_MARKER}; set -o pipefail; {inner}; code=$?; "
        f"echo \"[train_zipformer.sh selesai, exit code $code]\" | tee -a {REMOTE_LOG}"
    )
    if args.destroy_when_done:
        finish_env = [f"REPO_ZIPFORMER_DIR={REMOTE_REPO_DIR}/zipformer"]
        if args.results_remote:
            finish_env.append(f"RESULTS_REMOTE={shlex.quote(args.results_remote)}")
        inner += (
            f"; if [ $code -eq 0 ]; then {' '.join(finish_env)} bash {REMOTE_FINISH_SCRIPT} 2>&1 | tee -a {REMOTE_LOG}; "
            f"else echo '[--destroy-when-done: training tidak sukses, instance dibiarkan hidup]' | tee -a {REMOTE_LOG}; fi"
        )
    # Penanda bahwa sesi tmux ini sudah selesai (shell-nya tetap terbuka untuk membaca log)
    return inner + f"; touch {REMOTE_DONE_MARKER}"


def build_launch_command(args):
    """Command bash di box untuk menjalankan build_train_command() di sesi tmux."""
    inner = build_train_command(args)
    return (
        f"if tmux has-session -t {TMUX_SESSION} 2>/dev/null; then "
        f"if [ -f {REMOTE_DONE_MARKER} ]; then "
        f"echo 'Sesi tmux {TMUX_SESSION} dari run sebelumnya sudah selesai; ditutup.'; tmux kill-session -t {TMUX_SESSION}; "
        f"else echo 'Error: sesi tmux {TMUX_SESSION} masih menjalankan training. "
        f"Attach dengan: tmux attach -t {TMUX_SESSION} (atau tutup: tmux kill-session -t {TMUX_SESSION})'; exit 1; fi; fi; "
        f"tmux new-session -d -s {TMUX_SESSION} {shlex.quote(f'bash -c {shlex.quote(inner)}; exec bash')}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Jalankan training Zipformer (langkah 3 README) di box Vast.ai dari komputer lokal.",
    )
    parser.add_argument("--offer-id", type=int, help="buat instance baru dari offer Vast.ai ini")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help=f"Docker image untuk --offer-id (default: {DEFAULT_IMAGE})")
    parser.add_argument("--disk", type=float, default=DEFAULT_DISK,
                        help=f"disk (GB) untuk --offer-id (default: {DEFAULT_DISK:g})")
    parser.add_argument("--host", help="IP / hostname SSH box (default: instance pertama dari Vast.ai API)")
    parser.add_argument("--port", help="port SSH box (default: dari Vast.ai API, dicari berdasarkan --host jika diisi)")
    parser.add_argument("--wait-timeout", type=int, default=INSTANCE_WAIT_TIMEOUT,
                        help="tanpa --port: lama menunggu instance yang cocok muncul, dalam detik "
                             f"(default: {INSTANCE_WAIT_TIMEOUT})")
    parser.add_argument("--label", help=f"label instance baru untuk --offer-id (default: {DEFAULT_LABEL}), "
                                        "atau filter instance saat memakai Vast.ai API (default: VAST_LABEL)")
    parser.add_argument("--identity", default=os.path.expanduser("~/.ssh/id_rsa"), help="private key SSH")
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--branch", default="main", help="branch yang di-clone di box (default: main)")
    parser.add_argument("--env-file", default=os.path.join(ZIPFORMER_DIR, ".env"))
    parser.add_argument("--rclone-conf", default=os.path.expanduser("~/.config/rclone/rclone.conf"))
    parser.add_argument("--auto-restart", action="store_true",
                        help="set AUTO_RESTART=1 (ulangi training maks 3x jika crash)")
    parser.add_argument("--destroy-when-done", action="store_true",
                        help="setelah training sukses: upload output_dir ke Drive lalu hapus instance Vast.ai")
    parser.add_argument("--results-remote",
                        help="tujuan upload hasil untuk --destroy-when-done "
                             "(default: <rclone_checkpoint_path>-results)")
    parser.add_argument("--follow", action="store_true",
                        help="tampilkan log training setelah dijalankan (Ctrl+C tidak menghentikan training)")
    parser.add_argument("--dry-run", action="store_true", help="cetak command tanpa menjalankannya")
    parser.add_argument("train_args", nargs="*",
                        help="argumen tambahan untuk train_zipformer.sh, setelah '--' (mis. -- --push-to-hub)")
    args = parser.parse_args()

    check_local_prerequisites(args)
    host, port, created_id, from_api = resolve_box(args)
    try:
        launch_training(args, host, port, wait_ssh=from_api and not args.dry_run)
    except SystemExit as exc:
        if created_id is not None and exc.code:
            print(f"\nPERINGATAN: instance {created_id} sudah dibuat dan masih berjalan (ditagih).")
            print(f"  Jalankan ulang dengan --host {host} --port {port} setelah masalahnya diperbaiki,")
            print(f"  atau hapus: vastai destroy instance {created_id} -y")
        raise


def launch_training(args, host, port, wait_ssh=False):
    """Langkah 1-6 di box (lihat docstring modul)."""
    ssh_opts = ["-i", args.identity, "-o", "StrictHostKeyChecking=accept-new", "-o", "ServerAliveInterval=30"]
    ssh = ["ssh", "-p", port, *ssh_opts, f"root@{host}"]
    scp = ["scp", "-P", port, *ssh_opts]
    dry = args.dry_run

    print(f"--- Menjalankan training Zipformer di {host}:{port} ---\n")

    if wait_ssh and not wait_for_ssh(ssh):
        print("Error: SSH tidak bisa dihubungi. Pastikan public key dari --identity terdaftar di akun Vast.ai.")
        sys.exit(1)

    print("[1/6] Cek koneksi SSH & GPU")
    run_command(ssh + ["nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; df -h /root | tail -1"], dry)

    print("\n[2/6] Install git, tmux, curl (jika belum ada)")
    run_command(ssh + [
        "for c in git tmux curl; do command -v $c >/dev/null || missing=\"$missing $c\"; done; "
        "if [ -n \"$missing\" ]; then apt-get update && apt-get install -y $missing; fi"
    ], dry)

    print(f"\n[3/6] Clone / update repo ({args.branch}) di {REMOTE_REPO_DIR}")
    repo, branch = REMOTE_REPO_DIR, shlex.quote(args.branch)
    run_command(ssh + [
        f"if [ -d {repo}/.git ]; then "
        f"git -C {repo} fetch origin && git -C {repo} checkout {branch} && git -C {repo} pull --ff-only origin {branch}; "
        f"else git clone --branch {branch} {shlex.quote(args.repo_url)} {repo}; fi && "
        f"git -C {repo} log --oneline -1"
    ], dry)
    # Branch di GitHub bisa saja masih layout lama (belum di-push) tanpa zipformer/
    if run_command(ssh + [f"test -f {repo}/zipformer/train_zipformer.sh"], dry, check=False) != 0:
        print(f"Error: {repo}/zipformer/train_zipformer.sh tidak ada di hasil clone branch '{args.branch}'.")
        print(f"  Push dulu commit yang berisi folder zipformer/ ke GitHub (git push origin {args.branch}),")
        print("  lalu jalankan ulang skrip ini.")
        sys.exit(1)

    print("\n[4/6] Kirim konfigurasi rclone")
    run_command(ssh + ["mkdir -p /root/.config/rclone"], dry)
    run_command(scp + [args.rclone_conf, f"root@{host}:/root/.config/rclone/rclone.conf"], dry)
    run_command(ssh + ["chmod 600 /root/.config/rclone/rclone.conf"], dry)

    print("\n[5/6] Kirim .env" + (f" dan {FINISH_SCRIPT}" if args.destroy_when_done else ""))
    if os.path.isfile(args.env_file):
        run_command(scp + [args.env_file, f"root@{host}:{REMOTE_REPO_DIR}/zipformer/.env"], dry)
    else:
        print("Dilewati (.env lokal tidak ada).")
    if args.destroy_when_done:
        # Dikirim dari lokal (bukan dari clone) supaya tidak bergantung pada commit yang sudah di-push
        run_command(scp + [os.path.join(SCRIPT_DIR, FINISH_SCRIPT), f"root@{host}:{REMOTE_FINISH_SCRIPT}"], dry)

    print(f"\n[6/6] Jalankan train_zipformer.sh di sesi tmux '{TMUX_SESSION}'")
    run_command(ssh + [build_launch_command(args)], dry)

    ssh_str = shlex.join(["ssh", "-p", port, "-i", args.identity, f"root@{host}"])
    print("\n--- ✓ Training berjalan di box ---")
    print(f"  Lihat log      : {ssh_str} 'tail -f {REMOTE_LOG}'")
    print(f"  Attach tmux    : {ssh_str} -t 'tmux attach -t {TMUX_SESSION}'")
    if args.destroy_when_done:
        dest = args.results_remote or "<rclone_checkpoint_path>-results"
        print(f"  Setelah sukses : hasil di-upload ke {dest}, lalu instance dihapus")
    else:
        print(f"  Ambil hasil    : scp -P {port} -r root@{host}:{REMOTE_REPO_DIR}/zipformer/results-zipformer-v0/export ./")

    if args.follow:
        print("\nMenampilkan log (Ctrl+C hanya menghentikan tampilan log, bukan training)...\n")
        try:
            run_command(ssh + [f"tail -n 50 -F {REMOTE_LOG}"], dry, check=False)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

"""
send_files.py

Script otomatisasi untuk menginisiasi dan mentransfer file proyek ke server jarak jauh (remote server), 
seperti instance GPU yang disewa di Vast.ai. 

Kegunaan utama script ini:
1. Menginstal dependensi Python awal yang diperlukan di server.
2. Mentransfer kode proyek Icefall MGB2 (folder 'zipformer' dan 'local') dengan cara mengompresnya 
   di lokal, mengirimnya via SCP, lalu mengekstraknya di lingkungan kerja server.
3. Mentransfer konfigurasi rclone agar server memiliki akses ke cloud storage (Google Drive, S3, dll.)
   tanpa perlu autektikasi manual lagi.
4. Mentransfer script setup lanjutan (seperti prepare_dataset.py, train_model.py, dll.) ke 
   direktori root server untuk mempermudah eksekusi tahap selanjutnya.

Prasyarat:
- Pastikan rclone sudah dikonfigurasi di komputer lokal (~/.config/rclone/rclone.conf).
- Pastikan Anda sudah memiliki private key SSH (id_rsa) yang terdaftar di server tujuan.
"""

import os
import shlex
import subprocess
import sys
import time
from common_utils import (
    get_env_variable)

# --- KONFIGURASI VARIABLE ---
# Ubah nilai di bawah ini sesuai dengan detail koneksi server (misal: koneksi SSH dari Vast.ai)
INSTANCE_ID = get_env_variable("INSTANCE_ID", required=True, default="")                                # ID instance vast.ai (wajib diisi!)
PORT = get_env_variable("PORT", required=True, default="")                                   # Port SSH server jarak jauh
HOSTNAME = get_env_variable("HOSTNAME", required=True, default="")                         # Alamat IP server jarak jauh
ID_RSA = os.path.expanduser("~/.ssh/id_rsa")    # Path menuju file private key SSH lokal
PROJECT_HOME = "/Volumes/F/MachineLearning/icefall/" # Path lokal direktori dasar proyek icefall
AUTO_DESTROY = True  # Set True untuk aktifkan auto-destroy, False untuk matikan (debugging)

# --- KONFIGURASI TRANSAKSI TRANKSKRIPSI ---
# File audio di bawah `egs/mgb2/ASR/download/murattal/train/wav.ori/` yang akan ditranskrip
AUDIO_FILE_SUBPATH = "Abdullah_Basfar/Abdullah_Basfar-03.wav" 
# Nama file output JSON yang didinginkan
OUTPUT_JSON_NAME = "Abdullah_Basfar_WP_1-03.json"
# ----------------------------

def wait_for_ssh(ssh_command, max_retries=15, sleep_time=5):
    """
    Menunggu hingga koneksi SSH ke remote server benar-benar siap dan stabil.
    Server Vast.ai baru di-rent mungkin butuh puluhan detik untuk daemon SSH-nya menyala.
    """
    print(f"Menunggu koneksi SSH stabil dari server {HOSTNAME} (bisa memakan waktu beberapa saat)...")
    for attempt in range(1, max_retries + 1):
        try:
            # Gunakan subprocess.run dengan command echo simpel. Jika output ada 'SSH_READY', berhasil.
            result = subprocess.run(
                ssh_command + ["echo", "SSH_READY"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                timeout=10  # Tambahkan fallback timeout OS agar python tidak hang
            )
            if b"SSH_READY" in result.stdout:
                print("✓ Koneksi SSH stabil dan siap!\n")
                return True
            else:
                # Kasus sangat langka: SSH sukses exit 0 tapi string tidak kembali.
                # Kita lempar error paksa agar diproses di blok except dan retry jalan.
                raise subprocess.CalledProcessError(0, ssh_command)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            print(f"  -> [Percobaan {attempt}/{max_retries}] SSH belum siap (koneksi ditolak/timeout), mencoba lagi dalam {sleep_time} detik...")
            time.sleep(sleep_time)
            
    print("\nError: Koneksi SSH tidak bisa dijangkau setelah beberapa kali percobaan.")
    print("Pastikan Hostname dan Port benar, serta server Vast.ai sudah berstatus 'Running' di websitenya.")
    sys.exit(1)

def run_command(command, cwd=None):
    """
    Menjalankan command shell atau sistem operasi menggunakan library subprocess.
    
    Fungsi ini adalah wrapper untuk memudahkan eksekusi perintah terminal dari dalam Python.
    Jika perintah gagal (me-return exit code selain 0), script akan otomatis berhenti.
    
    Args:
        command (list): Daftar string bagian dari command yang akan dijalankan (contoh: ['ls', '-la']).
        cwd (str, optional): Working directory atau path folder di mana command tersebut harus dieksekusi.
    
    Raises:
        SystemExit: Berhenti total (exit status 1) jika command gagal dijalankan.
    """
    try:
        subprocess.run(
            command,
            cwd=cwd,
            check=True,
            shell=False  # Lebih aman menggunakan shell=False untuk menghindari shell injection
        )
    except subprocess.CalledProcessError as e:
        print(f"Error saat menjalankan command (gagal/crash): {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"Error: Program CLI tidak ditemukan di sistem: {e}")
        sys.exit(1)

def transfer_setup_scripts(scp_command):
    """
    Mentransfer kumpulan script otomatisasi Python lainnya (untuk setup lanjutan, 
    pembuatan dataset, training, dll) ke direktori root di remote server.

    File script Python (.py) yang kecil dikirim via SCP.
    File audio WAV yang berukuran besar dikirim via rsync karena rsync mendukung:
    - Resume transfer (tidak mulai dari awal jika koneksi putus di tengah jalan)
    - Deteksi perubahan file (hanya kirim bagian yang berubah)
    - Lebih tahan terhadap koneksi tidak stabil / 'Broken Pipe'
    
    Args:
        scp_command (list): Base command SCP untuk mentransfer file kecil.
    """
    mgb2_asr_path = os.path.join(PROJECT_HOME, "egs", "mgb2", "ASR")
    whisperx_dir = os.path.join(mgb2_asr_path, "tools", "whisperx")
    wavs_dir = os.path.join(mgb2_asr_path, "download","murattal","train","wav.ori")
    
    whisperx_transcribe_path = os.path.join(whisperx_dir, "whisperx_transcribe.py")
    wav_path = os.path.join(wavs_dir, AUDIO_FILE_SUBPATH)
    remote_dest = "/workspace/"  #TODO ubah path ini menjadi path default di container vast-ai

    # Transfer script Python (kecil) via SCP — cepat dan tidak butuh resume
    print("  Mengirim script whisperx_transcribe.py via SCP...")
    run_command(scp_command + [whisperx_transcribe_path, f"root@{HOSTNAME}:{remote_dest}"])

    # Transfer file WAV (besar) via rsync — tahan putus, bisa resume
    # Flag:
    #   --partial      : simpan file yang sudah terkirim sebagian jika putus (untuk resume)
    #   --progress     : tampilkan progres transfer
    #   -z             : kompres data saat transfer (menghemat bandwidth)
    #   -e             : tentukan transport SSH beserta opsi-opsinya
    print(f"  Mengirim file audio besar '{os.path.basename(wav_path)}' via rsync (mendukung resume)...")
    ssh_opt_str = f"ssh -p {PORT} -i {ID_RSA} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=30 -o ServerAliveCountMax=10"
    rsync_command = [
        "rsync",
        "--partial",   # simpan progres jika putus, agar bisa dilanjutkan
        "--progress",  # tampilkan progres
        "-z",          # kompresi data saat transfer
        "-e", ssh_opt_str,
        wav_path,
        f"root@{HOSTNAME}:{remote_dest}",
    ]
    run_command(rsync_command)
    print(f"Script dan audio berhasil dikirim ke {HOSTNAME}:{remote_dest}")

def transfer_rclone_config(scp_command, ssh_command):
    """
    Menyalin file konfigurasi otentikasi penyimpanan awan (Cloud Storage) milik aplikasi `rclone` 
    dari komputer lokal ke remote server. Dengan begini, server dapat langsung mengakses bucket 
    seperti AWS S3 atau Google Drive tanpa harus login token rclone ulang secara manual.
    
    Args:
        scp_command (list): Base command SCP.
        ssh_command (list): Base command SSH.
    """
    print("Memastikan direktori rclone di remote server tersedia...")
    remote_rclone_dir = "/root/.config/rclone/"
    # Buat direktori .config/rclone di server jika belum ada (-p)
    run_command(ssh_command + [f"mkdir -p {remote_rclone_dir}"])

    print("Mengirim konfigurasi rclone...")
    rclone_conf_path = os.path.expanduser("~/.config/rclone/rclone.conf")
    
    # Pastikan komputer lokal tempat menjalankan script ini sudah setup rclone
    if not os.path.isfile(rclone_conf_path):
        print("Error: File ~/.config/rclone/rclone.conf tidak ditemukan!")
        print("Pastikan rclone sudah dikonfigurasi di sistem lokal (komputer Anda).")
        sys.exit(1)
    
    # Kirim rclone config
    run_command(scp_command + [rclone_conf_path, f"root@{HOSTNAME}:{remote_rclone_dir}"])
    print(f"Konfigurasi rclone berhasil dikirim ke {HOSTNAME}")

def main():
    """
    Titik masuk utama eksekusi program.
    Mengorkestrasi seluruh tahapan transfer dan penyiapan file.
    """
    print(f"--- Memulai proses transfer data ke {HOSTNAME}:{PORT} ---\n")

    if not INSTANCE_ID:
        print("Error: Variabel INSTANCE_ID kosong.")
        print("Harap isi variabel INSTANCE_ID dengan ID instance vast.ai Anda di awal script.")
        sys.exit(1)

    # Siapkan perintah base (template) yang akan dipakai diseluruh fungsi 
    # untuk SCP (pemindahan file) dan SSH (menjalankan remot perintah).
    # Tambahkan opsi StrictHostKeyChecking agar otomatis mengabaikan prompt/warning SSH Fingerprint vast.ai baru
    # Tambahkan ConnectTimeout=5 agar tidak menggantung (hang) saat SSH port disetel tapi belum siap
    # Tambahkan ServerAliveInterval=30 agar koneksi SSH tidak diputus oleh router/NAT selama proses transkrip (karena idle diam tidak print apa-apa)
    # Tambahkan ServerAliveCountMax=10 agar klien SSH toleran terhadap 10 kali keepalive gagal sebelum menyerah (default hanya 3)
    ssh_opts = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=5", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=10"]
    scp_command = ["scp", "-P", PORT, "-i", ID_RSA] + ssh_opts
    ssh_command = ["ssh", "-p", PORT, "-i", ID_RSA] + ssh_opts + [f"root@{HOSTNAME}"]

    server_ready_for_destroy = False

    try:
        # Pra-Validasi (Pre-check) file lokal agar tidak terlanjur menghancurkan server 
        # karena kesalahan kecil seperti salah direktori lokal.
        rclone_conf_path = os.path.expanduser("~/.config/rclone/rclone.conf")
        if not os.path.isfile(rclone_conf_path):
            print("Error: File ~/.config/rclone/rclone.conf tidak ditemukan (LOKAL)!")
            sys.exit(1)

        if not os.path.isfile(ID_RSA):
            print(f"Error: Private Key SSH LOKAL tidak ditemukan di {ID_RSA}")
            sys.exit(1)
            
        mgb2_asr_path = os.path.join(PROJECT_HOME, "egs", "mgb2", "ASR")
        precheck_files = [
            os.path.join(mgb2_asr_path, "tools", "whisperx", "whisperx_transcribe.py"),
            os.path.join(mgb2_asr_path, "download", "murattal", "train", "wav.ori", AUDIO_FILE_SUBPATH)
        ]
        missing_precheck = [f for f in precheck_files if not os.path.exists(f)]
        if missing_precheck:
            print("Error: File script/audio belum ada secara (LOKAL):")
            for m_file in missing_precheck: print(f"  - {m_file}")
            sys.exit(1)

        # # Tahap 0: Jalankan ssh-agen dan tambahkan key
        # Tahap ini bisa diskip dengan perintah: ssh-add --apple-use-keychain ~/.ssh/id_rsa
        # print("Menjalankan ssh-agen dan menambahkan key...")
        # try:
        #     # Jalankan ssh-agent dan tangkap outputnya
        #     agent_output = subprocess.check_output(["ssh-agent", "-s"], text=True)
            
        #     # Parsing output ssh-agent (misal: SSH_AUTH_SOCK=xxx; export SSH_AUTH_SOCK; ...)
        #     # dan masukkan ke environment variable Python saat ini
        #     for line in agent_output.splitlines():
        #         if "=" in line and ";" in line:
        #             variable_part = line.split(";")[0]
        #             key, value = variable_part.split("=", 1)
        #             os.environ[key] = value
        #     # Tambahkan key ke agent
        #     # Gunakan input=b'\n' jika Anda ingin mencoba melewati prompt (tapi sebaiknya key tanpa password untuk otomasi)
        #     subprocess.run(["ssh-add", ID_RSA], check=True)
        #     print("✓ ssh-agent berhasil dijalankan dan key berhasil ditambahkan\n")
        # except Exception as e:
        #     print(f"Error saat menyiapkan ssh-agent: {e}")
        #     sys.exit(1)


        # Setelah file lokal aman, kita tunggu server menyala 
        wait_for_ssh(ssh_command)  # Agar bisa wait_for_ssh, pastikan file ID_RSA sudah tidak password!

        # Mulai dari poin ini, server telah diakses secara penuh.
        # Bila gagal di tahap berikutnya, kita wajib auto-destroy agar tagihan tidak jebol.
        server_ready_for_destroy = True

        # Tahap 1: Transfer autentikasi Cloud Storage 
        print("[1/4] Transfer konfigurasi rclone")
        transfer_rclone_config(scp_command, ssh_command)
        print()

        # Tahap 2: Transfer skrip runner pembantu lainnya
        print("[2/4] Transfer script setup")
        transfer_setup_scripts(scp_command)
        print()

        # Tahap 3: jalankan script whisperx_transcribe.py
        print("[3/4] Menjalankan script whisperx_transcribe.py")
        audio_filename = os.path.basename(AUDIO_FILE_SUBPATH)
        
        # [SEKURITI] Gunakan shlex.quote agar argument selalu di-escape dengan aman untuk Bash (mencegah shell injection)
        remote_audio_path = shlex.quote(f"/workspace/{audio_filename}")
        remote_output_path = shlex.quote(f"/workspace/{OUTPUT_JSON_NAME}")
        
        remote_transcribe_cmd = f"python3 /workspace/whisperx_transcribe.py --audio {remote_audio_path} --output {remote_output_path}"
        run_command(ssh_command + [remote_transcribe_cmd])
        print("Script whisperx_transcribe.py berhasil dijalankan")

        # Tahap 4: copy hasil json ke workstation
        print("[4/4] Copy hasil json ke workstation")
        output_dir = os.path.join(PROJECT_HOME, "tools", "outputs")
        os.makedirs(output_dir, exist_ok=True)
        # SCP juga mengeksekusi bash di remote server, jadi file pelacakan harus di-escape memakai shlex.quote
        scp_remote_target = f"root@{HOSTNAME}:{shlex.quote('/workspace/' + OUTPUT_JSON_NAME)}"
        run_command(scp_command + [scp_remote_target, output_dir])
        print("Hasil json berhasil dikirim ke workstation")

    finally:
        if AUTO_DESTROY and server_ready_for_destroy:
            # Tahap 5: Matikan server dan hapus otomatis (pasti dieksekusi walau gagal di tahap 1-4)
            print("\n[5/5] Memastikan server dimatikan (AUTO_DESTROY=True)...")
            print(f"Mengirim perintah terminasi instance dengan ID: {INSTANCE_ID}")
            try:
                # Gunakan subprocess.run agar error destroy instance tidak di-raise sebagai global exception
                # Tambahkan timeout untuk mencegah infinite hang jika vast.ai network down
                subprocess.run(["vastai", "destroy", "instance", str(INSTANCE_ID)], check=True, timeout=60)
                print("Server vast.ai berhasil dimatikan dan dihapus (Tagihan berhenti)")
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
                print(f"PERINGATAN: Gagal mematikan instance secara otomatis! Harap matikan manual dari Vast.ai. Error: {e}")
        elif not AUTO_DESTROY:
            print("\n[Info] AUTO_DESTROY dinonaktifkan. Server tetap menyala untuk pengecekan manual.")
        else:
            print("\n[Info] Sesi berakhir sebelum setup diinisialisasi secara penuh.")
            print("Server tidak dihapus/di-destroy otomatis agar Anda dapat memperbaiki error lokal terlebih dahulu.")

    # # Selesai
    # print("--- ✓ Semua proses transfer selesai! ---")
    # print("\nCatatan: Untuk menjalankan setup preparation di remote server, jalankan perintah ini dari terminal lokal Anda:")
    # print(f"  ssh -p {PORT} -i {ID_RSA} root@{HOSTNAME} 'python3 /root/prepare_dataset.py'")

if __name__ == "__main__":
    main()

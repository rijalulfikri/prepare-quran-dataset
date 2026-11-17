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
import subprocess
import sys

# --- KONFIGURASI VARIABLE ---
# Ubah nilai di bawah ini sesuai dengan detail koneksi server (misal: koneksi SSH dari Vast.ai)
PORT = "23330"                                   # Port SSH server jarak jauh
HOSTNAME = "45.81.32.13"                         # Alamat IP server jarak jauh
ID_RSA = os.path.expanduser("~/.ssh/id_rsa")    # Path menuju file private key SSH lokal
PROJECT_HOME = "/Volumes/F/MachineLearning/icefall/egs/" # Path lokal direktori dasar proyek icefall
# ----------------------------

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
        print(f"Error saat menjalankan command: {e}")
        sys.exit(1)

def transfer_zipformer_and_local(scp_command, ssh_command):
    """
    Mengompres (tar.gz), mentransfer, dan mengekstrak folder utama proyek 
    ('zipformer' dan 'local') ke remote server.
    
    Args:
        scp_command (list): Base command SCP untuk transfer file (menyertakan Port dan file key SSH).
        ssh_command (list): Base command SSH untuk eksekusi remote di bash server.
    """
    mgb2_asr_path = os.path.join(PROJECT_HOME, "mgb2", "ASR")
    zipformer_dir = os.path.join(mgb2_asr_path, "zipformer")
    local_dir = os.path.join(mgb2_asr_path, "local")
    
    print("Mengirim folder zipformer dan local...")
    
    # 1. Validasi memastikan direktori yang akan dikompres tersedia di lokal
    if not os.path.exists(mgb2_asr_path):
        print(f"Error: Direktori {mgb2_asr_path} tidak ditemukan.")
        sys.exit(1)
    
    if not os.path.exists(zipformer_dir):
        print(f"Error: Direktori {zipformer_dir} tidak ditemukan.")
        sys.exit(1)
    
    if not os.path.exists(local_dir):
        print(f"Error: Direktori {local_dir} tidak ditemukan.")
        sys.exit(1)

    # 2. Buat archive (tar.gz) secara lokal untuk mempercepat transfer file via SFTP/SCP
    print("Membuat archive zipformer.tar.gz...")
    run_command(["tar", "-czf", "zipformer.tar.gz", "zipformer"], cwd=mgb2_asr_path)
    print("Membuat archive local.tar.gz...")
    run_command(["tar", "-czf", "local.tar.gz", "local"], cwd=mgb2_asr_path)

    local_tar_path = os.path.join(mgb2_asr_path, "local.tar.gz")
    zipformer_tar_path = os.path.join(mgb2_asr_path, "zipformer.tar.gz")
    remote_dest_path = "/workspace/icefall/egs/mgb2/ASR" # Direktori kerja (workspace) target di server
    
    # 3. Transfer file archive ke remote server menggunakan SCP
    print(f"Mengirim archive ke {HOSTNAME}...")
    run_command(scp_command + [local_tar_path, zipformer_tar_path, f"root@{HOSTNAME}:{remote_dest_path}"])

    # 4. Hapus file archive di lokal (cleanup) untuk menghemat ruang
    print("Menghapus archive lokal (cleanup)...")
    if os.path.exists(local_tar_path):
        os.remove(local_tar_path)
    if os.path.exists(zipformer_tar_path):
        os.remove(zipformer_tar_path)

    # 5. Ekstrak archive yang telah sampai di remote server via command SSH
    print("Mengekstrak zipformer di remote server...")
    run_command(ssh_command + [f"cd {remote_dest_path} && tar xzf zipformer.tar.gz --warning=no-unknown-keyword"])
    print("Mengekstrak local di remote server...")
    run_command(ssh_command + [f"cd {remote_dest_path} && tar xzf local.tar.gz --warning=no-unknown-keyword"])

    # 6. Bersihkan file archive di remote server setelah diekstrak
    print("Menghapus archive tar.gz di remote server...")
    run_command(ssh_command + [f"rm {remote_dest_path}/local.tar.gz {remote_dest_path}/zipformer.tar.gz"])

def transfer_setup_scripts(scp_command):
    """
    Mentransfer kumpulan script otomatisasi Python lainnya (untuk setup lanjutan, 
    pembuatan dataset, training, dll) ke direktori root di remote server.
    
    Args:
        scp_command (list): Base command SCP untuk mentransfer file.
    """
    print("Mengirim script setup ke remote server...")
    
    # Ambil folder tempat file `send_files.py` ini berada
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Definisikan path absolut masing-masing file skrip yang akan di transport
    prepare_dataset_path = os.path.join(script_dir, "prepare_dataset.py")
    train_model_path = os.path.join(script_dir, "train_model.py")
    vast_api_path = os.path.join(script_dir, "vast_api.py")
    common_utils_path = os.path.join(script_dir, "common_utils.py")
    
    # Validasi apakah file-file script tersebut benar-benar ada secara lokal
    missing_files = []
    for file_path in [prepare_dataset_path, train_model_path, vast_api_path, common_utils_path]:
        if not os.path.exists(file_path):
            missing_files.append(os.path.basename(file_path))
    
    # Hentikan eksekusi jika ada script yang tidak ditemukan
    if missing_files:
        print(f"Error: File script tidak ditemukan secara lokal: {', '.join(missing_files)}")
        sys.exit(1)
    
    # Transfer keempat script sekaligus ke folder /root server via SCP
    remote_dest = "/root"
    run_command(scp_command + [prepare_dataset_path, train_model_path, vast_api_path, common_utils_path, f"root@{HOSTNAME}:{remote_dest}"])
    print(f"Script setup berhasil dikirim ke {HOSTNAME}:{remote_dest}")

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

    # Siapkan perintah base (template) yang akan dipakai diseluruh fungsi 
    # untuk SCP (pemindahan file) dan SSH (menjalankan remot perintah).
    scp_command = ["scp", "-P", PORT, "-i", ID_RSA]
    ssh_command = ["ssh", "-p", PORT, "-i", ID_RSA, f"root@{HOSTNAME}"]

    # Instal dependensi/library dasar Python di sistem server menggunakan pip via SSH
    print("Menginstal dependensi python di remote server...")
    run_command(ssh_command + [f"pip install lxml huggingface_hub nvidia-ml-py3 vastai"])

    # Tahap 1: Transfer inti proyek (source code)
    print("[1/3] Transfer folder zipformer dan local")
    transfer_zipformer_and_local(scp_command, ssh_command)
    print()

    # Tahap 2: Transfer autentikasi Cloud Storage 
    print("[2/3] Transfer konfigurasi rclone")
    transfer_rclone_config(scp_command, ssh_command)
    print()

    # Tahap 3: Transfer skrip runner pembantu lainnya
    print("[3/3] Transfer script setup")
    transfer_setup_scripts(scp_command)
    print()

    # Selesai
    print("--- ✓ Semua proses transfer selesai! ---")
    print("\nCatatan: Untuk menjalankan setup preparation di remote server, jalankan perintah ini dari terminal lokal Anda:")
    print(f"  ssh -p {PORT} -i {ID_RSA} root@{HOSTNAME} 'python3 /root/prepare_dataset.py'")

if __name__ == "__main__":
    main()

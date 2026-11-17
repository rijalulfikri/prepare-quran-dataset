#!/usr/bin/env python3

import subprocess
import os
import sys
import threading

from vast_api import get_first_instance_id, destroy_instance
from common_utils import (
    run_command,
    install_system_packages,
    install_rclone,
    create_download_directory,
    update_icefall_repo,
    install_icefall_requirements,
    download_from_google_drive
)

# Constants
SYNC_INTERVAL = 1200  # Interval sinkronisasi background dalam detik


def download_prepared_data_from_gdrive(download_dir, prepared_data="mgb2_data.tar.gz", remote_path="gdrive:ASR"):
    """Step 5: Download file dari Google Drive menggunakan Rclone.
    
    Args:
        download_dir (str): Direktori tujuan download.
        prepared_data (str): Nama file archive yang akan di-download.
        remote_path (str): Path remote di Google Drive (harus sudah dikonfigurasi).
    
    Note:
        Pastikan remote 'gdrive:' sudah dikonfigurasi sebelumnya.
    """
    remote_file = f"{remote_path}/{prepared_data}"
    download_from_google_drive(remote_file, download_dir, show_progress=True)


def extract_prepared_data(download_dir, prepared_data="mgb2_data.tar.gz", extract_to="/workspace/icefall/egs/mgb2/ASR"):
    """Step 6: Ekstraksi file archive menggunakan pigz untuk kecepatan maksimal.
    
    Args:
        download_dir (str): Direktori tempat file archive berada.
        prepared_data (str): Nama file archive yang akan diekstrak.
        extract_to (str): Direktori tujuan ekstraksi.
    """
    print(f"Mengekstrak {prepared_data} ke {extract_to}...")
    prepared_data_path = os.path.join(download_dir, prepared_data)
    run_command(["tar", "-I", "pigz", "-xf", prepared_data_path, "-C", extract_to])

    #Rename mv cuts_musan.jsonl.gz musan_cuts.jsonl.gz  
    musan_cuts_path = os.path.join(extract_to, "data/fbank/cuts_musan.jsonl.gz")
    musan_cuts_path_new = os.path.join(extract_to, "data/fbank/musan_cuts.jsonl.gz")
    run_command(["mv", musan_cuts_path, musan_cuts_path_new])
    
    #Hapus file archive setelah ekstraksi
    run_command(["rm", "-rf", prepared_data_path])





def background_sync(local_dir, remote_dir, interval=300, stop_event=None):
    """
    Fungsi background yang akan dijalankan di thread terpisah untuk sinkronisasi folder
    secara periodik menggunakan rclone.
    """
    print(f"[Sync] Monitoring background sync: {local_dir} -> {remote_dir}")
    print(f"[Sync] Interval: {interval} detik")
    
    while not stop_event.is_set():
        # Wait first, then sync (or sync first then wait? usually wait first to let files be created)
        # We wait with a timeout equal to interval so we can react to stop_event immediately
        if stop_event.wait(interval):
            break
            
        print(f"[Sync] Memulai sinkronisasi otomatis...")
        try:
            # Gunakan rclone copy agar file di remote tidak terhapus jika di local belum ada
            subprocess.run(
                ["rclone", "copy", local_dir, remote_dir], 
                check=False,
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )
            print(f"[Sync] Sinkronisasi selesai.")
        except Exception as e:
            print(f"[Sync] Error saat sinkronisasi: {e}")

    # Final sync saat training selesai
    print(f"[Sync] Melakukan sinkronisasi terakhir...")
    subprocess.run(["rclone", "copy", local_dir, remote_dir], check=False)
    print(f"[Sync] Sinkronisasi terakhir selesai.")


def run_asr_training(asr_dir):
    """Step 9: Menjalankan training dengan background sync ke GDrive.
    """
    print("Menjalankan training...")
    
    # Path output training (biasanya di zipformer/exp atau exp tergantung resep)
    # Sesuaikan dengan script training yang digunakan
    exp_dir = os.path.join(asr_dir, "zipformer", "exp")
    
    # Remote path di google drive
    remote_exp_path = "gdrive:ASR/zipformer/exp" 
    
    # Setup background sync
    stop_event = threading.Event()
    
    # Jalankan sync setiap 5 menit (300 detik)
    sync_thread = threading.Thread(
        target=background_sync, 
        args=(exp_dir, remote_exp_path, SYNC_INTERVAL, stop_event)
    )
    sync_thread.start()
    
    try:
        # TODO: Tambahkan command training di sini
        # Set GPU device
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        
        # Fix memory fragmentation issue
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        # run_command([
        #     "./zipformer/train.py", 
        #     "--world-size", "1", 
        #     "--num-epochs", "5",  #Coba 5 epoch dulu!
        #     "--start-epoch", "1", 
        #     "--use-fp16", "1",
        #     "--exp-dir", "zipformer/exp",
        #     "--max-duration", "700",
        #     "--tensorboard", "1",
        #     "--bpe-model", "data/lang_bpe_5000/bpe.model",
        #     "--num-buckets", "50",
        #     "--keep-last-k", "25",
        # ], cwd=asr_dir)

        # Jalankan training dengan causal Zipformer
        # max-duration dikurangi untuk menghindari OOM pada RTX 4090 48GB
        run_command([
            "./zipformer/train.py", 
            "--world-size", "1", 
            "--num-epochs", "30", 
            "--use-fp16", "1", 
            "--exp-dir", "zipformer/exp-causal", 
            "--causal", "1",
            "--chunk-size", "16,32,64,-1", 
            "--left-context-frames", "64,128,256,-1", 
            "--max-duration", "180",  # Dikurangi dari 360 untuk menghindari OOM
            "--tensorboard", "1", 
            "--bpe-model", "data/lang_bpe_5000/bpe.model",
        ], cwd=asr_dir)


        
    except Exception as e:
        print(f"Error saat training: {e}")
        raise e
    finally:
        # Hentikan background sync baik jika training sukses atau error
        print("Training selesai (atau error), menghentikan background sync...")
        stop_event.set()
        sync_thread.join()

def mainX():
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"  
    # run_command(["echo", "$CUDA_VISIBLE_DEVICES"], env=os.environ)
    # Menampilkan satu environment variable
    run_command(["echo $CUDA_VISIBLE_DEVICES"], capture_output=True, shell=True)
    # result = subprocess.run(
    #     ["printenv", "CUDA_VISIBLE_DEVICES"],
    #     capture_output=True,
    #     text=True,
    #     check=False  # False agar tidak error jika variable tidak ada
    # )
    # print(f"CUDA_VISIBLE_DEVICES: {result.stdout.strip()}")

def main():
    """Main function yang menjalankan semua langkah setup."""
    print("--- Memulai proses setup ---")
    
    # Define constants
    download_dir = "/workspace/icefall/egs/mgb2/ASR/download"
    icefall_dir = "/workspace/icefall"
    asr_dir = "/workspace/icefall/egs/mgb2/ASR"
    
    # Execute all steps
    install_system_packages()
    install_rclone()
    create_download_directory(download_dir)
    download_prepared_data_from_gdrive(download_dir)
    extract_prepared_data(download_dir)
    update_icefall_repo(icefall_dir)
    install_icefall_requirements(icefall_dir)
    
    # Jalankan training dengan sync
    run_asr_training(asr_dir)
    

    # Hapus script ini sendiri
    run_command(["rm", "--", sys.argv[0]])

    # Destroy instance
    try:
        instance_id = get_first_instance_id()
        destroy_instance(instance_id)
    except ValueError as e:
        print(f"Warning: Could not destroy instance - {e}")
    
    print("--- Proses Selesai! ---")

if __name__ == "__main__":
    main()

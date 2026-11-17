#!/usr/bin/env python3

import os
import sys
from vast_api import get_env_variable, get_first_instance_id, destroy_instance
from common_utils import (
    run_command,
    install_system_packages,
    install_rclone,
    create_download_directory,
    update_icefall_repo,
    install_icefall_requirements,
    remove_directory,
    upload_to_google_drive
)




def download_dataset_from_huggingface(hf_token, download_path):
    """Step 5: Mendownload dataset dari HuggingFace."""
    # We use subprocess to call python code to ensure the packages installed above are used 
    # and to match the original script's method (which avoids import errors if packages weren't present at start)
    print("Logging in to HuggingFace...")
    run_command([
        sys.executable, "-c",
        f"from huggingface_hub import login; login(token='{hf_token}')"
    ])
    
    print(f"Downloading dataset to {download_path}...")
    run_command([
        sys.executable, "-c",
        f"from huggingface_hub import snapshot_download; snapshot_download(repo_id='QCRI/mgb2', repo_type='dataset', local_dir='{download_path}')"
    ])


def extract_tar_archive(split_dir, tarfile_name, strip_components=True):
    """Helper function untuk extract dataset split (dev, test, train)."""
    if os.path.exists(split_dir):
        print(f"Extracting {tarfile_name} in {split_dir}...")
        cmd = ["tar", "-I", "pigz", "-xvf", tarfile_name]
        if strip_components:
            cmd.append("--strip-components=1")
        run_command(cmd, cwd=split_dir)
        
        # Hapus file tar setelah extract
        tar_path = os.path.join(split_dir, tarfile_name)
        if os.path.exists(tar_path):
            os.remove(tar_path)
            print(f"Removed {tar_path}")


def decompress_downloaded_dataset(mgb2_base_dir):
    """Step 6: Decompress hasil download."""
    print("Decompressing downloaded datasets...")
    
    # Dev
    dev_dir = os.path.join(mgb2_base_dir, "dev")
    extract_tar_archive(dev_dir, "dev.tar.gz", strip_components=True)
    
    # Test
    test_dir = os.path.join(mgb2_base_dir, "test")
    extract_tar_archive(test_dir, "test.tar.gz", strip_components=True)
    
    # Train
    # Note: The original script ran `tar -I pigz xfv train.tar.gz` in /workspace/icefall/egs/mgb2/ASR/download/mgb2
    # It did NOT include --strip-components=1 for train.
    if os.path.exists(mgb2_base_dir):
        print(f"Extracting train in {mgb2_base_dir}...")
        run_command(["tar", "-I", "pigz", "-xvf", "train.tar.gz"], cwd=mgb2_base_dir)


def run_prepare_script(asr_dir):
    """Step 9: Lakukan prepare."""
    print(f"Running prepare in {asr_dir}...")
    run_command(["./prepare", "--stage", "0"], cwd=asr_dir)


def zip_prepared_data(asr_dir, archive_name="mgb2_data.tar.gz"):
    """Step 11: Zip data yang sudah dihasilkan."""
    print("Zipping data directory...")
    run_command(["tar", "-I", "pigz", "-cf", archive_name, "data/"], cwd=asr_dir)
    print(f"Created {archive_name}")


def upload_prepared_data(asr_dir, archive_name="mgb2_data.tar.gz", remote_path="gdrive:ASR"):
    """Step 12: Kirim ke Google Drive via Rclone."""
    archive_path = os.path.join(asr_dir, archive_name)
    upload_to_google_drive(archive_path, remote_path, show_progress=False)




def main():
    """Main function yang menjalankan semua langkah setup."""
    print("--- Memulai proses setup ---")
    
    # Define constants
    download_dir = "/workspace/icefall/egs/mgb2/ASR/download"
    mgb2_download_dir = "/workspace/icefall/egs/mgb2/ASR/download/mgb2"
    icefall_dir = "/workspace/icefall"
    asr_dir = "/workspace/icefall/egs/mgb2/ASR"
    hf_token = get_env_variable('HF_TOKEN', required=False, default='')
    
    # Execute all steps
    install_system_packages()
    install_rclone()
    create_download_directory(download_dir)
    download_dataset_from_huggingface(hf_token, mgb2_download_dir)
    decompress_downloaded_dataset(mgb2_download_dir)
    update_icefall_repo(icefall_dir)
    install_icefall_requirements(icefall_dir)
    run_prepare_script(asr_dir)
    remove_directory(download_dir)
    zip_prepared_data(asr_dir)
    upload_prepared_data(asr_dir)

    # Hapus script ini sendiri
    run_command(["rm", "--", sys.argv[0]])
    # print("Script telah dihapus.")

    # Destroy instance
    try:
        instance_id = get_first_instance_id()
        destroy_instance(instance_id)
    except ValueError as e:
        print(f"Warning: Could not destroy instance - {e}")

    print("--- Proses Selesai! ---")

if __name__ == "__main__":
    main()

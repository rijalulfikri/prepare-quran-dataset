#!/usr/bin/env python3
"""
Common utility functions untuk automation scripts.
Berisi fungsi-fungsi yang digunakan bersama oleh prepare_dataset.py dan train_model.py.
"""

import os
import subprocess
import sys

def get_env_variable(var_name, required=True, default=None):
    """Get environment variable value with optional validation
    
    This is a generic helper function to retrieve environment variables
    with proper validation and error handling.
    
    Args:
        var_name (str): Name of the environment variable to retrieve
        required (bool): If True, raise ValueError if variable is not set or empty.
                        If False, return default value when variable is not set.
        default: Default value to return if variable is not set and required=False
        
    Returns:
        str: Value of the environment variable (stripped of whitespace)
        
    Raises:
        ValueError: If required=True and the environment variable is not set or empty
        
    Examples:
        >>> api_key = get_env_variable('VAST_API_KEY')  # Required
        >>> hf_token = get_env_variable('HF_TOKEN', required=False, default='')  # Optional
    """
    value = os.getenv(var_name)
    
    # Check if value exists and is not empty
    if value and value.strip():
        return value.strip()
    
    # Handle missing or empty value
    if required:
        raise ValueError(
            f"Environment variable '{var_name}' is required but not set or empty. "
            f"Please set it using: export {var_name}='your-value'"
        )
    
    return default
    
def run_command(command, shell=False, cwd=None, capture_output=False, env=None):
    """
    Menjalankan perintah shell dan akan berhenti jika terjadi error.
    
    Args:
        command: Command yang akan dijalankan (list atau string jika shell=True).
        shell: Apakah command dijalankan dalam shell.
        cwd: Working directory untuk menjalankan command.
        capture_output: Jika True, akan menangkap dan mengembalikan stdout/stderr.
    
    Returns:
        subprocess.CompletedProcess jika capture_output=True, None jika False.
    
    Raises:
        SystemExit: Jika command gagal dijalankan.
    """
    print(f"Running command: {' '.join(command) if isinstance(command, list) else command}")
    try:
        result = subprocess.run(
            command,
            shell=shell,
            check=True,
            cwd=cwd,
            text=True,
            capture_output=capture_output,
            env=env
        )
        if capture_output and result.stdout:
            print(result.stdout)
        return result if capture_output else None
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while executing: {command}")
        print(f"Return code: {e.returncode}")
        if hasattr(e, 'stdout') and e.stdout:
            print(f"STDOUT: {e.stdout}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"STDERR: {e.stderr}")
        sys.exit(1)


def install_system_packages():
    """Update package list dan install tools dasar (pigz & nano)."""
    print("Mengupdate sistem dan menginstall pigz & nano...")
    run_command(["apt-get", "update"])
    run_command(["apt-get", "install", "-y", "pigz", "nano"])


def install_rclone():
    """Install Rclone jika belum terinstall."""
    print("Menginstall Rclone...")
    
    # Check if rclone is already installed
    try:
        result = subprocess.run(["which", "rclone"], capture_output=True, text=True)
        if result.returncode == 0:
            print("Rclone sudah terinstall, skip instalasi.")
            # Get version
            version_result = subprocess.run(["rclone", "version"], capture_output=True, text=True)
            if version_result.stdout:
                print(version_result.stdout.split('\n')[0])  # Print first line (version info)
            return
    except (FileNotFoundError, subprocess.SubprocessError):
        pass  # If check fails, proceed with installation
    
    # Download the install script
    run_command(["curl", "-o", "/tmp/install_rclone.sh", "https://rclone.org/install.sh"])
    # Execute it with sudo (rclone install requires root access)
    run_command(["sudo", "bash", "/tmp/install_rclone.sh"])


def create_download_directory(download_dir):
    """
    Membuat direktori download jika belum ada.
    
    Args:
        download_dir (str): Path direktori download yang akan dibuat.
    """
    print(f"Membuat direktori download: {download_dir}...")
    os.makedirs(download_dir, exist_ok=True)


def update_icefall_repo(icefall_dir):
    """
    Git pull untuk update repository icefall.
    
    Args:
        icefall_dir (str): Path direktori icefall.
    """
    print(f"Git pull in {icefall_dir}...")
    run_command(["git", "pull"], cwd=icefall_dir)


def install_icefall_requirements(icefall_dir):
    """
    Install dependencies dari requirements.txt icefall.
    
    Args:
        icefall_dir (str): Path direktori icefall.
    """
    print("Installing icefall requirements...")
    run_command(["pip", "install", "-r", "requirements.txt"], cwd=icefall_dir)


def remove_directory(directory):
    """
    Hapus direktori secara rekursif.
    
    Args:
        directory (str): Path direktori yang akan dihapus.
    """
    print(f"Removing directory: {directory}...")
    if os.path.exists(directory):
        run_command(["rm", "-rf", directory])
        print(f"Removed {directory}")
    else:
        print(f"Directory {directory} tidak ada, skip penghapusan.")


def upload_to_google_drive(source_path, remote_path, show_progress=False):
    """
    Upload file/folder ke Google Drive menggunakan Rclone.
    
    Args:
        source_path (str): Path file/folder lokal yang akan diupload.
        remote_path (str): Path remote di Google Drive (format: 'gdrive:path').
        show_progress (bool): Tampilkan progress upload.
    """
    print(f"Uploading {source_path} to {remote_path}...")
    cmd = ["rclone", "copy", source_path, remote_path]
    if show_progress:
        cmd.append("-P")
    
    run_command(cmd)
    print("Upload complete!")


def download_from_google_drive(remote_path, local_dir, show_progress=False):
    """
    Download file/folder dari Google Drive menggunakan Rclone.
    
    Args:
        remote_path (str): Path remote di Google Drive (format: 'gdrive:path/file').
        local_dir (str): Direktori lokal tujuan download.
        show_progress (bool): Tampilkan progress download.
    """
    print(f"Downloading {remote_path} to {local_dir}...")
    cmd = ["rclone", "copy", remote_path, local_dir]
    if show_progress:
        cmd.append("-P")
    run_command(cmd)
    print("Download complete!")

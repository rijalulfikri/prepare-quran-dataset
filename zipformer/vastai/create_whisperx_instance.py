#!/usr/bin/env python3
"""
Create WhisperX Vast.ai Instance

Membuat instance Vast.ai dengan konfigurasi WhisperX menggunakan image Docker langsung
(tanpa template), setara dengan perintah:

  vastai create instance <OFFER_ID> \\
    --image pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime \\
    --env '-e DATA_DIRECTORY=/workspace/ -e JUPYTER_DIR=/' \\
    --onstart-cmd 'env >> /etc/environment;mkdir -p ${DATA_DIRECTORY:-/workspace/};
                   apt-get update && apt-get install -y ffmpeg vim nano;
                   pip install whisperx librosa torchcodec==0.7' \\
    --disk 20 \\
    --ssh \\
    --direct

Persyaratan environment variable:
    VAST_API_KEY  - API key Vast.ai (wajib)
    VAST_OFFER_ID - ID offer yang akan disewa (wajib, atau gunakan argumen --offer-id)
"""

import argparse
import sys
import traceback

from vast_api import (
    create_instance_with_ssh_info,
    destroy_instance,
    get_env_variable,
    get_offer_id_by_machine_id,
    wait_for_startup,
)


# ---------------------------------------------------------------------------
# Konfigurasi Default
# ---------------------------------------------------------------------------

DEFAULT_IMAGE = "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime"
DEFAULT_ENV = "-e DATA_DIRECTORY=/workspace/ -e JUPYTER_DIR=/"
DEFAULT_ONSTART_CMD = (
    "env >> /etc/environment;"
    "mkdir -p ${DATA_DIRECTORY:-/workspace/};"
    "apt-get update && apt-get install -y ffmpeg vim nano;"
    "pip install whisperx librosa torchcodec==0.7"
)
# DEFAULT_DISK = 20.0
DEFAULT_DISK = 10.0


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Buat Vast.ai instance WhisperX dengan image Docker langsung.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--offer-id",
        type=int,
        default=None,
        help=(
            "ID offer Vast.ai yang akan disewa. "
            "Jika tidak diberikan, dibaca dari env VAST_OFFER_ID."
        ),
    )
    parser.add_argument(
        "--machine-id",
        type=int,
        default=None,
        help=(
            "ID machine Vast.ai. Jika diberikan, digunakan untuk mencari offer ID "
            "secara otomatis menggunakan get_offer_id_by_machine_id(). "
            "Abaikan jika --offer-id atau VAST_OFFER_ID sudah diset."
        ),
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help=(
            "Jumlah GPU yang diinginkan. Digunakan bersama --machine-id untuk "
            "memfilter offer berdasarkan jumlah GPU."
        ),
    )
    parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE,
        help=f"Docker image (default: {DEFAULT_IMAGE})",
    )
    parser.add_argument(
        "--env",
        default=DEFAULT_ENV,
        help="Env variables / port mapping (format docker CLI, surround with quotes)",
    )
    parser.add_argument(
        "--onstart-cmd",
        default=DEFAULT_ONSTART_CMD,
        help="Perintah startup sebagai string",
    )
    parser.add_argument(
        "--disk",
        type=float,
        default=DEFAULT_DISK,
        help=f"Ukuran disk lokal dalam GB (default: {DEFAULT_DISK})",
    )
    parser.add_argument(
        "--no-ssh",
        action="store_true",
        help="Jangan aktifkan SSH (default: SSH aktif)",
    )
    parser.add_argument(
        "--no-direct",
        action="store_true",
        help="Jangan gunakan direct connection (default: direct aktif)",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Label opsional untuk instance",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Tunggu hingga instance berstatus 'running' setelah dibuat",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_offer_id(cli: argparse.Namespace) -> int:
    """Resolve the offer ID from CLI arg, environment variable, or machine ID lookup.

    Priority order: ``--offer-id`` > ``VAST_OFFER_ID`` env var > ``--machine-id`` lookup.

    Args:
        cli: Parsed CLI arguments.

    Returns:
        int: The resolved offer ID.

    Raises:
        SystemExit: If no valid offer ID can be determined.
    """
    # 1. Explicit CLI argument
    if cli.offer_id is not None:
        return cli.offer_id

    # 2. Environment variable
    offer_id_str = get_env_variable("VAST_OFFER_ID", required=False)
    if offer_id_str:
        try:
            return int(offer_id_str)
        except ValueError:
            print(f"[ERROR] VAST_OFFER_ID='{offer_id_str}' bukan angka yang valid.")
            sys.exit(1)

    # 3. Machine ID lookup
    if cli.machine_id is not None:
        print(f"[*] Mencari offer ID untuk machine ID: {cli.machine_id} ...")
        offer_id = get_offer_id_by_machine_id(
            machine_id=cli.machine_id,
            num_gpus=cli.num_gpus,
        )
        if offer_id is None:
            print(f"[ERROR] Offer ID tidak ditemukan untuk machine ID {cli.machine_id}.")
            sys.exit(1)
        return offer_id

    print(
        "[ERROR] Offer ID tidak ditemukan. "
        "Gunakan --offer-id <ID>, --machine-id <ID>, atau set VAST_OFFER_ID='<ID>'."
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    cli = parse_args()
    offer_id = _resolve_offer_id(cli)

    ssh_info = create_instance_with_ssh_info(
        offer_id=offer_id,
        image=cli.image,
        env=cli.env,
        onstart_cmd=cli.onstart_cmd,
        disk=cli.disk,
        ssh=not cli.no_ssh,
        direct=not cli.no_direct,
        label=cli.label,
    )

    if ssh_info is None:
        print("[!] Pembuatan instance gagal atau timeout tercapai.")
        sys.exit(1)

    print(
        f"\n[OK] Instance siap digunakan."
        f"\n    Instance ID : {ssh_info['instance_id']}"
        f"\n    IP          : {ssh_info['ip']}"
        f"\n    SSH Port    : {ssh_info['ssh_port']}"
        f"\n    SSH Command : ssh -p {ssh_info['ssh_port']} root@{ssh_info['ip']}"
    )

    if cli.wait:
        wait_for_startup(ssh_info["instance_id"])

    destroy_instance(ssh_info["instance_id"])


if __name__ == "__main__":
    try:
        main()
    except ValueError as e:
        print(f"\n[ERROR] {e}")
        print("\nPastikan environment variable sudah diset:")
        print("  export VAST_API_KEY='your-api-key'")
        print("  export VAST_OFFER_ID='<offer-id>'   # atau gunakan --offer-id")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n[!] Script dihentikan oleh user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)

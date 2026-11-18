#!/usr/bin/env python3
"""Mirror the data_dir tar shards from Hugging Face to an rclone folder (Google Drive).

Copies the exact bytes of every shard listed in the HF repo's `index.json`, so
both sources share one index and `train_zipformer.py --stage prepare` can fall
back from HF to Drive (`drive_data_shards_path` in config) and still get the
same data. Each shard is downloaded, checked against its sha256, uploaded
(rclone verifies the MD5 after upload), and deleted before the next, so peak
extra disk is `--workers` shards. Shards already on the destination with the
right size are skipped; `index.json` is uploaded last, so the destination only
looks complete once every shard is there.

Run after `vastai/pack_data_to_hf.py`, on a box with the HF token and rclone:
    python vastai/mirror_shards_to_drive.py --repo rijalulfikri/utluma-zipformer-data \
        --dest utluma-edit:RClone/Utluma/zipformer-shards
"""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from quran_zipformer.data_shards import INDEX_NAME, HfSource, _sha256  # noqa: E402

SHARD_DIR = "shards"  # same layout as vastai/pack_data_to_hf.py


def remote_sizes(dest: str) -> dict[str, int]:
    proc = subprocess.run(["rclone", "lsjson", "-R", "--files-only", dest], capture_output=True, text=True)
    if proc.returncode != 0:  # destination folder does not exist yet
        return {}
    return {e["Path"]: e["Size"] for e in json.loads(proc.stdout)}


def upload(src: Path, dst: str) -> None:
    subprocess.run(["rclone", "copyto", str(src), dst, "--drive-chunk-size", "256M"], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True, help="HF dataset repo with the shards")
    parser.add_argument("--dest", required=True, help="rclone remote:path to mirror into (needs write access)")
    parser.add_argument("--work-dir", default="/root/drive-mirror")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    dest = args.dest.rstrip("/")
    work = Path(args.work_dir)
    hf = HfSource(args.repo)
    index_path = hf.download(INDEX_NAME, work)
    index = json.loads(index_path.read_text())
    # Create the folders once up front: Drive allows duplicate folder names, so
    # parallel uploads into a missing folder would each create their own copy
    subprocess.run(["rclone", "mkdir", f"{dest}/{SHARD_DIR}"], check=True)
    existing = remote_sizes(dest)
    todo = [s for s in index["shards"] if existing.get(s["path"]) != s["size_bytes"]]
    print(f"{len(index['shards'])} shards, {len(todo)} to mirror to {dest}", flush=True)

    def mirror(shard: dict) -> None:
        t = time.time()
        tar = hf.download(shard["path"], work)
        try:
            if _sha256(tar) != shard["sha256"]:
                raise RuntimeError(f"{shard['name']} from HF failed sha256 verification")
            upload(tar, f"{dest}/{shard['path']}")
        finally:
            tar.unlink(missing_ok=True)
        print(f"  {shard['name']} mirrored ({time.time() - t:.0f}s)", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in [pool.submit(mirror, s) for s in todo]:
            future.result()

    final = remote_sizes(dest)
    missing = [s["path"] for s in index["shards"] if final.get(s["path"]) != s["size_bytes"]]
    if missing:
        sys.exit(f"Error: shard di {dest} belum lengkap: {missing}; index.json tidak di-upload")
    upload(index_path, f"{dest}/{INDEX_NAME}")
    print(f"done: {len(index['shards'])} shards + {INDEX_NAME} at {dest}", flush=True)


if __name__ == "__main__":
    main()

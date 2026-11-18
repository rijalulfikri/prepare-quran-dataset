#!/usr/bin/env python3
"""Pack a prepared `data_dir` into tar shards on a private Hugging Face dataset repo.

Run once on a box that already has the full data (e.g. after `rclone sync`).
Files are grouped deterministically (sorted paths, ~--shard-gib each) into
uncompressed tars (FLAC is already compressed), and each shard is built,
uploaded, and deleted before the next, so peak extra disk is one shard.
Re-running skips shards already in the repo. `index.json` (shard names,
sha256, sizes, file counts) is uploaded last; `train_zipformer.py --stage
prepare` uses it to download, verify, and extract (`hf_data_repo` in config).

Usage (on the box, inside zipformer/ with the venv and HUGGINGFACE_TOKEN set):
    python vastai/pack_data_to_hf.py --data-dir data-zipformer --repo rijalulfikri/utluma-zipformer-data
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from huggingface_hub import HfApi

INDEX_NAME = "index.json"
SHARD_DIR = "shards"


def plan_shards(data_dir: Path, shard_bytes: int) -> list[list[tuple[str, int]]]:
    """Sorted (relative path, size) lists, each group at most ~shard_bytes."""
    files = sorted(
        (str(p.relative_to(data_dir)), p.stat().st_size) for p in data_dir.rglob("*") if p.is_file()
    )
    shards, current, size = [], [], 0
    for rel, n in files:
        if current and size + n > shard_bytes:
            shards.append(current)
            current, size = [], 0
        current.append((rel, n))
        size += n
    if current:
        shards.append(current)
    return shards


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def build_tar(data_dir: Path, members: list[tuple[str, int]], out: Path) -> None:
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".lst") as lst:
        lst.write("\n".join(rel for rel, _ in members) + "\n")
    try:
        subprocess.run(
            ["tar", "-cf", str(out), "-C", str(data_dir), "--no-recursion", "-T", lst.name], check=True
        )
    finally:
        os.unlink(lst.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="data-zipformer")
    parser.add_argument("--repo", required=True, help="dataset repo id, e.g. user/utluma-zipformer-data")
    parser.add_argument("--shard-gib", type=float, default=4.0)
    parser.add_argument("--work-dir", default="/root/hf-pack", help="where each shard is built before upload")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    api = HfApi(token=os.environ.get("HUGGINGFACE_TOKEN") or None)
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
    existing = set(api.list_repo_files(args.repo, repo_type="dataset"))

    shards = plan_shards(data_dir, int(args.shard_gib * 2**30))
    total_files = sum(len(s) for s in shards)
    print(f"{total_files} files -> {len(shards)} shards", flush=True)

    # Shard metadata is kept across re-runs so already-uploaded shards need no rebuild
    state_path = work / "shards.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    entries = []
    for i, members in enumerate(shards):
        name = f"shard-{i:05d}.tar"
        repo_path = f"{SHARD_DIR}/{name}"
        first, last = members[0][0], members[-1][0]
        known = state.get(name)
        if repo_path in existing and known and known["first"] == first and known["last"] == last:
            print(f"[{i + 1}/{len(shards)}] {name} already uploaded, skipping", flush=True)
            entries.append(known)
            continue
        t = time.time()
        tar_path = work / name
        build_tar(data_dir, members, tar_path)
        entry = {
            "name": name, "path": repo_path, "sha256": sha256_file(tar_path),
            "size_bytes": tar_path.stat().st_size, "num_files": len(members), "first": first, "last": last,
        }
        api.upload_file(path_or_fileobj=str(tar_path), path_in_repo=repo_path,
                        repo_id=args.repo, repo_type="dataset", commit_message=f"add {name}")
        tar_path.unlink()
        state[name] = entry
        state_path.write_text(json.dumps(state, indent=1))
        entries.append(entry)
        print(f"[{i + 1}/{len(shards)}] {name}: {entry['size_bytes'] / 2**30:.2f} GiB, "
              f"{len(members)} files, {time.time() - t:.0f}s", flush=True)

    index = {"num_files": total_files, "total_bytes": sum(e["size_bytes"] for e in entries), "shards": entries}
    index_path = work / INDEX_NAME
    index_path.write_text(json.dumps(index, indent=1))
    api.upload_file(path_or_fileobj=str(index_path), path_in_repo=INDEX_NAME,
                    repo_id=args.repo, repo_type="dataset", commit_message="update index")
    stale = [f for f in existing if f.startswith(f"{SHARD_DIR}/") and f not in {e["path"] for e in entries}]
    if stale:
        print(f"PERINGATAN: shard lama tidak ada di index (tidak dihapus): {stale}", file=sys.stderr)
    print(f"done: {len(entries)} shards, {index['total_bytes'] / 2**30:.1f} GiB", flush=True)


if __name__ == "__main__":
    main()

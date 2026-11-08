"""Atomic, self-describing checkpoint store (plain torch, no icefall, no rclone).

A checkpoint ``<name>.pt`` is *complete* only once its sibling
``<name>.meta.json`` exists, parses, and records the ``.pt`` file's size. The
``.pt`` is always written (tmp + ``os.replace``) before its meta, so a crash at
any point leaves either a complete checkpoint or one that is ignored.

CLI::

    python -m prepare_quran_dataset.zipformer.checkpointing dump <pt>
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

Kind = Literal["batch", "epoch"]

_NAME_RE = re.compile(r"^(checkpoint|epoch)-(\d+)\.pt$")
_PREFIX_TO_KIND = {"checkpoint": "batch", "epoch": "epoch"}


@dataclass(frozen=True)
class CkptMeta:
    kind: Kind
    batch_idx_train: int
    epoch: int
    batches_consumed_in_epoch: int
    size_bytes: int
    sha256: str

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "CkptMeta | None":
        try:
            d = json.loads(text)
            meta = cls(
                kind=d["kind"],
                batch_idx_train=int(d["batch_idx_train"]),
                epoch=int(d["epoch"]),
                batches_consumed_in_epoch=int(d["batches_consumed_in_epoch"]),
                size_bytes=int(d["size_bytes"]),
                sha256=str(d["sha256"]),
            )
        except (ValueError, TypeError, KeyError):
            return None
        if meta.kind not in ("batch", "epoch"):
            return None
        return meta

    @property
    def name(self) -> str:
        return ckpt_name(self.kind, self.batch_idx_train, self.epoch)

    def sort_key(self) -> tuple[int, bool]:
        return (self.batch_idx_train, self.kind == "epoch")


def meta_path(pt: Path) -> Path:
    pt = Path(pt)
    return pt.with_name(pt.name.removesuffix(".pt") + ".meta.json")


def meta_name(pt_name: str) -> str:
    return pt_name.removesuffix(".pt") + ".meta.json"


def ckpt_name(kind: Kind, batch_idx_train: int, epoch: int) -> str:
    if kind == "batch":
        return f"checkpoint-{batch_idx_train}.pt"
    if kind == "epoch":
        return f"epoch-{epoch}.pt"
    raise ValueError(f"unknown checkpoint kind: {kind!r}")


def parse_name(pt_name: str) -> tuple[Kind, int] | None:
    """``checkpoint-N.pt`` -> ("batch", N); ``epoch-N.pt`` -> ("epoch", N); else None."""
    m = _NAME_RE.match(pt_name)
    if not m:
        return None
    return _PREFIX_TO_KIND[m.group(1)], int(m.group(2))  # type: ignore[return-value]


def name_number(meta: CkptMeta) -> int:
    """The number that appears in the checkpoint's file name."""
    return meta.batch_idx_train if meta.kind == "batch" else meta.epoch


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_checkpoint_atomic(state: dict, path: Path, meta_fields: dict) -> CkptMeta:
    """Write ``state`` to ``path`` atomically, then its meta (also atomically)."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    meta = CkptMeta(
        kind=meta_fields["kind"],
        batch_idx_train=int(meta_fields["batch_idx_train"]),
        epoch=int(meta_fields["epoch"]),
        batches_consumed_in_epoch=int(meta_fields["batches_consumed_in_epoch"]),
        size_bytes=path.stat().st_size,
        sha256=sha256_file(path),
    )
    _write_bytes_atomic(meta_path(path), meta.to_json().encode())
    return meta


def read_meta(pt: Path) -> CkptMeta | None:
    try:
        text = meta_path(pt).read_text()
    except (OSError, UnicodeDecodeError):
        return None
    return CkptMeta.from_json(text)


def verify_sha256(pt: Path, meta: CkptMeta) -> bool:
    try:
        return Path(pt).stat().st_size == meta.size_bytes and sha256_file(pt) == meta.sha256
    except OSError:
        return False


def list_complete(dir: Path) -> list[tuple[Path, CkptMeta]]:
    """Complete checkpoints in ``dir``, oldest first (epoch after batch at equal step)."""
    out: list[tuple[Path, CkptMeta]] = []
    for pt in Path(dir).glob("*.pt"):
        if parse_name(pt.name) is None:
            continue
        meta = read_meta(pt)
        if meta is None:
            continue
        try:
            if pt.stat().st_size != meta.size_bytes:
                continue
        except OSError:
            continue
        out.append((pt, meta))
    out.sort(key=lambda pm: pm[1].sort_key())
    return out


def latest_complete(dir: Path) -> tuple[Path, CkptMeta] | None:
    complete = list_complete(dir)
    return complete[-1] if complete else None


def _unlink(path: Path, deleted: list[Path]) -> None:
    try:
        path.unlink()
        deleted.append(path)
    except FileNotFoundError:
        pass


def prune(dir: Path, keep_last_k: int) -> list[Path]:
    """Per kind keep the newest ``keep_last_k`` complete checkpoints; return deleted paths.

    Also removes ``*.tmp`` leftovers and ``.pt`` files lacking a valid meta whose
    name number is older than the newest complete checkpoint of the same kind.
    Meta-less ``.pt`` files newer than that are left alone (possibly being written).
    """
    dir = Path(dir)
    deleted: list[Path] = []
    for tmp in dir.glob("*.tmp"):
        _unlink(tmp, deleted)

    complete = list_complete(dir)
    complete_names = {pt.name for pt, _ in complete}
    for kind in ("batch", "epoch"):
        of_kind = [(pt, m) for pt, m in complete if m.kind == kind]
        if not of_kind:
            continue
        drop = of_kind[: max(len(of_kind) - keep_last_k, 0)]
        for pt, _ in drop:
            _unlink(pt, deleted)
            _unlink(meta_path(pt), deleted)
        newest_number = name_number(of_kind[-1][1])
        for pt in dir.glob("*.pt"):
            parsed = parse_name(pt.name)
            if parsed is None or pt.name in complete_names:
                continue
            pkind, number = parsed
            if pkind == kind and number < newest_number:
                _unlink(pt, deleted)
                _unlink(meta_path(pt), deleted)
    return deleted


# ---------------------------------------------------------------- CLI


def _optimizer_sha256(opt_state: dict | None) -> str | None:
    if not isinstance(opt_state, dict):
        return None
    h = hashlib.sha256()
    per_param = opt_state.get("state", {}) or {}
    for key in sorted(per_param, key=str):
        fields = per_param[key]
        if not isinstance(fields, dict):
            continue
        for field in sorted(fields):
            value = fields[field]
            if isinstance(value, torch.Tensor):
                h.update(f"{key}/{field}".encode())
                t = value.detach().cpu().contiguous().reshape(-1)
                h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def dump_summary(pt: Path) -> dict:
    state = torch.load(pt, map_location="cpu", weights_only=False)
    opt = state.get("optimizer")
    lrs = [g["lr"] for g in opt.get("param_groups", [])] if isinstance(opt, dict) else []
    return {
        "batch_idx_train": state.get("batch_idx_train"),
        "epoch": state.get("epoch"),
        "batches_consumed_in_epoch": state.get("batches_consumed_in_epoch"),
        "lrs": lrs,
        "optimizer_sha256": _optimizer_sha256(opt),
        "sampler_state": state.get("sampler"),
        "wandb_run_id": state.get("wandb_run_id"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m prepare_quran_dataset.zipformer.checkpointing")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_dump = sub.add_parser("dump", help="print a JSON summary of a checkpoint")
    p_dump.add_argument("pt", type=Path)
    args = parser.parse_args(argv)
    if args.cmd == "dump":
        json.dump(dump_summary(args.pt), sys.stdout, default=str, sort_keys=True)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

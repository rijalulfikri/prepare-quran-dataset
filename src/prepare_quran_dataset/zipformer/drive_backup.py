"""Background rclone mirror of the checkpoint store (e.g. to Google Drive).

Remote layout mirrors the local one: ``<remote_dir>/<name>.pt`` plus
``<remote_dir>/<name>.meta.json``. The ``.pt`` is uploaded first and the meta
last, so on the remote a checkpoint is complete iff its meta exists.

Corrupt downloads raise :class:`RcloneError` (after deleting the bad local files).
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
from pathlib import Path

from prepare_quran_dataset.zipformer import checkpointing as ck

log = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 300.0


class RcloneError(RuntimeError):
    pass


def _join(remote_dir: str, name: str) -> str:
    if remote_dir.endswith((":", "/")):
        return f"{remote_dir}{name}"
    return f"{remote_dir}/{name}"


class Rclone:
    """Thin subprocess wrapper; every non-zero exit raises :class:`RcloneError`."""

    def __init__(self, binary: str = "rclone") -> None:
        self.binary = binary

    def _run(self, args: list[str], timeout: float) -> subprocess.CompletedProcess:
        cmd = [self.binary, *args]
        try:
            return subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise RcloneError(f"rclone {args[0]} timed out after {timeout}s") from exc
        except OSError as exc:
            raise RcloneError(f"cannot run {self.binary}: {exc}") from exc

    def _check(self, proc: subprocess.CompletedProcess, args: list[str]) -> None:
        if proc.returncode != 0:
            raise RcloneError(f"rclone {' '.join(args)} exited {proc.returncode}: {proc.stderr.strip()}")

    def copyto(self, src: str, dst: str, timeout: float) -> None:
        args = ["copyto", str(src), str(dst)]
        self._check(self._run(args, timeout), args)

    def lsjson(self, remote_dir: str, timeout: float = _DEFAULT_TIMEOUT) -> list[dict]:
        args = ["lsjson", remote_dir]
        proc = self._run(args, timeout)
        if proc.returncode == 3 and "directory not found" in proc.stderr:
            return []
        self._check(proc, args)
        return json.loads(proc.stdout)

    def cat(self, remote_file: str, timeout: float = _DEFAULT_TIMEOUT) -> str:
        args = ["cat", remote_file]
        proc = self._run(args, timeout)
        self._check(proc, args)
        return proc.stdout

    def deletefile(self, remote_file: str, timeout: float = _DEFAULT_TIMEOUT) -> None:
        args = ["deletefile", remote_file]
        self._check(self._run(args, timeout), args)


def _is_batch(pt: Path) -> bool:
    meta = ck.read_meta(pt)
    if meta is not None:
        return meta.kind == "batch"
    parsed = ck.parse_name(pt.name)
    return parsed is not None and parsed[0] == "batch"


class DriveBackup:
    def __init__(self, remote_dir: str, keep_last_k: int, timeout: float, rclone: Rclone | None = None) -> None:
        self.remote_dir = remote_dir
        self.keep_last_k = keep_last_k
        self.timeout = timeout
        self._rclone = rclone or Rclone()
        self.last_error: str | None = None
        self._pending: list[tuple[Path, bool]] = []  # (pt, is_batch)
        self._in_flight: Path | None = None
        self._cond = threading.Condition()
        self._worker = threading.Thread(target=self._run, name="drive-backup", daemon=True)
        self._worker.start()

    # ------------------------------------------------------------ queue

    def submit(self, pt: Path) -> None:
        """Queue ``pt`` for upload; never blocks on I/O.

        A pending (not yet started) batch upload is replaced by a newer batch
        submit; epoch uploads are never dropped.
        """
        pt = Path(pt)
        is_batch = _is_batch(pt)
        with self._cond:
            if is_batch:
                self._pending = [item for item in self._pending if not item[1]]
            self._pending.append((pt, is_batch))
            self._cond.notify_all()

    def flush(self, timeout: float) -> bool:
        """Wait until nothing is pending or in flight; ``False`` on timeout."""
        with self._cond:
            return self._cond.wait_for(lambda: not self._pending and self._in_flight is None, timeout=timeout)

    def pending_paths(self) -> set[Path]:
        """``.pt`` paths queued or being uploaded (must not be pruned locally)."""
        with self._cond:
            paths = {pt for pt, _ in self._pending}
            if self._in_flight is not None:
                paths.add(self._in_flight)
            return paths

    def _run(self) -> None:
        while True:
            with self._cond:
                self._cond.wait_for(lambda: bool(self._pending))
                pt, _ = self._pending.pop(0)
                self._in_flight = pt
            try:
                self._upload(pt)
            except Exception as exc:  # noqa: BLE001 - the backup must never kill training
                self.last_error = str(exc)
                log.warning("drive backup of %s failed: %s", pt, exc)
            finally:
                with self._cond:
                    self._in_flight = None
                    self._cond.notify_all()

    def _upload(self, pt: Path) -> None:
        meta = ck.meta_path(pt)
        if not meta.exists():
            raise RcloneError(f"{pt} has no meta file; refusing to upload an incomplete checkpoint")
        self._rclone.copyto(str(pt), _join(self.remote_dir, pt.name), self.timeout)
        self._rclone.copyto(str(meta), _join(self.remote_dir, meta.name), self.timeout)
        self.prune_remote()

    # ------------------------------------------------------------ remote view

    def _remote_names(self) -> dict[str, int]:
        """Remote file name -> size."""
        return {e["Name"]: int(e.get("Size", -1)) for e in self._rclone.lsjson(self.remote_dir, self.timeout)}

    def _remote_complete(self, names: dict[str, int]) -> list[tuple[str, ck.Kind, int]]:
        """(pt name, kind, number) of remote ``.pt`` files that have a meta sibling."""
        out = []
        for name in names:
            parsed = ck.parse_name(name)
            if parsed is not None and ck.meta_name(name) in names:
                out.append((name, parsed[0], parsed[1]))
        return out

    def prune_remote(self) -> list[str]:
        """Per kind keep the newest ``keep_last_k`` complete remote checkpoints.

        Deletes older ``.pt`` + meta, plus meta-less ``.pt`` files older than the
        newest complete one of their kind. Newer meta-less files are never touched
        (another process may be uploading them). Returns deleted remote paths.
        """
        names = self._remote_names()
        complete = self._remote_complete(names)
        deleted: list[str] = []
        for kind in ("batch", "epoch"):
            of_kind = sorted((n for n in complete if n[1] == kind), key=lambda n: n[2])
            if not of_kind:
                continue
            newest = of_kind[-1][2]
            doomed = [n[0] for n in of_kind[: max(len(of_kind) - self.keep_last_k, 0)]]
            complete_names = {n[0] for n in of_kind}
            for name in names:
                parsed = ck.parse_name(name)
                if parsed and parsed[0] == kind and name not in complete_names and parsed[1] < newest:
                    doomed.append(name)
            for pt_name in doomed:
                for target in (ck.meta_name(pt_name), pt_name):  # meta first: pt is incomplete from then on
                    if target in names:
                        path = _join(self.remote_dir, target)
                        self._rclone.deletefile(path, self.timeout)
                        deleted.append(path)
        return deleted

    def _remote_metas(self) -> list[ck.CkptMeta]:
        names = self._remote_names()
        metas = []
        for pt_name, _, _ in self._remote_complete(names):
            meta = ck.CkptMeta.from_json(self._rclone.cat(_join(self.remote_dir, ck.meta_name(pt_name)), self.timeout))
            if meta is None or meta.name != pt_name or names[pt_name] != meta.size_bytes:
                continue
            metas.append(meta)
        metas.sort(key=ck.CkptMeta.sort_key)
        return metas

    def remote_complete_metas(self) -> list[ck.CkptMeta]:
        """All complete remote checkpoints, newest first."""
        return self._remote_metas()[::-1]

    def latest_remote(self) -> ck.CkptMeta | None:
        metas = self._remote_metas()
        return metas[-1] if metas else None

    def download(self, meta: ck.CkptMeta, local_dir: Path) -> Path:
        """Copy ``.pt`` then meta into ``local_dir`` and verify the sha256.

        On mismatch both local files are deleted and :class:`RcloneError` is raised.
        """
        local_dir = Path(local_dir)
        pt = local_dir / meta.name
        meta_file = ck.meta_path(pt)
        self._rclone.copyto(_join(self.remote_dir, pt.name), str(pt), self.timeout)
        self._rclone.copyto(_join(self.remote_dir, meta_file.name), str(meta_file), self.timeout)
        local_meta = ck.read_meta(pt)
        if local_meta is None or not ck.verify_sha256(pt, local_meta):
            pt.unlink(missing_ok=True)
            meta_file.unlink(missing_ok=True)
            raise RcloneError(f"downloaded {pt.name} failed sha256 verification")
        return pt

    def download_missing_epochs(self, local_dir: Path) -> list[Path]:
        local_dir = Path(local_dir)
        got = []
        for meta in self._remote_metas():
            if meta.kind == "epoch" and not (local_dir / meta.name).exists():
                got.append(self.download(meta, local_dir))
        return got

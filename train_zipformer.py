"""Stage runner for the Zipformer2-CTC pipeline: prepare -> train -> eval -> export.

Every stage is idempotent, so re-running the same command after a crash or on a
new box continues where the last run stopped (training auto-resumes from local
or Drive checkpoints; eval / qdat_bench skip results that already exist).

Exit codes: 0 success, 143 graceful stop (`TrainingInterrupted`, e.g. SIGTERM;
checkpoints flushed), 1 any other failure.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

log = logging.getLogger("train_zipformer")

DEFAULT_CONFIG = "configs/train/zipformer/train_config.yml"
STAGES = ["prepare", "train", "eval", "export"]
EXIT_INTERRUPTED = 143
N_PARITY_WAVS = 5


class RcloneSetupError(RuntimeError):
    pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python train_zipformer.py", description=__doc__.splitlines()[0])
    p.add_argument("--config", default=DEFAULT_CONFIG, help=f"training YAML (default: {DEFAULT_CONFIG})")
    p.add_argument("--stage", choices=[*STAGES, "all"], default="all")
    p.add_argument(
        "--push-to-hub", action="store_true", help="upload export/, averaged .pt, config and results to hub_model_id"
    )
    p.add_argument("--rerun-testset", action="store_true", help="re-evaluate the test set even if results exist")
    p.add_argument("--rerun-qdat-bench", action="store_true", help="re-run qdat_bench even if results exist")
    p.add_argument("--device", default=None, help="default: cuda if available else cpu")
    return p.parse_args(argv)


def stages_to_run(stage: str) -> list[str]:
    return list(STAGES) if stage == "all" else [stage]


# ------------------------------------------------------------------ helpers


def _device(args: argparse.Namespace):
    import torch

    return torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))


def _tokens():
    from prepare_quran_dataset.zipformer.tokens import PhonemeTokens
    from prepare_quran_dataset.zipformer.train_loop import DEFAULT_VOCAB

    return PhonemeTokens.from_vocab_json(DEFAULT_VOCAB)


def _averaged_path(cfg) -> Path:
    return Path(cfg.output_dir) / f"averaged-{cfg.avg_last_k}.pt"


def _load_averaged_model(cfg, tokens):
    """CPU model with the averaged weights (averaging epoch checkpoints if needed)."""
    import torch

    from prepare_quran_dataset.zipformer.model import build_model
    from prepare_quran_dataset.zipformer.train_loop import average_last_epochs

    path = _averaged_path(cfg)
    if not path.exists():
        log.info("%s missing; averaging the last %d epoch checkpoints", path, cfg.avg_last_k)
        path = average_last_epochs(Path(cfg.output_dir), cfg.avg_last_k)
    model = build_model(cfg, tokens.vocab_size)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])
    model.eval()
    return model


def check_rclone() -> None:
    """Raise `RcloneSetupError` unless rclone is installed and has a remote."""
    if shutil.which("rclone") is None:
        raise RcloneSetupError("rclone not found on PATH; install it: curl https://rclone.org/install.sh | bash")
    res = subprocess.run(["rclone", "listremotes"], capture_output=True, text=True, check=False)
    if res.returncode != 0 or not res.stdout.strip():
        raise RcloneSetupError(
            "rclone has no remote configured; put rclone.conf at ~/.config/rclone/rclone.conf or set RCLONE_CONFIG"
        )


# ------------------------------------------------------------------ stages


def stage_prepare(cfg, args: argparse.Namespace) -> None:
    check_rclone()
    log.info("syncing %s -> %s", cfg.rclone_remote_path, cfg.data_dir)
    subprocess.run(["rclone", "sync", cfg.rclone_remote_path, str(cfg.data_dir), "--progress"], check=True)


def stage_train(cfg, args: argparse.Namespace) -> Path:
    from prepare_quran_dataset.zipformer.train_loop import run_training

    # run_training logs `resume: source=... checkpoint=... epoch=... batch_idx_train=...`
    path = run_training(cfg, Path(cfg.data_dir), _tokens(), _device(args))
    log.info("averaged model: %s", path)
    return path


def stage_eval(cfg, args: argparse.Namespace) -> None:
    out = Path(cfg.output_dir) / "test_results.json"
    if out.exists() and not args.rerun_testset:
        log.info("%s exists; skipping test eval (use --rerun-testset)", out)
        return
    from prepare_quran_dataset.zipformer.datamodule import eval_dataloader, load_cuts
    from prepare_quran_dataset.zipformer.decode import evaluate_split

    tokens, device = _tokens(), _device(args)
    model = _load_averaged_model(cfg, tokens).to(device)
    loader = eval_dataloader(cfg, load_cuts(Path(cfg.data_dir), "test"))
    results = evaluate_split(model, loader, tokens, device, cfg.export_chunk_size, cfg.export_left_context)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("test results -> %s: %s", out, results)


def _parity_wavs(cfg) -> list:
    """First `N_PARITY_WAVS` non-silence test cuts as float32 mono 16 kHz arrays."""
    from prepare_quran_dataset.zipformer.datamodule import load_cuts

    wavs = []
    for cut in load_cuts(Path(cfg.data_dir), "test"):
        if (cut.supervisions[0].custom or {}).get("input_type") == "silence":
            continue
        wavs.append(cut.load_audio()[0])
        if len(wavs) == N_PARITY_WAVS:
            break
    return wavs


def stage_export(cfg, args: argparse.Namespace) -> dict:
    from prepare_quran_dataset.zipformer.export_onnx import (
        OnnxStreamingDecoder,
        assert_onnx_matches_torch,
        export_streaming_onnx,
    )
    from prepare_quran_dataset.zipformer.qdat import run_qdat_bench

    output_dir = Path(cfg.output_dir)
    export_dir = output_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    tokens = _tokens()
    model = _load_averaged_model(cfg, tokens).cpu()  # the parity check runs torch on CPU
    paths = export_streaming_onnx(
        model,
        export_dir,
        tokens,
        cfg.export_chunk_size,
        cfg.export_left_context,
        output_downsampling_factor=cfg.output_downsampling_factor,
    )
    decoder = OnnxStreamingDecoder(paths["fp32"])
    wavs = _parity_wavs(cfg)
    assert_onnx_matches_torch(model, decoder, wavs, tokens, cfg.export_chunk_size, cfg.export_left_context)
    log.info("ONNX streaming decode matches PyTorch on %d test cuts", len(wavs))
    results = run_qdat_bench(decoder, tokens, output_dir, rerun=args.rerun_qdat_bench)
    log.info("qdat_bench: %s", results)
    for kind, path in paths.items():
        log.info("export %s: %s", kind, path)
    return paths


def push_to_hub(cfg) -> None:
    import os

    from dotenv import load_dotenv
    from huggingface_hub import HfApi

    load_dotenv()
    output_dir = Path(cfg.output_dir)
    repo = cfg.hub_model_id
    api = HfApi(token=os.environ.get("HUGGINGFACE_TOKEN") or None)
    api.create_repo(repo, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=output_dir / "export", repo_id=repo, path_in_repo="export", repo_type="model")
    files = [
        _averaged_path(cfg),
        output_dir / "config.yml",
        output_dir / "test_results.json",
        output_dir / "qdat_bench_test_results.json",
    ]
    for path in files:
        if not path.exists():
            log.warning("not uploading missing %s", path)
            continue
        api.upload_file(path_or_fileobj=path, path_in_repo=path.name, repo_id=repo, repo_type="model")
    log.info("pushed to https://huggingface.co/%s", repo)


# ------------------------------------------------------------------ entry point


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    from prepare_quran_dataset.zipformer.config import ZipformerTrainConfig
    from prepare_quran_dataset.zipformer.train_loop import TrainingInterrupted

    cfg = ZipformerTrainConfig.from_yaml(args.config)
    runners = {"prepare": stage_prepare, "train": stage_train, "eval": stage_eval, "export": stage_export}
    try:
        for name in stages_to_run(args.stage):
            log.info("===== stage: %s =====", name)
            runners[name](cfg, args)
        if args.push_to_hub:
            push_to_hub(cfg)
    except RcloneSetupError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except TrainingInterrupted as exc:
        log.warning("training interrupted, checkpoints flushed (%s); re-run the same command to resume", exc)
        return EXIT_INTERRUPTED
    except Exception:
        log.exception("stage failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

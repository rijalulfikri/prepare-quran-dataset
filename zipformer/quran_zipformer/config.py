"""Training configuration for the Zipformer2-CTC pipeline."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, model_validator


class ZipformerTrainConfig(BaseModel):
    # --- data ---
    train_moshaf_ids: list[str]
    test_moshaf_ids: list[str]
    devset_ratio: float = 0.1
    seed: int = 42
    max_audio_seconds: float | None = None
    noise_aug_ratio: float = 0.3
    max_noise_input_seconds: float = 40.0
    min_noise_input_seconds: float = 1.0
    data_dir: str = "./data-zipformer"
    rclone_remote_path: str = "gdrive:muaalem-zipformer-data"

    # --- model (icefall-style comma lists, one entry per encoder stack) ---
    num_encoder_layers: str = "2,2,3,4,3,2"
    downsampling_factor: str = "1,2,4,8,4,2"
    feedforward_dim: str = "512,768,1024,1536,1024,768"
    num_heads: str = "4,4,4,8,4,4"
    encoder_dim: str = "192,256,384,512,384,256"
    query_head_dim: str = "32"
    value_head_dim: str = "12"
    pos_head_dim: str = "4"
    pos_dim: int = 48
    encoder_unmasked_dim: str = "192,192,256,256,256,192"
    cnn_module_kernel: str = "31,31,15,15,15,31"
    output_downsampling_factor: Literal[1, 2] = 1
    causal: bool = True
    chunk_size: str = "8,16,24"
    left_context_frames: str = "64,128,256"
    feature_dim: int = 80

    # --- training ---
    max_duration: float = 600
    num_workers: int = 8
    num_epochs: int = 30
    base_lr: float = 0.045
    lr_batches: float = 7500
    lr_epochs: float = 3.5
    warmup_batches: int = 500
    keep_last_k: int = 3
    avg_last_k: int = 3
    bf16: bool = True
    log_every: int = 50
    dev_max_cuts: int | None = 2000
    cutmix_p: float = 0.5
    cutmix_snr: tuple[float, float] = (10.0, 20.0)
    speed_perturb: bool = True
    output_dir: str = "./results-zipformer-v0"

    # --- eval / export ---
    export_chunk_size: int = 24
    export_left_context: int = 256
    hub_model_id: str = "obadx/muaalem-zipformer-ctc-v0"
    wandb_project_name: str = "muaalem-zipformer-ctc-v0"

    # --- frame budget ---
    max_dropped_fraction: float = 0.01

    # --- fault tolerance ---
    save_every_n_batches: int = 2000
    rclone_checkpoint_path: str | None = None
    upload_timeout_seconds: int = 1800
    shutdown_flush_seconds: int = 120

    @model_validator(mode="after")
    def _validate(self) -> "ZipformerTrainConfig":
        if self.avg_last_k > self.keep_last_k:
            raise ValueError(
                f"avg_last_k ({self.avg_last_k}) must be <= keep_last_k ({self.keep_last_k})"
            )
        chunk_sizes = [int(x) for x in self.chunk_size.split(",")]
        if self.export_chunk_size not in chunk_sizes:
            raise ValueError(
                f"export_chunk_size ({self.export_chunk_size}) must be one of "
                f"chunk_size values {chunk_sizes}"
            )
        if self.save_every_n_batches < 1:
            raise ValueError("save_every_n_batches must be >= 1")
        if self.rclone_checkpoint_path is not None and ":" not in self.rclone_checkpoint_path:
            raise ValueError(
                "rclone_checkpoint_path must be an rclone `remote:path`, "
                f"got {self.rclone_checkpoint_path!r}"
            )
        return self

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "ZipformerTrainConfig":
        """Load the configuration from a YAML file."""
        with open(yaml_path, encoding="utf-8") as f:
            return cls(**yaml.safe_load(f))

    def to_yaml(self, yaml_path: str | Path) -> None:
        """Save the configuration to a YAML file (plain lists, no python tags)."""
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                self.model_dump(mode="json"),
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

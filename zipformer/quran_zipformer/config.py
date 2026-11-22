"""Training configuration for the Zipformer2-CTC pipeline."""

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, model_validator

SPEED_STEP = 0.05  # grid of drawn speed factors: few distinct factors, few cached resamplers
StressSet = Literal["vol-20db", "reverb", "speed1.25"]
_METRIC_KEY = re.compile(r"(per_phonemes|[\w.\-]+/per_phonemes)")


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
    rclone_remote_path: str = "utluma:RClone/Utluma/zipformer-data"
    # Tar shards of data_dir, tried in this order before the rclone sync fallback:
    # HF dataset repo (vastai/pack_data_to_hf.py), then an rclone folder with the
    # same shards (vastai/mirror_shards_to_drive.py)
    hf_data_repo: str | None = None
    drive_data_shards_path: str | None = None

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
    left_context_frames: str = "72,128,256"
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

    # --- augmentation (defaults reproduce the v0 pipeline) ---
    # One static copy of every train cut per range; a range's factor is drawn per
    # cut on a 0.05 grid from a seed and the cut id (see datamodule.copy_factors)
    speed_factor_ranges: list[tuple[float, float]] = [(0.9, 0.9), (1.0, 1.0), (1.1, 1.1)]
    reverb_p: float = 0.0  # simulated-RIR reverb, before the noise mix
    volume_p: float = 0.0  # gain drawn uniformly in dB, after the noise mix
    volume_db: tuple[float, float] = (-30.0, 6.0)
    # Consecutive segments of one sura joined into one training cut (the app hears
    # continuous recitation); a run starts at a cut with probability concat_p
    concat_p: float = 0.0
    concat_segments: tuple[int, int] = (2, 4)  # segments per joined cut, drawn uniformly
    concat_gap_seconds: tuple[float, float] = (0.0, 0.0)  # extra silence at each join

    # --- loss ---
    # CR-CTC (icefall): two SpecAugment views of each batch, CTC on both plus a
    # consistency KL between them. No new parameters: checkpoints and export are unchanged
    use_cr_ctc: bool = False
    cr_loss_scale: float = 0.2
    cr_time_mask_ratio: float = 2.5  # time masking x this, as the icefall recipe
    cr_warm_steps: int = 2000  # cr_loss_scale ramps up linearly over these batches

    # --- per-epoch evaluation sets (defaults: legacy dev only) ---
    dev_unseen_cuts_per_moshaf: int = 0  # fixed sample of cuts_test per test moshaf; 0 = off
    eval_qdat: bool = False  # obadx/qdat_bench through the same PyTorch path as dev
    stress_sets: list[StressSet] = []  # perturbed copies of the dev_unseen sample
    # dev_metrics.jsonl key the early-stop rule follows (vastai/zipformer_watchdog.sh,
    # zipformer_status.py); rows without it fall back to the legacy dev `per_phonemes`
    early_stop_metric: str = "qdat/per_phonemes"

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
        if self.causal:
            self._validate_chunk_context(chunk_sizes)
        if self.save_every_n_batches < 1:
            raise ValueError("save_every_n_batches must be >= 1")
        if self.rclone_checkpoint_path is not None and ":" not in self.rclone_checkpoint_path:
            raise ValueError(
                "rclone_checkpoint_path must be an rclone `remote:path`, "
                f"got {self.rclone_checkpoint_path!r}"
            )
        self._validate_augmentation()
        if self.stress_sets and self.dev_unseen_cuts_per_moshaf < 1:
            raise ValueError("stress_sets perturb the dev_unseen sample; set dev_unseen_cuts_per_moshaf >= 1")
        if self.dev_unseen_cuts_per_moshaf < 0:
            raise ValueError("dev_unseen_cuts_per_moshaf must be >= 0")
        if not _METRIC_KEY.fullmatch(self.early_stop_metric):
            raise ValueError(
                f"early_stop_metric must be 'per_phonemes' or '<set>/per_phonemes', got {self.early_stop_metric!r}"
            )
        return self

    def _validate_augmentation(self) -> None:
        low, high = self.concat_segments
        if not 2 <= low <= high:
            raise ValueError(f"concat_segments must be (low, high) with 2 <= low <= high, got {(low, high)}")
        low, high = self.concat_gap_seconds
        if not 0 <= low <= high:
            raise ValueError(f"concat_gap_seconds must be (low, high) with 0 <= low <= high, got {(low, high)}")
        if self.cr_loss_scale < 0 or self.cr_time_mask_ratio <= 0 or self.cr_warm_steps < 1:
            raise ValueError("cr_loss_scale must be >= 0, cr_time_mask_ratio > 0 and cr_warm_steps >= 1")
        for name in ("reverb_p", "volume_p", "cutmix_p", "concat_p"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("volume_db", "cutmix_snr"):
            low, high = getattr(self, name)
            if low > high:
                raise ValueError(f"{name} must be (low, high) with low <= high, got {(low, high)}")
        ranges = self.speed_factor_ranges
        if not ranges:
            raise ValueError("speed_factor_ranges needs at least one range")
        for low, high in ranges:
            on_grid = all(abs(x / SPEED_STEP - round(x / SPEED_STEP)) < 1e-6 for x in (low, high))
            if not (0.5 <= low <= high <= 2.0 and on_grid):
                raise ValueError(
                    f"speed_factor_ranges entry {(low, high)} must satisfy 0.5 <= low <= high <= 2.0 "
                    f"on a {SPEED_STEP} grid"
                )
        # disjoint ranges give every copy of a cut a different factor, hence a unique id
        if any(a[1] >= b[0] for a, b in zip(ranges, ranges[1:])):
            raise ValueError(f"speed_factor_ranges must be sorted and disjoint, got {ranges}")

    def _validate_chunk_context(self, chunk_sizes: list[int]) -> None:
        """Reject chunk / left-context pairs icefall's Zipformer2 asserts on.

        Training samples `chunk_size` and `left_context_frames` independently and
        uses `left // chunk` whole chunks of context; `_get_attn_mask` requires
        `chunk * (left // chunk) >= (kernel // 2) * downsampling` for every stack,
        so a bad pair otherwise only fails on the GPU when it is first drawn.
        """
        kernels = [int(x) for x in self.cnn_module_kernel.split(",")]
        factors = [int(x) for x in self.downsampling_factor.split(",")]
        need = max(k // 2 * d for k, d in zip(kernels, factors))
        lefts = [int(x) for x in self.left_context_frames.split(",")]
        pairs = {(c, left) for c in chunk_sizes for left in lefts if c > 0 and left >= 0}
        pairs.add((self.export_chunk_size, self.export_left_context))
        bad = sorted((c, left) for c, left in pairs if c * (left // c) < need)
        if bad:
            raise ValueError(
                f"chunk_size / left_context_frames pairs {bad} give fewer than {need} frames "
                "of left context (chunk * (left // chunk)), which icefall's Zipformer2 "
                "requires for cnn_module_kernel / downsampling_factor"
            )

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

from pathlib import Path

import pytest

from quran_zipformer.config import ZipformerTrainConfig

YAML = Path(__file__).resolve().parents[1] / "configs" / "train_config.yml"


def test_default_yaml_loads_and_matches_v3_dims():
    c = ZipformerTrainConfig.from_yaml(YAML)
    assert c.encoder_dim == "192,256,384,512,384,256"
    assert c.chunk_size == "8,16,24" and c.left_context_frames == "72,128,256"
    assert c.test_moshaf_ids == ["19.0", "29.0", "30.0"]
    assert len(c.train_moshaf_ids) == 23


def test_export_chunk_must_be_trained():
    with pytest.raises(ValueError):
        ZipformerTrainConfig(
            train_moshaf_ids=["0.0"], test_moshaf_ids=[], export_chunk_size=32
        )


def test_tiny_two_stack_config_is_valid():
    c = ZipformerTrainConfig(
        train_moshaf_ids=["0.0"],
        test_moshaf_ids=[],
        num_encoder_layers="1,1",
        chunk_size="8",
        export_chunk_size=8,
    )
    assert c.export_chunk_size == 8


def test_left_context_must_cover_conv_kernel_for_every_chunk():
    # icefall Zipformer2._get_attn_mask asserts
    # chunk * (left // chunk) >= (kernel // 2) * downsampling for every stack;
    # with the default dims that is 56, and 24 * (64 // 24) = 48 fails
    with pytest.raises(ValueError, match="left_context_frames"):
        ZipformerTrainConfig(
            train_moshaf_ids=["0.0"], test_moshaf_ids=[],
            chunk_size="8,16,24", left_context_frames="64,128,256",
        )


def test_avg_last_k_must_not_exceed_keep_last_k():
    with pytest.raises(ValueError):
        ZipformerTrainConfig(
            train_moshaf_ids=["0.0"], test_moshaf_ids=[], avg_last_k=4, keep_last_k=3
        )


def test_roundtrip(tmp_path):
    c = ZipformerTrainConfig(train_moshaf_ids=["0.0"], test_moshaf_ids=["1.0"])
    c.to_yaml(tmp_path / "c.yml")
    assert ZipformerTrainConfig.from_yaml(tmp_path / "c.yml") == c
    assert "!!python" not in (tmp_path / "c.yml").read_text()


def test_fault_tolerance_defaults():
    c = ZipformerTrainConfig.from_yaml(YAML)
    assert c.save_every_n_batches == 2000 and c.keep_last_k == 3
    assert c.rclone_checkpoint_path.startswith("utluma-edit:")
    default = ZipformerTrainConfig(train_moshaf_ids=["0.0"], test_moshaf_ids=[])
    assert default.rclone_checkpoint_path is None


def test_output_downsampling_default_and_choices():
    assert ZipformerTrainConfig.from_yaml(YAML).output_downsampling_factor == 1
    ZipformerTrainConfig(
        train_moshaf_ids=["0.0"], test_moshaf_ids=[], output_downsampling_factor=2
    )


@pytest.mark.parametrize(
    "bad",
    [
        {"save_every_n_batches": 0},
        {"rclone_checkpoint_path": "/no/remote"},
        {"output_downsampling_factor": 4},
    ],
)
def test_fault_tolerance_validation(bad):
    with pytest.raises(ValueError):
        ZipformerTrainConfig(train_moshaf_ids=["0.0"], test_moshaf_ids=[], **bad)


# ---------------------------------------------------------------- augmentation / eval sets (v1)


def _cfg(**kw):
    return ZipformerTrainConfig(train_moshaf_ids=["0.0"], test_moshaf_ids=[], **kw)


def test_defaults_are_the_v0_pipeline():
    c = _cfg()
    assert c.speed_perturb and c.speed_factor_ranges == [(0.9, 0.9), (1.0, 1.0), (1.1, 1.1)]
    assert (c.reverb_p, c.volume_p, c.cutmix_p, c.cutmix_snr) == (0.0, 0.0, 0.5, (10.0, 20.0))
    assert (c.dev_unseen_cuts_per_moshaf, c.eval_qdat, c.stress_sets) == (0, False, [])
    assert c.early_stop_metric == "qdat/per_phonemes"
    assert (c.concat_p, c.use_cr_ctc) == (0.0, False)


def test_train_yaml_turns_on_the_new_values_under_a_new_run_name():
    c = ZipformerTrainConfig.from_yaml(YAML)
    assert c.speed_factor_ranges == [(0.85, 0.95), (1.0, 1.0), (1.05, 1.25)]
    assert (c.reverb_p, c.volume_p, c.volume_db, c.cutmix_snr) == (0.3, 0.5, (-30.0, 6.0), (5.0, 20.0))
    assert (c.dev_unseen_cuts_per_moshaf, c.eval_qdat) == (1000, True)
    assert c.stress_sets == ["vol-20db", "reverb", "speed1.25"]
    assert c.early_stop_metric == "qdat/per_phonemes"
    assert (c.concat_p, c.use_cr_ctc) == (0.0, False)  # comparison-run options, off in v1
    # never the finished v0 run's folders: that would "resume" at its epoch 30
    assert "v0" not in c.output_dir and "v0" not in c.rclone_checkpoint_path


def test_baseline_yaml_matches_train_model_and_eval_sets():
    train = ZipformerTrainConfig.from_yaml(YAML)
    base = ZipformerTrainConfig.from_yaml(YAML.parent / "eval_baseline_v0.yml")
    model_fields = ["num_encoder_layers", "downsampling_factor", "feedforward_dim", "num_heads", "encoder_dim",
                    "query_head_dim", "value_head_dim", "pos_head_dim", "pos_dim", "encoder_unmasked_dim",
                    "cnn_module_kernel", "output_downsampling_factor", "causal", "chunk_size",
                    "left_context_frames", "feature_dim", "export_chunk_size", "export_left_context",
                    "dev_unseen_cuts_per_moshaf", "eval_qdat", "stress_sets", "seed", "dev_max_cuts",
                    "test_moshaf_ids", "data_dir"]
    assert {f: getattr(base, f) for f in model_fields} == {f: getattr(train, f) for f in model_fields}
    assert base.output_dir != train.output_dir and base.rclone_checkpoint_path != train.rclone_checkpoint_path
    assert (base.reverb_p, base.volume_p, base.speed_factor_ranges) == (0.0, 0.0, [(0.9, 0.9), (1.0, 1.0), (1.1, 1.1)])


def test_new_fields_survive_a_yaml_round_trip(tmp_path):
    c = ZipformerTrainConfig.from_yaml(YAML)
    c.to_yaml(tmp_path / "c.yml")
    assert ZipformerTrainConfig.from_yaml(tmp_path / "c.yml") == c


@pytest.mark.parametrize(
    "bad",
    [
        {"speed_factor_ranges": []},
        {"speed_factor_ranges": [(0.9, 1.0), (1.0, 1.1)]},  # overlap: copies could share an id
        {"speed_factor_ranges": [(1.1, 1.1), (0.9, 0.9)]},  # unsorted
        {"speed_factor_ranges": [(0.87, 0.95)]},  # off the 0.05 grid
        {"speed_factor_ranges": [(0.3, 0.5)]},
        {"reverb_p": 1.5},
        {"volume_p": -0.1},
        {"volume_db": (6.0, -30.0)},
        {"cutmix_snr": (20.0, 5.0)},
        {"stress_sets": ["reverb"]},  # needs a dev_unseen sample
        {"stress_sets": ["noise"], "dev_unseen_cuts_per_moshaf": 5},
        {"dev_unseen_cuts_per_moshaf": -1},
        {"early_stop_metric": "qdat/wer"},
        {"early_stop_metric": "loss"},
        {"concat_p": 1.5},
        {"concat_segments": (1, 3)},
        {"concat_segments": (4, 2)},
        {"concat_gap_seconds": (-0.1, 0.2)},
        {"cr_loss_scale": -0.1},
        {"cr_time_mask_ratio": 0.0},
        {"cr_warm_steps": 0},
    ],
)
def test_augmentation_and_eval_validation(bad):
    with pytest.raises(ValueError):
        _cfg(**bad)


@pytest.mark.parametrize("key", ["per_phonemes", "qdat/per_phonemes", "dev_unseen_speed1.25/per_phonemes"])
def test_early_stop_metric_accepts_set_keys(key):
    assert _cfg(early_stop_metric=key).early_stop_metric == key

import pytest

from prepare_quran_dataset.zipformer.config import ZipformerTrainConfig

YAML = "configs/train/zipformer/train_config.yml"


def test_default_yaml_loads_and_matches_v3_dims():
    c = ZipformerTrainConfig.from_yaml(YAML)
    assert c.encoder_dim == "192,256,384,512,384,256"
    assert c.chunk_size == "8,16,24" and c.left_context_frames == "64,128,256"
    assert c.test_moshaf_ids == ["19.0", "29.0", "30.0"]
    assert len(c.train_moshaf_ids) == 22


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
    assert c.rclone_checkpoint_path.startswith("gdrive:")
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

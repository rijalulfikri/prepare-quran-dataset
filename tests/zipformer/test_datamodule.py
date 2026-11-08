import json

import pytest
from lhotse import CutSet

from prepare_quran_dataset.zipformer import datamodule as dm
from prepare_quran_dataset.zipformer.data_prep import _write_cuts


def test_relative_paths_resolve_after_prefix(tiny_data_dir):
    cuts = dm.load_cuts(tiny_data_dir, "dev")
    assert cuts[0].load_audio().shape[-1] > 0


def test_load_cuts_empty_manifest_is_empty_cutset(tmp_path):
    _write_cuts([], tmp_path / "cuts_test.jsonl.gz")
    cuts = dm.load_cuts(tmp_path, "test")
    assert isinstance(cuts, CutSet) and len(cuts) == 0


def test_train_batch_shapes(tiny_data_dir, cfg):
    cfg.num_workers = 0
    cfg.max_duration = 10
    batch = next(iter(dm.train_dataloader(cfg, tiny_data_dir)))
    feats, lens, texts, cuts = dm.batch_to_tensors(batch)
    assert feats.ndim == 3 and feats.shape[-1] == 80
    assert lens.shape[0] == feats.shape[0] == len(texts) == len(cuts)
    assert feats.shape[1] == lens.max()


def test_cutmix_keeps_one_supervision_and_text_per_cut(tiny_data_dir, cfg):
    # noise_aug cuts carry an empty supervision that must not be mixed into the target
    cfg.num_workers = 0
    cfg.max_duration = 10
    cfg.cutmix_p = 1.0
    cfg.speed_perturb = False
    expected = {c.id: c.supervisions[0].text for c in dm.load_cuts(tiny_data_dir, "train")}
    batch = next(iter(dm.train_dataloader(cfg, tiny_data_dir)))
    _, _, texts, cuts = dm.batch_to_tensors(batch)
    assert [type(c).__name__ for c in cuts] == ["MixedCut"] * len(cuts)
    assert texts == [expected[c.id] for c in cuts]


def test_eval_loader_serves_every_cut_once(tiny_data_dir, cfg):
    cfg.num_workers = 0
    cuts = dm.load_cuts(tiny_data_dir, "dev")
    loader = dm.eval_dataloader(cfg, cuts, max_duration=7)
    ids = [c.id for b in loader for c in dm.batch_to_tensors(b)[3]]
    assert sorted(ids) == sorted(c.id for c in cuts)


def test_speed_perturb_triples_cut_count(tiny_data_dir, cfg):
    cfg.num_workers = 0
    n = len(dm.load_cuts(tiny_data_dir, "train"))
    assert len(dm._train_cuts(cfg, tiny_data_dir)) == 3 * n


def test_train_cuts_are_globally_shuffled_and_deterministic(tiny_data_dir_big, cfg):
    # block order [sp0.9, base, sp1.1] would skew bucket edges and batch order
    base = dm.load_cuts(tiny_data_dir_big, "train")
    blocks = [c.id for c in [*base.perturb_speed(0.9), *base, *base.perturb_speed(1.1)]]
    first = [c.id for c in dm._train_cuts(cfg, tiny_data_dir_big)]
    second = [c.id for c in dm._train_cuts(cfg, tiny_data_dir_big)]
    assert first == second
    assert sorted(first) == sorted(blocks)
    assert first != blocks


def _ids(batch):
    return [c.id for c in batch["supervisions"]["cut"]]


@pytest.mark.parametrize("workers", [0, 2])
def test_sampler_resume_continues_at_consumed_position(tiny_data_dir_big, cfg, workers):
    # tiny_data_dir_big: ~60 one-second cuts so an epoch has >= 10 batches at max_duration=4
    cfg.num_workers = workers
    cfg.max_duration = 4
    ref = [_ids(b) for b in dm.train_dataloader(cfg, tiny_data_dir_big, epoch=1)]
    assert len(ref) >= 10
    loader = dm.train_dataloader(cfg, tiny_data_dir_big, epoch=1)
    it = iter(loader)
    consumed = [_ids(next(it)) for _ in range(4)]
    state = loader.sampler.state_dict()  # drawn count may be > 4 with workers
    del it, loader
    resumed = dm.train_dataloader(
        cfg, tiny_data_dir_big, epoch=1, sampler_state=state, batches_consumed=4
    )
    rest = [_ids(b) for b in resumed]
    assert consumed + rest == ref  # no replay, no skip, same order


def test_restore_sampler_accepts_json_round_tripped_state(tiny_data_dir_big, cfg):
    # after a JSON round-trip the stats_per_epoch keys are strings
    cfg.num_workers = 0
    cfg.max_duration = 4
    loader = dm.train_dataloader(cfg, tiny_data_dir_big, epoch=1)
    it = iter(loader)
    consumed = [_ids(next(it)) for _ in range(3)]
    state = json.loads(json.dumps(loader.sampler.state_dict()))
    resumed = dm.train_dataloader(
        cfg, tiny_data_dir_big, epoch=1, sampler_state=state, batches_consumed=2
    )
    assert _ids(next(iter(resumed))) == consumed[2]


def test_train_dataset_has_specaugment_and_eval_has_no_input_transforms(tiny_data_dir, cfg):
    from lhotse.dataset import SpecAugment

    cfg.num_workers = 0
    train_ds = dm.train_dataloader(cfg, tiny_data_dir).dataset
    specs = [t for t in train_ds.input_transforms if isinstance(t, SpecAugment)]
    assert len(specs) == 1
    sa = specs[0]
    assert (
        sa.time_warp_factor,
        sa.num_frame_masks,
        sa.features_mask_size,
        sa.num_feature_masks,
        sa.frames_mask_size,
    ) == (80, 10, 27, 2, 100)
    eval_ds = dm.eval_dataloader(cfg, dm.load_cuts(tiny_data_dir, "dev")).dataset
    assert eval_ds.input_transforms == []


def test_cutmix_seed_is_randomized_per_worker(tiny_data_dir, cfg):
    # a fixed seed makes every persistent worker draw the same noise / SNR sequence
    from lhotse.dataset import CutMix

    cfg.num_workers = 2
    cfg.max_duration = 10
    cfg.cutmix_p = 1.0
    loader = dm.train_dataloader(cfg, tiny_data_dir)
    mixes = [t for t in loader.dataset.cut_transforms if isinstance(t, CutMix)]
    assert len(mixes) == 1 and mixes[0].seed == "randomized"
    # "randomized" needs Lhotse's worker_init_fn in each worker, else it asserts
    _, _, _, cuts = dm.batch_to_tensors(next(iter(loader)))
    assert {type(c).__name__ for c in cuts} == {"MixedCut"}


def test_restore_sampler_rejects_unknown_state_layout(tiny_data_dir_big, cfg):
    cfg.num_workers = 0
    cfg.max_duration = 4
    loader = dm.train_dataloader(cfg, tiny_data_dir_big, epoch=1)
    next(iter(loader))
    state = loader.sampler.state_dict()
    stats = state["diagnostics"]["stats_per_epoch"][1]
    stats["kept"] = stats.pop("kept_batches")
    with pytest.raises(RuntimeError, match="Lhotse sampler state layout changed"):
        dm.restore_sampler(loader.sampler, state, 1)


def test_train_dataloader_logs_rebuild_and_fast_forward(tiny_data_dir_big, cfg, caplog):
    cfg.num_workers = 0
    cfg.max_duration = 4
    loader = dm.train_dataloader(cfg, tiny_data_dir_big, epoch=1)
    it = iter(loader)
    next(it)
    next(it)
    state = loader.sampler.state_dict()
    with caplog.at_level("INFO", logger=dm.__name__):
        dm.train_dataloader(cfg, tiny_data_dir_big, epoch=1, sampler_state=state, batches_consumed=2)
    assert "rebuilding training cuts for epoch 1 (may take a few minutes)" in caplog.text
    assert "fast-forwarding sampler by 2 batches" in caplog.text

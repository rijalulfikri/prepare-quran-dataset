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

import json
import random

import numpy as np
import pytest
from lhotse import CutSet

from quran_zipformer import datamodule as dm
from quran_zipformer.data_prep import _write_cuts


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


def test_worker_augmentation_seed_changes_per_epoch(tiny_data_dir, cfg):
    # the loader is rebuilt every epoch; a constant worker seed would replay the
    # same CutMix / SpecAugment draws each epoch
    cfg.num_workers = 1
    seeds = [dm.train_dataloader(cfg, tiny_data_dir, epoch=e).worker_init_fn.keywords["seed"] for e in (0, 1)]
    assert seeds[0] != seeds[1]
    eval_seed = dm.eval_dataloader(cfg, dm.load_cuts(tiny_data_dir, "dev")).worker_init_fn.keywords["seed"]
    assert eval_seed == cfg.seed


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


# ---------------------------------------------------------------- augmentation (v1)


V1_AUG = dict(
    speed_factor_ranges=[(0.85, 0.95), (1.0, 1.0), (1.05, 1.25)],
    cutmix_snr=(5.0, 20.0),
    reverb_p=0.3,
    volume_p=0.5,
)


def test_defaults_rebuild_the_v0_train_cuts_exactly(tiny_data_dir_big, cfg):
    # v0: [*base.perturb_speed(0.9), *base, *base.perturb_speed(1.1)] shuffled with cfg.seed
    base = dm.load_cuts(tiny_data_dir_big, "train")
    v0 = CutSet.from_cuts([*base.perturb_speed(0.9), *base, *base.perturb_speed(1.1)])
    v0 = v0.shuffle(rng=random.Random(cfg.seed))
    new = dm._train_cuts(cfg, tiny_data_dir_big)
    assert [c.id for c in new] == [c.id for c in v0]
    assert [c.duration for c in new] == [c.duration for c in v0]


def test_defaults_keep_the_v0_cut_transforms(tiny_data_dir, cfg):
    from lhotse.dataset import CutMix

    cfg.num_workers = 0
    ds = dm.train_dataloader(cfg, tiny_data_dir).dataset
    assert [type(t) for t in ds.cut_transforms] == [CutMix]
    assert ds.cut_transforms[0].snr == (10.0, 20.0) and ds.cut_transforms[0].p == 0.5
    off = cfg.model_copy(update={"speed_perturb": False})
    assert [c.id for c in dm._train_cuts(off, tiny_data_dir)] == [c.id for c in dm.load_cuts(tiny_data_dir, "train")]


def test_augmentation_order_reverb_mix_gain(tiny_data_dir, cfg):
    from lhotse.dataset import CutMix

    cfg = cfg.model_copy(update=V1_AUG)
    cfg.num_workers = 0
    cfg.max_duration = 10
    ds = dm.train_dataloader(cfg, tiny_data_dir).dataset
    assert [type(t) for t in ds.cut_transforms] == [dm.RandomReverb, CutMix, dm.RandomGainDb]
    assert ds.cut_transforms[1].snr == (5.0, 20.0)
    eval_ds = dm.eval_dataloader(cfg, dm.load_cuts(tiny_data_dir, "dev")).dataset
    assert eval_ds.cut_transforms == []
    # a batch goes through all three
    _, _, texts, cuts = dm.batch_to_tensors(next(iter(dm.train_dataloader(cfg, tiny_data_dir))))
    assert len(texts) == len(cuts) > 0


def test_gain_is_uniform_in_db_not_linear(tiny_data_dir):
    gain = dm.RandomGainDb(1.0, (-30.0, 6.0), seed=0)
    gain.rng = random.Random(0)
    db = np.array([gain.draw_db() for _ in range(20000)])
    assert db.min() >= -30 and db.max() <= 6
    # uniform in dB: a quarter of the draws below -21 dB (linear 0.125..2 would give none)
    assert abs((db < -21).mean() - 0.25) < 0.02
    assert abs(db.mean() - (-12.0)) < 0.5
    cut = dm.load_cuts(tiny_data_dir, "train")[0]
    out = dm.RandomGainDb(1.0, (-20.0, -20.0), seed=0)(CutSet.from_cuts([cut]))[0]
    np.testing.assert_allclose(out.load_audio(), cut.load_audio() * 0.1, rtol=1e-5, atol=1e-7)
    assert out.id == cut.id  # ids are preserved


def test_transforms_apply_with_probability_p(tiny_data_dir):
    cuts = dm.load_cuts(tiny_data_dir, "train")
    untouched = dm.RandomReverb(0.0, seed=0)(cuts)
    assert [c.recording.transforms for c in untouched] == [c.recording.transforms for c in cuts]
    rev = dm.RandomReverb(1.0, seed=0)(cuts)
    assert all(type(c.recording.transforms[-1]).__name__ == "ReverbWithImpulseResponse" for c in rev)
    assert [c.id for c in rev] == [c.id for c in cuts]


def test_copy_factors_are_deterministic_on_grid_and_in_range(cfg):
    v1 = cfg.model_copy(update=V1_AUG)
    ids = [f"moshaf_6.0__{i}" for i in range(400)]
    factors = [dm.copy_factors(v1, i) for i in ids]
    assert factors == [dm.copy_factors(v1, i) for i in ids]
    slow, mid, fast = zip(*factors)
    assert set(slow) == {0.85, 0.9, 0.95} and set(mid) == {1.0} and set(fast) == {1.05, 1.1, 1.15, 1.2, 1.25}
    assert dm.copy_factors(cfg, "x") == [0.9, 1.0, 1.1]
    assert dm.copy_factors(cfg.model_copy(update={"speed_perturb": False}), "x") == [1.0]


def test_frame_budget_is_checked_per_copy_factor(cfg, make_cut_in_memory):
    # 1 s, 47 labels: (100 - 7) // 2 = 46 frames at 1.0 and (90 - 7) // 2 = 41 at 1.1 are
    # too few, 0.9 gives (111 - 7) // 2 = 52; the cut keeps its slow copy instead of
    # being dropped whole
    cut = make_cut_in_memory("c", 1.0, "بت" * 23 + "ب")
    ok = make_cut_in_memory("ok", 1.0, "ب")
    kept, dropped = dm.speed_copies(cfg, CutSet.from_cuts([cut, ok]))
    assert sorted(c.id for c in kept) == ["c_sp0.9", "ok", "ok_sp0.9", "ok_sp1.1"]
    assert sorted(dropped) == ["c", "c_sp1.1"]


def test_v1_epoch_length_stays_close_to_v0(tiny_data_dir_big, cfg):
    v0 = sum(c.duration for c in dm._train_cuts(cfg, tiny_data_dir_big))
    v1 = sum(c.duration for c in dm._train_cuts(cfg.model_copy(update=V1_AUG), tiny_data_dir_big))
    assert abs(v1 / v0 - 1) < 0.05


@pytest.mark.parametrize("workers, cutmix_p", [(0, 0.0), (2, 0.0), (0, 0.5), (2, 0.5)])
def test_sampler_resume_is_exact_with_v1_augmentation(tiny_data_dir_big, cfg, workers, cutmix_p):
    """Same cuts in every batch after a resume, with v1's mixed speed factors.

    With CutMix on, only the order *inside* a batch may differ: CutMix pads the
    cuts it mixes, the dataset sorts each batch by duration, and CutMix's RNG
    restarts with the resumed loader (true for v0 too; its equal-length test
    cuts hid it). Without CutMix the order is exact as well.
    """
    cfg = cfg.model_copy(update={**V1_AUG, "cutmix_p": cutmix_p})
    cfg.num_workers = workers
    cfg.max_duration = 4
    ref = [_ids(b) for b in dm.train_dataloader(cfg, tiny_data_dir_big, epoch=1)]
    assert len(ref) >= 10
    loader = dm.train_dataloader(cfg, tiny_data_dir_big, epoch=1)
    it = iter(loader)
    consumed = [_ids(next(it)) for _ in range(4)]
    state = loader.sampler.state_dict()
    del it, loader
    resumed = dm.train_dataloader(cfg, tiny_data_dir_big, epoch=1, sampler_state=state, batches_consumed=4)
    got = consumed + [_ids(b) for b in resumed]
    if cutmix_p == 0:
        assert got == ref
    else:
        assert [sorted(b) for b in got] == [sorted(b) for b in ref]


# ---------------------------------------------------------------- consecutive segments joined


def _seg(cut):
    return cut.supervisions[0].custom["segment_index"]


def _parts(cut):
    """Ids of the segments a (possibly joined) cut plays, in order."""
    if type(cut).__name__ != "MixedCut":
        return [cut.id]
    return [t.cut.id for t in cut.tracks if type(t.cut).__name__ == "MonoCut"]


def test_concat_off_is_identity(tiny_data_dir_concat, cfg):
    cuts = dm.load_cuts(tiny_data_dir_concat, "train")
    assert dm.concat_consecutive(cfg, cuts) is cuts


def test_concat_joins_only_adjacent_segments_of_one_sura(tiny_data_dir_concat, cfg):
    cfg = cfg.model_copy(update={"concat_p": 1.0, "concat_segments": (2, 4)})
    base = dm.load_cuts(tiny_data_dir_concat, "train")
    text = {c.id: c.supervisions[0].text for c in base}
    seg = {c.id: (_seg(c) if "segment_index" in c.supervisions[0].custom else None) for c in base}
    out = list(dm.concat_consecutive(cfg, base))
    joined = [c for c in out if type(c).__name__ == "MixedCut"]
    assert joined and len(out) < len(base)
    covered = []
    for c in out:
        ids = _parts(c)
        covered += ids
        if len(ids) > 1:
            assert 2 <= len(ids) <= 4
            keys = [tuple(int(x) for x in seg[i].split(".")) for i in ids]
            assert len({s for s, _ in keys}) == 1  # one sura
            assert [p for _, p in keys] == list(range(keys[0][1], keys[0][1] + len(ids)))  # adjacent
            assert len(c.supervisions) == 1
            sup = c.supervisions[0]
            assert sup.text == "".join(text[i] for i in ids)  # no separator
            assert (sup.start, sup.duration) == (0.0, pytest.approx(c.duration))
            assert sup.custom["num_labels"] == len(sup.text) and sup.custom["concat"] == len(ids)
    assert sorted(covered) == sorted(text)  # every cut once, silence untouched
    assert not any({seg[i] for i in _parts(c)} >= {"001.0007", "001.0009"} for c in joined)


def test_concat_is_deterministic_and_independent_of_manifest_order(tiny_data_dir_concat, cfg):
    cfg = cfg.model_copy(update={"concat_p": 0.5})
    base = dm.load_cuts(tiny_data_dir_concat, "train")
    a = sorted(c.id for c in dm.concat_consecutive(cfg, base))
    b = sorted(c.id for c in dm.concat_consecutive(cfg, base.shuffle(rng=random.Random(3))))
    assert a == b == sorted(c.id for c in dm.concat_consecutive(cfg, base))
    assert a != sorted(c.id for c in dm.concat_consecutive(cfg.model_copy(update={"seed": 7}), base))


def test_concat_gap_adds_silence_between_segments(tiny_data_dir_concat, cfg):
    cfg = cfg.model_copy(update={"concat_p": 1.0, "concat_segments": (2, 2), "concat_gap_seconds": (0.3, 0.3)})
    c = next(c for c in dm.concat_consecutive(cfg, dm.load_cuts(tiny_data_dir_concat, "train"))
             if type(c).__name__ == "MixedCut")
    assert c.duration == pytest.approx(2.3)
    assert c.load_audio().shape[-1] == round(2.3 * 16000)


def test_concat_cuts_go_through_speed_copies_and_the_loader(tiny_data_dir_concat, cfg):
    cfg = cfg.model_copy(update={"concat_p": 0.5, "reverb_p": 0.5, "volume_p": 0.5, "cutmix_p": 0.5,
                                 "speed_factor_ranges": [(0.85, 0.95), (1.0, 1.0), (1.05, 1.25)]})
    cfg.num_workers = 0
    cfg.max_duration = 8
    cuts = dm._train_cuts(cfg, tiny_data_dir_concat)
    assert any("_cat" in c.id and "_sp" in c.id for c in cuts)
    texts = []
    for batch in dm.train_dataloader(cfg, tiny_data_dir_concat, epoch=1):
        feats, lens, t, bcuts = dm.batch_to_tensors(batch)
        assert len(t) == len(bcuts) == feats.shape[0]
        texts += t
    assert max(len(t) for t in texts) > 4  # joined labels reach the batches


@pytest.mark.parametrize("workers", [0, 2])
def test_sampler_resume_is_exact_with_concat(tiny_data_dir_concat, cfg, workers):
    cfg = cfg.model_copy(update={"concat_p": 0.5, "cutmix_p": 0.0})
    cfg.num_workers = workers
    cfg.max_duration = 4
    ref = [_ids(b) for b in dm.train_dataloader(cfg, tiny_data_dir_concat, epoch=1)]
    assert len(ref) >= 6 and any("_cat" in i for b in ref for i in b)
    loader = dm.train_dataloader(cfg, tiny_data_dir_concat, epoch=1)
    it = iter(loader)
    consumed = [_ids(next(it)) for _ in range(3)]
    state = loader.sampler.state_dict()
    del it, loader
    resumed = dm.train_dataloader(cfg, tiny_data_dir_concat, epoch=1, sampler_state=state, batches_consumed=3)
    assert consumed + [_ids(b) for b in resumed] == ref


# ---------------------------------------------------------------- CR-CTC


def test_cr_ctc_moves_specaugment_out_of_the_dataset(tiny_data_dir, cfg):
    cfg = cfg.model_copy(update={"use_cr_ctc": True})
    cfg.num_workers = 0
    assert dm.train_dataloader(cfg, tiny_data_dir).dataset.input_transforms == []
    sa = dm.cr_spec_augment(cfg)
    assert (sa.time_warp_factor, sa.num_frame_masks, sa.max_frames_mask_fraction) == (0, 25, pytest.approx(0.375))


def test_supervision_segments_layout(tiny_data_dir, cfg):
    cfg.num_workers = 0
    cfg.max_duration = 10
    batch = next(iter(dm.train_dataloader(cfg, tiny_data_dir)))
    seg = dm.supervision_segments(batch)
    n = len(batch["supervisions"]["text"])
    assert seg.shape == (n, 3)
    assert seg[:, 0].tolist() == batch["supervisions"]["sequence_idx"].tolist()

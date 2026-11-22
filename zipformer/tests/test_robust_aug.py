"""Augmentation against the weaknesses the robustness audit found in v0:
non-speech around the speech, noisy gaps in joined cuts, clipping, tempo."""

import random

import numpy as np
import pytest
import torch

from quran_zipformer import datamodule as dm
from quran_zipformer import frame_budget as fb

SR = 16000


def _rms(audio, start, end):
    x = audio[..., round(start * SR): round(end * SR)]
    return float(np.sqrt(np.mean(np.square(x))))


def _pad_cfg(cfg, **kw):
    update = {"pad_p": 1.0, "pad_lead_seconds": (1.0, 1.0), "pad_trail_seconds": (0.5, 0.5),
              "nonspeech_noise_p": 1.0, "nonspeech_snr": (30.0, 30.0), **kw}
    return cfg.model_copy(update=update)


def _speech(cuts):
    return [c for c in cuts if not fb._is_silence(c)]


# ---------------------------------------------------------------- non-speech around the speech


def test_pad_off_returns_the_same_cuts(tiny_data_dir, cfg):
    cuts = list(dm.load_cuts(tiny_data_dir, "train"))
    out = dm.pad_non_speech(cfg, cuts, dm.load_cuts(tiny_data_dir, "noise_aug"))
    assert all(a is b for a, b in zip(out, cuts)) and len(out) == len(cuts)


def test_padded_cut_keeps_id_and_text_and_its_supervision_spans_all_of_it(tiny_data_dir, cfg):
    cuts = list(dm.load_cuts(tiny_data_dir, "train"))
    out = dm.pad_non_speech(_pad_cfg(cfg), cuts, dm.load_cuts(tiny_data_dir, "noise_aug"), epoch=1)
    assert [c.id for c in out] == [c.id for c in cuts]
    for before, after in zip(cuts, out):
        if fb._is_silence(before):
            assert after is before  # noise-only cuts stay as they are
            continue
        assert after.duration == pytest.approx(before.duration + 1.5)
        (sup,) = after.supervisions
        assert (sup.start, sup.duration) == (pytest.approx(0.0), pytest.approx(after.duration))
        assert sup.text == before.supervisions[0].text
        assert sup.custom["input_type"] == "speech"


def test_noise_fills_the_stretches_at_the_drawn_snr(tiny_data_dir, cfg):
    cut = _speech(dm.load_cuts(tiny_data_dir, "train"))[0]
    noise = dm.load_cuts(tiny_data_dir, "noise_aug")
    padded = dm.pad_non_speech(_pad_cfg(cfg), [cut], noise, epoch=1)[0]
    audio = padded.load_audio()
    speech = _rms(audio, 1.0, 4.0)
    np.testing.assert_allclose(audio[..., SR: 4 * SR], cut.load_audio(), atol=1e-6)  # speech untouched
    for start, end in ((0.0, 1.0), (4.0, 4.5)):
        assert 20 * np.log10(_rms(audio, start, end) / speech) == pytest.approx(-30.0, abs=1.5)


def test_without_noise_the_stretches_are_silent(tiny_data_dir, cfg):
    cut = _speech(dm.load_cuts(tiny_data_dir, "train"))[0]
    for kw, noise in (({"nonspeech_noise_p": 0.0}, dm.load_cuts(tiny_data_dir, "noise_aug")), ({}, None)):
        audio = dm.pad_non_speech(_pad_cfg(cfg, **kw), [cut], noise, epoch=1)[0].load_audio()
        assert _rms(audio, 0.0, 1.0) == 0.0 and _rms(audio, 4.0, 4.5) == 0.0
        assert _rms(audio, 1.0, 4.0) > 0.01


def test_a_short_noise_cut_sits_next_to_the_speech(tiny_data_dir_concat, cfg):
    # one-second noise cuts, two seconds of lead: silence first, then the noise, then the speech
    cut = _speech(dm.load_cuts(tiny_data_dir_concat, "train"))[0]
    cfg = _pad_cfg(cfg, pad_lead_seconds=(2.0, 2.0), pad_trail_seconds=(0.0, 0.0))
    audio = dm.pad_non_speech(cfg, [cut], dm.load_cuts(tiny_data_dir_concat, "noise_aug"), epoch=1)[0].load_audio()
    assert _rms(audio, 0.0, 0.95) == 0.0
    assert _rms(audio, 1.05, 2.0) > 0


def test_pad_is_redrawn_every_epoch_and_fixed_within_one(tiny_data_dir_big, cfg):
    cuts = list(dm.load_cuts(tiny_data_dir_big, "train"))
    noise = dm.load_cuts(tiny_data_dir_big, "noise_aug")
    cfg = cfg.model_copy(update={"pad_p": 0.5})

    def durations(epoch, order=None):
        out = dm.pad_non_speech(cfg, order or cuts, noise, epoch=epoch)
        return {c.id: c.duration for c in out}

    first = durations(1)
    assert first == durations(1) == durations(1, cuts[::-1])  # resume and cut order do not matter
    assert first != durations(2)
    padded = [i for i, d in first.items() if d > 1.01]
    assert 0.3 < len(padded) / len(_speech(cuts)) < 0.7
    assert all(1.5 <= first[i] <= 1.0 + 4.0 + 2.0 for i in padded)  # lead 0.5-4 s, trail 0-2 s


def test_padded_frames_are_inside_the_trained_length(tiny_data_dir, cfg):
    cfg = _pad_cfg(cfg, speed_perturb=False, cutmix_p=0.0)
    cfg.num_workers = 0
    cfg.max_duration = 20
    lens, texts = [], []
    for batch in dm.train_dataloader(cfg, tiny_data_dir, epoch=1):
        _, feat_lens, t, cuts = dm.batch_to_tensors(batch)
        lens += [n for n, c in zip(feat_lens.tolist(), cuts) if not fb._is_silence(c)]
        texts += t
    assert lens and set(lens) == {450}  # 1 s + 3 s + 0.5 s at 100 frames per second
    assert sorted(t for t in texts if t) == sorted(c.supervisions[0].text for c in _speech(
        dm.load_cuts(tiny_data_dir, "train")))


@pytest.mark.parametrize("workers", [0, 2])
def test_sampler_resume_is_exact_with_pad(tiny_data_dir_big, cfg, workers):
    cfg = cfg.model_copy(update={"pad_p": 0.5, "cutmix_p": 0.0})
    cfg.num_workers = workers
    cfg.max_duration = 6

    def ids(batch):
        return [c.id for c in batch["supervisions"]["cut"]]

    ref = [ids(b) for b in dm.train_dataloader(cfg, tiny_data_dir_big, epoch=2)]
    assert len(ref) >= 8
    loader = dm.train_dataloader(cfg, tiny_data_dir_big, epoch=2)
    it = iter(loader)
    consumed = [ids(next(it)) for _ in range(3)]
    state = loader.sampler.state_dict()
    del it, loader
    resumed = dm.train_dataloader(cfg, tiny_data_dir_big, epoch=2, sampler_state=state, batches_consumed=3)
    assert consumed + [ids(b) for b in resumed] == ref


def test_pad_works_on_joined_and_speed_perturbed_cuts(tiny_data_dir_concat, cfg):
    cfg = _pad_cfg(cfg, concat_p=0.5, concat_gap_seconds=(0.2, 0.6), reverb_p=0.5, volume_p=0.5, clip_p=0.5,
                   speed_factor_ranges=[(0.85, 0.95), (1.0, 1.0), (1.05, 1.25)])
    cfg.num_workers = 0
    cfg.max_duration = 12
    cuts = dm._train_cuts(cfg, tiny_data_dir_concat, noise=dm.load_cuts(tiny_data_dir_concat, "noise_aug"), epoch=1)
    joined = [c for c in cuts if "_cat" in c.id and "_sp" in c.id]
    assert joined and all(len(c.supervisions) == 1 for c in cuts)
    assert all(c.supervisions[0].duration == pytest.approx(c.duration) for c in joined)
    seen = 0
    for batch in dm.train_dataloader(cfg, tiny_data_dir_concat, epoch=1):
        feats, feat_lens, texts, bcuts = dm.batch_to_tensors(batch)
        assert len(texts) == len(bcuts) == feats.shape[0]
        assert torch.isfinite(feats).all()
        seen += len(texts)
    assert seen > 0


# ---------------------------------------------------------------- joined segments


def _first_joined(cfg, data_dir, noise):
    cuts = dm.concat_consecutive(cfg, dm.load_cuts(data_dir, "train"), noise)
    return next(c for c in cuts if type(c).__name__ == "MixedCut")


def test_concat_gap_holds_noise_only_when_noise_cuts_are_given(tiny_data_dir_concat, cfg):
    cfg = cfg.model_copy(update={"concat_p": 1.0, "concat_segments": (2, 2), "concat_gap_seconds": (0.5, 0.5),
                                 "nonspeech_noise_p": 1.0, "nonspeech_snr": (20.0, 20.0)})
    silent = _first_joined(cfg, tiny_data_dir_concat, None)
    noisy = _first_joined(cfg, tiny_data_dir_concat, dm.load_cuts(tiny_data_dir_concat, "noise_aug"))
    assert _rms(silent.load_audio(), 1.02, 1.48) == 0.0
    gap, speech = _rms(noisy.load_audio(), 1.02, 1.48), _rms(noisy.load_audio(), 0.0, 1.0)
    assert 20 * np.log10(gap / speech) == pytest.approx(-20.0, abs=1.5)
    # same cut for the frame budget either way
    assert (silent.id, silent.duration, silent.supervisions[0].text) == (
        noisy.id, noisy.duration, noisy.supervisions[0].text)
    (sup,) = noisy.supervisions
    assert (sup.start, sup.duration) == (0.0, pytest.approx(noisy.duration))


def test_concat_max_seconds_caps_the_joined_speech(tiny_data_dir_concat, cfg):
    base = dm.load_cuts(tiny_data_dir_concat, "train")
    cfg = cfg.model_copy(update={"concat_p": 1.0, "concat_segments": (4, 4)})
    assert max(c.duration for c in dm.concat_consecutive(cfg, base)) == pytest.approx(4.0)
    capped = dm.concat_consecutive(cfg.model_copy(update={"concat_max_seconds": 2.5}), base)
    assert max(c.duration for c in capped) == pytest.approx(2.0)
    assert sum(c.duration for c in capped) == pytest.approx(sum(c.duration for c in base))


# ---------------------------------------------------------------- clipping


def test_clip_at_is_relative_to_the_peak():
    audio = torch.tensor([[0.0, 0.2, -0.8, 0.4]])
    np.testing.assert_allclose(dm.clip_at(audio, 0.5).numpy(), [[0.0, 0.2, -0.4, 0.4]], rtol=1e-6)
    np.testing.assert_allclose(dm.clip_at(audio * 0.01, 0.5).numpy(), [[0.0, 0.002, -0.004, 0.004]], rtol=1e-6)
    assert dm.clip_at(torch.zeros(1, 4), 0.5).abs().sum() == 0
    assert dm.clip_at(np.array([[1.0, -1.0]], dtype="float32"), 0.25).abs().max() == pytest.approx(0.25)


def test_random_clip_applies_with_probability_p():
    audio = torch.linspace(-1, 1, 101)[None]
    assert dm.RandomClip(0.0, (0.1, 0.1), seed=0)(audio) is audio
    assert dm.RandomClip(1.0, (0.1, 0.1), seed=0)(audio).abs().max() == pytest.approx(0.1)
    clip = dm.RandomClip(0.5, (0.05, 0.5), seed=1)
    peaks = [float(clip(audio).abs().max()) for _ in range(400)]
    assert 0.4 < np.mean([p < 0.999 for p in peaks]) < 0.6
    assert min(peaks) >= 0.05 - 1e-6 and max(p for p in peaks if p < 0.999) <= 0.5 + 1e-6


def test_only_the_train_dataset_clips(tiny_data_dir, cfg):
    cfg.num_workers = 0
    dev = dm.load_cuts(tiny_data_dir, "dev")
    assert dm.train_dataloader(cfg, tiny_data_dir).dataset.input_strategy.wave_transforms == []
    on = cfg.model_copy(update={"clip_p": 0.3})
    (tf,) = dm.train_dataloader(on, tiny_data_dir).dataset.input_strategy.wave_transforms
    assert isinstance(tf, dm.RandomClip) and (tf.p, tf.peak_fraction) == (0.3, (0.05, 0.5))
    assert dm.eval_dataloader(on, dev).dataset.input_strategy.wave_transforms == []
    (fixed,) = dm.eval_dataloader(on, dev, wave_transforms=[dm.FixedClip(0.1)]).dataset.input_strategy.wave_transforms
    assert isinstance(fixed, dm.FixedClip)


# ---------------------------------------------------------------- tempo


def test_min_fbank_frames_is_the_inverse_of_output_frames():
    for factor in (1, 2):
        for text in ("ء", "ءَلِف", "ۦۦۦۦ", "ءَلِف" * 9):
            n = fb.min_fbank_frames(text, factor)
            assert fb.output_frames(n, factor) >= fb.required_frames(text)
            assert fb.output_frames(n - 1, factor) < fb.required_frames(text)
        assert fb.min_fbank_frames("", factor) == 0


def test_tempo_stretch_shortens_frames_and_lengths():
    ramp = torch.arange(200, dtype=torch.float32)[None, :, None].repeat(2, 1, 3)
    ramp[1, 120:] = -23.0  # padding after the second cut's 120 frames
    feats, lens = dm.tempo_stretch(ramp, torch.tensor([200, 120], dtype=torch.int32), 1.25)
    assert feats.shape == (2, 160, 3) and lens.tolist() == [160, 96] and lens.dtype == torch.int32
    assert (feats[0].diff(dim=0) > 0).all()  # time order kept, 1.25 input frames per output frame
    assert feats[0, 1, 0] - feats[0, 0, 0] == pytest.approx(1.25)
    assert feats[1, :96].min() >= 0  # no padding blended into the kept frames
    same, same_lens = dm.tempo_stretch(ramp, lens, 1.0)
    assert same is ramp and same_lens is lens


def test_apply_tempo_is_off_by_default_and_capped_by_the_frame_budget(cfg):
    feats = torch.randn(2, 400, 80)
    lens = torch.tensor([400, 300], dtype=torch.int32)
    out, out_lens = dm.apply_tempo(cfg, feats, lens, ["ءَلِف", "ءَ"], random.Random(0))
    assert out is feats and out_lens is lens  # tempo_p = 0
    on = cfg.model_copy(update={"tempo_p": 1.0, "tempo_range": (1.3, 1.3)})
    out, out_lens = dm.apply_tempo(on, feats, lens, ["ءَلِف", "ءَ"], random.Random(0))
    assert out.shape[1] == round(400 / 1.3) and out_lens.tolist() == [out.shape[1], 300 * out.shape[1] // 400]
    # a label needing 2 * 144 + 7 = 295 of the 300 frames allows no speed-up: left alone
    tight = "ءَ" * 72
    out, out_lens = dm.apply_tempo(on, feats, lens, ["ءَ", tight], random.Random(0))
    assert out is feats and out_lens is lens
    # a looser one caps the factor so the label still fits
    loose = "ءَ" * 55  # 227 frames needed of 300: at most 1.32 * 0.97 = 1.28 instead of 1.3
    out, out_lens = dm.apply_tempo(on, feats, lens, ["ءَ", loose], random.Random(0))
    assert round(400 / 1.3) < out.shape[1] < 400
    assert out_lens[1] >= fb.min_fbank_frames(loose, on.output_downsampling_factor)


def test_tempo_draw_follows_the_range_and_probability(cfg):
    on = cfg.model_copy(update={"tempo_p": 0.5, "tempo_range": (1.1, 1.4)})
    feats, lens = torch.zeros(1, 1000, 4), torch.tensor([1000], dtype=torch.int32)
    lengths = [dm.apply_tempo(on, feats, lens, ["ءَ"], random.Random(i))[0].shape[1] for i in range(300)]
    changed = [n for n in lengths if n != 1000]
    assert 0.4 < len(changed) / 300 < 0.6
    assert min(changed) >= round(1000 / 1.4) and max(changed) <= round(1000 / 1.1)

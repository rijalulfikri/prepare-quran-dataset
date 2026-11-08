import inspect
import time

import pytest
import torch

pytest.importorskip("k2")

from prepare_quran_dataset.zipformer import frame_budget as fb
from prepare_quran_dataset.zipformer.config import ZipformerTrainConfig
from prepare_quran_dataset.zipformer.model import (
    Zipformer2,
    Zipformer2OutDS,
    build_model,
    build_params,
    count_params,
)


def test_medium_model_matches_v3_size():
    m = build_model(ZipformerTrainConfig(train_moshaf_ids=["0.0"], test_moshaf_ids=[]), vocab_size=44)
    n = count_params(m)
    assert 63_000_000 < n < 68_000_000
    # icefall: ctc_output = Sequential(Dropout, Linear, LogSoftmax)
    linear = [l for l in m.ctc_output if isinstance(l, torch.nn.Linear)][0]
    assert m.use_ctc and not m.use_transducer and not m.use_attention_decoder
    assert linear.out_features == 44
    assert not hasattr(m, "decoder") and not hasattr(m, "joiner")


def test_build_params_overrides_from_cfg(tiny_cfg_factory):
    cfg = tiny_cfg_factory(causal=False, output_downsampling_factor=2)
    p = build_params(cfg, vocab_size=44)
    assert p.encoder_dim == "32,32" and p.pos_dim == 16 and p.chunk_size == "8"
    assert p.causal is False and p.output_downsampling_factor == 2
    assert p.vocab_size == 44 and p.blank_id == 0 and p.feature_dim == 80
    assert p.use_ctc and not p.use_transducer and not p.use_attention_decoder and not p.use_cr_ctc


@pytest.mark.parametrize("factor", [1, 2])
def test_output_frames_match_frame_budget(tiny_cfg_factory, factor):
    cfg = tiny_cfg_factory(output_downsampling_factor=factor)
    m = build_model(cfg, vocab_size=44).eval()
    assert isinstance(m.encoder, Zipformer2OutDS) and m.encoder.out_ds == factor
    for t_fbank in (107, 108, 200, 333):
        x = torch.randn(1, t_fbank, 80)
        lens = torch.tensor([t_fbank])
        with torch.no_grad():
            out, out_lens = m.forward_encoder(x, lens)
        assert out.shape[1] == out_lens.item() == fb.output_frames(t_fbank, factor)


def test_factor_one_replaces_output_downsample_with_identity(make_tiny_model):
    assert isinstance(make_tiny_model().encoder.downsample_output, torch.nn.Identity)
    assert not isinstance(
        make_tiny_model(output_downsampling_factor=2).encoder.downsample_output, torch.nn.Identity
    )


@pytest.mark.parametrize("factor", [1, 2])
def test_streaming_forward_lengths(make_tiny_model, factor):
    m = make_tiny_model(output_downsampling_factor=factor).eval()
    enc = m.encoder
    chunk = enc.chunk_size[0]  # one chunk of encoder_embed frames, as the exporter feeds
    x = torch.randn(chunk, 1, 32)
    x_lens = torch.tensor([chunk])
    mask = torch.zeros(1, enc.left_context_frames[0] + chunk, dtype=torch.bool)
    with torch.no_grad():
        states = enc.get_init_states(batch_size=1)
        out, out_lens, new_states = enc.streaming_forward(x, x_lens, states, mask)
    expected = x_lens if factor == 1 else (x_lens + 1) // 2
    assert out.shape[0] == out_lens.item() == expected.item()
    assert len(new_states) == len(states)


def test_modules_are_named(tiny_model):
    assert all(m.name == name for name, m in tiny_model.named_modules())


def test_tiny_model_forward_is_cheap(tiny_model):
    m = tiny_model.eval()
    x = torch.randn(1, 200, 80)
    with torch.no_grad():
        m.forward_encoder(x, torch.tensor([200]))  # warm-up
        t0 = time.perf_counter()
        m.forward_encoder(x, torch.tensor([200]))
    assert time.perf_counter() - t0 < 1.0


def test_icefall_still_hardcodes_factor_two():
    # canary: if icefall stops asserting/hard-coding 2, revisit Zipformer2OutDS
    from prepare_quran_dataset.zipformer.model import train

    assert "assert self.output_downsampling_factor == 2" in inspect.getsource(Zipformer2.forward)
    assert "assert self.output_downsampling_factor == 2" in inspect.getsource(
        Zipformer2.streaming_forward
    )
    assert "output_downsampling_factor=2" in inspect.getsource(train.get_encoder_model)

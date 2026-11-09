"""Zipformer2-CTC model construction from `ZipformerTrainConfig`.

Built from icefall's librispeech zipformer recipe, except the encoder is
`Zipformer2OutDS`, which also supports `output_downsampling_factor=1`
(20 ms output frames) without editing icefall.
"""

import argparse
from typing import List, Optional, Tuple

from torch import Tensor, nn

from prepare_quran_dataset.zipformer.config import ZipformerTrainConfig
from prepare_quran_dataset.zipformer.icefall_import import setup_icefall_paths

setup_icefall_paths()

import train  # noqa: E402  (icefall recipe)
from icefall.utils import AttributeDict  # noqa: E402
from model import AsrModel  # noqa: E402  (icefall recipe)
from scaling import ScheduledFloat  # noqa: E402  (icefall recipe)
from zipformer import Zipformer2  # noqa: E402  (icefall recipe)

MODEL_FIELDS = (
    "num_encoder_layers",
    "downsampling_factor",
    "feedforward_dim",
    "num_heads",
    "encoder_dim",
    "query_head_dim",
    "value_head_dim",
    "pos_head_dim",
    "pos_dim",
    "encoder_unmasked_dim",
    "cnn_module_kernel",
    "chunk_size",
    "left_context_frames",
)


class Zipformer2OutDS(Zipformer2):
    """`Zipformer2` with output downsampling factor 1 or 2.

    icefall's `forward`/`streaming_forward` assert factor 2 and halve the
    lengths, so the parent is always built with 2; for factor 1 the output
    downsampler becomes `Identity` and the input lengths are returned as is.
    """

    def __init__(self, output_downsampling_factor: int, **zipformer2_kwargs):
        if output_downsampling_factor not in (1, 2):
            raise ValueError(
                f"output_downsampling_factor must be 1 or 2, got {output_downsampling_factor}"
            )
        super().__init__(output_downsampling_factor=2, **zipformer2_kwargs)
        self.out_ds = output_downsampling_factor
        if output_downsampling_factor == 1:
            self.downsample_output = nn.Identity()

    def forward(
        self,
        x: Tensor,
        x_lens: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        x, lengths = super().forward(x, x_lens, src_key_padding_mask)
        if self.out_ds == 1:
            lengths = x_lens
        return x, lengths

    def streaming_forward(
        self,
        x: Tensor,
        x_lens: Tensor,
        states: List[Tensor],
        src_key_padding_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, List[Tensor]]:
        x, lengths, new_states = super().streaming_forward(
            x, x_lens, states, src_key_padding_mask
        )
        if self.out_ds == 1:
            lengths = x_lens
        return x, lengths, new_states


def _recipe_defaults() -> AttributeDict:
    parser = argparse.ArgumentParser()
    train.add_model_arguments(parser)
    params = train.get_params()
    params.update(vars(parser.parse_args([])))
    return params


def build_params(cfg: ZipformerTrainConfig, vocab_size: int) -> AttributeDict:
    """Recipe params (`get_params()` + model-arg defaults) overridden from `cfg`."""
    params = _recipe_defaults()
    params.update({f: getattr(cfg, f) for f in MODEL_FIELDS})
    params.update(
        use_transducer=False,
        use_ctc=True,
        use_attention_decoder=False,
        use_cr_ctc=False,
        causal=cfg.causal,
        feature_dim=cfg.feature_dim,
        vocab_size=vocab_size,
        blank_id=0,
        output_downsampling_factor=cfg.output_downsampling_factor,
    )
    return params


def build_encoder(params: AttributeDict) -> Zipformer2OutDS:
    """Recipe `get_encoder_model` with our class and output downsampling factor."""
    to_tuple = train._to_int_tuple
    return Zipformer2OutDS(
        output_downsampling_factor=params.output_downsampling_factor,
        downsampling_factor=to_tuple(params.downsampling_factor),
        num_encoder_layers=to_tuple(params.num_encoder_layers),
        encoder_dim=to_tuple(params.encoder_dim),
        encoder_unmasked_dim=to_tuple(params.encoder_unmasked_dim),
        query_head_dim=to_tuple(params.query_head_dim),
        pos_head_dim=to_tuple(params.pos_head_dim),
        value_head_dim=to_tuple(params.value_head_dim),
        pos_dim=params.pos_dim,
        num_heads=to_tuple(params.num_heads),
        feedforward_dim=to_tuple(params.feedforward_dim),
        cnn_module_kernel=to_tuple(params.cnn_module_kernel),
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
        warmup_batches=4000.0,
        causal=params.causal,
        chunk_size=to_tuple(params.chunk_size),
        left_context_frames=to_tuple(params.left_context_frames),
    )


def build_model(cfg: ZipformerTrainConfig, vocab_size: int) -> AsrModel:
    """CTC-only `AsrModel` (no decoder/joiner/attention decoder)."""
    params = build_params(cfg, vocab_size)
    model = AsrModel(
        encoder_embed=train.get_encoder_embed(params),
        encoder=build_encoder(params),
        decoder=None,
        joiner=None,
        attention_decoder=None,
        encoder_dim=max(train._to_int_tuple(params.encoder_dim)),
        decoder_dim=params.decoder_dim,
        vocab_size=vocab_size,
        use_transducer=False,
        use_ctc=True,
        use_attention_decoder=False,
    )
    # as in recipe `run()`: balancers/whiteners log by module name
    for name, m in model.named_modules():
        m.name = name
    return model


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

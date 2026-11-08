"""sherpa-onnx compatible streaming CTC export and an onnxruntime streaming decoder.

The export reuses icefall's `export-onnx-streaming-ctc.py` (`OnnxModel`,
`export_streaming_ctc_model_onnx`, `add_meta_data`), so the ONNX file has the
input/output names and metadata sherpa-onnx expects for `zipformer2` CTC models.
We add `output_downsampling_factor`, `subsampling_factor`, `output_frame_shift_ms`,
`vocab_size` and `blank_id`; `OnnxStreamingDecoder` reads the frame shift from
there instead of assuming 40 ms.
"""

import copy
import functools
import warnings
from pathlib import Path
from types import ModuleType

import numpy as np
import torch

from prepare_quran_dataset.zipformer.decode import (
    greedy_ctc,
    greedy_ctc_with_frames,
    set_streaming_context,
    token_times,
)
from prepare_quran_dataset.zipformer.icefall_import import (
    load_recipe_module,
    setup_icefall_paths,
)
from prepare_quran_dataset.zipformer.tokens import PhonemeTokens

SAMPLE_RATE = 16000
NUM_MEL_BINS = 80
TAIL_PADDING_S = 0.3  # as icefall `onnx_pretrained-streaming-ctc.py`
_SAMPLES_PER_FRAME = SAMPLE_RATE // 100  # 10 ms fbank hop


@functools.cache
def _exporter() -> ModuleType:
    return load_recipe_module("export-onnx-streaming-ctc.py")


def export_streaming_onnx(
    model,
    out_dir: Path,
    tokens: PhonemeTokens,
    chunk_size: int,
    left_context: int,
    output_downsampling_factor: int,
    opset: int = 13,
) -> dict[str, Path]:
    """Export `model` as a streaming CTC ONNX (fp32 + int8) plus `tokens.txt`.

    `chunk_size` / `left_context` are in 50 Hz frames (after `encoder_embed`).
    The export runs on a deep copy (`convert_scaled_to_non_scaled` mutates the
    model in place), so the caller's model is left untouched.
    """
    exporter = _exporter()
    setup_icefall_paths()
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from scaling_converter import convert_scaled_to_non_scaled  # icefall recipe

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32, int8 = out_dir / "model.onnx", out_dir / "model.int8.onnx"

    m = copy.deepcopy(model).cpu().eval()
    set_streaming_context(m, chunk_size, left_context)
    convert_scaled_to_non_scaled(m, inplace=True)
    onnx_model = exporter.OnnxModel(
        encoder=m.encoder, encoder_embed=m.encoder_embed, ctc_output=m.ctc_output
    )

    # The exporter patches `forward` on the encoder *class*; undo it afterwards
    # so every other instance of the class keeps its non-streaming forward.
    enc_cls = type(m.encoder)
    had_own_forward = "forward" in enc_cls.__dict__
    saved_forward = enc_cls.__dict__.get("forward")
    try:
        with warnings.catch_warnings(), torch.no_grad():
            warnings.simplefilter("ignore")
            exporter.export_streaming_ctc_model_onnx(
                onnx_model, str(fp32), opset_version=opset
            )
    finally:
        if had_own_forward:
            enc_cls.forward = saved_forward
        else:
            del enc_cls.forward

    f = output_downsampling_factor
    exporter.add_meta_data(
        str(fp32),
        {
            "output_downsampling_factor": str(f),
            "subsampling_factor": str(2 * f),
            "output_frame_shift_ms": str(20 * f),
            "vocab_size": str(tokens.vocab_size),
            "blank_id": str(tokens.blank_id),
        },
    )

    quantize_dynamic(
        str(fp32),
        str(int8),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul"],
    )
    int8_meta = _metadata(int8)
    missing = {k: v for k, v in _metadata(fp32).items() if int8_meta.get(k) != v}
    if missing:
        exporter.add_meta_data(str(int8), missing)

    tokens_txt = tokens.write_tokens_txt(out_dir / "tokens.txt")
    return {"fp32": fp32, "int8": int8, "tokens": tokens_txt}


def _metadata(onnx_path: Path) -> dict[str, str]:
    import onnx

    return {p.key: p.value for p in onnx.load(str(onnx_path)).metadata_props}


def _padded_features(
    wav: np.ndarray, sr: int, chunk_frames: int, hop: int
) -> torch.Tensor:
    """80-dim kaldi fbank `(n_frames, 80)` of `wav` plus 0.3 s of trailing zeros,
    extended with more zero samples so `(n_frames - chunk_frames) % hop == 0`,
    i.e. windows of `chunk_frames` advancing by `hop` cover every frame.

    `high_freq=-400` matches Lhotse's `FbankConfig` (training) and sherpa-onnx.
    """
    import torchaudio

    w = torch.as_tensor(np.asarray(wav, dtype=np.float32)).flatten()
    if sr != SAMPLE_RATE:
        w = torchaudio.functional.resample(w, sr, SAMPLE_RATE)
    n_samples = w.numel() + int(TAIL_PADDING_S * SAMPLE_RATE)
    # snip_edges=False yields round(n_samples / 160) frames
    n_frames = (n_samples + _SAMPLES_PER_FRAME // 2) // _SAMPLES_PER_FRAME
    n_frames = max(n_frames, chunk_frames)
    n_frames += (-(n_frames - chunk_frames)) % hop
    w = torch.nn.functional.pad(w, (0, n_frames * _SAMPLES_PER_FRAME - w.numel()))
    feats = torchaudio.compliance.kaldi.fbank(
        w[None],
        num_mel_bins=NUM_MEL_BINS,
        dither=0.0,
        snip_edges=False,
        sample_frequency=SAMPLE_RATE,
        high_freq=-400.0,
    )
    assert feats.shape[0] == n_frames, (feats.shape, n_frames)
    return feats


class OnnxStreamingDecoder:
    """Chunk-by-chunk greedy CTC decoding of an exported streaming ONNX model
    (batch 1), carrying every cache state between calls like sherpa-onnx."""

    def __init__(self, onnx_path: Path, blank_id: int = 0):
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        self.blank_id = blank_id
        meta = self.session.get_modelmeta().custom_metadata_map
        if "output_frame_shift_ms" not in meta:
            raise ValueError(
                f"{onnx_path} has no 'output_frame_shift_ms' metadata; "
                "export it with export_streaming_onnx()"
            )
        self.chunk_frames = int(meta["T"])
        self.decode_chunk_len = int(meta["decode_chunk_len"])
        self.frame_shift_s = int(meta["output_frame_shift_ms"]) / 1000
        factor = int(meta["output_downsampling_factor"])
        self.out_frames_per_chunk = self.decode_chunk_len // 2 // factor
        self._state_inputs = [i for i in self.session.get_inputs() if i.name != "x"]
        self._output_names = [o.name for o in self.session.get_outputs()]

    def _init_states(self) -> dict[str, np.ndarray]:
        states = {}
        for inp in self._state_inputs:
            shape = [d if isinstance(d, int) else 1 for d in inp.shape]  # "N" -> 1
            dtype = np.int64 if inp.type == "tensor(int64)" else np.float32
            states[inp.name] = np.zeros(shape, dtype=dtype)
        return states

    def log_probs(self, wav: np.ndarray, sr: int = SAMPLE_RATE) -> torch.Tensor:
        """`(1, T_out, V)` log-probs of all chunks, concatenated along time."""
        feats = _padded_features(
            wav, sr, self.chunk_frames, self.decode_chunk_len
        ).numpy()
        states = self._init_states()
        chunks = []
        last_start = feats.shape[0] - self.chunk_frames
        for start in range(0, last_start + 1, self.decode_chunk_len):
            x = feats[None, start : start + self.chunk_frames]
            out = self.session.run(self._output_names, {"x": x, **states})
            assert out[0].shape[1] == self.out_frames_per_chunk, (
                out[0].shape,
                self.out_frames_per_chunk,
            )
            chunks.append(out[0])
            # outputs: log_probs, then new_<state name> for each state input
            states = {
                name.removeprefix("new_"): v
                for name, v in zip(self._output_names[1:], out[1:])
            }
        return torch.from_numpy(np.concatenate(chunks, axis=1))

    def decode_with_times(
        self, wav: np.ndarray, sr: int = SAMPLE_RATE
    ) -> list[tuple[int, float]]:
        """Greedy `(token, start_seconds)`; repeats collapse across chunk borders."""
        lp = self.log_probs(wav, sr)
        lens = torch.tensor([lp.shape[1]])
        frames = greedy_ctc_with_frames(lp, lens, self.blank_id)[0]
        return token_times(frames, self.frame_shift_s)

    def decode(self, wav: np.ndarray, sr: int = SAMPLE_RATE) -> list[int]:
        return [tok for tok, _ in self.decode_with_times(wav, sr)]


@torch.no_grad()
def assert_onnx_matches_torch(
    model,
    decoder: OnnxStreamingDecoder,
    wavs: list[np.ndarray],
    tokens: PhonemeTokens,
    chunk_size: int,
    left_context: int,
) -> None:
    """Raise `AssertionError` (showing both sequences) unless the ONNX streaming
    decode equals PyTorch's simulated-streaming greedy decode for every wav.

    Both sides use the same padded fbank; the PyTorch log-probs are cut to the
    frames the ONNX chunks cover (the full forward has 3 extra right-context
    frames at 50 Hz). The model's streaming context and train mode are restored.
    """
    encoder = model.encoder
    saved = (encoder.chunk_size, encoder.left_context_frames)
    was_training = model.training
    model.eval()
    set_streaming_context(model, chunk_size, left_context)
    try:
        for i, wav in enumerate(wavs):
            feats = _padded_features(
                wav, SAMPLE_RATE, decoder.chunk_frames, decoder.decode_chunk_len
            )
            enc_out, _ = model.forward_encoder(
                feats[None], torch.tensor([feats.shape[0]])
            )
            onnx_lp = decoder.log_probs(wav)
            n = onnx_lp.shape[1]
            torch_lp = model.ctc_output(enc_out)[:, :n]
            lens = torch.tensor([n])
            want = greedy_ctc(torch_lp, lens, tokens.blank_id)[0]
            got = greedy_ctc(onnx_lp, lens, decoder.blank_id)[0]
            if got != want:
                raise AssertionError(
                    f"wav {i}: ONNX != PyTorch\n  onnx:  {got}\n  torch: {want}"
                )
    finally:
        encoder.chunk_size, encoder.left_context_frames = saved
        model.train(was_training)

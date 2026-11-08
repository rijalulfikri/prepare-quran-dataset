import numpy as np
import pytest

pytest.importorskip("k2")
pytest.importorskip("onnxruntime")

from prepare_quran_dataset.zipformer import export_onnx as ex


@pytest.mark.parametrize("factor", [1, 2])
def test_export_and_streaming_decode_matches_torch(tmp_path, tiny_cfg_factory, tokens, factor):
    from prepare_quran_dataset.zipformer.model import build_model

    model = build_model(tiny_cfg_factory(output_downsampling_factor=factor), 44).eval()
    paths = ex.export_streaming_onnx(
        model, tmp_path, tokens, chunk_size=8, left_context=32, output_downsampling_factor=factor
    )
    assert all(p.exists() for p in paths.values())
    dec = ex.OnnxStreamingDecoder(paths["fp32"], blank_id=0)
    assert dec.out_frames_per_chunk == 8 // factor
    assert dec.frame_shift_s == pytest.approx(0.02 * factor)
    wav = np.random.randn(16000 * 2).astype("float32") * 0.1
    ex.assert_onnx_matches_torch(model, dec, [wav], tokens, chunk_size=8, left_context=32)
    assert isinstance(dec.decode(wav), list)


def test_metadata_survives_int8(tmp_path, tiny_model, tokens):
    import onnxruntime as ort

    paths = ex.export_streaming_onnx(
        tiny_model, tmp_path, tokens, chunk_size=8, left_context=32, output_downsampling_factor=1
    )
    for k in ("fp32", "int8"):
        md = ort.InferenceSession(
            str(paths[k]), providers=["CPUExecutionProvider"]
        ).get_modelmeta().custom_metadata_map
        assert md["model_type"] == "zipformer2"
        assert md["output_frame_shift_ms"] == "20"
        assert md["subsampling_factor"] == "2"


def test_decoder_refuses_model_without_frame_shift(tmp_path, tiny_model, tokens):
    import onnx

    paths = ex.export_streaming_onnx(
        tiny_model, tmp_path, tokens, chunk_size=8, left_context=32, output_downsampling_factor=1
    )
    m = onnx.load(str(paths["fp32"]))
    kept = [p for p in m.metadata_props if p.key != "output_frame_shift_ms"]
    del m.metadata_props[:]
    m.metadata_props.extend(kept)
    stripped = tmp_path / "stripped.onnx"
    onnx.save(m, str(stripped))
    with pytest.raises(ValueError, match="output_frame_shift_ms"):
        ex.OnnxStreamingDecoder(stripped)

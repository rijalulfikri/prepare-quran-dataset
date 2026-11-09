import pytest

from prepare_quran_dataset.zipformer.tokens import PhonemeTokens

VOCAB = "vocab_streaming/vocab.json"


def test_blank_and_size():
    t = PhonemeTokens.from_vocab_json(VOCAB)
    assert t.blank_id == 0 and t.vocab_size == 44


def test_encode_decode_roundtrip():
    t = PhonemeTokens.from_vocab_json(VOCAB)
    s = "بِ"  # ba + kasra
    assert t.encode(s) == [3, 35]
    assert t.decode(t.encode(s)) == s
    assert t.encode("") == []


def test_decode_skips_blank_and_eos():
    t = PhonemeTokens.from_vocab_json(VOCAB)
    assert t.decode([0, 3, 1, 35, 0]) == "بِ"


def test_decode_unknown_id_raises():
    t = PhonemeTokens.from_vocab_json(VOCAB)
    with pytest.raises(KeyError):
        t.decode([999])


def test_unknown_symbol_raises():
    t = PhonemeTokens.from_vocab_json(VOCAB)
    with pytest.raises(KeyError, match="Z"):
        t.encode("Z")


def test_tokens_txt_roundtrip(tmp_path):
    t = PhonemeTokens.from_vocab_json(VOCAB)
    p = t.write_tokens_txt(tmp_path / "tokens.txt")
    lines = p.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "<blk> 0" and lines[1] == "[EOS] 1" and len(lines) == 44
    t2 = PhonemeTokens.from_tokens_txt(p)
    assert t2.encode("ب") == t.encode("ب")
    assert t2.vocab_size == t.vocab_size and t2.blank_id == 0
    assert t2.id_to_sym == t.id_to_sym


def test_fixture(tokens):
    assert tokens.vocab_size == 44

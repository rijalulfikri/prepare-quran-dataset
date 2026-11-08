"""Phoneme token table: vocab.json <-> ids <-> sherpa-onnx tokens.txt."""

import json
from collections.abc import Sequence
from pathlib import Path

PAD = "[PAD]"
BLANK_TOKEN = "<blk>"
_SKIP_IDS = frozenset({0, 1})  # [PAD]/blank and [EOS]


class PhonemeTokens:
    """Per-character phoneme vocabulary; id 0 is the CTC blank (`[PAD]`)."""

    def __init__(self, sym_to_id: dict[str, int]):
        self.sym_to_id = dict(sym_to_id)
        self.id_to_sym = {i: s for s, i in self.sym_to_id.items()}

    @classmethod
    def from_vocab_json(cls, path: str | Path) -> "PhonemeTokens":
        vocab = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(vocab["phonemes"])

    @classmethod
    def from_tokens_txt(cls, path: str | Path) -> "PhonemeTokens":
        sym_to_id: dict[str, int] = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            sym, _, idx = line.rpartition(" ")
            sym_to_id[PAD if sym == BLANK_TOKEN else sym] = int(idx)
        return cls(sym_to_id)

    @property
    def blank_id(self) -> int:
        return self.sym_to_id[PAD]

    @property
    def vocab_size(self) -> int:
        return len(self.sym_to_id)

    def encode(self, phonemes: str) -> list[int]:
        return [self.sym_to_id[ch] for ch in phonemes]

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(self.id_to_sym[i] for i in ids if i not in _SKIP_IDS)

    def write_tokens_txt(self, path: str | Path) -> Path:
        path = Path(path)
        lines = [f"{BLANK_TOKEN} {self.blank_id}"]
        lines += [
            f"{self.id_to_sym[i]} {i}"
            for i in sorted(self.id_to_sym)
            if i != self.blank_id
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

"""
Simple BPE tokenizer built on sentencepiece.

Example:
    # Train fresh
    tok = Tokenizer.train("data/corpus.txt", vocab_size=1024)
    ids = tok.encode("hello world")
    text = tok.decode(ids)

    # Load existing
    tok = Tokenizer.from_pretrained("pretrain/data/sp.model")
    ids = tok.encode("hello world")
    text = tok.decode(ids)
"""

from __future__ import annotations

import sentencepiece as spm
from pathlib import Path


class Tokenizer:
    """
    Wraps a SentencePiece model for fast in-memory encode/decode.

    Load with Tokenizer.from_pretrained(path) or create fresh with train().
    """

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self._sp: spm.SentencePieceProcessor | None = None

    def _ensure_loaded(self) -> spm.SentencePieceProcessor:
        if self._sp is None:
            self._sp = spm.SentencePieceProcessor()
            self._sp.load(str(self.model_path))
        return self._sp

    @property
    def vocab_size(self) -> int:
        return self._ensure_loaded().get_piece_size()

    def encode(self, text: str) -> list[int]:
        """Encode string → list of token IDs."""
        return self._ensure_loaded().encode(text, out_type=int)

    def decode(self, tokens: list[int]) -> str:
        """Decode list of token IDs → string."""
        return self._ensure_loaded().decode(tokens)

    @classmethod
    def from_pretrained(cls, model_path: str | Path) -> "Tokenizer":
        """Load an existing tokenizer from a .model file."""
        return cls(model_path)

    # ------------------------------------------------------------------
    # Class-level helpers that return a ready-to-use Tokenizer
    # ------------------------------------------------------------------

    @classmethod
    def train(
        cls,
        text_path: str | Path,
        vocab_size: int = 500,
        *,
        output_dir: str | Path = ".",
        name: str = "tokenizer",
    ) -> "Tokenizer":
        """
        Train a SentencePiece BPE model on text_path and return a Tokenizer.

        Example:
            tok = Tokenizer.train("data/corpus.txt", vocab_size=1024)
            ids = tok.encode("hello world")
        """
        text_path = Path(text_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_prefix = str(output_dir / name)

        spm.SentencePieceTrainer.train(
            input=str(text_path),
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="bpe",
            pad_id=0,
            unk_id=1,
            bos_id=-1,
            eos_id=-1,
            pad_piece="<pad>",
            unk_piece="<unk>",
            byte_fallback=True,
            add_dummy_prefix=False,
        )

        model_path = f"{model_prefix}.model"
        print(f"[tokenizer] trained — saved model → {model_path}")
        return cls(model_path)

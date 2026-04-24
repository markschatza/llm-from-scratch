"""
Simple BPE tokenizer built on sentencepiece.

Tokenizer class — load once, use in-memory for the rest of the session.
train() still returns a Tokenizer instance so callers don't have to
know about the model path.

Example:
    # Train fresh
    tok = train("data/corpus.txt", vocab_size=1024)
    ids = tok.encode("hello world")
    text = tok.decode(ids)

    # Load existing
    tok = Tokenizer.from_pretrained("pretrain/data/sp.model")
    ids = tok.encode("hello world")
"""

from __future__ import annotations

import json
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


# ------------------------------------------------------------------
# Module-level functions — built on top of Tokenizer for ergonomics
# ------------------------------------------------------------------

_trainer_cache: dict[str, Tokenizer] = {}


def encode(text: str, model_path: str | Path) -> list[int]:
    """Encode string → list of token IDs (loads model each call — prefer Tokenizer class)."""
    return Tokenizer.from_pretrained(model_path).encode(text)


def decode(tokens: list[int], model_path: str | Path) -> str:
    """Decode list of token IDs → string (loads model each call — prefer Tokenizer class)."""
    return Tokenizer.from_pretrained(model_path).decode(tokens)


def encode_file(
    text_path: str | Path,
    output_path: str | Path,
    *,
    model_path: str | Path,
    output_format: str = "bin",
) -> int:
    """
    Encode a text file and save tokens.

    output_format:
      "jsonl" — one JSON list of ints per line  (human-readable)
      "bin"   — raw little-endian uint32 binary (standard pretrain format)
    Returns the total number of tokens written.
    """
    import struct
    import numpy as np

    text_path = Path(text_path)
    output_path = Path(output_path)
    model_path = Path(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tok = Tokenizer.from_pretrained(model_path)
    n_tokens = 0

    if output_format == "jsonl":
        with open(text_path, "r", encoding="utf-8") as fin, \
             open(output_path, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.rstrip("\n")
                if not line:
                    continue
                tokens = tok.encode(line)
                fout.write(json.dumps(tokens) + "\n")
                n_tokens += len(tokens)

    elif output_format == "bin":
        all_tokens: list[int] = []
        with open(text_path, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.rstrip("\n")
                if not line:
                    continue
                all_tokens.extend(tok.encode(line))

        n_tokens = len(all_tokens)
        arr = np.array(all_tokens, dtype=np.uint32)
        arr.tofile(output_path)

    else:
        raise ValueError(f"Unknown output_format={output_format!r}")

    print(f"[tokenizer] encoded {text_path} → {output_path} ({n_tokens:,} tokens)")
    return n_tokens

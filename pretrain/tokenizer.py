"""
Simple BPE tokenizer built on sentencepiece.
train()     — learn a vocab + model from a text file
encode()    — tokenize text → token ids
decode()    — token ids → text
save() / load() — write/read the .model file
"""

from __future__ import annotations

import json
import sentencepiece as spm
from pathlib import Path


def train(
    text_path: str | Path,
    vocab_size: int = 500,
    *,
    output_dir: str | Path = ".",
    name: str = "tokenizer",
) -> dict:
    """
    Train a SentencePiece BPE model on text_path and save to output_dir.

    Returns a dict with vocab_size, n_train_tokens, and file paths.
    """
    text_path = Path(text_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_prefix = str(output_dir / name)

    # SentencePiece train arguments
    spm.SentencePieceTrainer.train(
        input=str(text_path),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        pad_id=0,
        unk_id=1,
        bos_id=-1,   # no BOS
        eos_id=-1,   # no EOS
        pad_piece="<pad>",
        unk_piece="<unk>",
        # byte-fallback: every token is valid UTF-8
        byte_fallback=True,
        add_dummy_prefix=False,
    )

    model_path = f"{model_prefix}.model"
    vocab_path = f"{model_prefix}.vocab"

    # Count training tokens via the trained model
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    n_train_tokens = len(sp.encode(str(text_path), out_type=int))

    meta = {
        "vocab_size": vocab_size,
        "name": name,
        "model_path": model_path,
        "vocab_path": vocab_path,
        "n_train_tokens": n_train_tokens,
    }

    print(f"[tokenizer] trained — vocab size: {vocab_size}, "
          f"train tokens: {n_train_tokens:,}")
    print(f"[tokenizer] saved model → {model_path}")

    return meta


def encode(text: str, model_path: str | Path) -> list[int]:
    """Encode string → list of token IDs."""
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))
    return sp.encode(text, out_type=int)


def decode(tokens: list[int], model_path: str | Path) -> str:
    """Decode list of token IDs → string."""
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))
    return sp.decode(tokens)


def encode_file(
    text_path: str | Path,
    output_path: str | Path,
    *,
    model_path: str | Path,
    output_format: str = "jsonl",
) -> int:
    """
    Encode a text file and save tokens.

    output_format:
      "jsonl" — one JSON list of ints per line  (human-readable)
      "bin"   — raw little-endian uint32 binary (standard pretrain format)
    Returns the total number of tokens written.
    """
    import json, struct, numpy as np

    text_path = Path(text_path)
    output_path = Path(output_path)
    model_path = Path(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))

    n_tokens = 0

    if output_format == "jsonl":
        with open(text_path, "r", encoding="utf-8") as fin, \
             open(output_path, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.rstrip("\n")
                if not line:
                    continue
                tokens = sp.encode(line, out_type=int)
                fout.write(json.dumps(tokens) + "\n")
                n_tokens += len(tokens)

    elif output_format == "bin":
        all_tokens: list[int] = []
        with open(text_path, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.rstrip("\n")
                if not line:
                    continue
                tokens = sp.encode(line, out_type=int)
                all_tokens.extend(tokens)

        n_tokens = len(all_tokens)
        arr = np.array(all_tokens, dtype=np.uint32)
        arr.tofile(output_path)

    else:
        raise ValueError(f"Unknown output_format={output_format!r}")

    print(f"[tokenizer] encoded {text_path} → {output_path} ({n_tokens:,} tokens)")
    return n_tokens

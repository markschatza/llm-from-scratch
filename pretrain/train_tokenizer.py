#!/usr/bin/env python3
"""
Train the tokenizer and encode the corpus.

Usage:
    python pretrain/train_tokenizer.py

Or with options:
    python pretrain/train_tokenizer.py --corpus data/van-life-story.txt --vocab-size 1024
"""

import argparse
import json
from pathlib import Path

# Ensure pretrain/ is on the path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pretrain.tokenizer import train, encode_file


def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer and encode corpus")
    parser.add_argument("--corpus", type=Path,
                        default=Path(__file__).parent.parent / "data" / "van-life-story.txt",
                        help="Path to training text file")
    parser.add_argument("--vocab-size", type=int, default=1024,
                        help="Vocabulary size (default: 1024)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).parent / "data",
                        help="Where to save the model and encoded files")
    parser.add_argument("--format", choices=["jsonl", "bin"], default="bin",
                        help="Output format for encoded corpus")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1 — Train
    meta = train(
        text_path=args.corpus,
        vocab_size=args.vocab_size,
        output_dir=output_dir,
        name="sp",
    )

    # 2 — Encode corpus to the chosen format
    corpus_stem = args.corpus.stem
    ext = "bin" if args.format == "bin" else "jsonl"
    out_path = output_dir / f"train.{corpus_stem}.{ext}"

    n_tokens = encode_file(
        text_path=args.corpus,
        output_path=out_path,
        model_path=meta["model_path"],
        output_format=args.format,
    )

    # 3 — Summary
    summary = {
        **meta,
        "output_format": args.format,
        "encoded_tokens": n_tokens,
        "encoded_path": str(out_path),
    }
    print(f"\n[done] tokenizer trained and corpus encoded.")
    print(f"       model  : {meta['model_path']}")
    print(f"       vocab  : {meta['vocab_path']}")
    print(f"       corpus : {out_path}  ({n_tokens:,} tokens)")

    return summary


if __name__ == "__main__":
    main()

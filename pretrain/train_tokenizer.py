#!/usr/bin/env python3
"""
Train the tokenizer and encode the corpus to disk.

Usage:
    python pretrain/train_tokenizer.py

Or with options:
    python pretrain/train_tokenizer.py --corpus data/van-life-story.txt --vocab-size 1024
"""

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pretrain.tokenizer import Tokenizer


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
    tok = Tokenizer.train(
        text_path=args.corpus,
        vocab_size=args.vocab_size,
        output_dir=output_dir,
        name="sp",
    )

    # 2 — Encode corpus
    corpus_stem = args.corpus.stem
    ext = "bin" if args.format == "bin" else "jsonl"
    out_path = output_dir / f"train.{corpus_stem}.{ext}"

    if args.format == "jsonl":
        n_tokens = 0
        with open(args.corpus, "r", encoding="utf-8") as fin, \
             open(out_path, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.rstrip("\n")
                if not line:
                    continue
                tokens = tok.encode(line)
                fout.write(json.dumps(tokens) + "\n")
                n_tokens += len(tokens)

    elif args.format == "bin":
        import numpy as np
        all_tokens: list[int] = []
        with open(args.corpus, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.rstrip("\n")
                if not line:
                    continue
                all_tokens.extend(tok.encode(line))
        n_tokens = len(all_tokens)
        arr = np.array(all_tokens, dtype=np.uint32)
        arr.tofile(out_path)

    print(f"\n[done] tokenizer trained and corpus encoded.")
    print(f"       model  : {tok.model_path}")
    print(f"       corpus : {out_path}  ({n_tokens:,} tokens)")

    return tok


if __name__ == "__main__":
    main()

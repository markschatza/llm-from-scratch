# LLM from Scratch

Building a language model from the ground up — tokenizer, pretraining pipeline, and transformer architecture.

## Project Structure

```
llm-from-scratch/
├── pretrain/
│   ├── tokenizer.py          # train(), encode(), decode(), encode_file()
│   ├── train_tokenizer.py    # CLI entry point
│   └── data/                 # tokenizer model + encoded corpus (gitignored)
├── data/
│   └── van-life-story.txt    # training corpus
├── main.py                   # placeholder
└── pyproject.toml
```

## Quick Start

### Install dependencies

```bash
uv sync
```

### Train tokenizer + encode corpus

```bash
uv run python pretrain/train_tokenizer.py
```

Options:

```bash
uv run python pretrain/train_tokenizer.py --vocab-size 2048 --format bin
```

### Use the tokenizer

```python
from pretrain.tokenizer import train, encode, decode

# Train on a text file
meta = train("data/my-corpus.txt", vocab_size=1024)

# Encode / decode
tokens = encode("Hello, world!", model_path="pretrain/data/sp.model")
text  = decode(tokens, model_path="pretrain/data/sp.model")
```

## What's Next

- Transformer architecture
- Pretraining loop
- Inference

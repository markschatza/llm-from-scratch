# LLM from Scratch

Building a language model from the ground up — tokenizer, pretraining pipeline, and transformer architecture.

## Project Structure

```
llm-from-scratch/
├── pretrain/
│   ├── tokenizer.py          # train(), encode(), decode(), encode_file()
│   ├── tokenizer.ipynb       # notebook version for Colab
│   ├── train_tokenizer.py    # CLI entry point
│   ├── train_tokenizer.ipynb # Colab tokenizer training workflow
│   └── data/                 # tokenizer model + encoded corpus (gitignored)
├── data/
│   └── van-life-story.txt    # training corpus
├── main.py                   # placeholder
├── main.ipynb                # notebook entry point
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

### Use on Google Colab

Open the matching notebook for the stage you want to run:

- `main.ipynb` - project entry point
- `pretrain/tokenizer.ipynb` - interactive tokenizer class and quick test
- `pretrain/train_tokenizer.ipynb` - train tokenizer and encode the corpus

The notebooks install `sentencepiece` and `numpy` inside Colab and keep outputs under `pretrain/data/`, matching the CLI folder structure.

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

# LLM from Scratch

Building a language model from the ground up — tokenizer, pretraining pipeline, and transformer architecture.

This repo is intentionally notebook-forward for learning, but reusable implementation should live in Python modules and be imported from the notebooks. The main compute target is Google Colab; local runs should stay small enough for CPU smoke tests.

## Project Structure

```
llm-from-scratch/
├── AGENTS.md                # how agents should help build this repo
├── pretrain/
│   ├── tokenizer.py          # Tokenizer class for train/load/encode/decode
│   ├── tokenizer.ipynb       # notebook version for Colab
│   ├── train_tokenizer.py    # CLI entry point
│   ├── train_tokenizer.ipynb # Colab tokenizer training workflow
│   └── data/                 # tokenizer model + encoded corpus (gitignored)
├── data/
│   └── van-life-story.txt    # training corpus
├── references/
│   └── youtube/              # workshop transcripts and source references
├── scripts/
│   └── download_youtube_transcript.py
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

For Colab, start with `pretrain/train_tokenizer.ipynb`. Colab opens one notebook at a time and does not automatically clone the whole repo when opening a notebook from GitHub, so that notebook's first setup cell clones or reuses the repository under `/content/llm-from-scratch`.

Notebook map:

- `main.ipynb` - project entry point
- `pretrain/tokenizer.ipynb` - interactive tokenizer class and quick test
- `pretrain/train_tokenizer.ipynb` - train tokenizer and encode the corpus

The training notebook installs `sentencepiece` and `numpy` inside Colab and keeps outputs under `pretrain/data/`, matching the CLI folder structure.

## Reference Workshop

The current reference workshop is:

- [Building LLMs from the Ground Up: A 3-hour Coding Workshop](https://www.youtube.com/watch?v=quh7z1q7-uc)
- Transcript: `references/youtube/quh7z1q7-uc/transcript.en-US.md`

To refresh the transcript locally, make sure `yt-dlp` is available and run:

```bash
uv run python scripts/download_youtube_transcript.py "https://www.youtube.com/watch?v=quh7z1q7-uc" --title "Building LLMs from the Ground Up: A 3-hour Coding Workshop"
```

### Use the tokenizer

```python
from pretrain.tokenizer import Tokenizer

# Train on a text file
tok = Tokenizer.train("data/my-corpus.txt", vocab_size=1024, output_dir="pretrain/data", name="sp")

# Encode / decode
tokens = tok.encode("Hello, world!")
text = tok.decode(tokens)
```

## What's Next

- Transformer architecture
- Pretraining loop
- Inference

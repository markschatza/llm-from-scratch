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
│   ├── batching.py           # tensor batching helpers for next-token prediction
│   ├── batching.ipynb        # walkthrough for x/y training batches
│   ├── transformer.py        # small GPT-style Transformer model
│   ├── transformer.ipynb     # forward pass, loss, tiny training loop
│   ├── train_tokenizer.py    # CLI entry point
│   ├── train_tokenizer.ipynb # Colab tokenizer training workflow
│   └── data/                 # tokenizer model + encoded corpus (gitignored)
├── model_inspection/
│   ├── gemma_explorer.py     # Hugging Face Gemma loading + inspection helpers
│   └── gemma_4_explorer.ipynb # load, chat with, and inspect Gemma 4 weights
├── data/
│   └── van-life-story.txt    # training corpus
├── references/
│   └── youtube/              # workshop transcripts and source references
├── scripts/
│   └── download_youtube_transcript.py
├── colab_setup.ipynb         # optional one-time setup notebook per Colab runtime
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

For Colab, the smooth path is:

1. Open `colab_setup.ipynb` first in a fresh runtime.
2. Run it once to clone or update the repo under `/content/llm-from-scratch` and install the current dependencies.
3. Open the lesson notebook you want and connect it to the same runtime.
4. Run the small bootstrap cell at the top of the lesson notebook.

The per-notebook bootstrap cell is intentionally still present. If Colab gives a notebook a fresh runtime, that cell can install dependencies and clone the repo by itself. If you already ran `colab_setup.ipynb` in the same runtime, it just reuses the existing setup.

Notebook map:

- `colab_setup.ipynb` - one-time setup for a fresh Colab runtime
- `main.ipynb` - project entry point
- `pretrain/tokenizer.ipynb` - interactive tokenizer class and quick test
- `pretrain/batching.ipynb` - turn token IDs into `x` and `y` tensors
- `pretrain/transformer.ipynb` - instantiate a tiny GPT-style Transformer
- `model_inspection/gemma_4_explorer.ipynb` - load a Gemma 4 checkpoint, chat with it, and inspect vocabulary, embedding, and attention weight shapes
- `pretrain/train_tokenizer.ipynb` - train tokenizer and encode the corpus

The notebooks install `sentencepiece`, `numpy`, and `torch` inside Colab and keep generated outputs under `pretrain/data/`, matching the CLI folder structure. The Gemma explorer notebook also installs `transformers>=5.5,<5.6`, `accelerate`, `safetensors`, `huggingface-hub`, and `pillow`.

On local Windows AMD GPU runs, keep the AMD PyTorch wheel installed manually in `.venv`. A plain `uv sync` can replace it with the normal PyPI CPU wheel; use `uv run --no-sync` or `.venv\Scripts\python.exe` after repairing the AMD wheel.

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

- Longer training loop with evaluation checkpoints
- Text generation quality checks
- Inference

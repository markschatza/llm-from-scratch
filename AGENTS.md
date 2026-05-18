# Agent Guide

This repo is for learning how to build an LLM from scratch, step by step. The user does not want to write the implementation code manually; the agent should do the coding while keeping the work understandable and notebook-driven.

## Working Style

- Build incrementally. Each change should preserve the learning path and avoid skipping important concepts.
- Prefer notebooks for walkthroughs and experiments, especially when the result will run in Google Colab.
- Put reusable logic in `.py` modules and call it from notebooks. Notebooks should explain and exercise the code; they should not become the only implementation.
- Keep each stage small enough to inspect: tokenizer, dataset preparation, batching, model blocks, training loop, evaluation, and inference.
- Use concrete checks after changes: run scripts when possible, validate notebooks structurally, and keep generated artifacts out of git unless they are intentionally small references.

## Project Assumptions

- Primary compute target: Google Colab.
- Local machine: Windows with an AMD GPU, so do not assume CUDA training is available locally.
- Local development should still support CPU smoke tests for small examples.
- Python best practices matter: type hints where useful, focused functions, module-level APIs, and simple CLI entry points for repeatable steps.
- Follow the reference workshop listed under `references/youtube/` without blindly copying structure. The repo should grow in a way that makes each concept learnable.

## Repo Conventions

- `pretrain/` contains tokenizer, data preparation, model, and training code for the pretraining path.
- `data/` contains small hand-curated input text that is safe to commit.
- `references/` contains source material used while learning, such as transcripts or notes.
- `scripts/` contains repo maintenance helpers that are not part of the model package itself.
- Generated model files, tokenized corpora, checkpoints, and large data files should stay ignored.

## How To Help Here

When asked to add the next piece, first inspect the current notebooks and modules, then implement the smallest durable step:

1. Add or update the Python module API.
2. Add or update the matching notebook cells for Colab usage.
3. Run a local CPU-sized smoke test.
4. Update README or reference notes only when the workflow changes.

Do not turn this into a polished framework too early. The goal is to learn the mechanics by building the parts in order.

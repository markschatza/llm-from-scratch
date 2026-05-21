"""Utilities for turning token IDs into language-model training batches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class BatchConfig:
    """Controls how many examples we sample and how long each example is."""

    batch_size: int = 4
    context_length: int = 8
    device: str = "cpu"


def load_token_ids(path: str | Path, *, dtype: np.dtype = np.uint32) -> torch.Tensor:
    """Load a flat `.bin` file of token IDs as a 1D int64 tensor."""
    token_path = Path(path)
    if not token_path.exists():
        raise FileNotFoundError(f"Token file does not exist: {token_path}")

    tokens = np.fromfile(token_path, dtype=dtype)
    if tokens.size == 0:
        raise ValueError(f"Token file is empty: {token_path}")

    return torch.from_numpy(tokens.astype(np.int64))


def split_tokens(
    tokens: torch.Tensor,
    *,
    train_fraction: float = 0.9,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a 1D token tensor into train and validation regions."""
    if tokens.ndim != 1:
        raise ValueError(f"Expected a 1D token tensor, got shape {tuple(tokens.shape)}")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")

    split_idx = int(len(tokens) * train_fraction)
    if split_idx < 2 or len(tokens) - split_idx < 2:
        raise ValueError("Not enough tokens to make non-empty train and validation splits")

    return tokens[:split_idx], tokens[split_idx:]


def get_batch(
    tokens: torch.Tensor,
    config: BatchConfig,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a batch of input and target tensors for next-token prediction.

    If a row in `x` is tokens[i:i+context_length], the matching row in `y` is
    tokens[i+1:i+context_length+1].
    """
    if tokens.ndim != 1:
        raise ValueError(f"Expected a 1D token tensor, got shape {tuple(tokens.shape)}")
    if config.batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if config.context_length < 1:
        raise ValueError("context_length must be at least 1")

    max_start = len(tokens) - config.context_length - 1
    if max_start < 1:
        raise ValueError(
            "Not enough tokens for the requested context_length. "
            f"Need at least {config.context_length + 2}, got {len(tokens)}."
        )

    starts = torch.randint(
        low=0,
        high=max_start + 1,
        size=(config.batch_size,),
        generator=generator,
    ).to(tokens.device)

    offsets = torch.arange(config.context_length + 1, device=tokens.device)
    batch = tokens[starts[:, None] + offsets[None, :]]
    x = batch[:, :-1]
    y = batch[:, 1:]
    return x.to(config.device), y.to(config.device)

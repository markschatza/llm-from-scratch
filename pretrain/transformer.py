"""A small GPT-style Transformer for next-token prediction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TransformerConfig:
    """Hyperparameters for a decoder-only Transformer language model."""

    vocab_size: int
    context_length: int
    n_embd: int = 64
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    """Multi-head masked self-attention."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.context_length, config.context_length)).view(
                1, 1, config.context_length, config.context_length
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, n_embd = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(n_embd, dim=2)

        q = q.view(batch_size, seq_len, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_size).transpose(1, 2)

        weights = q @ k.transpose(-2, -1) * (self.head_size**-0.5)
        weights = weights.masked_fill(self.causal_mask[:, :, :seq_len, :seq_len] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)
        weights = self.attn_dropout(weights)

        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, n_embd)
        out = self.resid_dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """Position-wise MLP used after attention in each block."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """One attention plus feed-forward block with residual connections."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffwd = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    """Decoder-only Transformer that predicts the next token at every position."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.context_length, config.n_embd)
        self.blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len = idx.shape
        if seq_len > self.config.context_length:
            raise ValueError(
                f"Cannot forward sequence of length {seq_len}; "
                f"context_length is {self.config.context_length}."
            )

        positions = torch.arange(seq_len, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(batch_size * seq_len, self.config.vocab_size),
                targets.view(batch_size * seq_len),
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Generate token IDs by repeatedly sampling the next-token distribution."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.context_length :]
            logits, _ = self(idx_cond)
            next_token_logits = logits[:, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx

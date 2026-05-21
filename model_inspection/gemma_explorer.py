"""Helpers for loading and inspecting Gemma-family Hugging Face models.

This module intentionally keeps the heavy model work behind functions so the
notebook can explain the ideas without turning into a pile of utility code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


DEFAULT_MODEL_ID = "google/gemma-4-E2B-it"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "hf_cache"
_MODEL_CACHE: dict[tuple[Any, ...], "GemmaChatModel"] = {}


@dataclass(frozen=True)
class GenerationSettings:
    """Small set of knobs for interactive text generation."""

    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = True


class GemmaChatModel:
    """Thin wrapper around a Hugging Face Gemma model and processor."""

    def __init__(self, model: Any, processor: Any):
        self.model = model
        self.processor = processor

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    @torch.no_grad()
    def reply(
        self,
        prompt: str,
        system_prompt: str | None = None,
        settings: GenerationSettings | None = None,
    ) -> str:
        """Generate one assistant reply from a plain text prompt."""
        settings = settings or GenerationSettings()
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[-1]
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
            top_p=settings.top_p,
            do_sample=settings.do_sample,
        )
        return self.processor.decode(outputs[0][input_len:], skip_special_tokens=True).strip()


def choose_dtype() -> torch.dtype:
    """Use bfloat16 on an accelerator, otherwise float32 for CPU inspection."""
    if torch.cuda.is_available():
        return torch.bfloat16
    return torch.float32


def _read_dotenv_token(env_path: Path | None = None) -> str | None:
    """Read HF_TOKEN from a local .env file without requiring python-dotenv."""
    import os

    env_path = env_path or Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return None

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip() in {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}:
            token = value.strip().strip('"').strip("'")
            if token:
                os.environ.setdefault("HF_TOKEN", token)
                os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
                return token
    return None


def resolve_hf_token(token: str | None = None) -> str | None:
    """Resolve a Hugging Face token from an explicit arg, env, or local .env."""
    import os

    return (
        token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or _read_dotenv_token()
    )


def get_hf_cache_dir(cache_dir: str | Path | None = None) -> Path:
    """Return the disk cache used for Hugging Face model files."""
    return Path(cache_dir).expanduser().resolve() if cache_dir else DEFAULT_CACHE_DIR


def load_gemma_config(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    token: str | None = None,
    cache_dir: str | Path | None = None,
) -> Any:
    """Load only the small Hugging Face config file using the shared cache."""
    from transformers import AutoConfig

    resolved_cache_dir = get_hf_cache_dir(cache_dir)
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)
    return AutoConfig.from_pretrained(
        model_id,
        token=resolve_hf_token(token),
        cache_dir=str(resolved_cache_dir),
    )


def load_gemma_chat_model(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    dtype: torch.dtype | str = "auto",
    device_map: str | dict[str, Any] = "auto",
    attn_implementation: str = "sdpa",
    load_in_4bit: bool = False,
    token: str | None = None,
    cache_dir: str | Path | None = None,
    reuse_loaded: bool = True,
) -> GemmaChatModel:
    """Load a Gemma chat model through Hugging Face Transformers.

    `load_in_4bit=True` is useful on NVIDIA/CUDA setups with bitsandbytes. On
    the user's AMD Windows ROCm setup, start with `False` and the E2B model.
    """
    from transformers import AutoModelForCausalLM, AutoProcessor

    torch_dtype = choose_dtype() if dtype == "auto" else dtype
    quantization_config = None
    resolved_token = resolve_hf_token(token)
    resolved_cache_dir = get_hf_cache_dir(cache_dir)
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)

    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        torch_dtype = None

    cache_key = (
        model_id,
        str(torch_dtype),
        str(device_map),
        attn_implementation,
        load_in_4bit,
        str(resolved_cache_dir),
        bool(resolved_token),
    )
    if reuse_loaded and cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    processor = AutoProcessor.from_pretrained(
        model_id,
        padding_side="left",
        token=resolved_token,
        cache_dir=str(resolved_cache_dir),
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        quantization_config=quantization_config,
        token=resolved_token,
        cache_dir=str(resolved_cache_dir),
    )
    model.eval()
    chat_model = GemmaChatModel(model=model, processor=processor)
    if reuse_loaded:
        _MODEL_CACHE[cache_key] = chat_model
    return chat_model


def summarize_config(config: Any) -> dict[str, Any]:
    """Pull the architecture fields that matter most for a first inspection."""
    text_config = getattr(config, "text_config", config)
    hidden_size = getattr(text_config, "hidden_size", None)
    num_heads = getattr(text_config, "num_attention_heads", None)
    head_dim = getattr(text_config, "head_dim", None)
    if head_dim is None and hidden_size is not None and num_heads:
        head_dim = hidden_size // num_heads

    return {
        "model_type": getattr(config, "model_type", None),
        "text_model_type": getattr(text_config, "model_type", None),
        "vocab_size": getattr(text_config, "vocab_size", None),
        "hidden_size": hidden_size,
        "intermediate_size": getattr(text_config, "intermediate_size", None),
        "num_hidden_layers": getattr(text_config, "num_hidden_layers", None),
        "num_attention_heads": num_heads,
        "num_key_value_heads": getattr(text_config, "num_key_value_heads", None),
        "head_dim": head_dim,
        "max_position_embeddings": getattr(text_config, "max_position_embeddings", None),
        "sliding_window": getattr(text_config, "sliding_window", None),
        "rope_theta": getattr(text_config, "rope_theta", None),
        "hidden_activation": getattr(text_config, "hidden_activation", None),
    }


def parameter_summary(model: torch.nn.Module) -> dict[str, Any]:
    """Count parameters and show how much has landed on each device/dtype."""
    total = 0
    trainable = 0
    by_dtype: dict[str, int] = {}
    by_device: dict[str, int] = {}

    for param in model.parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
        by_dtype[str(param.dtype)] = by_dtype.get(str(param.dtype), 0) + count
        by_device[str(param.device)] = by_device.get(str(param.device), 0) + count

    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "by_dtype": by_dtype,
        "by_device": by_device,
    }


def find_weight_shapes(
    model: torch.nn.Module,
    name_fragments: tuple[str, ...],
    *,
    limit: int = 40,
) -> list[dict[str, Any]]:
    """Return named parameter shapes whose names contain any fragment."""
    rows: list[dict[str, Any]] = []
    for name, param in model.named_parameters():
        if any(fragment in name for fragment in name_fragments):
            rows.append(
                {
                    "name": name,
                    "shape": tuple(param.shape),
                    "dtype": str(param.dtype),
                    "device": str(param.device),
                    "parameters": param.numel(),
                }
            )
            if len(rows) >= limit:
                break
    return rows


def attention_projection_shapes(model: torch.nn.Module, *, limit: int = 64) -> list[dict[str, Any]]:
    """Find the Q/K/V/O projection matrices used by attention layers."""
    return find_weight_shapes(
        model,
        ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight", "qkv.weight"),
        limit=limit,
    )


def embedding_shapes(model: torch.nn.Module) -> list[dict[str, Any]]:
    """Find token embedding and language-model-head matrices."""
    return find_weight_shapes(
        model,
        ("embed_tokens", "embed_tokens_per_layer", "lm_head.weight"),
        limit=20,
    )

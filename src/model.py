"""Trainable temporal pooling and classification components for turn detection."""
from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor, nn


class MeanPooling(nn.Module):
    """Average-pool across the time axis, masking padded frames when provided."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(f"Expected [batch, time, hidden], got {tuple(hidden_states.shape)}")
        if attention_mask is not None:
            if attention_mask.shape != hidden_states.shape[:2]:
                raise ValueError("attention_mask must have shape [batch, time]")
            valid = attention_mask.to(dtype=torch.bool, device=hidden_states.device)
            masked = hidden_states.masked_fill(~valid.unsqueeze(-1), 0.0)
            denom = valid.sum(dim=1, keepdim=True).clamp_min(1)
            return masked.sum(dim=1) / denom
        return hidden_states.mean(dim=1)


class MaxPooling(nn.Module):
    """Max-pool across the time axis, masking padded frames when provided."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(f"Expected [batch, time, hidden], got {tuple(hidden_states.shape)}")
        if attention_mask is not None:
            if attention_mask.shape != hidden_states.shape[:2]:
                raise ValueError("attention_mask must have shape [batch, time]")
            valid = attention_mask.to(dtype=torch.bool, device=hidden_states.device)
            masked = hidden_states.masked_fill(~valid.unsqueeze(-1), torch.finfo(hidden_states.dtype).min)
            return masked.max(dim=1).values
        return hidden_states.max(dim=1).values


class LastFramePooling(nn.Module):
    """Select the last valid frame, ignoring padding."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(f"Expected [batch, time, hidden], got {tuple(hidden_states.shape)}")
        if attention_mask is None:
            return hidden_states[:, -1, :]
        if attention_mask.shape != hidden_states.shape[:2]:
            raise ValueError("attention_mask must have shape [batch, time]")
        valid = attention_mask.to(dtype=torch.bool, device=hidden_states.device)
        if not valid.any(dim=1).all():
            raise ValueError("Each item must have at least one unmasked frame")
        last_index = valid.long().sum(dim=1) - 1
        batch_index = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        return hidden_states[batch_index, last_index, :]


class AttentionPooling(nn.Module):
    """Mask-aware learned weighted pooling for encoder states shaped ``[B, T, H]``."""

    def __init__(self, hidden_size: int, attention_size: int | None = None) -> None:
        super().__init__()
        attention_size = attention_size or max(32, hidden_size // 2)
        self.score = nn.Sequential(
            nn.Linear(hidden_size, attention_size), nn.Tanh(), nn.Linear(attention_size, 1)
        )

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(f"Expected [batch, time, hidden], got {tuple(hidden_states.shape)}")
        scores = self.score(hidden_states).squeeze(-1)
        if attention_mask is not None:
            if attention_mask.shape != scores.shape:
                raise ValueError("attention_mask must have shape [batch, time]")
            valid = attention_mask.to(dtype=torch.bool, device=scores.device)
            if not valid.any(dim=1).all():
                raise ValueError("Each item must have at least one unmasked frame")
            scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        return torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)


class TurnMLPHead(nn.Module):
    """Small task-specific binary END-logit head."""

    def __init__(self, hidden_size: int = 384, widths: Iterable[int] = (128,), dropout: float = 0.1) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_features = hidden_size
        for width in widths:
            layers += [nn.Linear(in_features, width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(dropout)]
            in_features = width
        layers.append(nn.Linear(in_features, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, embeddings: Tensor) -> Tensor:
        return self.network(embeddings).squeeze(-1)


class TemporalTurnClassifier(nn.Module):
    """Temporal pooling followed by a compact MLP; accepts frame-level encoder states."""

    def __init__(self, hidden_size: int = 384, widths: Iterable[int] = (128,), dropout: float = 0.1, pooling: str = "attention") -> None:
        super().__init__()
        poolers = {
            "mean": MeanPooling(),
            "max": MaxPooling(),
            "last": LastFramePooling(),
            "attention": AttentionPooling(hidden_size),
        }
        if pooling not in poolers:
            raise ValueError(f"Unsupported pooling type: {pooling}")
        self.pooling = poolers[pooling]
        self.head = TurnMLPHead(hidden_size, widths, dropout)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        return self.head(self.pooling(hidden_states, attention_mask))


def configure_whisper_finetuning(encoder: nn.Module, mode: str = "frozen", final_layers: int = 4) -> int:
    """Freeze/unfreeze a Whisper encoder reproducibly and return trainable parameter count."""
    if mode == "frozen":
        for parameter in encoder.parameters():
            parameter.requires_grad = False
    elif mode == "full":
        for parameter in encoder.parameters():
            parameter.requires_grad = True
    elif mode == "partial":
        for parameter in encoder.parameters():
            parameter.requires_grad = False
        layers = getattr(getattr(encoder, "encoder", encoder), "layers", None)
        if layers is None:
            raise ValueError("Could not locate encoder layers for partial fine-tuning")
        for layer in list(layers)[-final_layers:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
    else:
        raise ValueError("mode must be frozen, partial, or full")
    return sum(p.numel() for p in encoder.parameters() if p.requires_grad)

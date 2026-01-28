from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


def masked_mean_pool(last_hidden_state: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if attention_mask is None:
        return last_hidden_state.mean(dim=1)
    mask = attention_mask.to(dtype=last_hidden_state.dtype).unsqueeze(-1)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return summed / denom


class MLPHead(nn.Module):
    def __init__(self, d_model: int, num_labels: int = 2, hidden_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        return self.fc2(x)


@dataclass(frozen=True)
class ForwardOutput:
    logits: torch.Tensor
    pooled: torch.Tensor


class FrozenBackboneClassifier(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        adapter: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.adapter = adapter
        self.head = head

    @torch.no_grad()
    def freeze_backbone_(self) -> "FrozenBackboneClassifier":
        for p in self.backbone.parameters():
            p.requires_grad = False
        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> ForwardOutput:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs["last_hidden_state"]
        pooled = masked_mean_pool(last_hidden_state, outputs.get("attention_mask", attention_mask))
        if self.adapter is not None:
            pooled = self.adapter(pooled)
        logits = self.head(pooled)
        return ForwardOutput(logits=logits, pooled=pooled)

    def trainable_state_dict(self) -> Dict[str, Dict[str, torch.Tensor]]:
        out: Dict[str, Dict[str, torch.Tensor]] = {"head": self.head.state_dict()}
        if self.adapter is not None:
            out["adapter"] = self.adapter.state_dict()
        return out


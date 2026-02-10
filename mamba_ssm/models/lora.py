from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LoRAConfig:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target: Tuple[str, ...] = ("in_proj",)

    def scaling(self) -> float:
        return float(self.alpha) / float(max(int(self.r), 1))

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @staticmethod
    def from_json_file(path: str | Path) -> "LoRAConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        target = data.get("target", ("in_proj",))
        if isinstance(target, list):
            target = tuple(str(x) for x in target)
        return LoRAConfig(
            r=int(data.get("r", 8)),
            alpha=int(data.get("alpha", 16)),
            dropout=float(data.get("dropout", 0.05)),
            target=tuple(target),
        )


class LoRALinear(nn.Module):
    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        bias: bool,
        r: int,
        alpha: int,
        dropout: float,
        device=None,
        dtype=None,
        lora_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        lora_factory_kwargs = {"device": device, "dtype": lora_dtype}
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.r = int(r)
        self.alpha = int(alpha)
        self.dropout = float(dropout)

        self.weight = nn.Parameter(torch.empty((self.out_features, self.in_features), **factory_kwargs))
        self.bias = nn.Parameter(torch.empty((self.out_features,), **factory_kwargs)) if bias else None

        if self.r > 0:
            self.lora_A = nn.Parameter(torch.empty((self.r, self.in_features), **lora_factory_kwargs))
            self.lora_B = nn.Parameter(torch.zeros((self.out_features, self.r), **lora_factory_kwargs))
        else:
            self.lora_A = None
            self.lora_B = None

        self.reset_parameters()

    @staticmethod
    def from_linear(linear: nn.Linear, cfg: LoRAConfig) -> "LoRALinear":
        out = LoRALinear(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            r=cfg.r,
            alpha=cfg.alpha,
            dropout=cfg.dropout,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
            lora_dtype=torch.float32,
        )
        with torch.no_grad():
            out.weight.copy_(linear.weight)
            if out.bias is not None and linear.bias is not None:
                out.bias.copy_(linear.bias)
        out.weight.requires_grad = False
        if out.bias is not None:
            out.bias.requires_grad = False
        return out

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / fan_in**0.5 if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        if self.lora_A is not None:
            nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        if self.lora_A is None or self.lora_B is None:
            return y
        x_d = F.dropout(x, p=self.dropout, training=self.training)
        x_l = x_d.to(dtype=self.lora_A.dtype)
        delta = F.linear(F.linear(x_l, self.lora_A, bias=None), self.lora_B, bias=None)
        delta = delta.to(dtype=y.dtype)
        return y + delta * (float(self.alpha) / float(max(int(self.r), 1)))


def inject_lora(model: nn.Module, cfg: LoRAConfig, *, module_name_allowlist: Optional[Iterable[str]] = None) -> List[str]:
    targets = tuple(str(x) for x in cfg.target)
    allow = None if module_name_allowlist is None else set(str(x) for x in module_name_allowlist)
    replaced: List[str] = []

    def matches(full_name: str) -> bool:
        if allow is not None and full_name not in allow:
            return False
        return any(full_name.endswith(t) for t in targets)

    def walk(parent: nn.Module, prefix: str) -> None:
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear) and matches(full_name):
                setattr(parent, child_name, LoRALinear.from_linear(child, cfg))
                replaced.append(full_name)
                continue
            walk(child, full_name)

    walk(model, "")
    return replaced


def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    sd = model.state_dict()
    return {k: v for k, v in sd.items() if k.endswith(".lora_A") or k.endswith(".lora_B")}


def load_lora_state_dict(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    current = model.state_dict()
    with torch.no_grad():
        for k, v in state.items():
            if k not in current:
                continue
            current[k].copy_(v.to(device=current[k].device, dtype=current[k].dtype))


def trainable_parameters(model: nn.Module) -> List[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def count_parameters(params: Iterable[nn.Parameter]) -> int:
    return int(sum(int(p.numel()) for p in params))

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn

from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.ops.triton.layernorm_gated import RMSNorm


@dataclass(frozen=True)
class Mamba2BackboneConfig:
    d_model: int
    n_layer: int
    vocab_size: int
    ssm_cfg: Dict[str, Any]
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8

    @staticmethod
    def from_json_file(path: str | Path) -> "Mamba2BackboneConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return Mamba2BackboneConfig(
            d_model=int(data["d_model"]),
            n_layer=int(data["n_layer"]),
            vocab_size=int(data["vocab_size"]),
            ssm_cfg=dict(data.get("ssm_cfg", {})),
            rms_norm=bool(data.get("rms_norm", True)),
            residual_in_fp32=bool(data.get("residual_in_fp32", True)),
            fused_add_norm=bool(data.get("fused_add_norm", True)),
            pad_vocab_size_multiple=int(data.get("pad_vocab_size_multiple", 8)),
        )

    def padded_vocab_size(self) -> int:
        m = max(int(self.pad_vocab_size_multiple), 1)
        return int(math.ceil(self.vocab_size / m) * m)


class Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        ssm_cfg: Optional[Dict[str, Any]] = None,
        residual_in_fp32: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.residual_in_fp32 = residual_in_fp32
        self.norm = RMSNorm(d_model, eps=1e-5, **factory_kwargs)
        self.mixer = Mamba2(d_model=d_model, **(ssm_cfg or {}), device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.mixer(x)
        if self.residual_in_fp32:
            return (residual.float() + x.float()).to(dtype=residual.dtype)
        return residual + x


def _strip_known_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    prefixes = ("model.", "backbone.", "module.", "transformer.")
    for p in prefixes:
        if all(k.startswith(p) for k in state_dict.keys()):
            return {k[len(p) :]: v for k, v in state_dict.items()}
    return state_dict


def _try_import_safetensors():
    try:
        from safetensors.torch import load_file  # type: ignore
    except Exception:
        return None
    return load_file


def _try_import_safetensors_safe_open():
    try:
        from safetensors import safe_open  # type: ignore
    except Exception:
        return None
    return safe_open


class Mamba2Backbone(nn.Module):
    def __init__(self, config: Mamba2BackboneConfig, device=None, dtype=None):
        super().__init__()
        self.config = config
        factory_kwargs = {"device": device, "dtype": dtype}
        self.gradient_checkpointing = False

        vocab_padded = config.padded_vocab_size()
        self.embedding = nn.Embedding(vocab_padded, config.d_model, **factory_kwargs)

        self.layers = nn.ModuleList(
            [
                Block(
                    d_model=config.d_model,
                    ssm_cfg=config.ssm_cfg,
                    residual_in_fp32=config.residual_in_fp32,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(config.n_layer)
            ]
        )

        self.norm_f = RMSNorm(config.d_model, eps=1e-5, **factory_kwargs) if config.rms_norm else None

    @torch.no_grad()
    def freeze_(self) -> "Mamba2Backbone":
        for p in self.parameters():
            p.requires_grad = False
        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        x = self.embedding(input_ids)
        if self.training and bool(self.gradient_checkpointing):
            x = x.detach()
            x.requires_grad_(True)
            from torch.utils.checkpoint import checkpoint  # type: ignore

            for layer in self.layers:
                try:
                    x = checkpoint(layer, x, use_reentrant=False)
                except TypeError:
                    x = checkpoint(layer, x)
        else:
            for layer in self.layers:
                x = layer(x)
        if self.norm_f is not None:
            x = self.norm_f(x)
        return {"last_hidden_state": x, "attention_mask": attention_mask}

    def enable_gradient_checkpointing_(self) -> "Mamba2Backbone":
        self.gradient_checkpointing = True
        return self

    @staticmethod
    def load_pretrained(
        pretrained_dir: str | Path,
        device=None,
        dtype=None,
        strict: bool = False,
        load_weights: bool = True,
    ) -> Tuple["Mamba2Backbone", Dict[str, Any]]:
        pretrained_dir = Path(pretrained_dir)
        config = Mamba2BackboneConfig.from_json_file(pretrained_dir / "config.json")
        model = Mamba2Backbone(config, device=device, dtype=dtype)

        safetensors_path = pretrained_dir / "model.safetensors"
        safe_open = _try_import_safetensors_safe_open()
        if safe_open is None:
            raise RuntimeError(
                "缺少依赖 safetensors，无法加载 model.safetensors。请安装: pip install safetensors"
            )
        model_state_keys = set(model.state_dict().keys())
        missing_keys = set(model_state_keys)
        unexpected_keys: Iterable[str] = []

        with safe_open(str(safetensors_path), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            prefixed = {k: k for k in keys}
            stripped = _strip_known_prefixes({k: torch.empty(0) for k in keys})
            if stripped.keys() != prefixed.keys():
                reverse_map = {}
                for original_key in keys:
                    for prefix in ("model.", "backbone.", "module.", "transformer."):
                        if original_key.startswith(prefix):
                            reverse_map[original_key[len(prefix) :]] = original_key
                            break
                prefixed = reverse_map

            file_key_set = set(prefixed.keys())
            unexpected_keys = sorted(file_key_set - model_state_keys)

            if load_weights:
                for name, param in model.named_parameters():
                    file_key = prefixed.get(name)
                    if file_key is None:
                        continue
                    tensor = f.get_tensor(file_key)
                    if tensor.shape != param.shape:
                        continue
                    param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))
                    missing_keys.discard(name)

                for name, buf in model.named_buffers():
                    file_key = prefixed.get(name)
                    if file_key is None:
                        continue
                    tensor = f.get_tensor(file_key)
                    if tensor.shape != buf.shape:
                        continue
                    buf.copy_(tensor.to(device=buf.device, dtype=buf.dtype))
                    missing_keys.discard(name)
            else:
                missing_keys = missing_keys - file_key_set

        if strict and (missing_keys or unexpected_keys):
            raise RuntimeError(
                f"strict=True 加载失败: missing={len(missing_keys)}, unexpected={len(unexpected_keys)}"
            )

        info: Dict[str, Any] = {
            "missing_keys": sorted(missing_keys),
            "unexpected_keys": list(unexpected_keys),
        }
        return model, info

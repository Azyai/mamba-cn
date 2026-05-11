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
    bidirectional_layers: int = 0
    bidirectional_fusion: str = "gate"
    bidirectional_share_mixer: bool = True

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
            bidirectional_layers=int(data.get("bidirectional_layers", 0)),
            bidirectional_fusion=str(data.get("bidirectional_fusion", "gate")),
            bidirectional_share_mixer=bool(data.get("bidirectional_share_mixer", True)),
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
        self.d_model = int(d_model)
        self.ssm_cfg = dict(ssm_cfg or {})
        self.norm = RMSNorm(d_model, eps=1e-5, **factory_kwargs)
        self.mixer = Mamba2(d_model=d_model, **(ssm_cfg or {}), device=device, dtype=dtype)
        self.bidirectional_enabled = False
        self.bidirectional_fusion = "gate"
        self.bidirectional_share_mixer = True
        self.backward_mixer: Optional[Mamba2] = None
        self.bidirectional_gate: Optional[nn.Linear] = None
        self.bidirectional_proj: Optional[nn.Linear] = None

    @staticmethod
    def _reverse_valid_tokens(x: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if attention_mask is None:
            return torch.flip(x, dims=(1,))
        out = x.clone()
        lengths = attention_mask.to(device=x.device, dtype=torch.long).sum(dim=1).clamp_min(0)
        for i, length in enumerate(lengths.tolist()):
            if length > 0:
                out[i, :length] = torch.flip(x[i, :length], dims=(0,))
        return out

    def enable_bidirectional_(
        self,
        *,
        fusion: str = "gate",
        share_mixer: bool = True,
        copy_forward: bool = True,
    ) -> "Block":
        fusion = str(fusion).strip().lower()
        if fusion not in {"add", "gate", "concat"}:
            raise ValueError(f"Unsupported bidirectional fusion: {fusion}")
        self.bidirectional_enabled = True
        self.bidirectional_fusion = fusion
        self.bidirectional_share_mixer = bool(share_mixer)

        ref = next(self.mixer.parameters())
        factory_kwargs = {"device": ref.device, "dtype": ref.dtype}

        if not self.bidirectional_share_mixer and self.backward_mixer is None:
            self.backward_mixer = Mamba2(d_model=self.d_model, **self.ssm_cfg, **factory_kwargs)
            if copy_forward:
                self.backward_mixer.load_state_dict(self.mixer.state_dict(), strict=True)

        if fusion == "gate" and self.bidirectional_gate is None:
            self.bidirectional_gate = nn.Linear(self.d_model * 2, self.d_model, **factory_kwargs)
            nn.init.zeros_(self.bidirectional_gate.weight)
            nn.init.zeros_(self.bidirectional_gate.bias)
        elif fusion == "concat" and self.bidirectional_proj is None:
            self.bidirectional_proj = nn.Linear(self.d_model * 2, self.d_model, **factory_kwargs)
            nn.init.zeros_(self.bidirectional_proj.weight)
            nn.init.zeros_(self.bidirectional_proj.bias)
            with torch.no_grad():
                idx = torch.arange(self.d_model, device=self.bidirectional_proj.weight.device)
                self.bidirectional_proj.weight[idx, idx] = 0.5
                self.bidirectional_proj.weight[idx, idx + self.d_model] = 0.5
        return self

    def set_bidirectional_trainable_(self, *, train_fusion: bool = True, train_backward_mixer: bool = False) -> "Block":
        for module in (self.bidirectional_gate, self.bidirectional_proj):
            if module is None:
                continue
            for p in module.parameters():
                p.requires_grad = bool(train_fusion)
        if self.backward_mixer is not None:
            for p in self.backward_mixer.parameters():
                p.requires_grad = bool(train_backward_mixer)
        return self

    def _fuse_bidirectional(self, fwd: torch.Tensor, bwd: torch.Tensor) -> torch.Tensor:
        if self.bidirectional_fusion == "add":
            return 0.5 * (fwd + bwd)
        cat = torch.cat([fwd, bwd], dim=-1)
        if self.bidirectional_fusion == "concat":
            if self.bidirectional_proj is None:
                raise RuntimeError("bidirectional_proj is not initialized")
            return self.bidirectional_proj(cat)
        if self.bidirectional_gate is None:
            raise RuntimeError("bidirectional_gate is not initialized")
        gate = torch.sigmoid(self.bidirectional_gate(cat))
        return gate * fwd + (1.0 - gate) * bwd

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        y = self.mixer(x)
        if self.bidirectional_enabled:
            x_rev = self._reverse_valid_tokens(x, attention_mask)
            backward_mixer = self.mixer if self.bidirectional_share_mixer else self.backward_mixer
            if backward_mixer is None:
                raise RuntimeError("backward_mixer is not initialized")
            y_rev = backward_mixer(x_rev)
            y_bwd = self._reverse_valid_tokens(y_rev, attention_mask)
            x = self._fuse_bidirectional(y, y_bwd)
        else:
            x = y
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
        if int(config.bidirectional_layers) > 0:
            self.enable_bidirectional_(
                num_layers=int(config.bidirectional_layers),
                fusion=config.bidirectional_fusion,
                share_mixer=bool(config.bidirectional_share_mixer),
            )

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
                    x = checkpoint(layer, x, attention_mask, use_reentrant=False)
                except TypeError:
                    x = checkpoint(layer, x, attention_mask)
        else:
            for layer in self.layers:
                x = layer(x, attention_mask=attention_mask)
        if self.norm_f is not None:
            x = self.norm_f(x)
        return {"last_hidden_state": x, "attention_mask": attention_mask}

    def enable_gradient_checkpointing_(self) -> "Mamba2Backbone":
        self.gradient_checkpointing = True
        return self

    def enable_bidirectional_(
        self,
        *,
        num_layers: int,
        fusion: str = "gate",
        share_mixer: bool = True,
    ) -> "Mamba2Backbone":
        num_layers = max(0, min(int(num_layers), len(self.layers)))
        if num_layers <= 0:
            return self
        for layer in self.layers[-num_layers:]:
            layer.enable_bidirectional_(fusion=fusion, share_mixer=share_mixer)
        self.config = Mamba2BackboneConfig(
            d_model=self.config.d_model,
            n_layer=self.config.n_layer,
            vocab_size=self.config.vocab_size,
            ssm_cfg=dict(self.config.ssm_cfg),
            rms_norm=self.config.rms_norm,
            residual_in_fp32=self.config.residual_in_fp32,
            fused_add_norm=self.config.fused_add_norm,
            pad_vocab_size_multiple=self.config.pad_vocab_size_multiple,
            bidirectional_layers=num_layers,
            bidirectional_fusion=str(fusion).strip().lower(),
            bidirectional_share_mixer=bool(share_mixer),
        )
        return self

    def set_bidirectional_trainable_(self, *, train_fusion: bool = True, train_backward_mixer: bool = False) -> "Mamba2Backbone":
        for layer in self.layers:
            if getattr(layer, "bidirectional_enabled", False):
                layer.set_bidirectional_trainable_(
                    train_fusion=train_fusion,
                    train_backward_mixer=train_backward_mixer,
                )
        return self

    def bidirectional_state_dict(self) -> Dict[str, torch.Tensor]:
        keys = (".backward_mixer.", ".bidirectional_gate.", ".bidirectional_proj.")
        return {k: v for k, v in self.state_dict().items() if any(part in k for part in keys)}

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

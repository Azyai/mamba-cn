from __future__ import annotations

import json
import sys
import types
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch


def _install_megatron_stubs() -> None:
    megatron = types.ModuleType("megatron")
    core = types.ModuleType("megatron.core")
    enums = types.ModuleType("megatron.core.enums")

    class ModelType(Enum):
        encoder_or_decoder = 1
        encoder_and_decoder = 2

    enums.ModelType = ModelType

    sys.modules.setdefault("megatron", megatron)
    sys.modules.setdefault("megatron.core", core)
    sys.modules.setdefault("megatron.core.enums", enums)


def _infer_ngroups_and_d_state(group_state: int) -> Tuple[int, int]:
    candidates = [128, 64, 256, 32, 16]
    for d_state in candidates:
        if group_state % d_state != 0:
            continue
        ngroups = group_state // d_state
        if ngroups in {1, 2, 4, 8, 16, 32}:
            return ngroups, d_state
    return 1, group_state


def _infer_ssm_cfg(megatron_model: Dict[str, torch.Tensor], d_model: int) -> Dict[str, object]:
    dt_bias = megatron_model["decoder.layers.0.mixer.dt_bias"]
    conv1d_weight = megatron_model["decoder.layers.0.mixer.conv1d.weight"]
    out_proj_weight = megatron_model["decoder.layers.0.mixer.out_proj.weight"]

    nheads = int(dt_bias.numel())
    d_conv = int(conv1d_weight.shape[-1])
    d_inner = int(out_proj_weight.shape[1])
    if d_model <= 0 or d_inner % d_model != 0:
        raise RuntimeError(f"无法推断 expand（d_inner={d_inner}, d_model={d_model}）")
    expand = int(d_inner // d_model)
    headdim = d_inner // max(nheads, 1)

    channels = int(conv1d_weight.shape[0])
    d_ssm = d_inner
    rem = channels - d_ssm
    if rem < 0 or rem % 2 != 0:
        ngroups, d_state = 1, 128
    else:
        group_state = rem // 2
        ngroups, d_state = _infer_ngroups_and_d_state(group_state)

    return {
        "d_state": int(d_state),
        "d_conv": int(d_conv),
        "expand": expand,
        "headdim": int(headdim),
        "ngroups": int(ngroups),
        "rmsnorm": True,
    }


def _infer_backbone_config(obj: Dict[str, object]) -> Dict[str, object]:
    megatron_model = obj["model"]
    args = obj.get("args", None)

    emb = megatron_model["embedding.word_embeddings.weight"]
    vocab_size = int(emb.shape[0])
    d_model = int(emb.shape[1])

    n_layer = None
    if args is not None and hasattr(args, "num_layers") and getattr(args, "num_layers") is not None:
        n_layer = int(getattr(args, "num_layers"))
    if n_layer is None:
        n_layer = max(int(k.split(".")[2]) for k in megatron_model.keys() if k.startswith("decoder.layers.")) + 1

    ssm_cfg = _infer_ssm_cfg(megatron_model, d_model=d_model)

    return {
        "d_model": d_model,
        "n_layer": int(n_layer),
        "vocab_size": vocab_size,
        "ssm_cfg": ssm_cfg,
        "rms_norm": True,
        "residual_in_fp32": True,
        "fused_add_norm": True,
        "pad_vocab_size_multiple": 8,
    }


def _map_megatron_key_to_mamba_backbone(key: str) -> Optional[str]:
    if key == "embedding.word_embeddings.weight":
        return "embedding.weight"
    if key.startswith("decoder.layers."):
        return "layers." + key[len("decoder.layers.") :]
    if key == "decoder.final_norm.weight":
        return "norm_f.weight"
    return None


def convert_nvidia_mamba2_8b_megatron_checkpoint(
    *,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
) -> Path:
    try:
        from safetensors.torch import save_file  # type: ignore
    except Exception as e:
        raise RuntimeError("缺少依赖 safetensors。请安装: pip install safetensors") from e

    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "config.json"
    weights_path = output_dir / "model.safetensors"
    if not overwrite and config_path.exists() and weights_path.exists():
        return output_dir

    _install_megatron_stubs()
    obj = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(obj, dict) or "model" not in obj:
        raise RuntimeError("不支持的 checkpoint 格式：缺少 model 字段")

    cfg = _infer_backbone_config(obj)
    config_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    megatron_model = obj["model"]
    state_dict: Dict[str, torch.Tensor] = {}
    for k, v in megatron_model.items():
        if not isinstance(v, torch.Tensor):
            continue
        mapped = _map_megatron_key_to_mamba_backbone(k)
        if mapped is None:
            continue
        state_dict[mapped] = v

    save_file(state_dict, str(weights_path))
    return output_dir

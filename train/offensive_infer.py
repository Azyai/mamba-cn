from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from mamba_ssm.models.lora import LoRAConfig, inject_lora, load_lora_state_dict
from mamba_ssm.models.mamba2_backbone import Mamba2Backbone, Mamba2BackboneConfig
from mamba_ssm.models.offensive_classifier import FrozenBackboneClassifier, MLPHead
from train.sentencepiece_tokenizer import SentencePieceTokenizer, SentencePieceTokenizerConfig


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dtype_from_name(name: str) -> torch.dtype:
    n = str(name).strip().lower()
    if n in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if n in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.float32


def _load_metrics(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "best_metrics.json"
    if p.exists():
        return _read_json(p)
    return {}


def _extract_calibrated_threshold(metrics: Dict[str, Any], *, dataset: str) -> Optional[float]:
    if "calibrated_threshold" in metrics:
        try:
            return float(metrics["calibrated_threshold"])
        except Exception:
            return None
    evals = metrics.get("eval", None)
    if isinstance(evals, dict):
        ds = evals.get(dataset, None)
        if isinstance(ds, dict) and "calibrated_threshold" in ds:
            try:
                return float(ds["calibrated_threshold"])
            except Exception:
                return None
    return None


def _maybe_load_transformers_tokenizer(name_or_path: str):
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as e:
        raise RuntimeError("缺少依赖 transformers，无法加载 tokenizer_name_or_path。") from e
    return AutoTokenizer.from_pretrained(str(name_or_path), use_fast=True)


def _infer_head_hidden_dim(head_state: Dict[str, torch.Tensor]) -> int:
    w = head_state.get("fc1.weight", None)
    if w is None:
        raise RuntimeError("无法从 head state 推断 hidden_dim（缺少 fc1.weight）。")
    return int(w.shape[0])


def _disable_mem_eff_path_if_needed(backbone: Mamba2Backbone, lora_targets: Iterable[str]) -> None:
    if "out_proj" not in set(str(x) for x in lora_targets):
        return
    for layer in getattr(backbone, "layers", []):
        mixer = getattr(layer, "mixer", None)
        if mixer is not None and hasattr(mixer, "use_mem_eff_path"):
            setattr(mixer, "use_mem_eff_path", False)


def _build_backbone_from_config(config_dict: Dict[str, Any], *, device: torch.device, dtype: torch.dtype) -> Mamba2Backbone:
    cfg = Mamba2BackboneConfig(
        d_model=int(config_dict["d_model"]),
        n_layer=int(config_dict["n_layer"]),
        vocab_size=int(config_dict["vocab_size"]),
        ssm_cfg=dict(config_dict.get("ssm_cfg", {})),
        rms_norm=bool(config_dict.get("rms_norm", True)),
        residual_in_fp32=bool(config_dict.get("residual_in_fp32", True)),
        fused_add_norm=bool(config_dict.get("fused_add_norm", True)),
        pad_vocab_size_multiple=int(config_dict.get("pad_vocab_size_multiple", 8)),
    )
    return Mamba2Backbone(cfg, device=device, dtype=dtype)


@dataclass(frozen=True)
class PredictResult:
    p_toxic: List[float]
    labels: List[int]
    threshold: float
    threshold_mode: str
    latency_ms: float


class OffensivePredictor:
    def __init__(
        self,
        *,
        model: FrozenBackboneClassifier,
        tokenizer: object,
        max_length: int,
        device: torch.device,
        dtype: torch.dtype,
        calibrated_threshold: Optional[float],
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.device = device
        self.dtype = dtype
        self.calibrated_threshold = calibrated_threshold

    def _encode(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        if isinstance(self.tokenizer, SentencePieceTokenizer):
            enc = self.tokenizer(
                texts,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )
            return {"input_ids": enc["input_ids"], "attention_mask": enc.get("attention_mask", None)}
        tok = self.tokenizer
        enc = tok(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        return {"input_ids": enc["input_ids"], "attention_mask": enc.get("attention_mask", None)}

    @torch.no_grad()
    def predict_proba(self, texts: List[str], *, batch_size: int = 8) -> List[float]:
        if not texts:
            return []
        self.model.eval()
        out: List[float] = []
        for i in range(0, len(texts), int(batch_size)):
            chunk = texts[i : i + int(batch_size)]
            enc = self._encode(chunk)
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc.get("attention_mask", None)
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = attention_mask.to(self.device)
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=-1)[:, 1].to(torch.float32).detach().cpu().tolist()
            out.extend(float(x) for x in probs)
        return out

    def predict(
        self,
        texts: List[str],
        *,
        threshold_mode: str = "calibrated",
        threshold: float = 0.5,
        dataset_for_threshold: str = "toxicn",
        batch_size: int = 8,
    ) -> PredictResult:
        t0 = time.time()
        p = self.predict_proba(texts, batch_size=batch_size)
        mode = str(threshold_mode).strip().lower()
        thr = float(threshold)
        if mode in {"calibrated", "cal"}:
            if self.calibrated_threshold is not None:
                thr = float(self.calibrated_threshold)
            else:
                thr = float(threshold)
        elif mode in {"argmax", "0.5"}:
            thr = 0.5
        labels = [1 if float(x) >= float(thr) else 0 for x in p]
        dt = (time.time() - t0) * 1000.0
        return PredictResult(p_toxic=p, labels=labels, threshold=float(thr), threshold_mode=mode, latency_ms=float(dt))


def load_offensive_predictor(
    *,
    run_dir: str | Path,
    device: str = "cuda",
    dtype: str = "fp16",
    dataset_for_threshold: str = "toxicn",
    pretrained_dir: str | Path | None = None,
) -> OffensivePredictor:
    run_path = Path(run_dir).expanduser().resolve()
    if not run_path.exists():
        raise FileNotFoundError(f"run_dir 不存在: {run_path}")

    dev = torch.device(str(device))
    dt = _dtype_from_name(dtype)
    if dev.type == "cpu":
        dt = torch.float32

    metrics = _load_metrics(run_path)
    calibrated_thr = _extract_calibrated_threshold(metrics, dataset=str(dataset_for_threshold))

    full_model = run_path / "full_model.pt"
    if full_model.exists():
        ckpt = torch.load(str(full_model), map_location="cpu")
        config_dict = dict(ckpt["config"])
        backbone = _build_backbone_from_config(config_dict, device=dev, dtype=dt)
        backbone.load_state_dict(ckpt["backbone"], strict=True)
        backbone = backbone.to(dev)
        backbone.eval()

        head_state = ckpt["head"]
        hidden_dim = _infer_head_hidden_dim(head_state)
        head = MLPHead(d_model=int(config_dict["d_model"]), hidden_dim=hidden_dim, dropout=0.0).to(dev, dtype=dt)
        head.load_state_dict(head_state, strict=True)
        head.eval()

        tok = None
        max_length = int(ckpt.get("max_length", 256))
        if "tokenizer_model_path" in ckpt:
            tok_cfg = SentencePieceTokenizerConfig(model_file=str(ckpt["tokenizer_model_path"]))
            tok = SentencePieceTokenizer(tok_cfg)
        else:
            tok = _maybe_load_transformers_tokenizer(str(ckpt["tokenizer_name_or_path"]))

        model = FrozenBackboneClassifier(backbone=backbone, head=head).to(dev)
        model.eval()
        return OffensivePredictor(model=model, tokenizer=tok, max_length=max_length, device=dev, dtype=dt, calibrated_threshold=calibrated_thr)

    best_head = run_path / "best_head.pt"
    if not best_head.exists():
        raise FileNotFoundError(f"run_dir 下缺少 best_head.pt: {best_head}")
    ckpt = torch.load(str(best_head), map_location="cpu")
    config_dict = dict(ckpt["config"])

    resolved_pretrained = None
    txt = run_path / "backbone_converted_dir.txt"
    if txt.exists():
        resolved_pretrained = Path(txt.read_text(encoding="utf-8").strip()).expanduser()
    if pretrained_dir is not None:
        resolved_pretrained = Path(pretrained_dir).expanduser()
    if resolved_pretrained is None:
        raise RuntimeError("未找到 backbone 权重目录：缺少 backbone_converted_dir.txt 且未显式传入 pretrained_dir。")
    if not resolved_pretrained.is_absolute():
        resolved_pretrained = (run_path.parent / resolved_pretrained).resolve()

    backbone, _ = Mamba2Backbone.load_pretrained(resolved_pretrained, device=dev, dtype=dt, strict=False)
    backbone = backbone.to(dev)
    backbone.eval()

    lora_cfg_dict = ckpt.get("lora_cfg", None)
    if isinstance(lora_cfg_dict, dict) and "lora" in ckpt:
        targets = lora_cfg_dict.get("target", ("in_proj",))
        if isinstance(targets, list):
            targets = tuple(str(x) for x in targets)
        lora_cfg = LoRAConfig(
            r=int(lora_cfg_dict.get("r", 8)),
            alpha=int(lora_cfg_dict.get("alpha", 16)),
            dropout=float(lora_cfg_dict.get("dropout", 0.05)),
            target=tuple(str(x) for x in targets),
        )
        inject_lora(backbone, lora_cfg)
        load_lora_state_dict(backbone, ckpt["lora"])
        _disable_mem_eff_path_if_needed(backbone, lora_cfg.target)

    head_state = ckpt["head"]
    hidden_dim = _infer_head_hidden_dim(head_state)
    head = MLPHead(d_model=int(config_dict["d_model"]), hidden_dim=hidden_dim, dropout=0.0).to(dev, dtype=dt)
    head.load_state_dict(head_state, strict=True)
    head.eval()

    max_length = int(ckpt.get("max_length", 256))
    tok = None
    if "tokenizer_model_path" in ckpt:
        tok_cfg = SentencePieceTokenizerConfig(model_file=str(ckpt["tokenizer_model_path"]))
        tok = SentencePieceTokenizer(tok_cfg)
    else:
        tok = _maybe_load_transformers_tokenizer(str(ckpt["tokenizer_name_or_path"]))

    model = FrozenBackboneClassifier(backbone=backbone, head=head).to(dev)
    model.eval()
    return OffensivePredictor(model=model, tokenizer=tok, max_length=max_length, device=dev, dtype=dt, calibrated_threshold=calibrated_thr)

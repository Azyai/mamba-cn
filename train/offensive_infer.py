from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from mamba_ssm.models.lora import LoRAConfig, inject_lora, load_lora_state_dict
from mamba_ssm.models.mamba2_backbone import Mamba2Backbone, Mamba2BackboneConfig
from mamba_ssm.models.offensive_classifier import FrozenBackboneClassifier, MLPHead, MultimodalClassifier
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


def _maybe_load_transformers_tokenizer(name_or_path: str, cache_dir: str = None):
    try:
        from transformers import AutoTokenizer
        try:
            tok = AutoTokenizer.from_pretrained(str(name_or_path), use_fast=True, cache_dir=cache_dir, local_files_only=True)
        except Exception:
            tok = AutoTokenizer.from_pretrained(str(name_or_path), use_fast=True, cache_dir=cache_dir)
            
        if tok.pad_token is None:
            if tok.eos_token is not None:
                tok.pad_token = tok.eos_token
            else:
                tok.add_special_tokens({'pad_token': '[PAD]'})
        return tok
    except Exception as e:
        raise RuntimeError("缺少依赖 transformers，或者无法加载 tokenizer。") from e
    try:
        from transformers import AutoTokenizer
        try:
            tok = AutoTokenizer.from_pretrained(str(name_or_path), use_fast=True, cache_dir=cache_dir, local_files_only=True)
        except Exception:
            tok = AutoTokenizer.from_pretrained(str(name_or_path), use_fast=True, cache_dir=cache_dir)
            
        if tok.pad_token is None:
            if tok.eos_token is not None:
                tok.pad_token = tok.eos_token
            else:
                tok.add_special_tokens({'pad_token': '[PAD]'})
        return tok
    except Exception as e:
        raise RuntimeError("缺少依赖 transformers，或者无法加载 tokenizer。") from e
    try:
        from transformers import AutoTokenizer
        try:
            tok = AutoTokenizer.from_pretrained(str(name_or_path), use_fast=True, cache_dir=cache_dir, local_files_only=True)
        except Exception:
            tok = AutoTokenizer.from_pretrained(str(name_or_path), use_fast=True, cache_dir=cache_dir)
            
        if tok.pad_token is None:
            if tok.eos_token is not None:
                tok.pad_token = tok.eos_token
            else:
                tok.add_special_tokens({'pad_token': '[PAD]'})
        return tok
    except Exception as e:
        raise RuntimeError("缺少依赖 transformers，或者无法加载 tokenizer。") from e
    try:
        from transformers import AutoTokenizer
        try:
            return AutoTokenizer.from_pretrained(str(name_or_path), use_fast=True, cache_dir=cache_dir, local_files_only=True)
        except Exception:
            return AutoTokenizer.from_pretrained(str(name_or_path), use_fast=True, cache_dir=cache_dir)
    except Exception as e:
        raise RuntimeError("缺少依赖 transformers，或者无法加载 tokenizer。") from e




def _infer_head_hidden_dim(head_state: Dict[str, torch.Tensor]) -> int:
    w = head_state.get("fc1.weight", None)
    if w is None:
        raise RuntimeError("无法从 head state 推断 hidden_dim（缺少 fc1.weight）。")
    return int(w.shape[0])


def _infer_head_input_dim(head_state: Dict[str, torch.Tensor]) -> int:
    w = head_state.get("fc1.weight", None)
    if w is None:
        raise RuntimeError("无法从 head state 推断 input_dim（缺少 fc1.weight）。")
    return int(w.shape[1])


def _load_image_backbone(name_or_path: str, *, cache_dir: Path, device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    from transformers import AutoModel

    try:
        model = AutoModel.from_pretrained(str(name_or_path), cache_dir=str(cache_dir), local_files_only=True)
    except Exception:
        model = AutoModel.from_pretrained(str(name_or_path), cache_dir=str(cache_dir))
    vision = getattr(model, "vision_model", None)
    if vision is not None:
        model = vision
    return model.to(device, dtype=dtype)


def _disable_mem_eff_path_if_needed(backbone: Mamba2Backbone, lora_targets: Iterable[str]) -> None:
    if "out_proj" not in set(str(x) for x in lora_targets):
        return
    for layer in getattr(backbone, "layers", []):
        for mixer_name in ("mixer", "backward_mixer"):
            mixer = getattr(layer, mixer_name, None)
            if mixer is not None and hasattr(mixer, "use_mem_eff_path"):
                setattr(mixer, "use_mem_eff_path", False)


def _apply_bidirectional_config(backbone: Mamba2Backbone, config_dict: Dict[str, Any]) -> None:
    num_layers = int(config_dict.get("bidirectional_layers", 0) or 0)
    if num_layers <= 0:
        return
    backbone.enable_bidirectional_(
        num_layers=num_layers,
        fusion=str(config_dict.get("bidirectional_fusion", "gate")),
        share_mixer=bool(config_dict.get("bidirectional_share_mixer", True)),
    )


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
        bidirectional_layers=int(config_dict.get("bidirectional_layers", 0) or 0),
        bidirectional_fusion=str(config_dict.get("bidirectional_fusion", "gate")),
        bidirectional_share_mixer=bool(config_dict.get("bidirectional_share_mixer", True)),
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
        model: torch.nn.Module,
        tokenizer: object,
        image_processor: Optional[object],
        audio_processor: Optional[object],
        max_length: int,
        device: torch.device,
        dtype: torch.dtype,
        calibrated_threshold: Optional[float],
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.audio_processor = audio_processor
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

    def predict_proba(self, texts: List[str], images: List[Optional[str]] = None, audios: List[Optional[str]] = None, *, batch_size: int = 8) -> List[float]:
        if not texts:
            return []
        if images is None:
            images = [None] * len(texts)
        if audios is None:
            audios = [None] * len(texts)
            
        self.model.eval()
        out: List[float] = []
        for i in range(0, len(texts), int(batch_size)):
            chunk_texts = texts[i : i + int(batch_size)]
            chunk_images = images[i : i + int(batch_size)]
            chunk_audios = audios[i : i + int(batch_size)]
            
            enc = self._encode(chunk_texts)
            input_ids = enc["input_ids"].to(self.device, dtype=torch.long)
            # 解决完全没有输入文字导致序列长度为0，从而mamba backbone使用 Conv1D 崩溃的问题
            if input_ids.shape[1] == 0:
                # 填充一个PAD符号（如果是gpt2则默认填充eos_token的id)，以保证文本轴的存在
                pad_id = self.tokenizer.pad_token_id if hasattr(self.tokenizer, "pad_token_id") and self.tokenizer.pad_token_id is not None else 0
                input_ids = torch.full((input_ids.shape[0], 1), pad_id, dtype=torch.long, device=self.device)
                
            attention_mask = enc.get("attention_mask", None)
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = attention_mask.to(self.device)
                if attention_mask.shape[1] == 0:
                     attention_mask = torch.ones((attention_mask.shape[0], 1), dtype=torch.long, device=self.device)
                
            pixel_values = None
            image_mask = None
            if self.image_processor is not None:
                import PIL.Image
                imgs = []
                masks = []
                for img_path in chunk_images:
                    if img_path and Path(img_path).exists():
                        try:
                            imgs.append(PIL.Image.open(img_path).convert("RGB"))
                            masks.append(True)
                        except:
                            imgs.append(PIL.Image.new("RGB", (224, 224)))
                            masks.append(False)
                    else:
                        imgs.append(PIL.Image.new("RGB", (224, 224)))
                        masks.append(False)
                img_enc = self.image_processor(images=imgs, return_tensors="pt")
                pixel_values = img_enc["pixel_values"].to(self.device, dtype=self.dtype)
                image_mask = torch.tensor(masks, dtype=torch.bool).to(self.device)
                
            input_values = None
            audio_mask = None
            if self.audio_processor is not None:
                import torchaudio
                auds = []
                masks = []
                for aud_path in chunk_audios:
                    if aud_path and Path(aud_path).exists():
                        try:
                            waveform, sr = torchaudio.load(aud_path)
                            if sr != self.audio_processor.sampling_rate:
                                resampler = torchaudio.transforms.Resample(sr, self.audio_processor.sampling_rate)
                                waveform = resampler(waveform)
                            auds.append(waveform[0].numpy())
                            masks.append(True)
                        except:
                            auds.append(torch.zeros(16000).numpy())
                            masks.append(False)
                    else:
                        auds.append(torch.zeros(16000).numpy())
                        masks.append(False)
                aud_enc = self.audio_processor(auds, sampling_rate=self.audio_processor.sampling_rate, return_tensors="pt", padding=True)
                input_values = aud_enc["input_values"].to(self.device, dtype=self.dtype)
                audio_mask = torch.tensor(masks, dtype=torch.bool).to(self.device)

            if isinstance(self.model, MultimodalClassifier):
                logits = self.model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_mask=image_mask,
                    input_values=input_values,
                    audio_mask=audio_mask
                ).logits
            else:
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
                
            probs = torch.softmax(logits, dim=-1)[:, 1].to(torch.float32).detach().cpu().tolist()
            out.extend(float(x) for x in probs)
        return out

    def predict(
        self,
        texts: List[str],
        images: List[Optional[str]] = None,
        audios: List[Optional[str]] = None,
        *,
        threshold_mode: str = "calibrated",
        threshold: float = 0.5,
        dataset_for_threshold: str = "toxicn",
        batch_size: int = 8,
    ) -> PredictResult:
        t0 = time.time()
        p = self.predict_proba(texts, images=images, audios=audios, batch_size=batch_size)
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
        lora_cfg_dict = ckpt.get("lora_cfg", None)
        if isinstance(lora_cfg_dict, dict):
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
            _disable_mem_eff_path_if_needed(backbone, lora_cfg.target)
        backbone.load_state_dict(ckpt["backbone"], strict=True)
        backbone = backbone.to(dev)
        backbone.eval()

        head_state = ckpt["head"]
        hidden_dim = _infer_head_hidden_dim(head_state)
        tok = None
        max_length = int(ckpt.get("max_length", 256))
        if "tokenizer_model_path" in ckpt:
            tok_cfg = SentencePieceTokenizerConfig(model_file=str(ckpt["tokenizer_model_path"]))
            tok = SentencePieceTokenizer(tok_cfg)
        else:
            tok_path = str(ckpt["tokenizer_name_or_path"])
            p_tok = Path(tok_path)
            if not p_tok.is_absolute():
                if not p_tok.exists():
                    p_tok = (run_path.parent.parent / p_tok).resolve()
            if p_tok.exists():
                tok_path = str(p_tok)
            tok = _maybe_load_transformers_tokenizer(tok_path, cache_dir=str((run_path.parent.parent / "predict/gpt2").resolve()))

        if "classifier_state" in ckpt:
            from transformers import Wav2Vec2Model, AutoImageProcessor, Wav2Vec2FeatureExtractor
            
            image_backbone = None
            audio_backbone = None
            image_processor = None
            audio_processor = None
            
            if ckpt.get("vit_name_or_path"):
                image_processor = AutoImageProcessor.from_pretrained(ckpt["vit_name_or_path"], cache_dir=str((run_path.parent.parent / "predict/multimodal").resolve()), local_files_only=True)
                image_backbone = _load_image_backbone(ckpt["vit_name_or_path"], cache_dir=(run_path.parent.parent / "predict/multimodal").resolve(), device=dev, dtype=dt)
                image_backbone.eval()
                
            if ckpt.get("wav2vec2_name_or_path"):
                audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(ckpt["wav2vec2_name_or_path"], cache_dir=str((run_path.parent.parent / "predict/multimodal").resolve()), local_files_only=True)
                audio_backbone = Wav2Vec2Model.from_pretrained(ckpt["wav2vec2_name_or_path"], cache_dir=str((run_path.parent.parent / "predict/multimodal").resolve()), local_files_only=True).to(dev, dtype=dt)
                audio_backbone.eval()

            input_dim = _infer_head_input_dim(head_state)
            head = MLPHead(d_model=input_dim, hidden_dim=hidden_dim, dropout=0.0).to(dev, dtype=dt)
            head.load_state_dict(head_state, strict=True)
            head.eval()

            model = MultimodalClassifier(
                text_backbone=backbone,
                head=head,
                image_backbone=image_backbone,
                audio_backbone=audio_backbone,
                text_dim=config_dict["d_model"],
                image_dim=768,
                audio_dim=768
            ).to(dev, dtype=dt)
            model.load_state_dict(ckpt["classifier_state"], strict=False)
        else:
            image_processor = None
            audio_processor = None
            head = MLPHead(d_model=int(config_dict["d_model"]), hidden_dim=hidden_dim, dropout=0.0).to(dev, dtype=dt)
            head.load_state_dict(head_state, strict=True)
            head.eval()
            model = FrozenBackboneClassifier(backbone=backbone, head=head).to(dev, dtype=dt)

        model.eval()
        return OffensivePredictor(
            model=model, 
            tokenizer=tok, 
            image_processor=image_processor,
            audio_processor=audio_processor,
            max_length=max_length, 
            device=dev, 
            dtype=dt, 
            calibrated_threshold=calibrated_thr
        )

    best_head = run_path / "best_head.pt"
    if not best_head.exists():
        raise FileNotFoundError(f"run_dir 下缺少 best_head.pt: {best_head}")
    ckpt = torch.load(str(best_head), map_location="cpu", weights_only=False)
    config_dict = dict(ckpt["config"])

    resolved_pretrained = None
    txt = run_path / "backbone_converted_dir.txt"
    if txt.exists():
        resolved_pretrained = Path(txt.read_text(encoding="utf-8").strip()).expanduser(); print("DEBUG pre:", resolved_pretrained, "exists:", resolved_pretrained.exists(), "cwd:", __import__("os").getcwd()); print("DEBUG pre:", resolved_pretrained, "exists:", resolved_pretrained.exists(), "cwd:", __import__("os").getcwd())
    if pretrained_dir is not None:
        resolved_pretrained = Path(pretrained_dir).expanduser()
    if resolved_pretrained is None:
        raise RuntimeError("未找到 backbone 权重目录：缺少 backbone_converted_dir.txt 且未显式传入 pretrained_dir。")
    if not resolved_pretrained.is_absolute():
        if not resolved_pretrained.exists():
            # Fallback to resolving relative to the project root, assuming run_path is in `runs/xx`
            # Try resolving from cwd
            if Path(resolved_pretrained).exists():
                resolved_pretrained = Path(resolved_pretrained).resolve()
            else:
                resolved_pretrained = (run_path.parent.parent / resolved_pretrained).resolve()
            print("DEBUG pre modified:", resolved_pretrained)
        else:
            resolved_pretrained = resolved_pretrained.resolve()

    backbone, _ = Mamba2Backbone.load_pretrained(resolved_pretrained, device=dev, dtype=dt, strict=False)
    backbone = backbone.to(dev)
    _apply_bidirectional_config(backbone, config_dict)
    if int(config_dict.get("bidirectional_layers", 0) or 0) > 0 and isinstance(ckpt.get("bidirectional_state", None), dict):
        backbone.load_state_dict(ckpt["bidirectional_state"], strict=False)
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
    tok = None
    max_length = int(ckpt.get("max_length", 256))
    if "tokenizer_model_path" in ckpt:
        tok_cfg = SentencePieceTokenizerConfig(model_file=str(ckpt["tokenizer_model_path"]))
        tok = SentencePieceTokenizer(tok_cfg)
    else:
        tok_path = str(ckpt["tokenizer_name_or_path"])
        p_tok = Path(tok_path)
        if not p_tok.is_absolute():
            if not p_tok.exists():
                p_tok = (run_path.parent.parent / p_tok).resolve()
        if p_tok.exists():
            tok_path = str(p_tok)
        tok = _maybe_load_transformers_tokenizer(tok_path, cache_dir=str((run_path.parent.parent / "predict/gpt2").resolve()))

    if "classifier_state" in ckpt:
        from transformers import Wav2Vec2Model, AutoImageProcessor, Wav2Vec2FeatureExtractor
        
        image_backbone = None
        audio_backbone = None
        image_processor = None
        audio_processor = None
        
        if ckpt.get("vit_name_or_path"):
            image_processor = AutoImageProcessor.from_pretrained(ckpt["vit_name_or_path"], cache_dir=str((run_path.parent.parent / "predict/multimodal").resolve()), local_files_only=True)
            image_backbone = _load_image_backbone(ckpt["vit_name_or_path"], cache_dir=(run_path.parent.parent / "predict/multimodal").resolve(), device=dev, dtype=dt)
            image_backbone.eval()
            
        if ckpt.get("wav2vec2_name_or_path"):
            audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(ckpt["wav2vec2_name_or_path"], cache_dir=str((run_path.parent.parent / "predict/multimodal").resolve()), local_files_only=True)
            audio_backbone = Wav2Vec2Model.from_pretrained(ckpt["wav2vec2_name_or_path"], cache_dir=str((run_path.parent.parent / "predict/multimodal").resolve()), local_files_only=True).to(dev, dtype=dt)
            audio_backbone.eval()

        input_dim = _infer_head_input_dim(head_state)
        head = MLPHead(d_model=input_dim, hidden_dim=hidden_dim, dropout=0.0).to(dev, dtype=dt)
        head.load_state_dict(head_state, strict=True)
        head.eval()

        model = MultimodalClassifier(
            text_backbone=backbone,
            head=head,
            image_backbone=image_backbone,
            audio_backbone=audio_backbone,
            text_dim=config_dict["d_model"],
            image_dim=768,
            audio_dim=768
        ).to(dev, dtype=dt)
        model.load_state_dict(ckpt["classifier_state"], strict=False)
    else:
        image_processor = None
        audio_processor = None
        head = MLPHead(d_model=int(config_dict["d_model"]), hidden_dim=hidden_dim, dropout=0.0).to(dev, dtype=dt)
        head.load_state_dict(head_state, strict=True)
        head.eval()
        model = FrozenBackboneClassifier(backbone=backbone, head=head).to(dev, dtype=dt)

    model.eval()
    return OffensivePredictor(
        model=model, 
        tokenizer=tok, 
        image_processor=image_processor,
        audio_processor=audio_processor,
        max_length=max_length, 
        device=dev, 
        dtype=dt, 
        calibrated_threshold=calibrated_thr
    )

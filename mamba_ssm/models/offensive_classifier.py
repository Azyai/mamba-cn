from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm.models.paer_module import PAERModule


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
    base_logits: Optional[torch.Tensor] = None
    paer_aux: Optional[Dict[str, torch.Tensor]] = None


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


class MultimodalClassifier(nn.Module):
    def __init__(
        self,
        text_backbone: nn.Module,
        head: nn.Module,
        image_backbone: Optional[nn.Module] = None,
        audio_backbone: Optional[nn.Module] = None,
        text_dim: int = 2560,
        image_dim: int = 768,
        audio_dim: int = 768,
        paer_enable: bool = False,
        paer_span_kernel_size: int = 5,
        paer_topk: int = 3,
        paer_beta: float = 1.0,
        paer_lambda_logit: float = 1.0,
        paer_dropout: float = 0.1,
        paer_span_pooling: str = "topk",
        paer_use_modality_mask: bool = True,
        paer_balance_logits: bool = False,
        paer_toxic_label_id: int = 1,
        paer_calibration_mode: str = "residual",
        paer_max_delta: float = 2.0,
    ):
        super().__init__()
        self.text_backbone = text_backbone
        self.image_backbone = image_backbone
        self.audio_backbone = audio_backbone
        self.head = head

        # Learnable blank tokens for missing modalities
        self.blank_image = nn.Parameter(torch.zeros(1, image_dim)) if image_backbone else None
        self.blank_audio = nn.Parameter(torch.zeros(1, audio_dim)) if audio_backbone else None
        
        if self.blank_image is not None:
            nn.init.normal_(self.blank_image, std=0.02)
        if self.blank_audio is not None:
            nn.init.normal_(self.blank_audio, std=0.02)

        # 降维投影层
        self.image_proj = nn.Linear(image_dim, text_dim) if image_backbone else None
        self.audio_proj = nn.Linear(audio_dim, text_dim) if audio_backbone else None
        
        # Missing-aware reliability gated fusion (MRGF)
        self.image_conf_proj = nn.Linear(text_dim, 1) if image_backbone else None
        self.audio_conf_proj = nn.Linear(text_dim, 1) if audio_backbone else None
        self.image_gate = nn.Sequential(nn.Linear(text_dim * 2 + 3, text_dim), nn.Sigmoid()) if image_backbone else None
        self.audio_gate = nn.Sequential(nn.Linear(text_dim * 2 + 3, text_dim), nn.Sigmoid()) if audio_backbone else None
        self.paer_module = (
            PAERModule(
                text_hidden_size=text_dim,
                fused_size=text_dim,
                num_labels=2,
                toxic_label_id=paer_toxic_label_id,
                span_kernel_size=paer_span_kernel_size,
                topk=paer_topk,
                beta=paer_beta,
                lambda_logit=paer_lambda_logit,
                dropout=paer_dropout,
                span_pooling=paer_span_pooling,
                use_modality_mask=paer_use_modality_mask,
                balance_logits=paer_balance_logits,
                calibration_mode=paer_calibration_mode,
                max_delta=paer_max_delta,
            )
            if paer_enable
            else None
        )

    @torch.no_grad()
    def freeze_backbones_(self) -> "MultimodalClassifier":
        for p in self.text_backbone.parameters():
            p.requires_grad = False
        if self.image_backbone is not None:
            for p in self.image_backbone.parameters():
                p.requires_grad = False
        if self.audio_backbone is not None:
            for p in self.audio_backbone.parameters():
                p.requires_grad = False
        return self

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_mask: Optional[torch.Tensor] = None,
        input_values: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
    ) -> ForwardOutput:
        batch_size = 1
        device = self.head.fc1.weight.device
        
        # 1. Text Features
        if input_ids is not None:
            outputs = self.text_backbone(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs["last_hidden_state"]
            text_feat = masked_mean_pool(last_hidden_state, outputs.get("attention_mask", attention_mask))
            batch_size = text_feat.size(0)
        else:
            raise ValueError("Currently text input_ids is required to infer batch size.")

        # 2. Image Features
        if self.image_backbone is not None:
            if pixel_values is not None:
                img_outputs = self.image_backbone(pixel_values=pixel_values)
                img_feat = img_outputs.pooler_output if hasattr(img_outputs, "pooler_output") and img_outputs.pooler_output is not None else img_outputs.last_hidden_state.mean(dim=1)
                if image_mask is not None:
                    # image_mask: (B, ) boolean tensor, True if image is valid
                    blank_expand = self.blank_image.expand(batch_size, -1)
                    image_mask_expanded = image_mask.unsqueeze(1).to(device)
                    image_feat = torch.where(image_mask_expanded, img_feat, blank_expand)
            else:
                image_feat = self.blank_image.expand(batch_size, -1)
        else:
            image_feat = None

        # 3. Audio Features
        if self.audio_backbone is not None:
            if input_values is not None:
                aud_outputs = self.audio_backbone(input_values=input_values)
                aud_feat = aud_outputs.last_hidden_state.mean(dim=1)
                if audio_mask is not None:
                    blank_expand = self.blank_audio.expand(batch_size, -1)
                    audio_mask_expanded = audio_mask.unsqueeze(1).to(device)
                    audio_feat = torch.where(audio_mask_expanded, aud_feat, blank_expand)
            else:
                audio_feat = self.blank_audio.expand(batch_size, -1)
        else:
            audio_feat = None

        # Gated Fusion
        image_mask_f = None
        audio_mask_f = None

        if image_feat is not None:
            image_feat = self.image_proj(image_feat)
            if image_mask is None:
                image_mask_f = torch.zeros((batch_size, 1), device=device, dtype=image_feat.dtype) if pixel_values is None else torch.ones((batch_size, 1), device=device, dtype=image_feat.dtype)
            else:
                image_mask_f = image_mask.to(dtype=image_feat.dtype, device=device).unsqueeze(1)
            image_conf = torch.sigmoid(self.image_conf_proj(image_feat)) if self.image_conf_proj is not None else torch.zeros((batch_size, 1), device=device, dtype=image_feat.dtype)
            image_sim = F.cosine_similarity(text_feat, image_feat, dim=-1).unsqueeze(1)
            gate_in = torch.cat([text_feat, image_feat, image_mask_f, image_conf, image_sim], dim=-1)
            g_img = self.image_gate(gate_in) if self.image_gate is not None else torch.ones_like(text_feat)
        else:
            g_img = None

        if audio_feat is not None:
            audio_feat = self.audio_proj(audio_feat)
            if audio_mask is None:
                audio_mask_f = torch.zeros((batch_size, 1), device=device, dtype=audio_feat.dtype) if input_values is None else torch.ones((batch_size, 1), device=device, dtype=audio_feat.dtype)
            else:
                audio_mask_f = audio_mask.to(dtype=audio_feat.dtype, device=device).unsqueeze(1)
            audio_conf = torch.sigmoid(self.audio_conf_proj(audio_feat)) if self.audio_conf_proj is not None else torch.zeros((batch_size, 1), device=device, dtype=audio_feat.dtype)
            audio_sim = F.cosine_similarity(text_feat, audio_feat, dim=-1).unsqueeze(1)
            gate_in = torch.cat([text_feat, audio_feat, audio_mask_f, audio_conf, audio_sim], dim=-1)
            g_aud = self.audio_gate(gate_in) if self.audio_gate is not None else torch.ones_like(text_feat)
        else:
            g_aud = None

        pooled = text_feat
        if image_feat is not None and g_img is not None and image_mask_f is not None:
            pooled = pooled + (image_mask_f * g_img * image_feat)
        if audio_feat is not None and g_aud is not None and audio_mask_f is not None:
            pooled = pooled + (audio_mask_f * g_aud * audio_feat)
            
        base_logits = self.head(pooled)
        logits = base_logits
        paer_aux = None
        if self.paer_module is not None:
            logits, paer_aux = self.paer_module(
                text_hidden_states=last_hidden_state,
                fused_feat=pooled,
                base_logits=base_logits,
                attention_mask=outputs.get("attention_mask", attention_mask),
                image_mask=image_mask_f,
                audio_mask=audio_mask_f,
            )
        return ForwardOutput(logits=logits, pooled=pooled, base_logits=base_logits, paer_aux=paer_aux)

    def paer_config_dict(self) -> Optional[Dict[str, object]]:
        if self.paer_module is None:
            return None
        return self.paer_module.config_dict()

    def trainable_state_dict(self) -> Dict[str, object]:
        out = {"head": self.head.state_dict()}
        if self.image_gate is not None:
            out["image_gate"] = self.image_gate.state_dict()
        if self.audio_gate is not None:
            out["audio_gate"] = self.audio_gate.state_dict()
        if self.image_conf_proj is not None:
            out["image_conf_proj"] = self.image_conf_proj.state_dict()
        if self.audio_conf_proj is not None:
            out["audio_conf_proj"] = self.audio_conf_proj.state_dict()
        if self.image_proj is not None:
            out["image_proj"] = self.image_proj.state_dict()
        if self.audio_proj is not None:
            out["audio_proj"] = self.audio_proj.state_dict()
        if self.blank_image is not None:
            out["blank_image"] = self.blank_image.data.detach().cpu()
        if self.blank_audio is not None:
            out["blank_audio"] = self.blank_audio.data.detach().cpu()
        if self.paer_module is not None:
            out["paer_module"] = self.paer_module.state_dict()
            out["paer_config"] = self.paer_module.config_dict()
        return out

    def load_trainable_state_dict(self, state: Dict[str, object], strict: bool = False) -> None:
        if not isinstance(state, dict):
            return
        module_names = [
            "head",
            "image_gate",
            "audio_gate",
            "image_conf_proj",
            "audio_conf_proj",
            "image_proj",
            "audio_proj",
            "paer_module",
        ]
        for name in module_names:
            module = getattr(self, name, None)
            module_state = state.get(name)
            if module is not None and isinstance(module_state, dict):
                module.load_state_dict(module_state, strict=strict)
        if self.blank_image is not None and isinstance(state.get("blank_image"), torch.Tensor):
            self.blank_image.data.copy_(state["blank_image"].to(device=self.blank_image.device, dtype=self.blank_image.dtype))
        if self.blank_audio is not None and isinstance(state.get("blank_audio"), torch.Tensor):
            self.blank_audio.data.copy_(state["blank_audio"].to(device=self.blank_audio.device, dtype=self.blank_audio.dtype))


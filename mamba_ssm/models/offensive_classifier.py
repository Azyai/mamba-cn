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
        
        fusion_dim = text_dim
        if image_backbone: fusion_dim += text_dim
        if audio_backbone: fusion_dim += text_dim
        
        # 门控机制：抑制噪声特征
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim, text_dim),
            nn.Sigmoid()
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
        if image_feat is not None:
            image_feat = self.image_proj(image_feat)
        if audio_feat is not None:
            audio_feat = self.audio_proj(audio_feat)
        
        feats = [text_feat]
        if image_feat is not None:
            feats.append(image_feat)
        if audio_feat is not None:
            feats.append(audio_feat)
            
        concat_feat = torch.cat(feats, dim=-1)
        gated_weights = self.gate(concat_feat)
        
        # 门控相加融合，文本作为骨干基底
        pooled = text_feat
        if image_feat is not None:
            pooled = pooled + (image_feat * gated_weights)
        if audio_feat is not None:
            pooled = pooled + (audio_feat * gated_weights)
            
        logits = self.head(pooled)
        return ForwardOutput(logits=logits, pooled=pooled)

    def trainable_state_dict(self) -> Dict[str, object]:
        out = {"head": self.head.state_dict(), "gate": self.gate.state_dict()}
        if self.image_proj is not None:
            out["image_proj"] = self.image_proj.state_dict()
        if self.audio_proj is not None:
            out["audio_proj"] = self.audio_proj.state_dict()
        if self.blank_image is not None:
            out["blank_image"] = self.blank_image.data.detach().cpu()
        if self.blank_audio is not None:
            out["blank_audio"] = self.blank_audio.data.detach().cpu()
        return out


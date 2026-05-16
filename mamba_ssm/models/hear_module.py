from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _topk_mean(probs: torch.Tensor, mask: torch.Tensor, topk: int) -> torch.Tensor:
    probs = probs * mask.to(dtype=probs.dtype)
    if probs.size(1) == 0:
        return probs.new_zeros((probs.size(0), 1))
    k = min(max(int(topk), 1), int(probs.size(1)))
    vals, _ = torch.topk(probs, k=k, dim=1)
    return vals.mean(dim=1, keepdim=True)


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    logits = logits.masked_fill(mask == 0, -1e4)
    return F.softmax(logits, dim=dim)


class SourceAwareSequenceTagging(nn.Module):
    """Inject source and position information into token states."""

    def __init__(
        self,
        hidden_size: int,
        *,
        num_sources: int = 4,
        max_position: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_sources = int(num_sources)
        self.max_position = int(max_position)
        self.source_embedding = nn.Embedding(self.num_sources, hidden_size)
        self.position_embedding = nn.Embedding(self.max_position, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor, source_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seqlen, _ = hidden_states.shape
        device = hidden_states.device
        if source_ids is None:
            source_ids = torch.zeros((bsz, seqlen), dtype=torch.long, device=device)
        else:
            source_ids = source_ids.to(device=device, dtype=torch.long)
        source_ids = source_ids.clamp(min=0, max=self.num_sources - 1)

        position_ids = torch.arange(seqlen, device=device).unsqueeze(0).expand(bsz, seqlen)
        position_ids = position_ids.clamp(max=self.max_position - 1)

        out = hidden_states + self.source_embedding(source_ids) + self.position_embedding(position_ids)
        return self.dropout(self.norm(out))


class HierarchicalToxicEvidenceMining(nn.Module):
    """Mine token-, span-, and segment-level toxic evidence."""

    def __init__(
        self,
        hidden_size: int,
        *,
        span_kernel_sizes: Sequence[int] = (3, 5, 7),
        topk: int = 5,
        max_segments: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.topk = int(topk)
        self.max_segments = int(max_segments)
        kernels = tuple(int(k) for k in span_kernel_sizes) or (3, 5, 7)
        mid_dim = max(64, hidden_size // 2)

        self.token_scorer = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, 1),
        )
        self.span_convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=hidden_size,
                    out_channels=hidden_size,
                    kernel_size=k,
                    padding=k // 2,
                )
                for k in kernels
            ]
        )
        self.span_scorer = nn.Sequential(
            nn.LayerNorm(hidden_size * len(kernels)),
            nn.Linear(hidden_size * len(kernels), mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, 1),
        )
        self.segment_scorer = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, 1),
        )
        self.evidence_fusion = nn.Sequential(
            nn.LayerNorm(hidden_size * 3 + 3),
            nn.Linear(hidden_size * 3 + 3, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )

    def _segment_pool(
        self,
        hidden_states: torch.Tensor,
        segment_ids: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seqlen, dim = hidden_states.shape
        device = hidden_states.device
        if segment_ids is None:
            segment_ids = torch.zeros((bsz, seqlen), dtype=torch.long, device=device)
        else:
            segment_ids = segment_ids.to(device=device, dtype=torch.long)
        segment_ids = segment_ids.masked_fill(attention_mask == 0, -1)

        reprs = []
        masks = []
        for idx in range(self.max_segments):
            seg_mask = ((segment_ids == idx) & (attention_mask > 0)).to(dtype=hidden_states.dtype)
            denom = seg_mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
            seg_repr = (hidden_states * seg_mask.unsqueeze(-1)).sum(dim=1) / denom
            reprs.append(seg_repr)
            masks.append((seg_mask.sum(dim=1) > 0).to(dtype=hidden_states.dtype))
        segment_reprs = torch.stack(reprs, dim=1).reshape(bsz, self.max_segments, dim)
        segment_mask = torch.stack(masks, dim=1)
        return segment_reprs, segment_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        segment_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        attention_mask = attention_mask.to(device=hidden_states.device, dtype=hidden_states.dtype)

        token_logits = self.token_scorer(hidden_states).squeeze(-1)
        token_logits = token_logits.masked_fill(attention_mask == 0, -1e4)
        token_probs = torch.sigmoid(token_logits) * attention_mask
        token_alpha = _masked_softmax(token_logits, attention_mask, dim=1)
        z_token = (hidden_states * token_alpha.unsqueeze(-1)).sum(dim=1)
        p_token = _topk_mean(token_probs, attention_mask, self.topk)

        conv_in = hidden_states.transpose(1, 2)
        span_features = [conv(conv_in).transpose(1, 2) for conv in self.span_convs]
        span_hidden = torch.cat(span_features, dim=-1)
        span_logits = self.span_scorer(span_hidden).squeeze(-1)
        span_logits = span_logits.masked_fill(attention_mask == 0, -1e4)
        span_probs = torch.sigmoid(span_logits) * attention_mask
        span_alpha = _masked_softmax(span_logits, attention_mask, dim=1)
        z_span = (hidden_states * span_alpha.unsqueeze(-1)).sum(dim=1)
        p_span = _topk_mean(span_probs, attention_mask, self.topk)

        segment_reprs, segment_mask = self._segment_pool(hidden_states, segment_ids, attention_mask)
        segment_logits = self.segment_scorer(segment_reprs).squeeze(-1)
        segment_logits = segment_logits.masked_fill(segment_mask == 0, -1e4)
        segment_probs = torch.sigmoid(segment_logits) * segment_mask
        segment_alpha = _masked_softmax(segment_logits, segment_mask, dim=1)
        z_segment = (segment_reprs * segment_alpha.unsqueeze(-1)).sum(dim=1)
        p_segment = _topk_mean(segment_probs, segment_mask, self.topk)

        p_toxic = torch.maximum(torch.maximum(p_token, p_span), p_segment)
        fusion_input = torch.cat([z_token, z_span, z_segment, p_token, p_span, p_segment], dim=-1)
        z_toxic = self.evidence_fusion(fusion_input)

        aux = {
            "token_logits": token_logits,
            "token_probs": token_probs,
            "token_alpha": token_alpha,
            "span_logits": span_logits,
            "span_probs": span_probs,
            "segment_logits": segment_logits,
            "segment_probs": segment_probs,
            "segment_mask": segment_mask,
            "p_token": p_token,
            "p_span": p_span,
            "p_segment": p_segment,
            "p_toxic": p_toxic,
            "z_token": z_token,
            "z_span": z_span,
            "z_segment": z_segment,
            "z_toxic": z_toxic,
        }
        return z_toxic, aux


class EvasionIntentDetection(nn.Module):
    """Detect disclaimer, pseudo-instruction, and suffix evasion signals."""

    def __init__(self, hidden_size: int, *, topk: int = 5, dropout: float = 0.1) -> None:
        super().__init__()
        self.topk = int(topk)
        mid_dim = max(64, hidden_size // 2)
        self.evasion_scorer = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, 1),
        )
        self.evasion_fusion = nn.Sequential(
            nn.LayerNorm(hidden_size + 2),
            nn.Linear(hidden_size + 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        attention_mask = attention_mask.to(device=hidden_states.device, dtype=hidden_states.dtype)
        logits = self.evasion_scorer(hidden_states).squeeze(-1)
        logits = logits.masked_fill(attention_mask == 0, -1e4)
        probs = torch.sigmoid(logits) * attention_mask
        alpha = _masked_softmax(logits, attention_mask, dim=1)
        z_evasion_base = (hidden_states * alpha.unsqueeze(-1)).sum(dim=1)

        p_evasion = _topk_mean(probs, attention_mask, self.topk)
        seqlen = hidden_states.size(1)
        if seqlen == 0:
            suffix_weight = probs.new_zeros(probs.shape)
        else:
            suffix_weight = torch.linspace(0.0, 1.0, seqlen, device=hidden_states.device, dtype=probs.dtype)
            suffix_weight = suffix_weight.unsqueeze(0).expand_as(probs)
        suffix_probs = probs * suffix_weight
        p_suffix = _topk_mean(suffix_probs, attention_mask, self.topk)

        z_evasion = self.evasion_fusion(torch.cat([z_evasion_base, p_evasion, p_suffix], dim=-1))
        aux = {
            "evasion_logits": logits,
            "evasion_probs": probs,
            "evasion_alpha": alpha,
            "suffix_weight": suffix_weight,
            "p_evasion": p_evasion,
            "p_suffix_evasion": p_suffix,
            "z_evasion": z_evasion,
        }
        return z_evasion, aux


class EvidenceRetentionAdapter(nn.Module):
    """Inject HEAR evidence back into the MRGF fused representation."""

    def __init__(
        self,
        text_hidden_size: int,
        fused_size: int,
        *,
        adapter_hidden: int = 256,
        dropout: float = 0.1,
        use_modality_mask: bool = True,
        init_gate_bias: float = -2.0,
    ) -> None:
        super().__init__()
        self.use_modality_mask = bool(use_modality_mask)
        modality_dim = 2 if self.use_modality_mask else 0
        evidence_input_dim = text_hidden_size * 2 + 3 + modality_dim

        self.evidence_proj = nn.Sequential(
            nn.LayerNorm(evidence_input_dim),
            nn.Linear(evidence_input_dim, adapter_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_hidden, fused_size),
        )
        adapter_input_dim = fused_size * 2
        self.adapter = nn.Sequential(
            nn.LayerNorm(adapter_input_dim),
            nn.Linear(adapter_input_dim, adapter_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_hidden, fused_size),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(adapter_input_dim),
            nn.Linear(adapter_input_dim, adapter_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_hidden, fused_size),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, float(init_gate_bias))

    @staticmethod
    def _normalize_mask(mask: Optional[torch.Tensor], batch_size: int, device, dtype: torch.dtype) -> torch.Tensor:
        if mask is None:
            return torch.zeros((batch_size, 1), device=device, dtype=dtype)
        if mask.dim() == 1:
            mask = mask.unsqueeze(-1)
        return mask.to(device=device, dtype=dtype)

    def forward(
        self,
        fused_feat: torch.Tensor,
        z_toxic: torch.Tensor,
        z_evasion: torch.Tensor,
        p_toxic: torch.Tensor,
        p_evasion: torch.Tensor,
        p_suffix_evasion: torch.Tensor,
        image_mask: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        bsz = fused_feat.size(0)
        evidence_inputs = [z_toxic, z_evasion, p_toxic, p_evasion, p_suffix_evasion]
        if self.use_modality_mask:
            evidence_inputs.extend(
                [
                    self._normalize_mask(image_mask, bsz, fused_feat.device, fused_feat.dtype),
                    self._normalize_mask(audio_mask, bsz, fused_feat.device, fused_feat.dtype),
                ]
            )
        evidence_input = torch.cat(evidence_inputs, dim=-1)
        evidence_feat = self.evidence_proj(evidence_input)
        adapter_input = torch.cat([fused_feat, evidence_feat], dim=-1)
        delta = self.adapter(adapter_input)
        gate = torch.sigmoid(self.gate(adapter_input))
        fused_feat_hear = fused_feat + gate * delta
        return fused_feat_hear, {"evidence_feat": evidence_feat, "adapter_delta": delta, "adapter_gate": gate}


class HEARModule(nn.Module):
    """Hierarchical Evidence Anti-evasion Retention module."""

    def __init__(
        self,
        text_hidden_size: int,
        fused_size: int,
        *,
        num_sources: int = 4,
        max_position: int = 512,
        max_segments: int = 16,
        span_kernel_sizes: Sequence[int] = (3, 5, 7),
        topk: int = 5,
        adapter_hidden: int = 256,
        dropout: float = 0.1,
        use_modality_mask: bool = True,
    ) -> None:
        super().__init__()
        self.source_tagger = SourceAwareSequenceTagging(
            hidden_size=text_hidden_size,
            num_sources=num_sources,
            max_position=max_position,
            dropout=dropout,
        )
        self.toxic_miner = HierarchicalToxicEvidenceMining(
            hidden_size=text_hidden_size,
            span_kernel_sizes=span_kernel_sizes,
            topk=topk,
            max_segments=max_segments,
            dropout=dropout,
        )
        self.evasion_detector = EvasionIntentDetection(hidden_size=text_hidden_size, topk=topk, dropout=dropout)
        self.retention_adapter = EvidenceRetentionAdapter(
            text_hidden_size=text_hidden_size,
            fused_size=fused_size,
            adapter_hidden=adapter_hidden,
            dropout=dropout,
            use_modality_mask=use_modality_mask,
        )

    def forward(
        self,
        text_hidden_states: torch.Tensor,
        fused_feat: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        source_ids: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
        image_mask: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Dict[str, torch.Tensor] | torch.Tensor]]:
        bsz, seqlen, _ = text_hidden_states.shape
        if attention_mask is None:
            attention_mask = torch.ones((bsz, seqlen), device=text_hidden_states.device, dtype=text_hidden_states.dtype)
        attention_mask = attention_mask.to(device=text_hidden_states.device, dtype=text_hidden_states.dtype)

        tagged_hidden = self.source_tagger(text_hidden_states, source_ids=source_ids)
        z_toxic, toxic_aux = self.toxic_miner(tagged_hidden, attention_mask=attention_mask, segment_ids=segment_ids)
        z_evasion, evasion_aux = self.evasion_detector(tagged_hidden, attention_mask=attention_mask)
        fused_feat_hear, adapter_aux = self.retention_adapter(
            fused_feat=fused_feat,
            z_toxic=z_toxic,
            z_evasion=z_evasion,
            p_toxic=toxic_aux["p_toxic"],
            p_evasion=evasion_aux["p_evasion"],
            p_suffix_evasion=evasion_aux["p_suffix_evasion"],
            image_mask=image_mask,
            audio_mask=audio_mask,
        )
        aux: Dict[str, Dict[str, torch.Tensor] | torch.Tensor] = {
            "tagged_hidden": tagged_hidden,
            "toxic": toxic_aux,
            "evasion": evasion_aux,
            "adapter": adapter_aux,
        }
        return fused_feat_hear, aux

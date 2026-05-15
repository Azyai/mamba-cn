from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PAERModule(nn.Module):
    """Parallel Anti-Evasion Evidence Retention module.

    PAER reads token-level text hidden states in parallel with MRGF fusion and
    calibrates the toxic logit when local toxic evidence co-occurs with evasion
    or disclaimer-like signals.
    """

    def __init__(
        self,
        text_hidden_size: int,
        fused_size: int,
        num_labels: int = 2,
        toxic_label_id: int = 1,
        span_kernel_size: int = 5,
        topk: int = 3,
        beta: float = 1.0,
        lambda_logit: float = 1.0,
        dropout: float = 0.1,
        span_pooling: str = "topk",
        use_modality_mask: bool = True,
        balance_logits: bool = False,
    ) -> None:
        super().__init__()
        if num_labels != 2:
            raise ValueError("PAERModule currently supports binary classification only.")
        if toxic_label_id not in (0, 1):
            raise ValueError("toxic_label_id must be 0 or 1.")
        if span_pooling not in {"topk", "noisy_or"}:
            raise ValueError("span_pooling must be 'topk' or 'noisy_or'.")
        if span_kernel_size < 1:
            raise ValueError("span_kernel_size must be positive.")
        if span_kernel_size % 2 == 0:
            raise ValueError("span_kernel_size must be odd to keep sequence length unchanged.")

        self.text_hidden_size = int(text_hidden_size)
        self.fused_size = int(fused_size)
        self.num_labels = int(num_labels)
        self.toxic_label_id = int(toxic_label_id)
        self.non_toxic_label_id = 1 - int(toxic_label_id)
        self.span_kernel_size = int(span_kernel_size)
        self.topk = int(topk)
        self.beta = float(beta)
        self.lambda_logit = float(lambda_logit)
        self.dropout = float(dropout)
        self.span_pooling = str(span_pooling)
        self.use_modality_mask = bool(use_modality_mask)
        self.balance_logits = bool(balance_logits)

        mid_dim = max(64, self.text_hidden_size // 2)

        self.token_evidence_scorer = nn.Sequential(
            nn.LayerNorm(self.text_hidden_size),
            nn.Linear(self.text_hidden_size, mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, 1),
        )

        self.span_conv = nn.Conv1d(
            in_channels=self.text_hidden_size,
            out_channels=self.text_hidden_size,
            kernel_size=self.span_kernel_size,
            padding=self.span_kernel_size // 2,
        )
        self.span_scorer = nn.Sequential(
            nn.LayerNorm(self.text_hidden_size),
            nn.Linear(self.text_hidden_size, mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, 1),
        )

        self.evasion_scorer = nn.Sequential(
            nn.LayerNorm(self.text_hidden_size),
            nn.Linear(self.text_hidden_size, mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, 1),
        )

        gate_input_dim = self.fused_size + self.text_hidden_size + 2
        if self.use_modality_mask:
            gate_input_dim += 2
        self.risk_gate = nn.Sequential(
            nn.LayerNorm(gate_input_dim),
            nn.Linear(gate_input_dim, mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, 1),
            nn.Sigmoid(),
        )

    def config_dict(self) -> Dict[str, object]:
        return {
            "text_hidden_size": self.text_hidden_size,
            "fused_size": self.fused_size,
            "num_labels": self.num_labels,
            "toxic_label_id": self.toxic_label_id,
            "span_kernel_size": self.span_kernel_size,
            "topk": self.topk,
            "beta": self.beta,
            "lambda_logit": self.lambda_logit,
            "dropout": self.dropout,
            "span_pooling": self.span_pooling,
            "use_modality_mask": self.use_modality_mask,
            "balance_logits": self.balance_logits,
        }

    @staticmethod
    def _normalize_mask(mask: Optional[torch.Tensor], batch_size: int, device, dtype) -> torch.Tensor:
        if mask is None:
            return torch.zeros(batch_size, 1, device=device, dtype=dtype)
        if mask.dim() == 1:
            mask = mask.unsqueeze(-1)
        return mask.to(device=device, dtype=dtype)

    @staticmethod
    def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
        masked_logits = logits.masked_fill(mask <= 0, -1e4)
        return F.softmax(masked_logits, dim=dim)

    def _pool_span_probability(self, span_probs: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        span_probs = span_probs * attention_mask
        if self.span_pooling == "noisy_or":
            return 1.0 - torch.prod(1.0 - span_probs.clamp(min=0.0, max=0.999), dim=1, keepdim=True)

        k = min(max(self.topk, 1), span_probs.size(1))
        topk_vals, _ = torch.topk(span_probs, k=k, dim=1)
        return topk_vals.mean(dim=1, keepdim=True)

    def forward(
        self,
        text_hidden_states: torch.Tensor,
        fused_feat: torch.Tensor,
        base_logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_mask: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, seq_len, _ = text_hidden_states.shape
        device = text_hidden_states.device
        dtype = text_hidden_states.dtype

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=device, dtype=dtype)
        else:
            attention_mask = attention_mask.to(device=device, dtype=dtype)

        token_evidence_logits = self.token_evidence_scorer(text_hidden_states).squeeze(-1)
        token_alpha = self._masked_softmax(token_evidence_logits, attention_mask, dim=1)
        z_toxic = torch.sum(text_hidden_states * token_alpha.unsqueeze(-1), dim=1)

        span_hidden = self.span_conv(text_hidden_states.transpose(1, 2)).transpose(1, 2)
        span_logits = self.span_scorer(span_hidden).squeeze(-1)
        span_logits = span_logits.masked_fill(attention_mask <= 0, -1e4)
        span_probs = torch.sigmoid(span_logits) * attention_mask
        p_span = self._pool_span_probability(span_probs=span_probs, attention_mask=attention_mask)

        evasion_logits = self.evasion_scorer(text_hidden_states).squeeze(-1)
        evasion_logits = evasion_logits.masked_fill(attention_mask <= 0, -1e4)
        evasion_probs = torch.sigmoid(evasion_logits) * attention_mask
        k = min(max(self.topk, 1), seq_len)
        evasion_topk_vals, _ = torch.topk(evasion_probs, k=k, dim=1)
        p_evasion = evasion_topk_vals.mean(dim=1, keepdim=True)

        gate_inputs = [fused_feat, z_toxic, p_span, p_evasion]
        if self.use_modality_mask:
            image_mask_f = self._normalize_mask(image_mask, batch_size, device, dtype)
            audio_mask_f = self._normalize_mask(audio_mask, batch_size, device, dtype)
            gate_inputs.extend([image_mask_f, audio_mask_f])
        gate_input = torch.cat(gate_inputs, dim=-1)
        risk_gate = self.risk_gate(gate_input)

        risk_delta = self.lambda_logit * risk_gate * p_span * (1.0 + self.beta * p_evasion)
        final_logits = base_logits.clone()
        final_logits[:, self.toxic_label_id] = final_logits[:, self.toxic_label_id] + risk_delta.squeeze(-1)
        if self.balance_logits:
            final_logits[:, self.non_toxic_label_id] = (
                final_logits[:, self.non_toxic_label_id] - 0.5 * risk_delta.squeeze(-1)
            )

        aux_outputs = {
            "base_logits": base_logits,
            "final_logits": final_logits,
            "token_evidence_logits": token_evidence_logits,
            "token_evidence_alpha": token_alpha,
            "z_toxic": z_toxic,
            "span_logits": span_logits,
            "span_probs": span_probs,
            "p_span": p_span,
            "evasion_logits": evasion_logits,
            "evasion_probs": evasion_probs,
            "p_evasion": p_evasion,
            "risk_gate": risk_gate,
            "risk_delta": risk_delta,
        }
        return final_logits, aux_outputs


class PAERLoss(nn.Module):
    """Optional PAER auxiliary loss wrapper.

    The project can train PAER with plain CE/Focal on final logits. This helper
    is available for experiments that add token rationale labels, evasion labels,
    or clean/augmented consistency pairs later.
    """

    def __init__(
        self,
        lambda_span: float = 0.2,
        lambda_evasion: float = 0.2,
        lambda_consistency: float = 0.5,
    ) -> None:
        super().__init__()
        self.lambda_span = float(lambda_span)
        self.lambda_evasion = float(lambda_evasion)
        self.lambda_consistency = float(lambda_consistency)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        paer_aux: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        token_rationale_labels: Optional[torch.Tensor] = None,
        evasion_labels: Optional[torch.Tensor] = None,
        clean_logits: Optional[torch.Tensor] = None,
        aug_logits: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        loss_cls = self.ce_loss(logits, labels)
        loss = loss_cls
        loss_dict: Dict[str, torch.Tensor] = {"loss_cls": loss_cls.detach()}

        if token_rationale_labels is not None:
            token_logits = paer_aux["token_evidence_logits"]
            rationale = token_rationale_labels.to(device=token_logits.device, dtype=token_logits.dtype)
            if attention_mask is not None:
                mask = attention_mask.to(device=token_logits.device).bool()
                loss_span = F.binary_cross_entropy_with_logits(token_logits[mask], rationale[mask])
            else:
                loss_span = F.binary_cross_entropy_with_logits(token_logits, rationale)
            loss = loss + self.lambda_span * loss_span
            loss_dict["loss_span"] = loss_span.detach()
        else:
            loss_dict["loss_span"] = torch.zeros((), device=logits.device)

        if evasion_labels is not None:
            p_evasion = paer_aux["p_evasion"].clamp(min=1e-6, max=1.0 - 1e-6)
            evasion = evasion_labels.to(device=p_evasion.device, dtype=p_evasion.dtype)
            if evasion.dim() == 1:
                evasion = evasion.unsqueeze(-1)
            loss_evasion = F.binary_cross_entropy(p_evasion, evasion)
            loss = loss + self.lambda_evasion * loss_evasion
            loss_dict["loss_evasion"] = loss_evasion.detach()
        else:
            loss_dict["loss_evasion"] = torch.zeros((), device=logits.device)

        if clean_logits is not None and aug_logits is not None:
            clean_prob = F.softmax(clean_logits.detach(), dim=-1)
            aug_prob = F.softmax(aug_logits, dim=-1)
            loss_consistency = F.mse_loss(aug_prob, clean_prob)
            loss = loss + self.lambda_consistency * loss_consistency
            loss_dict["loss_consistency"] = loss_consistency.detach()
        else:
            loss_dict["loss_consistency"] = torch.zeros((), device=logits.device)

        loss_dict["loss_total"] = loss
        return loss_dict

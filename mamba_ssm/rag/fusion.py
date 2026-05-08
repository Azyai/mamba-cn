from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class FusionWeights:
    model: float = 0.6
    bm25: float = 0.2
    vector: float = 0.15
    rule: float = 0.05


@dataclass(frozen=True)
class FusionResult:
    score: float
    label: int
    components: Dict[str, float]


def _normalize_weights(weights: FusionWeights) -> FusionWeights:
    total = float(weights.model + weights.bm25 + weights.vector + weights.rule)
    if total <= 0:
        return FusionWeights(model=1.0, bm25=0.0, vector=0.0, rule=0.0)
    return FusionWeights(
        model=float(weights.model / total),
        bm25=float(weights.bm25 / total),
        vector=float(weights.vector / total),
        rule=float(weights.rule / total),
    )


def fuse_scores(
    *,
    model_score: float,
    bm25_score: float,
    vector_score: float,
    rule_score: float,
    weights: FusionWeights,
    threshold: float = 0.5,
) -> FusionResult:
    w = _normalize_weights(weights)
    score = (
        float(model_score) * w.model
        + float(bm25_score) * w.bm25
        + float(vector_score) * w.vector
        + float(rule_score) * w.rule
    )
    label = 1 if score >= float(threshold) else 0
    return FusionResult(
        score=float(score),
        label=int(label),
        components={
            "model": float(model_score),
            "bm25": float(bm25_score),
            "vector": float(vector_score),
            "rule": float(rule_score),
        },
    )

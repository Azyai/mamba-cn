from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class RagDocument:
    doc_id: str
    title: str
    text: str
    doc_type: str
    tags: List[str] = field(default_factory=list)
    source: str = ""


@dataclass(frozen=True)
class RagHit:
    doc_id: str
    title: str
    doc_type: str
    score_bm25: float
    score_vec: float
    score_fused: float
    text_snippet: str
    source: str = ""


@dataclass(frozen=True)
class RagQueryResult:
    hits: List[RagHit]
    bm25_score: float
    vector_score: float
    rule_score: float
    rule_hits: List[str]
    meta: Dict[str, object]
    query: str = ""


@dataclass(frozen=True)
class RagRequest:
    query: str
    top_k: int = 5
    with_rules: bool = True
    max_snippet_chars: int = 180
    min_score: float = 0.0
    filters: Optional[Dict[str, object]] = None

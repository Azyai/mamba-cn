from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .text_utils import normalize_text, tokenize
from .types import RagDocument, RagHit, RagQueryResult, RagRequest


class _KeywordMatcher:
    def __init__(self, terms: Sequence[str]) -> None:
        self._terms = [t for t in (normalize_text(t) for t in terms) if t]
        self._use_aho = False
        self._aho = None
        try:
            import ahocorasick  # type: ignore

            automaton = ahocorasick.Automaton()
            for term in self._terms:
                automaton.add_word(term, term)
            automaton.make_automaton()
            self._aho = automaton
            self._use_aho = True
        except Exception:
            self._use_aho = False
            self._aho = None

    def find(self, text: str, *, max_hits: int = 20) -> List[str]:
        s = normalize_text(text)
        if not s:
            return []
        if self._use_aho and self._aho is not None:
            hits = []
            for _, term in self._aho.iter(s):
                hits.append(term)
                if len(hits) >= max_hits:
                    break
            return list(dict.fromkeys(hits))
        hits: List[str] = []
        for term in self._terms:
            if term in s:
                hits.append(term)
                if len(hits) >= max_hits:
                    break
        return hits


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _minmax_normalize(scores: Sequence[float]) -> List[float]:
    if not scores:
        return []
    min_s = min(scores)
    max_s = max(scores)
    if max_s <= min_s:
        return [0.0 for _ in scores]
    scale = max_s - min_s
    return [(float(s) - min_s) / scale for s in scores]


def _build_snippet(text: str, *, max_chars: int) -> str:
    if text is None:
        return ""
    s = str(text)
    if len(s) <= max_chars:
        return s
    return s[: max(0, max_chars - 3)] + "..."


class RagRetriever:
    def __init__(
        self,
        *,
        docs: List[RagDocument],
        tokens: List[List[str]],
        bm25,
        faiss_index,
        embedder,
        lexicon_terms: Sequence[str],
        index_meta: Dict[str, object],
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
        max_rule_hits: int = 20,
    ) -> None:
        self.docs = docs
        self.tokens = tokens
        self.bm25 = bm25
        self.faiss_index = faiss_index
        self.embedder = embedder
        self.lexicon_terms = list(lexicon_terms)
        self.index_meta = dict(index_meta)
        self.bm25_weight = float(bm25_weight)
        self.vector_weight = float(vector_weight)
        self.max_rule_hits = int(max_rule_hits)
        self.keyword_matcher = _KeywordMatcher(self.lexicon_terms)

    @classmethod
    def load(
        cls,
        index_dir: str | Path,
        *,
        device: str = "cpu",
        embedding_model: Optional[str] = None,
        max_rule_hits: int = 20,
    ) -> "RagRetriever":
        base = Path(index_dir).expanduser().resolve()
        if not base.exists():
            raise FileNotFoundError(f"RAG index dir not found: {base}")

        manifest_path = base / "manifest.json"
        manifest = _read_json(manifest_path) if manifest_path.exists() else {}
        model_name = embedding_model or str(manifest.get("embedding_model", ""))

        docs_path = base / "docs.jsonl"
        if docs_path.exists():
            rows = _read_jsonl(docs_path)
        else:
            rows = _read_json(base / "docs.json")
        docs = [RagDocument(**row) for row in rows]

        tokens_path = base / "bm25_tokens.json"
        tokens_raw = _read_json(tokens_path)
        tokens = [list(x) for x in tokens_raw]

        lexicon_terms: List[str] = []
        lexicon_path = base / "lexicon_terms.json"
        if lexicon_path.exists():
            lexicon_terms = [str(x) for x in _read_json(lexicon_path)]

        try:
            from rank_bm25 import BM25Okapi
        except Exception as exc:
            raise RuntimeError("Missing dependency: rank_bm25") from exc

        bm25 = BM25Okapi(tokens)

        faiss_index = None
        if (base / "faiss.index").exists():
            try:
                import faiss  # type: ignore
            except Exception as exc:
                raise RuntimeError("Missing dependency: faiss") from exc
            faiss_index = faiss.read_index(str(base / "faiss.index"))

        embedder = None
        if faiss_index is not None:
            if not model_name:
                raise RuntimeError("Embedding model missing in manifest or args")
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as exc:
                raise RuntimeError("Missing dependency: sentence-transformers") from exc
            embedder = SentenceTransformer(model_name, device=str(device))

        return cls(
            docs=docs,
            tokens=tokens,
            bm25=bm25,
            faiss_index=faiss_index,
            embedder=embedder,
            lexicon_terms=lexicon_terms,
            index_meta=manifest,
            bm25_weight=float(manifest.get("bm25_weight", 0.5)),
            vector_weight=float(manifest.get("vector_weight", 0.5)),
            max_rule_hits=max_rule_hits,
        )

    def _vector_search(self, query: str, *, top_k: int) -> Tuple[List[int], List[float]]:
        if self.faiss_index is None or self.embedder is None:
            return [], []
        if not query:
            return [], []
        vectors = self.embedder.encode([query], normalize_embeddings=True)
        try:
            import numpy as np
        except Exception as exc:
            raise RuntimeError("Missing dependency: numpy") from exc
        vec = np.asarray(vectors, dtype="float32")
        scores, indices = self.faiss_index.search(vec, int(top_k))
        if indices.size == 0:
            return [], []
        return [int(i) for i in indices[0].tolist()], [float(s) for s in scores[0].tolist()]

    def _filter_doc(self, doc: RagDocument, filters: Optional[Dict[str, object]]) -> bool:
        if not filters:
            return True
        doc_type = filters.get("doc_type")
        if doc_type is not None:
            if isinstance(doc_type, list):
                if doc.doc_type not in doc_type:
                    return False
            else:
                if doc.doc_type != str(doc_type):
                    return False
        tags = filters.get("tags")
        if tags is not None:
            want = {str(t) for t in tags} if isinstance(tags, list) else {str(tags)}
            if not want.intersection(set(doc.tags)):
                return False
        return True

    def query(self, request: RagRequest) -> RagQueryResult:
        t0 = time.time()
        query = normalize_text(request.query)
        tokens = tokenize(query)

        bm25_scores: List[float] = []
        if self.bm25 is not None and tokens:
            bm25_scores = [float(x) for x in self.bm25.get_scores(tokens)]
        bm25_norm = _minmax_normalize(bm25_scores) if bm25_scores else [0.0 for _ in self.docs]

        bm25_top = []
        if bm25_scores:
            bm25_top = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[: request.top_k]

        vec_idx, vec_scores = self._vector_search(query, top_k=request.top_k)
        vec_norm = [0.0 for _ in self.docs]
        if vec_scores:
            norm_scores = _minmax_normalize(vec_scores)
            for i, idx in enumerate(vec_idx):
                if 0 <= idx < len(vec_norm):
                    vec_norm[idx] = norm_scores[i]

        candidate_idx = set(bm25_top + vec_idx)
        if not candidate_idx and self.docs:
            candidate_idx = set(range(min(len(self.docs), request.top_k)))

        hits: List[RagHit] = []
        for idx in candidate_idx:
            doc = self.docs[idx]
            if not self._filter_doc(doc, request.filters):
                continue
            fused = float(bm25_norm[idx]) * self.bm25_weight + float(vec_norm[idx]) * self.vector_weight
            if fused < float(request.min_score):
                continue
            hits.append(
                RagHit(
                    doc_id=doc.doc_id,
                    title=doc.title,
                    doc_type=doc.doc_type,
                    score_bm25=float(bm25_norm[idx]),
                    score_vec=float(vec_norm[idx]),
                    score_fused=float(fused),
                    text_snippet=_build_snippet(doc.text, max_chars=int(request.max_snippet_chars)),
                    source=doc.source,
                )
            )

        hits.sort(key=lambda h: h.score_fused, reverse=True)
        hits = hits[: request.top_k]

        rule_hits: List[str] = []
        rule_score = 0.0
        if request.with_rules and self.keyword_matcher is not None:
            rule_hits = self.keyword_matcher.find(query, max_hits=self.max_rule_hits)
            rule_score = min(1.0, float(len(rule_hits)) / max(1.0, float(self.max_rule_hits)))

        bm25_score = max((h.score_bm25 for h in hits), default=0.0)
        vector_score = max((h.score_vec for h in hits), default=0.0)

        return RagQueryResult(
            hits=hits,
            bm25_score=float(bm25_score),
            vector_score=float(vector_score),
            rule_score=float(rule_score),
            rule_hits=rule_hits,
            meta={
                "elapsed_ms": (time.time() - t0) * 1000.0,
                "doc_count": len(self.docs),
                "bm25_weight": float(self.bm25_weight),
                "vector_weight": float(self.vector_weight),
                "index_meta": self.index_meta,
            },
            query=query,
        )

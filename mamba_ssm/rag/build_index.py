from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Tuple

from .text_utils import normalize_text, tokenize
from .types import RagDocument


def _normalize_device_name(device: str) -> str:
    value = str(device).strip().lower()
    if value in {"gpu", "cuda", "cuda:0"}:
        return "cuda"
    return value or "cpu"


def _iter_text_files(base: Path) -> Iterable[Path]:
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in {".txt", ".md", ".csv", ".tsv", ".json", ".jsonl"}:
            yield path


def _load_terms_from_text(path: Path) -> List[str]:
    terms: List[str] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        for sep in ("\t", ",", " "):
            if sep in raw:
                raw = raw.split(sep)[0]
        term = raw.strip()
        if term:
            terms.append(term)
    return terms


def _load_terms_from_json(path: Path) -> List[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    terms: List[str] = []
    if isinstance(data, list):
        for row in data:
            if isinstance(row, str):
                terms.append(row)
            elif isinstance(row, dict):
                for key in ("term", "word", "text", "name"):
                    if key in row:
                        terms.append(str(row[key]))
                        break
    elif isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, str):
                terms.append(value)
            else:
                terms.append(str(key))
    return terms


def load_lexicon_terms(lexicon_dir: Path, *, min_len: int = 2) -> Tuple[List[str], List[Tuple[str, str]]]:
    terms: List[str] = []
    sources: List[Tuple[str, str]] = []
    for path in _iter_text_files(lexicon_dir):
        if path.suffix.lower() == ".json":
            rows = _load_terms_from_json(path)
        else:
            rows = _load_terms_from_text(path)
        for term in rows:
            if len(term.strip()) < min_len:
                continue
            terms.append(term.strip())
            sources.append((term.strip(), path.name))
    dedup = {}
    for term in terms:
        dedup[term] = True
    return list(dedup.keys()), sources


def load_rule_docs(rules_dir: Path) -> List[RagDocument]:
    docs: List[RagDocument] = []
    idx = 0
    for path in _iter_text_files(rules_dir):
        idx += 1
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        docs.append(
            RagDocument(
                doc_id=f"rule-{idx}",
                title=path.stem,
                text=text,
                doc_type="rule_doc",
                tags=["rule"],
                source=str(path),
            )
        )
    return docs


def build_docs_from_lexicon(terms: List[str], sources: List[Tuple[str, str]]) -> List[RagDocument]:
    docs: List[RagDocument] = []
    source_map = {term: src for term, src in sources}
    for i, term in enumerate(terms, start=1):
        docs.append(
            RagDocument(
                doc_id=f"lexicon-{i}",
                title=term,
                text=term,
                doc_type="lexicon",
                tags=["lexicon"],
                source=source_map.get(term, ""),
            )
        )
    return docs


def build_index(
    *,
    docs: List[RagDocument],
    output_dir: Path,
    embedding_model: str,
    device: str,
    batch_size: int,
    bm25_weight: float,
    vector_weight: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    docs_path = output_dir / "docs.jsonl"
    with docs_path.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")

    tokenized = [tokenize(doc.text) for doc in docs]
    (output_dir / "bm25_tokens.json").write_text(json.dumps(tokenized, ensure_ascii=False), encoding="utf-8")

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise RuntimeError("Missing dependency: sentence-transformers") from exc
    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError("Missing dependency: numpy") from exc
    try:
        import faiss  # type: ignore
    except Exception as exc:
        raise RuntimeError("Missing dependency: faiss") from exc

    embedder = SentenceTransformer(embedding_model, device=_normalize_device_name(device))
    texts = [normalize_text(doc.text) for doc in docs]
    embeddings = embedder.encode(texts, batch_size=int(batch_size), normalize_embeddings=True)
    vecs = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, str(output_dir / "faiss.index"))

    manifest = {
        "embedding_model": embedding_model,
        "vector_dim": int(vecs.shape[1]),
        "doc_count": len(docs),
        "created_at": time.time(),
        "bm25_weight": float(bm25_weight),
        "vector_weight": float(vector_weight),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon_dir", type=str, required=True)
    ap.add_argument("--rules_dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--embedding_model", type=str, default="BAAI/bge-base-zh-v1.5")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--min_term_len", type=int, default=2)
    ap.add_argument("--bm25_weight", type=float, default=0.5)
    ap.add_argument("--vector_weight", type=float, default=0.5)
    args = ap.parse_args()

    lexicon_dir = Path(args.lexicon_dir).expanduser().resolve()
    rules_dir = Path(args.rules_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not lexicon_dir.exists():
        raise FileNotFoundError(f"lexicon_dir not found: {lexicon_dir}")
    if not rules_dir.exists():
        raise FileNotFoundError(f"rules_dir not found: {rules_dir}")

    terms, sources = load_lexicon_terms(lexicon_dir, min_len=int(args.min_term_len))
    lex_docs = build_docs_from_lexicon(terms, sources)
    rule_docs = load_rule_docs(rules_dir)
    docs = lex_docs + rule_docs

    build_index(
        docs=docs,
        output_dir=output_dir,
        embedding_model=str(args.embedding_model),
        device=_normalize_device_name(str(args.device)),
        batch_size=int(args.batch_size),
        bm25_weight=float(args.bm25_weight),
        vector_weight=float(args.vector_weight),
    )

    (output_dir / "lexicon_terms.json").write_text(json.dumps(terms, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

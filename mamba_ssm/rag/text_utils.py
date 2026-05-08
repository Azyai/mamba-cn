from __future__ import annotations

import re
from typing import List

try:
    import jieba
except Exception:  # pragma: no cover - optional dependency
    jieba = None


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    s = str(text)
    s = s.replace("\u3000", " ")
    s = _WHITESPACE_RE.sub(" ", s)
    return s.strip().lower()


def tokenize(text: str) -> List[str]:
    s = normalize_text(text)
    if not s:
        return []
    if jieba is not None:
        return [t for t in jieba.lcut(s) if t.strip()]
    return re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", s)

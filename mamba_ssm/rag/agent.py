from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from .types import RagQueryResult


@dataclass(frozen=True)
class AgentResponse:
    content: str
    model: str
    latency_ms: float
    used_llm: bool


def _format_hits(rag: RagQueryResult, *, max_hits: int = 5) -> str:
    lines: List[str] = []
    for hit in rag.hits[:max_hits]:
        lines.append(f"- [{hit.doc_type}] {hit.title}: {hit.text_snippet}")
    return "\n".join(lines) if lines else "- (no hits)"


def _build_prompt(
    *,
    query: str,
    rag: Optional[RagQueryResult],
    model_score: float,
    fusion_score: float,
    fusion_label: int,
) -> str:
    rag_block = ""
    if rag is not None:
        rag_block = (
            "检索证据：\n"
            f"{_format_hits(rag)}\n\n"
            f"规则命中：{', '.join(rag.rule_hits) if rag.rule_hits else '无'}\n"
            f"BM25 分数：{rag.bm25_score:.4f} | 向量分数：{rag.vector_score:.4f} | 规则分数：{rag.rule_score:.4f}\n"
        )
    return (
        "你是一个中文安全分析助手，负责判断文本、图片和音频中是否存在攻击性或毒性表达。"
        "请始终用中文回答，语气简洁、专业、可直接给用户阅读。"
        "如果提供了检索证据，请优先结合证据解释，不要输出英文。\n\n"
        f"输入内容：\n{query}\n\n"
        f"模型分数（有毒概率）：{model_score:.4f}\n"
        f"融合分数：{fusion_score:.4f} | 融合标签：{fusion_label}\n\n"
        f"{rag_block}\n"
        "请按以下结构输出 3 到 6 句中文：\n"
        "1. 先给出结论，说明是否存在毒性或攻击性风险。\n"
        "2. 再说明依据，若有检索证据请点明最关键的证据或规则命中。\n"
        "3. 最后给出建议，例如如何改写、如何进一步确认或如何处理。"
    )


class AgentClient:
    def __init__(
        self,
        *,
        model: str,
        api_key: Optional[str],
        base_url: Optional[str],
        timeout_s: float = 20.0,
        temperature: float = 0.2,
    ) -> None:
        self.model = str(model)
        self.api_key = api_key
        self.base_url = base_url
        self.timeout_s = float(timeout_s)
        self.temperature = float(temperature)
        self._client = None
        self._enabled = False

        if not self.api_key:
            return
        try:
            from langchain_openai import ChatOpenAI
        except Exception:
            return
        self._client = ChatOpenAI(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            timeout=self.timeout_s,
        )
        self._enabled = True

    @classmethod
    def from_env(cls) -> "AgentClient":
        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.deepseek.com"
        model = os.getenv("DEEPSEEK_MODEL") or "deepseek-chat"
        timeout_s = float(os.getenv("DEEPSEEK_TIMEOUT_S", "20"))
        temperature = float(os.getenv("DEEPSEEK_TEMPERATURE", "0.2"))
        return cls(model=model, api_key=api_key, base_url=base_url, timeout_s=timeout_s, temperature=temperature)

    @property
    def enabled(self) -> bool:
        return bool(self._enabled)

    def analyze(
        self,
        *,
        query: str,
        rag: Optional[RagQueryResult],
        model_score: float,
        fusion_score: float,
        fusion_label: int,
    ) -> AgentResponse:
        prompt = _build_prompt(
            query=query,
            rag=rag,
            model_score=float(model_score),
            fusion_score=float(fusion_score),
            fusion_label=int(fusion_label),
        )

        if not self._enabled or self._client is None:
            fallback = (
                f"模型分数：{model_score:.4f}。 "
                f"融合分数：{fusion_score:.4f}（标签={fusion_label}）。 "
                "当前未配置 LLM，返回的是中文模板结果。"
            )
            return AgentResponse(content=fallback, model="none", latency_ms=0.0, used_llm=False)

        t0 = time.time()
        try:
            response = self._client.invoke(prompt)
            content = getattr(response, "content", str(response))
        except Exception as exc:
            content = f"LLM call failed: {exc}"
        dt = (time.time() - t0) * 1000.0
        return AgentResponse(content=str(content), model=self.model, latency_ms=float(dt), used_llm=True)

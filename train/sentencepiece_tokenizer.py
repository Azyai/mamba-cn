from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch


@dataclass(frozen=True)
class SentencePieceTokenizerConfig:
    model_file: str
    add_bos: bool = False
    add_eos: bool = True


class SentencePieceTokenizer:
    def __init__(self, cfg: SentencePieceTokenizerConfig):
        try:
            import sentencepiece as spm  # type: ignore
        except Exception as e:
            raise RuntimeError("缺少依赖 sentencepiece。请安装: pip install sentencepiece") from e

        self.cfg = cfg
        self.model_path = Path(cfg.model_file).expanduser().resolve()
        self.sp = spm.SentencePieceProcessor(model_file=str(self.model_path))

        self.bos_id = int(self.sp.bos_id()) if self.sp.bos_id() >= 0 else None
        self.eos_id = int(self.sp.eos_id()) if self.sp.eos_id() >= 0 else None
        self.pad_id = int(self.sp.pad_id()) if self.sp.pad_id() >= 0 else None
        if self.pad_id is None:
            self.pad_id = self.eos_id if self.eos_id is not None else 0

    def encode(self, text: str) -> List[int]:
        ids: List[int] = list(self.sp.encode(text, out_type=int))
        if self.cfg.add_bos and self.bos_id is not None:
            ids = [self.bos_id] + ids
        if self.cfg.add_eos and self.eos_id is not None:
            ids = ids + [self.eos_id]
        return ids

    def __call__(
        self,
        texts: List[str],
        *,
        truncation: bool = True,
        max_length: int = 256,
        padding: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        if return_tensors != "pt":
            raise ValueError("SentencePieceTokenizer 仅支持 return_tensors='pt'")

        encoded = [self.encode(t) for t in texts]
        if truncation:
            encoded = [ids[:max_length] for ids in encoded]

        if not padding:
            input_ids = torch.tensor(encoded, dtype=torch.long)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        max_len = max((len(ids) for ids in encoded), default=0)
        max_len = min(max_len, max_length) if truncation else max_len
        if max_len <= 0:
            max_len = 1

        input_ids = torch.full((len(encoded), max_len), int(self.pad_id), dtype=torch.long)
        attention_mask = torch.zeros((len(encoded), max_len), dtype=torch.long)
        for i, ids in enumerate(encoded):
            n = min(len(ids), max_len)
            if n > 0:
                input_ids[i, :n] = torch.tensor(ids[:n], dtype=torch.long)
                attention_mask[i, :n] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


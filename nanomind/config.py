"""NanoMindConfig：Tier0/1/2 共用的模型設定類別。

這個檔案本身不含任何神經網路實作（那是 Phase 3 `nanomind/model.py` 的
工作），只負責「一組超參數合不合法、以及缺省值怎麼推導」這件事，屬於
確定性強、適合 TDD 的工程元件。
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def compute_intermediate_size(hidden_size: int) -> int:
    """SwiGLU FFN 的中間維度，沿用 minimind 的公式：
    ceil(hidden_size * pi / 64) * 64
    （取 64 的倍數，方便之後的矩陣運算對齊）。
    """
    return math.ceil(hidden_size * math.pi / 64) * 64


@dataclass
class NanoMindConfig:
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_seq_len: int
    intermediate_size: int | None = None
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "hidden_size 必須能被 num_attention_heads 整除，"
                f"目前 hidden_size={self.hidden_size}, "
                f"num_attention_heads={self.num_attention_heads}"
            )
        if self.num_key_value_heads > self.num_attention_heads:
            raise ValueError(
                "num_key_value_heads 不能大於 num_attention_heads（GQA 分組數"
                f"不合理），目前 num_key_value_heads={self.num_key_value_heads}, "
                f"num_attention_heads={self.num_attention_heads}"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads 必須能被 num_key_value_heads 整除"
                "（GQA 分組必須平均），目前 "
                f"num_attention_heads={self.num_attention_heads}, "
                f"num_key_value_heads={self.num_key_value_heads}"
            )
        if self.intermediate_size is None:
            self.intermediate_size = compute_intermediate_size(self.hidden_size)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NanoMindConfig:
        return cls(**data)

    @classmethod
    def from_json(cls, path: str | Path) -> NanoMindConfig:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

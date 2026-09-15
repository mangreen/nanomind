"""NanoMindConfig：Tier0 / Tier1 / Tier2 三種模型規模共用的設定類別。

## 這個檔案負責什麼
NanoMind 有三種預先設計好的模型規模（Tier0 最小、Tier1 是主力訓練目標、
Tier2 是進階版），但它們的「架構種類」其實一樣，差別只在幾個數字（隱藏
維度、層數、詞表大小...）。這個檔案定義的 `NanoMindConfig` 就是用來裝
這些數字的容器，並且負責兩件事：

1. **驗證這組數字合不合理**（例如 attention head 數量要能整除隱藏維度），
   如果不合理就直接報錯，而不是讓程式默默算出錯誤的結果。
2. **自動推導沒有明確指定的數字**（例如 FFN 的中間維度，通常不需要使用者
   手動指定，可以從 hidden_size 自動算出來）。

這個檔案**不包含任何神經網路的實際運算**（那是 Phase 3 `nanomind/model.py`
的工作，會用到這裡定義的 config 來決定要建多大的模型）。之所以先把「設定」
和「模型運算」分成兩個檔案，是因為「一組數字合不合法」這件事是完全確定性
的（同樣輸入永遠得到同樣結果），很適合寫測試去保護；但「模型怎麼運算」
牽涉到浮點數運算與隨機性，測試方式完全不同（詳見 tests/test_config.py
開頭的說明，以及 CLAUDE.md 的「測試哲學」章節）。

## 名詞小抄（如果你是第一次接觸 Transformer 語言模型）
- **hidden_size（隱藏維度）**：模型內部用來表示「一個 token 的意思」的
  向量長度。這個數字越大，模型能表達的資訊越豐富，但運算量也越大。
- **num_hidden_layers（層數）**：模型堆疊了幾層「理解→加工」的區塊。
- **num_attention_heads（注意力頭數）**：每一層裡，模型會同時用好幾個
  「視角」去看輸入的每個字之間的關聯性，這個數字就是視角的數量。
- **num_key_value_heads（KV 頭數，GQA 的核心概念）**：正常來說每個
  attention head 都要有自己的一份 Key/Value（模型內部用來比對「誰跟誰
  相關」的資料），但這樣很佔記憶體。GQA（Grouped-Query Attention，分組
  查詢注意力）讓好幾個 attention head **共用**同一份 Key/Value，藉此省
  記憶體。`num_key_value_heads` 就是「實際有幾組獨立的 Key/Value」，
  一定要能被 `num_attention_heads` 整除（例如 4 個 attention head 共用
  2 組 KV，就是每 2 個一組）。
- **intermediate_size（FFN 中間維度）**：每一層裡除了 attention，還有
  一個「前饋神經網路」（Feed-Forward Network, FFN）負責進一步加工資訊，
  這個數字是它內部暫時展開的維度，通常比 hidden_size 大好幾倍。
- **SwiGLU**：目前主流語言模型（包括 minimind 和 NanoMind）採用的 FFN
  設計方式，比最原始的 FFN 多一個「閘門」機制去控制資訊流量，效果比較
  好。我們不會在這個檔案裡實作 SwiGLU 本身（那在 Phase 3 的
  `nanomind/model.py`），這裡只是先把它需要的「中間維度」這個數字算好。
- **vocab_size（詞表大小）**：Tokenizer 能認得的「詞元（token）」總共有
  幾種。這個數字會直接影響模型最後一層（把內部向量轉成「這個字是誰」的
  機率分布）的運算量，所以 Phase 2 才會特別討論詞表大小的取捨。
- **max_seq_len（最長序列長度）**：模型一次最多能處理幾個 token。
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def compute_intermediate_size(hidden_size: int) -> int:
    """算出 SwiGLU FFN 的中間維度，不用使用者自己手動決定這個數字。

    公式沿用 minimind 專案的慣例：``ceil(hidden_size * pi / 64) * 64``。

    白話解釋這個公式在幹嘛：
    - ``hidden_size * pi``：SwiGLU 因為多了一個「閘門」，實際需要的參數量
      跟傳統 FFN 不一樣，經驗上抓「hidden_size 的 π 倍」左右可以讓
      SwiGLU 版本跟同樣參數量的傳統 FFN 打平，不會因為多了閘門機制而讓
      整體模型變胖太多。這是業界（包括 LLaMA 系列）常用的經驗公式，不是
      NanoMind 自己發明的。
    - ``ceil(... / 64) * 64``：把算出來的數字「無條件進位到 64 的倍數」。
      這是為了讓矩陣運算的維度對齊到電腦運算比較有效率的邊界（64 是常見
      的 SIMD/快取對齊單位），不是為了美觀。

    Args:
        hidden_size: 模型的隱藏維度（見本檔案開頭的名詞小抄）。

    Returns:
        算出來的 FFN 中間維度，保證是 64 的倍數。
    """
    return math.ceil(hidden_size * math.pi / 64) * 64


@dataclass
class NanoMindConfig:
    """一組完整、合法的 NanoMind 模型規模設定。

    建立這個物件時，`__post_init__`（見下方）會自動檢查所有數字是否合理，
    不合理會直接丟出 `ValueError`，而不是讓你在幾百個 training step 之後
    才發現某個維度對不上而崩潰——**愈早報錯，除錯成本愈低**，這是這個
    class 存在的核心價值。
    """

    # --- 必填欄位：沒有「合理的預設值」，一定要由呼叫者明確指定 ---
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_seq_len: int

    # --- 選填欄位：有預設值，或是可以從其他欄位自動推導 ---
    # intermediate_size 留 None 代表「請自動計算」，見 __post_init__。
    intermediate_size: int | None = None
    # RMSNorm（一種正規化技巧）計算時避免除以 0 的小常數，通常不需要調整。
    rms_norm_eps: float = 1e-5
    # RoPE（Rotary Position Embedding，旋轉位置編碼）用來讓模型知道
    # 「每個 token 在句子裡的位置」的一個超參數，通常不需要調整。
    rope_theta: float = 10000.0
    # 是否讓「輸入詞嵌入」和「輸出預測層」共用同一份權重矩陣。共用可以
    # 省下 vocab_size * hidden_size 這麼多參數，NanoMind 這種小模型
    # 幾乎都會開啟（minimind 的預設也是開啟）。
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        """在物件建立完成的當下，立刻檢查所有數字合不合理。

        dataclass 會在 `__init__` 的最後自動呼叫這個方法，所以我們不需要
        額外呼叫，只要 `NanoMindConfig(...)` 建立成功，就代表這組設定
        已經通過以下所有檢查。
        """
        # 檢查 1：hidden_size 要能平均分給每一個 attention head。
        # 例如 hidden_size=128、num_attention_heads=4，代表每個 head
        # 分到 128/4=32 維，這是合理的；如果分不整除，代表這組數字組合
        # 在數學上就是錯的，不可能建出模型。
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "hidden_size 必須能被 num_attention_heads 整除，"
                f"目前 hidden_size={self.hidden_size}, "
                f"num_attention_heads={self.num_attention_heads}"
            )

        # 檢查 2：GQA 的 KV 頭數不能比 attention head 數量還多。
        # GQA 的精神是「多個 attention head 共用一組 KV」，所以 KV 組數
        # 只能「少於或等於」attention head 數量，不可能反過來。
        if self.num_key_value_heads > self.num_attention_heads:
            raise ValueError(
                "num_key_value_heads 不能大於 num_attention_heads（GQA 分組數"
                f"不合理），目前 num_key_value_heads={self.num_key_value_heads}, "
                f"num_attention_heads={self.num_attention_heads}"
            )

        # 檢查 3：attention head 數量要能平均分給每一組 KV。
        # 例如 4 個 attention head、2 組 KV，代表「每 2 個 head 共用 1 組
        # KV」，分組平均、合理；如果像 5 個 head 配 2 組 KV，就會有一組
        # 要負責 3 個 head、另一組只負責 2 個，分組不平均，實作上會很
        # 麻煩且沒有必要，所以直接視為不合法設定。
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads 必須能被 num_key_value_heads 整除"
                "（GQA 分組必須平均），目前 "
                f"num_attention_heads={self.num_attention_heads}, "
                f"num_key_value_heads={self.num_key_value_heads}"
            )

        # 如果使用者沒有明確指定 intermediate_size，就自動幫他算一個
        # 合理的預設值，使用者不需要自己背 SwiGLU 的維度公式。
        if self.intermediate_size is None:
            self.intermediate_size = compute_intermediate_size(self.hidden_size)

    def to_dict(self) -> dict[str, Any]:
        """把這組設定轉成一般的 Python dict，方便存成 JSON 檔案。

        Phase 3 會把 Tier0/1/2 各自的設定存成
        `config/tier0.json` / `tier1.json` / `tier2.json`，
        就是靠這個方法把物件變成可以寫進 JSON 檔的格式。
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NanoMindConfig:
        """從一個 dict 建立 NanoMindConfig（`to_dict` 的相反操作）。

        因為會重新呼叫 `__init__`（連帶觸發 `__post_init__` 的驗證），
        所以就算 dict 裡的數字被人手動改壞了，載入時一樣會被檢查出來，
        不會偷偷放行一組不合法的設定。
        """
        return cls(**data)

    @classmethod
    def from_json(cls, path: str | Path) -> NanoMindConfig:
        """直接從一個 JSON 檔案路徑載入設定，內部就是「讀檔 + from_dict」。

        Args:
            path: JSON 檔案的路徑，例如 ``config/tier1.json``。
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

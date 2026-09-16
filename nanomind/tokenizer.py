"""訓練並載入 NanoMind 自己的小型 Byte-Level BPE tokenizer。

## Tokenizer 是什麼、為什麼需要它
語言模型的神經網路只認得數字（向量），不認得文字，所以需要一個「翻譯」
的角色：把人類寫的文字切成一個一個「詞元（token）」，再把每個詞元對應
到一個固定的整數編號。這個「怎麼切、怎麼編號」的規則，就叫做
tokenizer。模型訓練完之後，這份 tokenizer 的規則也必須跟著模型一起保存
下來——如果訓練時用一套切法，之後推論時用另一套切法，模型會完全看不懂
輸入（因為同一個編號在兩套規則裡可能代表不同的意思）。

## 為什麼 NanoMind 要自己訓練一個小的，而不是直接用 minimind 現成的
詳細成本分析在 `docs/decisions/MEM-0002-tokenizer-strategy.md`。這裡
只提結論：minimind 官方 tokenizer 的詞表有 6400 個詞元，這個數字會
直接決定模型最後一層（把內部向量轉換成「這個詞元是誰」的機率分布）的
運算量，對 NanoMind 想要的「小而快」規模來說太浪費了，所以我們用這個
檔案，在自己的訓練語料上，訓練一個小很多的詞表（Tier1 目標 1536）。

## 為什麼選 BPE，又為什麼是「Byte-Level」
BPE（Byte Pair Encoding）的想法很直覺：一開始把文字拆成最小單位，然後
統計「哪兩個單位最常黏在一起出現」，把出現頻率最高的那一對合併成一個新
的、更大的單位，不斷重複這個過程，直到詞表大小達到我們設定的目標
（例如 1536 個）。常見的合併結果，剛好就是「常用的字、詞、詞綴」。

「Byte-Level」指的是「一開始的最小單位」選擇「原始 UTF-8 位元組」（總共
只有 256 種可能），而不是「Unicode 字元」。這個選擇的好處是**保證任何
文字都能無損地編碼、解碼**——不管是中文、日文、emoji，甚至是亂碼，
反正任何文字最終都能拆解成 UTF-8 位元組，詞表裡一定找得到對應的
基礎單位，不會遇到「這個字元沒在詞表裡」的窘境（這是 char-level
tokenizer 常見的痛點）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

# NanoMind 目前需要的特殊 token，數量刻意保持精簡：
# - <unk>：遇到 tokenizer 完全無法辨識的內容時的保底代表（理論上因為是
#   byte-level，幾乎不會真的用到，但 BPE 的模型定義要求一定要指定一個）
# - <pad>：訓練時要把長度不一的句子湊成同一個 batch，短的句子後面要用
#   什麼「填充」，就是這個 token（模型看到它會知道「這裡不是真正的內容，
#   不用列入 loss 計算」）
# - <|im_start|> / <|im_end|>：sft 對話資料的「一輪對話開始/結束」標記，
#   沿用 minimind 的慣例命名，讓 Phase 5 SFT 的 chat template 設計可以
#   直接複用同一套慣例，不用再發明一套新的。
#
# 如果之後的 Phase 需要更多特殊 token（例如某種推理標記），必須重新
# 訓練 tokenizer 才能加進去——HF 的 BPE tokenizer 存好之後，特殊 token
# 的集合就固定了，不能事後偷加。
SPECIAL_TOKENS = ["<unk>", "<pad>", "<|im_start|>", "<|im_end|>"]


def extract_text_from_record(record: dict[str, Any]) -> str | None:
    """從一行已經解析好的 JSON 資料裡，取出可以拿去訓練 tokenizer 的純文字。

    NanoMind 的訓練資料有兩種 schema（沿用 minimind 的慣例，詳見
    `CLAUDE.md`）：
    - pretrain 格式：``{"text": "一大段純文字"}``
    - sft 對話格式：``{"conversations": [{"role": ..., "content": ...}, ...]}``

    Args:
        record: 已經用 `json.loads` 解析好的一筆資料。

    Returns:
        可以拿去訓練 tokenizer 的文字字串；如果這筆資料兩種格式都不符合，
        回傳 `None`（由呼叫端決定要不要跳過，這裡不負責報錯，因為單一
        一行資料格式不對，不該讓整個訓練流程中斷）。
    """
    if "text" in record:
        return record["text"]

    if "conversations" in record:
        # 把每一輪對話包上 <|im_start|>角色\n內容<|im_end|> 的格式，
        # 讓 tokenizer 訓練時看到的排版方式，跟 Phase 5 SFT 階段真正
        # 餵給模型的格式一致。這樣做的好處是：如果某個角色名稱
        # （例如 "assistant"）常常緊接著某個標點符號出現，BPE 訓練器
        # 有機會學到「這一整段常見的排版方式」對應的合併，而不是完全沒
        # 看過這種格式，等到真正 SFT 訓練時才第一次遇到。
        parts = []
        for turn in record["conversations"]:
            role = turn.get("role", "")
            content = turn.get("content", "")
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        return "".join(parts)

    return None


def iter_corpus_texts(paths: list[str | Path]):
    """從一或多個 JSONL 檔案，逐行讀出可以拿去訓練 tokenizer 的文字。

    這是一個 generator（用 `yield`，不是 `return` 一個完整的 list），
    好處是不需要把整個檔案的內容一次讀進記憶體——對 GB 等級的原始資料
    檔案來說，這個差異很重要（雖然 Phase 2 訓練 tokenizer 用的是已經
    抽樣過的小檔案，用 generator 仍然是比較穩健、不會因為檔案變大就要
    重寫的寫法）。

    遇到壞掉的 JSON 行或空行，會直接跳過，不會讓整個訓練流程中斷——
    1.2GB 的真實資料裡難免會有幾行格式怪怪的資料。

    Args:
        paths: 一或多個 JSONL 檔案的路徑，會依序讀取。
    """
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = extract_text_from_record(record)
                if text:
                    yield text


def train_bpe_tokenizer(
    corpus_paths: list[str | Path],
    vocab_size: int,
    output_dir: str | Path,
    special_tokens: list[str] | None = None,
) -> PreTrainedTokenizerFast:
    """在給定的語料上訓練一個 Byte-Level BPE tokenizer，並存到指定資料夾。

    Args:
        corpus_paths: 一或多個 JSONL 語料檔案路徑（例如
            `dataset/nano_pretrain.jsonl` 跟 `dataset/nano_sft.jsonl`）。
        vocab_size: 希望訓練出來的詞表大小上限（實際結果可能略小於這個
            數字，如果語料本身的變化不夠豐富，BPE 訓練器會提早停止，
            這是正常現象）。
        output_dir: 訓練好的 tokenizer 要存到哪個資料夾。
        special_tokens: 要加入詞表的特殊 token 列表，預設用本檔案定義的
            `SPECIAL_TOKENS`。

    Returns:
        一個包好、可以直接拿來 `.encode()` / `.decode()` 的
        `PreTrainedTokenizerFast` 物件（HuggingFace transformers 生態系
        通用的 tokenizer 介面，方便之後 Phase 3+ 直接搭配使用）。
    """
    if special_tokens is None:
        special_tokens = SPECIAL_TOKENS

    # `models.BPE(unk_token=...)`：底層的 BPE 演算法本體，`unk_token`
    # 是理論上遇到完全無法辨識內容時的保底代表（byte-level 幾乎不會真的
    # 用到，但 API 要求一定要指定）。
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # `pre_tokenizers.ByteLevel`：在真正做 BPE 合併之前，先把輸入文字
    # 轉成一串 UTF-8 位元組（每個位元組會被映射成一個看起來像亂碼、但
    # 彼此不會混淆的顯示字元，這是 GPT-2 等模型開創的慣例寫法）。
    # `add_prefix_space=False`：不要在每段文字前面自動加一個空白字元，
    # 我們的訓練資料（尤其中文）前面加空白沒有意義。
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # decoder 要跟 pre_tokenizer 對應，負責把「一串詞元 id」還原成人類看
    # 得懂的原始文字，也就是 encode 的反向操作。
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        # `initial_alphabet`：一開始就把「256 種可能的位元組」全部塞進
        # 詞表的起點，確保任何輸入（不管多罕見的字元）都至少有基礎的
        # byte-level 表示方式可以退回去用，這是 byte-level BPE「保證
        # 無損還原」這個特性的關鍵。
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )

    tokenizer.train_from_iterator(iter_corpus_texts(corpus_paths), trainer=trainer)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_json_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_json_path))

    # 把底層的 `tokenizers.Tokenizer` 包成 HuggingFace transformers 生態
    # 系通用的 `PreTrainedTokenizerFast`，這樣 Phase 3+ 的模型程式碼可以
    # 用業界標準的介面（`.encode()`, `.decode()`, `.save_pretrained()`,
    # `.from_pretrained()`...）操作它，不用自己重新發明一套 API。
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_json_path),
        unk_token="<unk>",
        pad_token="<pad>",
    )
    hf_tokenizer.save_pretrained(str(output_dir))

    return hf_tokenizer


def load_tokenizer(tokenizer_dir: str | Path) -> PreTrainedTokenizerFast:
    """載入一個之前用 `train_bpe_tokenizer` 訓練好、存檔過的 tokenizer。

    Args:
        tokenizer_dir: 之前傳給 `train_bpe_tokenizer` 的 `output_dir`。
    """
    return PreTrainedTokenizerFast.from_pretrained(str(tokenizer_dir))

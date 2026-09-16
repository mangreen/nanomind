# MEM: Tokenizer 策略——自訓練小型 Byte-Level BPE
Date: 2026-09-16
Status: Approved
Tags: [tokenizer, phase-2, vocab-size]

## Summary
NanoMind 不沿用 minimind 官方內建的 tokenizer（vocab=6400），改為在自己
的訓練語料上，用 `tokenizers` 函式庫訓練一個小很多的 Byte-Level BPE
tokenizer。這份文件記錄具體的實作決策；原始的成本效益分析（為什麼詞表
大小很重要）已經在規劃階段完整討論過，完整保留在
`docs/decisions/MEM-0000-planning-draft.md`，這裡不重複貼一次，只記錄
「後來實際怎麼做」與「跟原始規劃相比有沒有調整」。

## Details

### Context & Evolution（脈絡與演進）
MEM-0000 的原始分析：沿用 minimind 6400 詞表時，LM-head 的輸出投影矩陣
運算量（batch × seq × hidden × vocab）在 Tier1 規模下約 33.5 億次
乘加，換成自訓練的 1536 詞表可以降到約 5.4 億次，縮小約 6 倍。這個結論
在 Phase 2 實作階段沒有變化，維持原本的決策：**自己訓練小 vocab**。

### Specification

#### 各 Tier 的目標 vocab_size
| Tier | vocab_size | 是否需要真的訓練 tokenizer |
|---|---|---|
| Tier0 | （不需要） | **否**——Tier0 的煙霧測試（Phase 4）直接用合成的整數
  token id 序列驗證訓練迴圈機制，不牽涉真實文字，所以不需要真的訓練
  tokenizer。這點是實作階段才確認的細節，MEM-0000 原始規劃沒有明確
  講到這一層，在此補充記錄。 |
| Tier1 | 1536 | 是（Phase 2 本次） |
| Tier2 | 4096 | 是（留到 Phase 10 Tier2 capstone 階段才訓練） |

#### 演算法選擇：Byte-Level BPE
- **BPE（Byte Pair Encoding）**：業界主流選擇（GPT 系列、LLaMA 系列、
  minimind 皆使用），從最小單位開始反覆合併高頻共現的 pair，直到達到
  目標詞表大小。
- **Byte-Level（而非 char-level 或 word-level）**：初始的最小單位選
  UTF-8 位元組（256 種可能），保證任何輸入文字都能無損編碼再解碼，不會
  遇到「詞表裡沒有這個字元」的問題。這對 NanoMind 特別重要，因為我們的
  語料是中英夾雜，如果用 char-level 又沒有把所有可能出現的中文字都涵蓋
  進初始詞表，會有解碼失敗的風險；byte-level 從根本上避免這個問題。
- 使用的函式庫：`tokenizers`（HuggingFace 官方的 Rust 實作，速度快，
  且原本就是 `transformers` 的相依套件，不用額外引入新的相依）。

#### 特殊 Token 集合
```
<unk>, <pad>, <|im_start|>, <|im_end|>
```
- `<unk>`：BPE 模型 API 要求必須指定一個，byte-level 架構下實務上幾乎
  不會真的用到。
- `<pad>`：訓練時把不同長度的序列湊成同一個 batch 用。
- `<|im_start|>` / `<|im_end|>`：對話輪次的開始/結束標記，**沿用
  minimind 的命名慣例**（而不是自己發明新名稱），這樣 Phase 5 SFT 的
  chat template 設計可以直接參考 minimind 既有的做法，降低之後的設計
  成本。

**重要限制，記錄下來避免之後忘記**：HuggingFace 的 BPE tokenizer 一旦
訓練、存檔完成，特殊 token 的集合就固定了，不能事後偷加。如果未來某個
Phase（例如涉及思考過程標記的進階實驗）需要新的特殊 token，必須整份
tokenizer 重新訓練，不是加一行設定就能解決的小事。目前的 4 個特殊
token 是根據「已經確定會用到」的需求（SFT chat template）決定的，
沒有為了「以後可能用到」而預先塞一堆用不到的特殊 token 進去（YAGNI
原則：You Aren't Gonna Need It）。

#### 訓練語料的文字抽取方式
- pretrain 資料（`{"text": ...}`）：直接取 `text` 欄位內容。
- sft 對話資料（`{"conversations": [...]}`）：把每一輪對話包成
  `<|im_start|>{role}\n{content}<|im_end|>\n` 的格式再串接。這樣訓練
  tokenizer 時看到的排版方式，會跟 Phase 5 真正餵給模型的格式一致，讓
  BPE 訓練器有機會學到這種排版方式常見的合併模式（例如角色名稱後面
  接換行字元的組合）。
- Tier0 的 pretrain 資料如果未來需要文件分隔符號，會重用
  `<|im_end|>` 而不是另外發明新 token（見上方 YAGNI 原則），實際怎麼
  分隔留給 Phase 4 的 dataset loader 決定，本文件先確保 token 已經
  存在、可以被重用。

#### 捨棄的替代方案
- **`datasets` 函式庫的串流讀取**：沒有採用，`nanomind/tokenizer.py`
  的 `iter_corpus_texts` 自己用純 Python generator 實作串流讀取
  JSONL，理由跟 MEM-0001 提到的「捨棄 HuggingFace datasets 套件」一致
  ——資料量本來就是幾 MB 等級，純 Python 已經足夠，不需要 Arrow 後端
  的額外相依與啟動成本。

## When to Use
任何時候要理解「NanoMind 的 tokenizer 為什麼長這樣」、或是要決定「要不要
加新的特殊 token」，先看這份文件；原始的成本效益數字計算過程看
MEM-0000。

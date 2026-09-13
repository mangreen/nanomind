# NanoMind 開發學習計畫（草案，待您確認）

我已經把三份參考資料都讀過一輪，先跟您同步一下我讀到的關鍵事實，再給出完整計畫。這份先只是**規劃**，不會動手寫任何程式碼或建立檔案。

## 0. 我從三份資料裡確認到的事實

**minimind-master**（我們的骨幹範本）
- 核心是 `model/model_minimind.py`：一個乾淨的 Llama 風格 decoder-only Transformer（RMSNorm + RoPE + GQA + SwiGLU + 可選 MoE），單檔案、無花俏抽象，非常適合當「每一行都看得懂」的教學基底。
- 官方最小配置是 `minimind-3`：8 層、d_model=768、**64M 參數**，號稱「3 塊人民幣、2 小時」，但那是**單張 3090**的數字，而且連「mini」資料集都是 `pretrain_t2t_mini.jsonl`（1.2GB）+ `sft_t2t_mini.jsonl`（1.6GB）——這對您的機器完全不現實。
- `trainer/train_pretrain.py`、`train_full_sft.py`、`train_tokenizer.py`、`dataset/lm_dataset.py` 的資料格式很單純：pretrain 是 `{"text": ...}`、SFT 是 `{"conversations":[{"role":..,"content":..}]}`。這個 schema 我們會沿用。

**deepseek-recipe-main**（跟您想的可能不太一樣）
- 這**不是訓練程式**，而是一個 Rust/Python 函式庫，功能是把 OpenAI 風格的 API 請求轉成 DeepSeek V4/V4.1 的 prompt 格式（serving 端的協議轉換層）。
- 對 NanoMind 真正有用的地方只有兩個：① 它把「對話 → 結構化 prompt 編碼／解碼」拆成獨立模組的設計方式，可以借鏡到我們的 chat template 設計；② 當作「Ports & Adapters」概念的實例教材。**我不會假裝拿它來訓練模型**。

**DeepSeek_V4.1_Tech_Report.pdf**
- 552B 參數、CED、CSA2、Engram、DSpark、FP4 KV cache…這些技術解決的是「百萬 token 上下文、HBM/SSD 頻寬」等問題。NanoMind 目標是 context 128–384 tokens、參數量 <10M，規模差 5–7 個數量級。**這份報告我會當成「概念詞彙表」寫成學習筆記，而不是實作目標**——這點我會在計畫最後說明怎麼處理，才不會浪費這份資料。

## 1. 硬體與環境現實檢查（Phase 0 的核心風險）

您的機器是 **2014 年 13" MacBook Pro（雙核 i7 3GHz、16GB RAM、Intel Iris 1536MB，macOS Big Sur 11.7.11）**：

| 項目 | 現實限制 | 對應決策 |
|---|---|---|
| GPU | Intel Iris（非 NVIDIA、非 Apple Silicon） | **完全 CPU 訓練**，不用談 CUDA/MPS |
| PyTorch | PyTorch 從 **2.3.0** 起不再發布 macOS Intel(x86_64) wheel，最後可用版本是 **2.2.2**（需 Python ≤3.12） | 鎖定 `torch==2.2.2`，Python 建議 3.10 或 3.11 |
| minimind requirements.txt | 鎖 `transformers==4.57.6`（很新版本），是否與 `torch==2.2.2` 相容**未經驗證** | 列為 Phase 0 待驗證風險，若不相容則另鎖較舊版本，並記成 ERR 文件 |
| CPU 核心數 | 實體雙核（4 邏輯執行緒） | DataLoader `num_workers` 建議設 0–1，避免搶核心；訓練前先做 benchmark 校準 batch size |
| RAM 16GB | 對 <10M 參數模型完全綽綽有餘 | 這不是瓶頸，不需要特別優化 |

這些都會在 **Phase 0** 用實測方式驗證，而不是我在這裡憑空保證數字。

## 2. 開發規範 v3.0 → 本專案 Scope 對照（Trade-off Analysis）

您說「可以根據 scope 調整，但 git commit、架構設計、設計修改、錯誤要完整記錄」——這正是規範 v3.0 第 1 節「pragmatic, not dogmatic」的精神。以下是我建議保留 / 簡化 / 捨棄的對照：

| 規範項目 | 原規範要求 | NanoMind 本專案做法 | 理由 |
|---|---|---|---|
| Git 語意化 commit | feat/fix/test/refactor/docs/chore | **完整保留，不簡化** | 低成本高價值，且您明確要求 |
| MEM / ERR / SKILL / todo.md | 多服務、分散管理 | **完整保留**，但 ADR 併入 MEM（同一份文件兼任架構決策記錄，避免概念重複） | 單一專案不需要 ADR+MEM 兩套並行 |
| TDD Red-Green-Refactor | 全面套用，含大量整合測試 | 只在**確定性強**的工程元件套用（tokenizer/config/dataset/checkpoint/shape）；訓練迴圈本身改用「**小樣本過擬合測試**」驗證正確性 | 訓練是統計過程，`assert loss==x` 沒意義；但「能否把幾筆資料背到 loss≈0」是 ML 工程界公認驗證 training loop 正確性的標準手法 |
| 覆蓋率 80% 門檻 | CI 強制 | 只對 engineering 模組（tokenizer/dataset/config/checkpoint）要求 ≥70%；訓練/生成迴圈不列入覆蓋率統計，改用煙霧測試 | 訓練迴圈覆蓋率數字沒有工程意義 |
| Clean/Hexagonal/DDD 分層 | Domain/Application/Infrastructure 三層 + Ports&Adapters | 簡化為 4 個扁平模組（tokenizer/model/dataset/trainer），介面用簡單約定取代正式抽象層 | 單人單機 CLI 腳本，過度分層違反 KISS |
| Observability（trace-id/全域例外處理） | 微服務等級 | 簡化為本機 JSONL log（step/loss/lr/tokens_per_sec） | 沒有分散式場景 |
| 機密管理（.env/gitleaks） | 完整規範 | 不適用（無 API 金鑰），僅保留 `.gitignore` 排除大型資料/checkpoint | 專案不涉及任何憑證 |
| 日報/週報/月報 | 依日曆週期 | 改為「依 Phase 產出」的 `docs/reports/phase-N.md` | 這是零碎時間的個人學習專案，不綁日曆節奏 |
| Docker/K8s/K6 壓測/前端 MVP | production-grade | **全部移除** | 沒有部署與壓測場景 |
| 六階段開發迴圈 + 每階段確認關卡 | 通用軟體六階段 | **精神完全保留**，階段改對應 ML pipeline（見下） | 這正是您要的「先規劃、每步確認」流程 |

## 3. NanoMind 技術決策（含具體 Trade-off 計算）

### 3.1 詞表策略：不要直接沿用 minimind 的 6400 詞表

`[Trade-off Analysis]`
- **選項A：沿用 minimind 內建 tokenizer（vocab=6400）**
  - Pros：現成、已含 chat template、不用重訓
  - Cons：Embedding 綁定 LM head 時，**輸出投影矩陣的運算量 = batch × seq × hidden × vocab**。以 Tier1（hidden=128, seq=256, batch=16）估算：16×256×128×6400 ≈ **33.5 億次乘加**——這會讓輸出層計算量遠超過整個 Transformer 主體，完全違背「nano 模型應該快」的初衷
- **選項B：用 minimind 的 `train_tokenizer.py` 邏輯，自己在小語料上訓練 vocab≈1024–1536 的小 BPE**
  - Pros：同樣算式下運算量降到約 5.4 億次乘加（**縮小約 6 倍**），且詞表大小本來就該跟語料規模匹配
  - Cons：多一個「訓練詞表」的步驟，需要驗證中英文編碼正確性
- **決策：採用選項B**，理由：這是決定「nano 是不是真的 nano」的關鍵一步，多花一個步驟換取數倍訓練效率划算。（會記錄為 MEM-0002）

### 3.2 模型規模分級（三個 Tier，實際參數量會在 Phase 3 用程式驗證）

| Tier | hidden | layers | heads/kv_heads | vocab | seq_len | 估計參數量 | 定位 |
|---|---|---|---|---|---|---|---|
| Tier0（煙霧測試） | 64 | 2 | 4/2 | 512 | 128 | **≈0.15M** | 驗證整條 pipeline 通不通，分鐘級 |
| **Tier1（建議預設）** | 128 | 4 | 4/2 | 1536 | 256 | **≈1.1M** | 真正的 NanoMind，正式訓練目標 |
| Tier2（進階選配） | 256 | 6 | 8/2 | 4096 | 384 | **≈5.9M** | 只有 Tier1 順利且時間充裕才做 |

（估算方法沿用 minimind config 公式 `intermediate_size = ceil(hidden×π/64)×64`，詳細算式會寫進 MEM 文件）

### 3.3 資料策略：自製小語料，不下載 minimind 的 GB 級檔案

`[Trade-off Analysis]`
- **選項A：下載 minimind 的 `*_mini.jsonl`（合計 2.8GB）再抽樣一小份**：方便，但要下載近 3GB 只為了用其中不到 1%，也違背「從 0 開始、自己理解每一步資料」的學習精神
- **選項B：自行準備／挑選公版語料（幾 MB 等級），照 minimind 的 schema 存成 `nano_pretrain.jsonl` / `nano_sft.jsonl`**：需要自己動手整理，但完全對齊「從0學習」目標，也是 DeepSeek 報告裡強調的「資料建構是第一線工作」精神的縮影版練習
- **決策：預設採用選項B**，選項A留作備援（如果您想先求快）

### 3.4 其他工程決策
- **Vendor 而非依賴 minimind 套件**：把 `model_minimind.py` 複製並精簡進 `nanomind/model.py`，移除 MoE 分支（連帶移除 `MoeCausalLMOutputWithPast` 的 import），降低對特定 `transformers` 版本的耦合（MEM-0001）。
- **捨棄 HuggingFace `datasets` 套件**：我們的資料只有幾 MB，用純 Python 讀 JSONL 就夠，不需要 Arrow 後端，也少一個在舊 Intel Mac 上可能踩雷的依賴（MEM-0004）。

## 4. 專案目錄結構

```
nanomind/
├── README.md / CHANGELOG.md / todo.md
├── requirements-nano.txt          # 精簡過，鎖 torch==2.2.2
├── .gitignore                     # 排除 dataset/*.jsonl、out/、checkpoints/
├── docs/
│   ├── architecture/              # mermaid 圖
│   ├── decisions/                 # MEM-*.md（兼ADR）
│   ├── errors/                    # ERR-*.md
│   ├── skills/                    # SKILL-*.md
│   ├── experiments/               # loss曲線、超參數快照、benchmark結果
│   ├── reports/                   # phase-0.md ~ phase-6.md
│   └── concept-notes/             # DeepSeek V4.1 概念對照筆記（學習用）
├── config/tier0.json / tier1.json / tier2.json
├── nanomind/                      # 核心套件（vendored & 簡化）
│   ├── tokenizer.py / model.py / dataset.py / checkpoint.py
├── trainer/train_pretrain.py / train_sft.py / bench_cpu.py
├── scripts/chat_cli.py
├── dataset/raw/（含來源與授權說明）, nano_pretrain.jsonl, nano_sft.jsonl
└── tests/test_tokenizer.py / test_model.py / test_dataset.py / test_smoke_train.py
```

## 5. 開發階段（每階段結束都會停下來跟您確認）

| Phase | 目標 | 關鍵交付物 | 過關條件 |
|---|---|---|---|
| **0. 環境與基準** | 驗證 torch 2.2.2 + transformers 相容性；量測 CPU tokens/sec | MEM-0001、`bench_cpu.py`、git 初始化 | 能跑完一次 forward+backward 無錯，拿到實測效能數字 |
| **1. 骨架與規範** | 目錄結構、pytest、todo.md、MEM/ERR/SKILL 資料夾 | 專案骨架、CI（可選） | `pytest` 綠燈 |
| **2. Tokenizer** | 決定並訓練小 BPE、準備 nano 語料 | tokenizer、nano_pretrain/sft.jsonl、MEM-0002 | 中英文編解碼正確、vocab 在目標範圍 |
| **3. 模型架構** | Vendor + 裁剪 model_minimind.py，定義 tier0/1/2 | `nanomind/model.py`、config 三份、MEM-0001/0003 | 三個 tier 都能 forward，參數量與估算一致 |
| **4. 預訓練迴圈** | 簡化版 train_pretrain；先過 Tier0 過擬合測試再上 Tier1 | `train_pretrain.py`、loss 曲線紀錄 | Tier0 能把 loss 訓到趨近 0；Tier1 loss 穩定下降無 NaN |
| **5. SFT + 對話 CLI** | 小型對話資料、chat template、推論 CLI | `train_sft.py`、`chat_cli.py` | 模型對訓練過的問題能給出合理回覆 |
| **6. 收斂與延伸** | README、DeepSeek概念對照筆記、（選配）DPO-lite/量化推論 | 完整文件、concept-notes | 別人只看 README 就能重現整個流程 |

## 6. DeepSeek V4.1 概念會怎麼被「用到」

不會實作，但會在 Phase 6 產出一份對照筆記，例如：

| 技術 | 目的 | NanoMind 是否適用 | 處理方式 |
|---|---|---|---|
| MoE / 384 專家 | 增加容量、控制啟動參數 | 否，規模差太多 | 保持 Dense；想體驗可做 2-expert 玩具實驗（選配） |
| CSA2 / KV cache 壓縮 | 解決百萬 token 上下文的 HBM/SSD 壓力 | 否，context 只有 128–384 tokens | 標準 dense causal attention |
| CED（Causal Encoder-Decoder） | 降低長 prompt prefill 成本 | 否，prefill 本身已極快 | 單一 decoder-only |
| Engram / DSpark / FP4 KV | 記憶體查表、投機解碼、量化 KV | 否 | 略過；「量化」概念可作選配延伸（INT8 動態量化推論） |
| Reasoning Effort 可控機制 | 用 RL 調整輸出長度成本 | 概念可借鏡 | 不做 RL，但 SFT 資料可設計長短回覆風格 |

---

## 後訓練完整地圖：納入 / 簡化 / 捨棄一覽

| 環節 | 標準做法（minimind / DeepSeek 對應） | NanoMind 決策 | 取捨理由 |
|---|---|---|---|
| **SFT** | `train_full_sft.py`，全參數微調 | ✅ 納入，不簡化 | 本來就是最基本、成本最低的一步 |
| **離線偏好優化（DPO）** | `train_dpo.py`，actor+ref 雙模型 | ✅ 納入，不簡化 | 雙模型在 nano 規模（幾 MB）完全不是負擔，跟一般大模型 DPO 的痛點不同 |
| **線上強化學習：PPO vs GRPO** | minimind 兩者都支援 | ✅ 納入 **GRPO**，捨棄 PPO | GRPO 不需要額外訓練 critic/value network，工程與計算量都更低；DeepSeek 系列本身也是走 GRPO 這條路線，教學代表性更高，符合「精簡」 |
| **RL 的 Reward 來源** | minimind 用「規則式懲罰（長度/重複）+ 外部學習型 reward model（`LMForRewardModel`，需下載 GB 級模型並 GPU 推論）」 | ✅ 納入規則式 reward，❌ **捨棄學習型 reward model** | 我實際看了 `trainer_utils.py`，`LMForRewardModel` 是用 `AutoModel.from_pretrained` 載入一個獨立的大型 reward model checkpoint——這在 CPU-only、零成本前提下完全不現實；而且「訓練/取得一個 reward model」本身工程量跟再做一次 DPO 一樣重，性價比很低。我們改用**可驗證的規則式 reward**（RLVR 風格，例如簡單四則運算對不對、格式對不對），這正是 minimind README 自己在 RLAIF 章節提到的「非人類、可自動取得的訊號」路線 |
| **Rollout 推論引擎** | minimind 支援純 PyTorch (`TorchRolloutEngine`) 或外接 SGLang server (`SGLangRolloutEngine`) | ✅ 納入 **TorchRolloutEngine**，捨棄 SGLang | SGLang 是為大模型高吞吐服務設計的推論引擎，nano 模型 + CPU 環境下用純 PyTorch generate 已經夠快，架 SGLang server 是不必要的複雜度 |
| **知識蒸餾** | `train_distillation.py`，token-level KL（`teacher_logits[..., :vocab_size_student]`，靠**共用同一份 tokenizer** 才能對齊 token id） | ✅ 納入，但有一個關鍵限制要解決（見下方說明） | 這是完整後訓練清單裡確實該有的一環，也呼應 DeepSeek 報告 5.2.4 節「on-policy distillation 是後訓練的最後一步」 |
| **工具使用 / Agentic RL** | `train_agent.py`，需要 mock 工具環境、沙箱、多輪任務合成（DeepSeek 報告整個 5.1 節都在講這個基礎設施） | ❌ 不納入，僅寫入 concept-notes | 這本質上是「特定能力的資料/環境工程」，而不是「後訓練演算法」本身；做下去專案重心會偏移到寫模擬環境，而不是學後訓練技術 |
| **LoRA / 參數高效微調** | `train_lora.py` | ❌ 刻意捨棄（不是忽略） | LoRA 的價值是「攤薄對大模型微調的成本」；NanoMind 全參數微調本身已經是幾秒到幾分鐘等級，LoRA 在這個規模反而是「為了精簡而增加的複雜度」，違反 KISS |
| **量化推論** | 無專門腳本，PyTorch 內建 | ✅ 保留，但**明確標註它不是後訓練**，是後訓練完成後的部署最佳化 | 回應您上一則問題本身 |

### 知識蒸餾的一個關鍵技術限制，需要您知道

我實際讀了 `train_distillation.py`：minimind 的蒸餾是 **token-level KL divergence**，而且程式碼直接對 teacher logits 做 `[..., :vocab_size_student]` 切片——這**只有在 student 和 teacher 共用同一份 tokenizer（同樣的 vocab 對應）時才是對的**，因為 minimind 所有規模的模型本來就共用同一份 tokenizer。

但我們在 Phase 2 已經決定 NanoMind 要**訓練自己的小 vocab**（為了降低 LM-head 運算量）。這代表：

`[Trade-off Analysis]`
- **選項A（貼近原生做法）**：這個蒸餾實驗**特別**讓學生模型改用 minimind 官方 tokenizer（vocab=6400），教師用官方最小的預訓練模型 `minimind2-small`（26M參數，可下載），直接沿用 minimind 的 token-level KL 蒸餾程式碼
  - Pros：技術上「正確」、真正學到 soft-label 蒸餾的本質、程式碼可以幾乎照搬
  - Cons：這個實驗用的模型必須用大 vocab，跟我們核心 Tier0/1/2 用的小 vocab 系列不一致
  - 但因為這階段的計算瓶頸其實是 **teacher（26M參數）的推論**，不是 student 的 vocab 大小，所以「vocab=6400 拖慢訓練」這個顧慮在這裡影響有限
- **選項B（tokenizer-agnostic 簡化版）**：維持我們自己的小 vocab，蒸餾方式改成「教師生成文字 → 當作額外的 SFT pseudo-label 資料」（sequence-level 而非 token-level KL）
  - Pros：完全不用碰 tokenizer 對齊問題，工程最簡單
  - Cons：失去「真正的機率分布蒸餾」這個核心技術，比較像是「用教師資料做多一輪 SFT」，教學價值打折
- **決策：採用選項A**，作為一個**獨立、隔離的蒸餾實驗**（不影響我們主線 Tier0/1/2 的小 vocab 決策），理由是要學就學真的 token-level 蒸餾，而且成本瓶頸分析顯示這樣做並不會犧牲太多效率。這件事會記錄成 MEM-000X，明確標註「此為刻意的例外」。

## 更新後的完整 Phase 清單（0–11）

Phase 0–5 維持不變（環境 → 骨架 → Tokenizer/資料裁切 → 模型架構 → 預訓練 → SFT），後訓練與延伸部分更新如下：

| Phase | 內容 | 屬於後訓練？ | 備註 |
|---|---|---|---|
| 6 | **DPO-lite**（偏好對齊） | ✅ 是 | 沿用前版規劃 |
| 7 | **GRPO-lite**（線上強化學習） | ✅ 是 | 新增；規則式 verifiable reward，`TorchRolloutEngine`，需自己合成一小批「答案可程式驗證」的 prompt（如簡單算術），這是唯一一個**不從 minimind 資料集裁切**的階段，因為 reward 機制不同，需要自己造資料 |
| 8 | **知識蒸餾**（Distillation） | ✅ 是（且是最後一步，呼應 DeepSeek 報告順序） | 新增；教師=`minimind2-small`，見上方 Trade-off |
| 9 | 量化推論封裝 | ❌ 否，是部署最佳化 | 沿用前版規劃，明確標註非後訓練 |
| 10 | Tier2 規模化復現（capstone） | — | 視 Phase 0–9 實測時間，決定 Tier2 要不要也跑完整後訓練，或只做 pretrain+SFT |
| 11 | 文件收斂 | — | README/CHANGELOG/concept-notes（含工具使用/Agentic RL/LoRA 為何不做的說明）/todo 清空 |

**時間預算煞車機制維持不變**：每個 Phase 結束都會回報實測花費時間，Phase 7（GRPO，需要 rollout 生成，每步成本比 SFT/DPO 高）跟 Phase 8（蒸餾，需要跑 teacher 推論）是最可能超出「極少訓練時間」預期的兩關，屆時會如實回報，由您決定要不要繼續、縮減規模，或跳過。

---

三個選擇都收到了，整理一下對計畫的實際影響，並把方案更新成最終版：

## 確認後的方向
1. **語料**：直接用 minimind 官方 `pretrain_t2t_mini.jsonl` / `sft_t2t_mini.jsonl` 裁切一小份，不自己找文本（語言分佈就跟著原資料集走，本身即中英混合）。
2. **規模**：Tier0 → Tier1 → Tier2 依序做完，不是只做一個。
3. **範圍**：預訓練 + SFT 是必做核心，再加碼 DPO-lite 與量化推論。

這把原本 0–6 的七個 Phase 擴展成 **0–9 共十個 Phase**，多出來的都是小步快跑、每步都會停下來給您確認。以下只列出跟前一版**有變動或新增**的部分。

### Phase 2 更新：資料取得與裁切（改動最大的部分）

- **來源**（二選一，優先 ModelScope，HuggingFace 備援）：
  - ModelScope：`https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files`
  - HuggingFace：`https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main`
  - 只需要下載 `pretrain_t2t_mini.jsonl`（1.2GB）+ `sft_t2t_mini.jsonl`（1.6GB），**一次性頻寬/硬碟成本約 2.8GB**，不需要金錢成本。裁切完成後，原始大檔可以刪除，只留下裁切後的小檔進 repo。
- **裁切方式**：寫一個 `scripts/sample_dataset.py`，用固定 random seed 做等機率抽樣，抽出目標大小（例如 pretrain 抽到 ~10MB、SFT 抽到 ~3MB），輸出 `dataset/nano_pretrain.jsonl` / `dataset/nano_sft.jsonl`。
- **可追溯性**（呼應您要「設計思路要完整記錄」）：會產生 `dataset/raw/MANIFEST.md`，記錄來源檔案、檔案大小、SHA256、抽樣 seed、抽樣行數、抽樣時間——這樣任何人都能重現一模一樣的子集，這件事本身也會寫成 MEM-0002。
- Tier0 的「煙霧測試」語料**維持用合成資料**（例如重複的固定 pattern），因為 Tier0 的目的是驗證 training loop 機制正不正確，不是驗證真實語言能力，用合成資料更快也更容易斷言正確性。

### Phase 6：DPO-lite
- 資料：`dpo.jsonl`（53MB，比前兩個檔小很多，抽樣一次即可，格式是 `{"chosen":[...], "rejected":[...]}` 各自一組對話）。
- 方法：沿用 minimind `train_dpo.py` 的邏輯——**同時載入一份凍結的 reference model + 一份可訓練的 policy model**，用 DPO loss（`-log σ(β·(policy對數比 - reference對數比))`）微調 Tier1 SFT 完成的模型。
- 硬體備註：在 ~1M 參數規模下，兩份模型同時存在記憶體裡完全不是問題（總共幾 MB），這點跟一般 DPO 在大模型上會佔用雙倍 GPU 記憶體的痛點完全不同，值得在文件裡記一筆對比。

### Phase 9：量化推論封裝
- 用 PyTorch 內建 `torch.quantization.quantize_dynamic`（僅需 CPU，剛好完全符合我們的硬體）對 Linear 層做動態 INT8 量化。
- 會誠實記錄一個預期：Tier1 模型本身 fp32 只有幾 MB，量化後的「省下來的量」不會很戲劇性，這個 Phase **重點是體驗技術本身**，跟 DeepSeek 報告裡 FP4 KV cache 那種「為了解決真實頻寬瓶頸」的動機不同規格，會在 concept-notes 裡對照說明。
- 交付物：量化前後的檔案大小、推論延遲（tokens/sec）對比表。

### Phase 10：Tier2 規模化復現（capstone）
- 目的：驗證整條 pipeline（tokenizer 訓練腳本、pretrain/SFT trainer）是否真的可重用、可參數化，而不是為 Tier1 寫死。
- 做法：重新跑一次 tokenizer 訓練（vocab=4096）、pretrain、SFT，套用 Tier2 config。
- **明確的煞車機制**：這個 Phase **只有在 Phase 4/5 實測時間在可接受範圍內才會進行**——會先跟您報告 Tier1 實際花了多久，再一起決定要不要投入 Tier2（DPO/量化在 Tier2 上是否要做，屆時再議，非強制）。這是為了不違背「極少訓練時間」的核心目標。

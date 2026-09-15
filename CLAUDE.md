# CLAUDE.md

給任何後續在這個 repo 裡工作的 Claude（或人類開發者）的說明。開始改任何
東西之前，請先讀完這份文件，並先看 `todo.md` 確認目前在哪個 Phase、哪個
git branch。

## 這是什麼專案

NanoMind：從 0 開始、用極少成本與極少訓練時間，在一台 **2014 年 13"
MacBook Pro（雙核 Intel i7 3GHz、16GB RAM、Intel Iris 1536MB、macOS Big
Sur 11.7.11，無 CUDA/MPS 加速）**上訓練規模超迷你（Tier0 ~0.15M /
Tier1 ~1M / Tier2 ~6M 參數）的語言模型，並走過完整但精簡的後訓練流程
（SFT → DPO → GRPO → 知識蒸餾），最後做量化推論封裝。

架構與資料 pipeline 裁剪自 [minimind](https://github.com/jingyaogong/minimind)；
對話協議設計理念參考 deepseek-ai/deepseek-recipe（注意：那是 serving 端
的協議轉換函式庫，不是訓練程式）；概念對照參考 DeepSeek-V4.1 技術報告
（`docs/concept-notes/`，那些技術規模比 NanoMind 大 5-7 個數量級，
**大部分不會被實作**，只當學習筆記用）。

## 黃金原則：Pragmatic, not dogmatic

本專案的工程規範改編自一份通用的企業級規範（"AI Loop/Graph Engineering
v3.0"），但這是單人、單機、CPU-only 的學習專案，**微服務/K8s/前端MVP/
K6壓測/機密管理等等級的規範已被刻意簡化或整段捨棄**。完整的取捨對照與
理由記錄在規劃階段的對話中，日後若有疑問可回頭看
`docs/decisions/MEM-0001-*.md` 之後陸續累積的 MEM 文件。

遇到「規範說要 XXX，但這裡明顯用不到」的狀況：預設「不適用」的判斷通常
是對的，**但要把理由寫進 MEM，不要默默跳過**。

## Git 工作流程（每次開工前先確認自己在哪個 branch）

- **每個 Phase 開一個新 branch**，命名 `phase-N-<簡短描述>`
  （例：`phase-1-scaffold-and-conventions`）。
- 不要直接在 `main` 上做開發。`main` 只接受 phase 分支的 merge。
  （`phase-0-env-setup` 是例外：當時還沒訂出這個規則，是完成後才用
  `git branch phase-0-env-setup` 做的事後快照。）
- Phase 內的工作照常用下面的語意化 commit 逐步累積，**不要囤積成一個
  大 commit 最後才交**。
- Phase 完成、經確認後，用 `git merge --no-ff phase-N-xxx` 合併回
  `main`。**絕對不要 squash**——每一個 commit（包含走過的彎路、
  修正過程）都要完整保留在歷史裡。
- Merge commit 訊息要總結這個 phase 做了什麼、對應哪些 MEM/ERR 文件。

## Commit 規範

```
<type>(<scope>): <subject>

<body：做了什麼、為什麼>

Ref: MEM-xxxx / ERR-xxxx（如果適用）
```

`type` ∈ {feat, fix, test, refactor, docs, chore}。

## 三種記憶文件（不要另外發明新的筆記檔案格式）

- **`docs/decisions/MEM-XXXX-topic.md`**：架構/技術決策記錄（兼任 ADR
  角色，本專案規模不需要 ADR/MEM 兩套並行）。四位數流水號遞增。做出
  「以後可能會被問『當初為什麼這樣做』」的決策時就該寫一份。
- **`docs/errors/ERR-YYYYMMDD-topic.md`**：真正卡關過、有明確 root cause
  可寫的錯誤，尤其是環境/相依套件類的坑（這類問題很容易被重踩）。不是
  每個小 bug 都要寫，但值得記錄的錯誤不能只留在對話紀錄裡。
- **`docs/skills/SKILL-topic.md`**：可重複使用的流程，等真的出現重複性
  工作再開始寫，目前還沒有。
- **`todo.md`**：目前 Phase 的 sprint backlog，`[ ]` / `[x]`。
- **`CHANGELOG.md`**：Keep a Changelog 格式，每個 Phase 收尾時更新。

## 測試哲學

- **確定性強、可斷言的工程元件**（tokenizer / config / dataset /
  checkpoint / shape）：正常 TDD，Red → Green → Refactor。
- **訓練迴圈本身**：不要寫 `assert loss == 常數` 這種沒意義的測試，改用
  「小樣本過擬合測試」——能不能在幾百 step 內把一組已知答案的合成資料
  背到 loss≈0，這才是驗證 training loop 正確性的標準做法。
- 覆蓋率：只要求 engineering 模組 ≥70%，訓練/生成迴圈不列入覆蓋率統計
  （用整合煙霧測試取代）。

## 程式碼註解規範（2026-09-15 補充）

**註解要盡量完整，讓新手也看得懂，不能只寫給已經懂的人看。**具體來說：

- 每個檔案開頭的 docstring 要說明「這個檔案負責什麼、為什麼需要它、跟
  其他哪些檔案有關係」，不是只有一行函式簽名式的描述。
- 遇到領域知識或縮寫（RoPE、GQA、SwiGLU、BPE、KL divergence、
  perplexity……）第一次出現時，用一兩句話解釋是什麼、為什麼在這裡用，
  不能假設讀者已經知道。
- 數學公式或魔術數字（例如 `ceil(hidden*pi/64)*64`）旁邊要有白話文說明
  這個公式在算什麼、為什麼長這樣，不能只丟公式。
- 寫「為什麼這樣做」比寫「這行在做什麼」更重要——程式碼本身已經說明了
  「做什麼」，註解要補的是「什麼」背後的「為什麼」與「不這樣做會怎樣」。
- 測試檔案的每個 test case 也要有一行說明「這個測試在保護什麼行為、
  為什麼這個行為重要」，不能只靠函式名稱自我解釋。
- 寧可稍微囉唆，也不要讓新手看不懂——這是一個學習型專案，程式碼本身
  就是教材的一部分。

## 環境地雷（已知，不要重踩）

- `torch` 只能鎖 `==2.2.2`（macOS Intel x86_64 最後支援版本，2.3.0 起
  官方不再提供 mac x86_64 wheel）。細節：`docs/decisions/MEM-0001-*.md`。
- `numpy` 必須 `<2.0`（鎖 `==1.26.4`），否則 torch 的 numpy interop 會
  整個壞掉（`RuntimeError: Numpy is not available`）。細節：
  `docs/errors/ERR-20260912-numpy2-breaks-torch222.md`。
- 這台機器**沒有 CUDA/MPS**，任何程式碼都不該假設有 GPU 可用，也不要
  加 `.cuda()` / `device="mps"` 這類分支（除非明確是條件式 fallback）。
- Python 版本建議 3.10 / 3.11（3.12 也可以，3.13 不行，torch==2.2.2
  沒有對應 wheel）。

## Phase 路線圖（目前狀態以 `todo.md` 為準，這裡只是總覽）

| Phase | 內容 | 狀態 |
|---|---|---|
| 0 | 環境建置與基準驗證 | ✅ 已完成（branch: `phase-0-env-setup`） |
| 1 | 骨架與規範（ruff + pytest + TDD 第一輪：NanoMindConfig） | 進行中 |
| 2 | Tokenizer + 資料裁切（自訓練小 vocab，裁切 minimind 官方資料集） | 未開始 |
| 3 | NanoMind 模型架構（vendor 並簡化 minimind，tier0/1/2 config） | 未開始 |
| 4 | 預訓練（Tier0 過擬合驗證 → Tier1 正式預訓練） | 未開始 |
| 5 | SFT + 對話 CLI | 未開始 |
| 6 | DPO-lite（偏好對齊） | 未開始 |
| 7 | GRPO-lite（線上強化學習，規則式 verifiable reward） | 未開始 |
| 8 | 知識蒸餾（teacher=minimind2-small，獨立實驗分支，共用官方 tokenizer） | 未開始 |
| 9 | 量化推論封裝（非後訓練，屬部署最佳化） | 未開始 |
| 10 | Tier2 規模化復現（capstone，視實測時間決定範圍） | 未開始 |
| 11 | 文件收斂 | 未開始 |

## 明確捨棄、不要重新提案的項目（已有明確理由，別重複討論）

- **LoRA**：全參數微調在這個規模已經很便宜，LoRA 反而增加不必要複雜度。
- **Agentic RL / 工具呼叫訓練**：需要環境模擬/沙箱基礎設施，跟「學後
  訓練演算法」的目標不同軸線，只留在 `docs/concept-notes/`。
- **GRPO 階段的學習型 reward model**：需要下載/訓練額外的大型模型，
  改用規則式 verifiable reward（如四則運算對不對）。
- **SGLang rollout engine**：用純 PyTorch `TorchRolloutEngine` 就夠。
- **HuggingFace `datasets` 套件**：資料量幾 MB 等級，用純 Python JSONL
  讀取即可，不需要 Arrow 後端。
- **Docker / K8s / K6 壓測 / 前端 MVP**：沒有部署或壓測場景。

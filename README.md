# NanoMind

從 0 開始、用極少成本與極少訓練時間，訓練一個規模超迷你（Tier0 ~0.15M /
Tier1 ~1M / Tier2 ~6M 參數）的語言模型，並走過完整但精簡的後訓練流程
（SFT → DPO → GRPO → Distillation），最後做量化推論封裝。

架構與資料 pipeline 主要參考、裁剪自 [minimind](https://github.com/jingyaogong/minimind)；
對話協議設計理念參考 deepseek-ai/deepseek-recipe；概念對照參考
DeepSeek-V4.1 技術報告（詳見 `docs/concept-notes/`）。

## 目標硬體

2014 年 13" MacBook Pro（雙核 Intel i7 3GHz、16GB RAM、Intel Iris 1536MB、
macOS Big Sur 11.7.11）。**純 CPU 訓練，無 CUDA/MPS 加速**。所有規模與時間
預算都是針對這個限制設計的，細節見 `docs/decisions/MEM-0001-*.md`。

## 目前狀態：Phase 0 已完成 ✅，進行中：Phase 1（骨架與規範）

Phase 0 已在真實目標硬體（MacBook Pro Mid 2014, macOS Big Sur）驗證通過，
詳見 `docs/reports/phase-0-env-setup.md` 與 `docs/decisions/MEM-0001-*.md`。
進度與待辦見 `todo.md`。

## 開發規範

本專案的工程規範改編自一份通用的「AI Loop/Graph Engineering v3.0」規範，
依專案規模做了大量簡化（移除微服務/K8s/前端MVP等不適用項目），但保留：

- 語意化 git commit
- `docs/decisions/`（MEM，架構決策記錄）
- `docs/errors/`（ERR，重大錯誤與解法記錄）
- `docs/skills/`（SKILL，可重用流程）
- 每個 Phase 結束才進到下一個 Phase，且會產出 `docs/reports/phase-N.md`

完整的規範取捨對照表在對話規劃階段已產出，之後會整理進
`docs/decisions/` 中的一份 MEM。

## 快速開始（環境建置）

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-nano.txt
python trainer/bench_cpu.py
```

若 `pip install` 或執行時遇到問題，先查
`docs/errors/ERR-20260912-numpy2-breaks-torch222.md`。

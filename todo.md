# todo.md（Active Sprint: Phase 2 收尾中，等待使用者真實資料執行結果）

## Phase 0：環境建置與基準驗證 — ✅ 已完成（branch: `phase-0-env-setup`）
## Phase 1：骨架與規範 — ✅ 已完成（已 merge 進 main）

## Phase 2：Tokenizer + 資料裁切 — 🟡 工程部分完成，等待使用者操作
- [x] 開新 branch `phase-2-tokenizer-and-data`
- [x] `nanomind/sampling.py`（reservoir sampling / SHA256 / manifest 寫入），9+2=11 tests
- [x] `scripts/sample_dataset.py` CLI，4 tests
- [x] `nanomind/tokenizer.py`（Byte-Level BPE 訓練/載入），13 tests
- [x] `scripts/train_tokenizer.py` CLI，3 tests
- [x] `docs/decisions/MEM-0002-tokenizer-strategy.md`
- [x] `docs/reports/phase-2.md`（含給使用者的操作步驟）
- [ ] **需要使用者動作**：下載 minimind 官方資料集、執行
      `sample_dataset.py` x2、`train_tokenizer.py` x1，回報結果
- [ ] 依真實結果補完 MEM-0002，正式關閉 Phase 2 並 merge 進 main

## Phase 3（下一步預告，尚未開始）
- [ ] 開新 branch `phase-3-model-architecture`
- [ ] vendor 並簡化 minimind 的 model_minimind.py 成 nanomind/model.py
      （移除 MoE 分支）
- [ ] 用 Phase 1 的 NanoMindConfig 定義 config/tier0.json / tier1.json /
      tier2.json
- [ ] forward pass shape 測試、causal mask 正確性測試、參數量驗證測試
- [ ] checkpoint save/load round-trip 測試

# todo.md（Active Sprint: Phase 2）

## Phase 0：環境建置與基準驗證 — ✅ 已完成（branch: `phase-0`）
## Phase 1：骨架與規範 — ✅ 已完成（branch: `phase-1-scaffold-and-conventions`，待 merge）
- [x] ruff + pytest 設定（pyproject.toml, requirements-dev.txt）
- [x] TDD 第一輪：NanoMindConfig（11 tests, Red→Green→Refactor）
- [x] 修正 bench_cpu.py import 順序（ruff --fix）
- [x] ERR: git add -A 誤加 .venv-dev 的教訓與修正
- [x] docs/reports/phase-1.md

## Phase 2：Tokenizer + 資料裁切（下一步，尚未開始）
- [ ] 開新 branch `phase-2-tokenizer-and-data`
- [ ] 下載 minimind 官方 pretrain_t2t_mini.jsonl / sft_t2t_mini.jsonl
- [ ] 寫 `scripts/sample_dataset.py`（固定 seed 抽樣）
- [ ] 產出 `dataset/nano_pretrain.jsonl` / `dataset/nano_sft.jsonl`
- [ ] `dataset/raw/MANIFEST.md`（來源、checksum、抽樣方法記錄）
- [ ] 訓練 Tier1 用的小 vocab BPE tokenizer（沿用 minimind
      train_tokenizer.py 邏輯改造）
- [ ] tokenizer round-trip 測試（TDD）
- [ ] MEM-0002：tokenizer 策略決策記錄

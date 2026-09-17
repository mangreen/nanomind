# todo.md（Active Sprint: Phase 3）

## Phase 0：環境建置與基準驗證 — ✅ 已完成（branch: `phase-0-env-setup`）
## Phase 1：骨架與規範 — ✅ 已完成（已 merge 進 main）
## Phase 2：Tokenizer + 資料裁切 — ✅ 已完成（branch: `phase-2-tokenizer-and-data`，待 merge）
- 真實資料：nano_pretrain.jsonl 13,927 行/12.95MB、nano_sft.jsonl 1,106行/2.01MB
- Tier1 tokenizer：vocab_size 1536/1536（滿版命中）
- pytest 42 passed, ruff check 全過

## Phase 3：NanoMind 模型架構（下一步）
- [ ] 開新 branch `phase-3-model-architecture`
- [ ] vendor 並簡化 minimind 的 model_minimind.py 成 nanomind/model.py
      （移除 MoE 分支，理由見 MEM-0001）
- [ ] 用 Phase 1 的 NanoMindConfig 定義 config/tier0.json（vocab=512,
      不需真實tokenizer）/ tier1.json（vocab=1536，對應真實 tokenizer）/
      tier2.json（vocab=4096，Phase 10 才用）
- [ ] forward pass shape 測試
- [ ] causal mask 正確性測試（用簡單可驗證的合成序列）
- [ ] 參數量驗證測試（斷言 numel 在預期範圍，並跟 Phase 0 bench_cpu.py
      的 proxy 估算對照）
- [ ] checkpoint save/load round-trip 測試
- [ ] RoPE/GQA 正確性測試

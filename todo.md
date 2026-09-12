# todo.md（Active Sprint: Phase 0）

## Phase 0：環境建置與基準驗證
- [x] git repo 初始化
- [x] 專案目錄骨架
- [x] `.gitignore`
- [x] `requirements-nano.txt`（鎖定版本）
- [x] 相容性驗證：torch==2.2.2 支援的 Python 版本範圍（3.9–3.12，macOS Intel）
- [x] 相容性驗證：發現並修好 numpy>=2.0 vs torch==2.2.2 的 ABI 問題
- [x] 相容性驗證：transformers==4.57.6 所需元件（含 MoeCausalLMOutputWithPast）可正常 import
- [x] `trainer/bench_cpu.py`（proxy 效能基準腳本），已在 Linux 沙盒驗證腳本本身能跑
- [ ] **需要使用者動作**：在實際 MacBook Pro (Mid 2014) 上安裝環境並執行
      `bench_cpu.py`，回報結果
- [ ] 依使用者回報的真實數字，補完 MEM-0001 並正式關閉 Phase 0

## Phase 1（下一步預告，尚未開始）
- [ ] pytest 骨架 + 第一個測試（TDD：先寫 `test_config.py` 再寫 config）
- [ ] ruff/lint 設定
- [ ] （選配）GitHub Actions CI：pytest + ruff

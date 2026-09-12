# todo.md（Active Sprint: Phase 1）

## Phase 0：環境建置與基準驗證 — ✅ 已完成關閉
- [x] git repo 初始化 / 專案目錄骨架 / .gitignore
- [x] requirements-nano.txt（鎖定版本）
- [x] 相容性驗證：torch==2.2.2 支援的 Python 版本範圍（3.9–3.12，macOS Intel）
- [x] 相容性驗證：修好 numpy>=2.0 vs torch==2.2.2 的 ABI 問題
- [x] 相容性驗證：transformers==4.57.6 所需元件可正常 import
- [x] trainer/bench_cpu.py（proxy 效能基準腳本）
- [x] 在真實 MacBook Pro (Mid 2014) 上驗證安裝與執行，全程無需編譯，無新增 ERR
- [x] 依真實數字補完 MEM-0001，正式關閉 Phase 0

## Phase 1：骨架與規範（現在開始）
- [ ] 選定並設定 lint 工具（ruff）
- [ ] pytest 骨架 + pytest.ini / pyproject.toml 設定
- [ ] TDD 練習第一輪：`tests/test_config.py`（Red）→ 最小可行的 config
      dataclass（Green）→ refactor
- [ ] （選配）GitHub Actions CI：pytest + ruff，只在 push 時跑，不含任何
      部署/雲端資源
- [ ] 更新 README「開發規範」章節，補上完整的規範取捨對照表（從規劃階段的
      對話內容整理進來，正式成為專案文件而不只是聊天記錄）

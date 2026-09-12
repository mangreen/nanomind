# Phase 1 報告：骨架與規範

## ✅ 狀態：完成

## 做了什麼
1. **Lint/測試工具鏈**：`pyproject.toml`（ruff + pytest 設定）、
   `requirements-dev.txt`（ruff==0.16.7、pytest==9.1.1，已驗證有 macOS
   Intel x86_64 wheel）。
2. **TDD 第一輪練習：`NanoMindConfig`**
   - Red：先寫 `tests/test_config.py`（11 個測試），涵蓋
     `intermediate_size` 自動計算公式、GQA 相關合法性驗證、
     `tie_word_embeddings` 預設值、JSON round-trip。
   - Green：實作 `nanomind/config.py` 讓測試全過。過程中**測試本身**
     抓到一個資料設計錯誤（`hidden_size=64, num_attention_heads=5` 會
     先撞到別的驗證邏輯而測不到真正想測的東西），已修正。
   - Refactor：跑 ruff，抓到並修正 `bench_cpu.py` 的 import 順序問題。
3. **這個 `NanoMindConfig` 不是練習用完即丟的程式碼**——它就是 Phase 3
   要用來產生 `config/tier0.json` / `tier1.json` / `tier2.json` 的那個
   類別，直接對齊之前討論過的規模估算公式。
4. **一個插曲（如實記錄）**：收尾時 `git add -A` 不小心把新建的
   `.venv-dev/` 虛擬環境（14,212 個檔案）誤 commit 進去，因為
   `.gitignore` 沒有涵蓋這個新的虛擬環境資料夾名稱。已用
   `git reset --hard` 清掉（未推送到任何地方，修正安全），並把
   `.gitignore` 規則改成萬用字元、記錄成
   `ERR-20260912-git-add-dash-a-staged-venv.md`，之後一律用明確檔名
   `git add`。

## 驗證結果
```
pytest: 11 passed
ruff check .: All checks passed!
```

## Branch 狀態
本 Phase 在 `phase-1-scaffold-and-conventions` branch 上完成，待您確認
後 merge 回 `master`（`git merge --no-ff`，保留所有 commit 歷史，不
squash）。

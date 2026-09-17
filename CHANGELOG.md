# Changelog

本專案的變更記錄，格式參考 [Keep a Changelog](https://keepachangelog.com/)。

## [Phase 0] - 2026-09-12 - 環境建置與基準驗證（已關閉）

### Added
- 專案骨架與目錄結構
- `requirements-nano.txt`：鎖定 torch==2.2.2 / numpy==1.26.4 /
  transformers==4.57.6 / tokenizers==0.22.2
- `trainer/bench_cpu.py`：Phase 3 之前的 proxy CPU 效能基準測試腳本
- `docs/decisions/MEM-0001-environment-and-dependency-baseline.md`
- `docs/errors/ERR-20260912-numpy2-breaks-torch222.md`
- `docs/experiments/hardware-baseline.md`：MacBook Pro (Mid 2014) 真實
  benchmark 結果

### Fixed
- 修正 `numpy>=2.0` 與 `torch==2.2.2` 的相容性問題（鎖定 numpy==1.26.4）

### Verified（在 MacBook Pro Mid 2014 / macOS Big Sur 11.7.11 實測）
- `pip install -r requirements-nano.txt` 全程下載預編譯 wheel，無需編譯
- `trainer/bench_cpu.py` 執行成功，tier1 proxy ~0.52 sec/step
  （~8.65 分鐘/1000 steps），tier2 proxy ~3.55 sec/step
  （~59 分鐘/1000 steps）——時間預算比預期寬裕

## [Phase 1] - 2026-09-12 - 骨架與規範（完成，待 merge）

### Added
- `pyproject.toml`（ruff + pytest 設定）、`requirements-dev.txt`
- `nanomind/config.py`：`NanoMindConfig`（TDD 完整走過 Red→Green→Refactor）
- `tests/test_config.py`：11 個測試
- `CLAUDE.md`：專案規範/工作流程說明文件
- `docs/errors/ERR-20260912-git-add-dash-a-staged-venv.md`

### Fixed
- `bench_cpu.py` import 順序（ruff --fix）
- `.gitignore` 虛擬環境規則改用萬用字元，修正 `git add -A` 誤加
  `.venv-dev/` 的問題

## [Phase 2] - 2026-09-17 - Tokenizer + 資料裁切（已關閉）

### Added
- `nanomind/sampling.py`：reservoir sampling、串流 SHA256、
  MANIFEST.md 寫入
- `scripts/sample_dataset.py`：資料抽樣 CLI（支援 `--lines` 或
  `--target-mb`）
- `nanomind/tokenizer.py`：Byte-Level BPE tokenizer 訓練/載入，特殊
  token `<unk>/<pad>/<|im_start|>/<|im_end|>`
- `scripts/train_tokenizer.py`：tokenizer 訓練 CLI
- `docs/decisions/MEM-0002-tokenizer-strategy.md`
- `dataset/raw/MANIFEST.md`：真實資料抽樣記錄

### Verified（用合成假資料於開發階段，真實資料於使用者機器驗證）
- 開發階段：中文/英文/中英混合＋未見過內容皆無損 round-trip（42 tests）
- 真實資料（2026-09-17）：nano_pretrain.jsonl 13,927行/12.95MB、
  nano_sft.jsonl 1,106行/2.01MB，Tier1 tokenizer vocab_size 1536/1536
  （滿版命中），pytest 42 passed，ruff check 全過

### Decisions
- `nanomind_artifacts/`（tokenizer 訓練產物）不進 git：BPE 訓練無
  隨機性，可完全從 MANIFEST.md 記錄的來源雜湊+seed+程式碼重現

## [Unreleased] - Phase 3

# Changelog

本專案的變更記錄，格式參考 [Keep a Changelog](https://keepachangelog.com/)。

## [Unreleased] - Phase 0

### Added
- 專案骨架與目錄結構
- `requirements-nano.txt`：鎖定 torch==2.2.2 / numpy==1.26.4 /
  transformers==4.57.6 / tokenizers==0.22.2
- `trainer/bench_cpu.py`：Phase 3 之前的 proxy CPU 效能基準測試腳本
- `docs/decisions/MEM-0001-environment-and-dependency-baseline.md`
- `docs/errors/ERR-20260912-numpy2-breaks-torch222.md`

### Fixed
- 修正 `numpy>=2.0` 與 `torch==2.2.2` 的相容性問題（鎖定 numpy==1.26.4）

### Pending（需要使用者在目標 macOS 機器上完成才能關閉 Phase 0）
- 在實際的 MacBook Pro (Mid 2014) 上驗證環境安裝與 `bench_cpu.py` 真實數字

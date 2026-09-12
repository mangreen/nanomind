# MEM: 環境與相依套件版本鎖定
Date: 2026-09-12
Status: Approved
Tags: [environment, dependencies, macos-intel, phase-0]

## Summary
NanoMind 開發機是 2014 年 13" MacBook Pro（雙核 Intel i7 3GHz、16GB RAM、
Intel Iris 1536MB、macOS Big Sur 11.7.11）。這台機器**沒有 CUDA GPU，也不是
Apple Silicon（無 MPS 加速）**，訓練只能是純 CPU、fp32。本文件記錄為了讓
minimind 的技術堆疊能在這台機器上正常運作，所做的版本鎖定與相容性驗證結果。

## Details

### Context & Evolution（脈絡與演進）
minimind 官方 `requirements.txt` 鎖 `transformers==4.57.6`，但註解掉了
`torch==2.6.0`，代表 torch 版本要使用者自行依硬體選擇。問題是：

- **PyTorch 從 2.3.0 版起，官方不再發布 macOS Intel(x86_64) 的 wheel**
  （只剩 Apple Silicon macOS wheel）。最後一個支援 macOS Intel 的版本是
  **2.2.2**。這代表我們不能用 minimind 建議的新版 torch，必須鎖定舊版。
- 鎖定 torch==2.2.2 之後，還要確認 Python 版本相容範圍、以及跟
  transformers==4.57.6 這種「很新」的套件搭配「很舊」的 torch 會不會出問題。

### 已驗證的事實（在 Linux 沙盒環境測試，Python/套件版本相容性與作業系統
無關，可視為跨平台成立；但**實際 CPU 效能數字仍需在目標 macOS 機器上
另外量測**，見「待辦」）

1. **torch==2.2.2 支援的 Python 版本（含 macOS Intel x86_64 wheel）**：
   實測 `pip download --platform macosx_10_9_x86_64` 確認 **Python 3.9 /
   3.10 / 3.11 / 3.12 皆有對應 wheel，3.13 沒有**。
   → 開發機建議使用 Python 3.10 或 3.11（3.12 也可以，但 3.10/3.11 是
   minimind 生態圈更常見的測試版本，風險更低）。

2. **numpy 版本是關鍵地雷**：預設 `pip install transformers` 會連帶解析出
   `numpy>=2.0`，但 `torch==2.2.2` 是用 NumPy 1.x 的 C-API 編譯的，兩者
   搭配會讓 `torch` 內部所有牽涉 numpy 互轉的功能直接壞掉
   （`RuntimeError('Numpy is not available')`）。
   → **必須明確鎖定 `numpy==1.26.4`**（剛好等於 minimind 原始
   requirements.txt 的版本，代表官方應該也踩過這個坑）。
   → 完整重現過程見 `docs/errors/ERR-20260912-numpy2-breaks-torch222.md`。

3. **確認 `torch==2.2.2` + `numpy==1.26.4` + `transformers==4.57.6` 這個
   組合下，minimind `model_minimind.py` 需要的 transformers 元件都能正常
   import**：`ACT2FN`、`PreTrainedModel`、`GenerationMixin`、
   `PretrainedConfig`、`MoeCausalLMOutputWithPast` 皆存在且可用。
   → 這代表 Phase 3「vendor 並移除 MoE 分支」這個決策，理由不是
   「MoeCausalLMOutputWithPast 在這個 transformers 版本不存在」（它其實
   存在），而單純是「我們用不到 MoE，移除以降低版本耦合與程式複雜度」
   ——這點會在 Phase 3 的 MEM 裡另外說明，避免理由寫錯。

4. 基本 forward + backward（`nn.Linear` 堆疊 + autograd）在上述組合下
   運作正常。

### Specification
最終鎖定版本（見 `requirements-nano.txt`）：

```
torch==2.2.2
numpy==1.26.4
transformers==4.57.6
tokenizers==0.22.2
```

相較 minimind 原始 `requirements.txt`，移除 `datasets`（Phase 2 會說明改用
純 Python JSONL 讀取的理由）、`wandb`、`streamlit`、`trl`、`peft`、
`modelscope`、`sentence_transformers`、`jieba` 等本專案 scope 不需要的套件。

### 待辦（需要在真正的 macOS 機器上完成，我這邊的 Linux 沙盒無法代勞）
- [ ] 在 MacBook Pro (Mid 2014) 上，用 pyenv 或官方安裝檔裝 Python 3.10 或
      3.11
- [ ] `python3 -m venv .venv && source .venv/bin/activate`
- [ ] `pip install -r requirements-nano.txt`
- [ ] 確認 `pip install` 過程沒有出現需要編譯（compile from source）的警告
      （若有，代表某個套件缺 macOS Intel 的 prebuilt wheel，需要另外處理）
- [ ] `python trainer/bench_cpu.py`，把終端機輸出與
      `docs/experiments/hardware-baseline.md` 的內容回報回來，我會補進本
      文件的「已驗證的事實」區塊，正式關閉 Phase 0

## When to Use
任何時候要重新建置 NanoMind 開發環境（換機器、環境壞掉重裝），都先看這份
文件，不要直接照 minimind 官方 README 的安裝方式做（那是為有 CUDA GPU 的
機器寫的）。

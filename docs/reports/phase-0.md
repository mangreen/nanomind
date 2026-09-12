# Phase 0 報告：環境建置與基準驗證

## 做了什麼

1. **專案骨架**：git repo、目錄結構（`docs/{decisions,errors,skills,
   experiments,reports,concept-notes,architecture}`、`config/`、
   `nanomind/`、`trainer/`、`scripts/`、`dataset/raw/`、`tests/`）。
2. **相依套件版本鎖定**（`requirements-nano.txt`），並在 Linux 沙盒環境
   實測驗證（版本相容性問題與作業系統無關，可視為跨平台成立）：
   - 確認 `torch==2.2.2` 是最後一個有 macOS Intel(x86_64) wheel 的版本，
     且支援 Python 3.9–3.12（3.13 沒有 wheel）。
   - **抓到一個真的會炸的相容性問題**：`numpy>=2.0` 會讓 `torch==2.2.2`
     的 numpy 互轉功能整個壞掉（`RuntimeError: Numpy is not available`）。
     根本原因是 torch 2.2.2 用 NumPy 1.x 的 C-API 編譯，而
     `pip install transformers` 預設會裝到 numpy 2.x。已修復：鎖定
     `numpy==1.26.4`。完整記錄在
     `docs/errors/ERR-20260912-numpy2-breaks-torch222.md`。
   - 確認 `torch==2.2.2` + `numpy==1.26.4` + `transformers==4.57.6` 這個
     組合下，minimind 模型檔需要的 `PreTrainedModel` /
     `MoeCausalLMOutputWithPast` 等元件都能正常 import。
3. **`trainer/bench_cpu.py`**：Phase 3 真正模型架構出現之前，先用形狀
   相近的通用 Linear-layer 堆疊當 proxy，量測這台機器每個 step 大概要
   多久。已在 Linux 沙盒跑過確認腳本本身沒有 bug（沙盒只有 1 個邏輯核心，
   跟您的雙核機器條件不同，所以**沙盒數字不能拿來當真實依據**，僅供驗證
   腳本正確性）：

   ```
   tier0: 0.0196 sec/step, ~52,300 tokens/sec（沙盒單核）
   tier1: 0.3766 sec/step, ~10,900 tokens/sec（沙盒單核）
   tier2: 2.2185 sec/step, ~2,800 tokens/sec（沙盒單核）
   ```

## 需要您做的事（這是 Phase 0 真正的 exit criteria）

我這邊是 Linux 容器，**不是**您的 macOS 機器，沒辦法代替您驗證真實情況。
麻煩您在您的 MacBook Pro (Mid 2014) 上：

```bash
python3 --version                 # 確認是 3.10 / 3.11 / 3.12
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-nano.txt
python trainer/bench_cpu.py
```

請把以下內容回報給我：
1. `pip install` 的完整輸出（特別留意有沒有出現「Building wheel...」這種
   代表要現場編譯的訊息——正常情況下所有套件都應該是下載預編譯的
   wheel，不用編譯）
2. `python trainer/bench_cpu.py` 的完整終端機輸出
3. 產生出來的 `docs/experiments/hardware-baseline.md` 內容

拿到這些之後，我會：
- 把真實數字補進 `MEM-0001`，正式關閉 Phase 0
- 根據真實的 tokens/sec，重新校準 Phase 4/6/7/8 的時間預算是否要調整
  規模（例如若 Tier1 一輪要跑好幾小時，我們會在那個 Phase 重新討論要不要
  縮小 config 或減少訓練步數）
- 開始 Phase 1（骨架與規範：pytest + lint）

## 風險與備註
- 若 `pip install` 過程中任何一個套件出現「需要編譯」的訊息，代表該版本
  在您的 macOS 版本上可能沒有現成 wheel，請把完整錯誤訊息貼給我，我會
  另外開一份 ERR 記錄並找替代版本。
- 若 Python 版本是 3.13 或更新，`torch==2.2.2` 會直接裝不上，需要先用
  `pyenv install 3.11` 之類的方式裝一個相容版本。

# ERR: numpy>=2.0 讓 torch==2.2.2 的 numpy interop 整個壞掉
Date: 2026-09-12
Severity: High（會擋掉幾乎所有牽涉 tensor<->numpy 互轉的功能，包含很多
`transformers`/`tokenizers` 內部流程）

## Description
在乾淨環境依序執行：

```bash
pip install torch==2.2.2
pip install transformers==4.57.6   # 這一步預設會連帶裝 numpy>=2.0
```

之後只要碰到任何 tensor↔numpy 互轉，就會出現：

```
UserWarning: Failed to initialize NumPy: _ARRAY_API not found
RuntimeError: Numpy is not available
```

重現方式（三行就能重現）：
```python
import torch
x = torch.randn(2, 3)
x.numpy()   # RuntimeError: Numpy is not available
```

## Root Cause
`torch==2.2.2`（2024年3月釋出）的官方 wheel 是用 **NumPy 1.x 的 C-API**
編譯的。NumPy 2.0（2024年6月釋出）改了 C-API/ABI，舊版 torch 沒有針對
NumPy 2.x 重新編譯，兩者的二進位介面對不上。

而 `transformers==4.57.6` 本身沒有把 `numpy<2` 寫進它的相依限制式裡，所以
`pip install transformers` 在一個還沒裝 numpy 的環境裡，會直接解析到當下
最新的 numpy（2.5.x），而不會自動退讓給 torch 的舊限制——pip 的相依解析
只看「宣告的版本限制」，不會知道「torch 這個特定版本的二進位跟 numpy 2.x
真的合不來」這種執行期才會爆的問題。

## Solution
在 `requirements-nano.txt` 明確鎖定：
```
numpy==1.26.4
```
並確保**安裝順序上 numpy 的版本鎖定會生效**（無論先裝 torch 還是先裝
transformers，只要最後 `pip install -r requirements-nano.txt` 是用同一份
鎖定檔案一次裝完，pip 就會尊重明確的版本號，不會被 transformers 的隱含解析
覆蓋掉）。

修復後驗證：
```python
import torch, numpy
print(numpy.__version__)   # 1.26.4
torch.randn(2, 3).numpy()  # 正常運作，不再拋錯
```

## Prevention
1. `requirements-nano.txt` 已明確鎖定 `numpy==1.26.4`，且在檔案開頭用註解
   標明「這一行不能刪，理由見本 ERR 文件」。
2. `trainer/bench_cpu.py` 在 `main()` 一開始就呼叫
   `check_numpy_torch_compat()`，主動偵測 `numpy>=2.0` 搭配 `torch<2.3`
   的組合並直接印出這份 ERR 文件的路徑後 exit，不讓使用者去猜一串難懂的
   C-API traceback。之後 Phase 3+ 的所有進入點腳本（pretrain/SFT/DPO/
   GRPO/distillation 的 trainer）都會沿用同一個檢查函式。
3. 日後若要升級 torch（例如哪天 macOS Intel 又重新有新版 wheel，或換到
   Apple Silicon 機器可以用新版），升級前一併检查 numpy 版本限制是否能
   放寬，不要只改 torch 版本號。

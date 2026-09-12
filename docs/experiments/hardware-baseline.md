# Hardware Baseline（Phase 0，實測，MacBook Pro Mid 2014）

> ⚠️ 這是用形狀相近的通用 Linear-layer 堆疊做的 **proxy** 測試，
> 在 Phase 3 真正的 NanoMind 模型（含 RoPE/GQA/真實 attention）
> 完成之前先量測。真實訓練會比這裡的數字慢一些（attention/RoPE
> 有這個 proxy 沒有算到的額外成本），這裡的數字只拿來抓數量級，
> 不是精確 ETA。
>
> **這份是在目標硬體（MacBook Pro Mid 2014, macOS Big Sur）上的真實測量結果**，
> 取代先前 Linux 沙盒的示範性輸出。

## Environment

- **platform**: macOS-11.7.11-x86_64-i386-64bit
- **processor**: i386
- **python_version**: 3.11.14
- **torch_version**: 2.2.2
- **cpu_count_logical**: 4（雙核 i7 + hyperthreading，符合硬體規格）
- **torch_num_threads**: 2（torch 預設抓到實體核心數）

## Results

| Tier | Proxy Params | sec/step | tokens/sec | est. min / 1000 steps |
|---|---|---|---|---|
| tier0 | 197,120 | 0.0315 | 32,524.6 | 0.52 |
| tier1 | 1,345,536 | 0.5188 | 7,895.5 | 8.65 |
| tier2 | 7,510,016 | 3.5518 | 1,729.8 | 59.2 |

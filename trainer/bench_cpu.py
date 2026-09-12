"""
NanoMind - Phase 0 CPU Benchmark
==================================
在 Phase 3 真正的 NanoMind 模型架構還沒寫出來之前，用「形狀跟 Tier0/1/2
config 相近的通用 Linear-layer 堆疊」當 proxy，量測這台機器每個 training
step 大概要花多久。目的不是精確模擬 RoPE/GQA attention 的真實成本，而是
先抓一個「數量級」的概念，讓我們在真正投入 Phase 3/4 之前，就知道
「值不值得跑、要跑多久」。

用法：
    python trainer/bench_cpu.py
    python trainer/bench_cpu.py --iters 20   # 想量測更穩定可以加大迭代數

會把結果寫入 docs/experiments/hardware-baseline.md（同時印到終端機）。

⚠️ 這支腳本本身已經在 Linux 沙盒環境驗證過可以正常執行（見
docs/reports/phase-0.md），但沙盒不是目標硬體。真正要記錄進
MEM-0001 的數字，必須是您在您自己的 MacBook Pro (Mid 2014) 上
實際跑出來的結果。
"""
import os
import sys
import time
import platform
import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def check_numpy_torch_compat() -> None:
    """防呆檢查：對應 docs/errors/ERR-20260912-numpy2-breaks-torch222.md
    torch==2.2.2 是用 NumPy 1.x 的 C-API 編譯的，若環境裝到 numpy>=2.0，
    torch 內部的 numpy interop 會直接壞掉。與其讓使用者看到一串難懂的
    C-API traceback，不如在腳本一開始就先攔下來、給明確的修法。
    """
    try:
        import numpy  # noqa: F401

        numpy_major = int(numpy.__version__.split(".")[0])
        torch_ver_core = torch.__version__.split("+")[0]
        torch_major, torch_minor = (int(x) for x in torch_ver_core.split(".")[:2])
        if numpy_major >= 2 and (torch_major, torch_minor) < (2, 3):
            print("!! 偵測到 numpy>=2.0 搭配 torch<2.3，這個組合已知會壞掉。")
            print(f"!!   目前 numpy={numpy.__version__}, torch={torch.__version__}")
            print("!! 詳見 docs/errors/ERR-20260912-numpy2-breaks-torch222.md")
            print("!! 修法： pip install 'numpy==1.26.4'")
            sys.exit(1)
        # 實際跑一次 tensor<->numpy 互轉，確認真的沒問題（不是只看版本號猜的）
        _ = torch.randn(2, 2).numpy()
    except RuntimeError as e:
        print(f"!! numpy/torch interop 自我檢查失敗: {e}")
        print("!! 詳見 docs/errors/ERR-20260912-numpy2-breaks-torch222.md")
        sys.exit(1)


@dataclass
class TierShape:
    name: str
    hidden: int
    layers: int
    ffn: int
    seq_len: int
    batch_size: int
    vocab: int


# 這三組形狀對應 docs/decisions/ 未來 Tier0/1/2 config 的估算值
# （Phase 3 會用同一批數字，這裡先拿來做 proxy 量測，之後可以互相對照）
TIERS = [
    TierShape("tier0", hidden=64, layers=2, ffn=256, seq_len=128, batch_size=8, vocab=512),
    TierShape("tier1", hidden=128, layers=4, ffn=448, seq_len=256, batch_size=16, vocab=1536),
    TierShape("tier2", hidden=256, layers=6, ffn=832, seq_len=384, batch_size=16, vocab=4096),
]


class ProxyBlock(nn.Module):
    """不是真正的 attention/RoPE/SwiGLU 實作，只是矩陣乘法量級相近的替身，
    純粹用來在 Phase 3 之前先抓時間數量級。"""

    def __init__(self, hidden: int, ffn: int):
        super().__init__()
        self.q = nn.Linear(hidden, hidden, bias=False)
        self.k = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, hidden, bias=False)
        self.o = nn.Linear(hidden, hidden, bias=False)
        self.gate = nn.Linear(hidden, ffn, bias=False)
        self.up = nn.Linear(hidden, ffn, bias=False)
        self.down = nn.Linear(ffn, hidden, bias=False)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_like = self.o(self.q(h) + self.k(h) + self.v(h))
        x = x + attn_like
        h2 = self.norm2(x)
        ffn_out = self.down(F.silu(self.gate(h2)) * self.up(h2))
        return x + ffn_out


class ProxyModel(nn.Module):
    def __init__(self, shape: TierShape):
        super().__init__()
        self.embed = nn.Embedding(shape.vocab, shape.hidden)
        self.blocks = nn.ModuleList(
            [ProxyBlock(shape.hidden, shape.ffn) for _ in range(shape.layers)]
        )
        self.head = nn.Linear(shape.hidden, shape.vocab, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(ids)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def bench_one(shape: TierShape, warmup: int, iters: int) -> dict:
    torch.manual_seed(42)
    model = ProxyModel(shape)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ids = torch.randint(0, shape.vocab, (shape.batch_size, shape.seq_len))
    labels = torch.randint(0, shape.vocab, (shape.batch_size, shape.seq_len))

    def step() -> float:
        opt.zero_grad(set_to_none=True)
        logits = model(ids)
        loss = F.cross_entropy(logits.reshape(-1, shape.vocab), labels.reshape(-1))
        loss.backward()
        opt.step()
        return loss.item()

    for _ in range(warmup):
        step()

    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    elapsed = time.perf_counter() - t0

    sec_per_step = elapsed / iters
    tokens_per_step = shape.batch_size * shape.seq_len
    params = sum(p.numel() for p in model.parameters())
    return {
        "tier": shape.name,
        "params_proxy": params,
        "sec_per_step": round(sec_per_step, 4),
        "tokens_per_sec": round(tokens_per_step / sec_per_step, 1),
        "est_minutes_per_1000_steps": round(sec_per_step * 1000 / 60, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=10, help="每個 tier 計時的迭代數")
    parser.add_argument("--warmup", type=int, default=3, help="計時前的暖身迭代數")
    parser.add_argument(
        "--out", type=str, default="docs/experiments/hardware-baseline.md", help="結果輸出檔"
    )
    args = parser.parse_args()

    check_numpy_torch_compat()

    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cpu_count_logical": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
    }

    print("=== NanoMind Phase 0 CPU Benchmark (proxy, 非真實模型架構) ===")
    for k, v in info.items():
        print(f"{k}: {v}")
    print()

    results = []
    for shape in TIERS:
        print(f"Benchmarking {shape.name} ...")
        r = bench_one(shape, warmup=args.warmup, iters=args.iters)
        results.append(r)
        print(f"  -> {r}")
    print()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# Hardware Baseline（Phase 0）\n\n")
        f.write(
            "> ⚠️ 這是用形狀相近的通用 Linear-layer 堆疊做的 **proxy** 測試，\n"
            "> 在 Phase 3 真正的 NanoMind 模型（含 RoPE/GQA/真實 attention）\n"
            "> 完成之前先量測。真實訓練會比這裡的數字慢一些（attention/RoPE\n"
            "> 有這個 proxy 沒有算到的額外成本），這裡的數字只拿來抓數量級，\n"
            "> 不是精確 ETA。\n\n"
        )
        f.write("## Environment\n\n")
        for k, v in info.items():
            f.write(f"- **{k}**: {v}\n")
        f.write("\n## Results\n\n")
        f.write("| Tier | Proxy Params | sec/step | tokens/sec | est. min / 1000 steps |\n")
        f.write("|---|---|---|---|---|\n")
        for r in results:
            f.write(
                f"| {r['tier']} | {r['params_proxy']:,} | {r['sec_per_step']} "
                f"| {r['tokens_per_sec']} | {r['est_minutes_per_1000_steps']} |\n"
            )

    print(f"結果已寫入 {args.out}")


if __name__ == "__main__":
    main()

"""從 minimind 官方大型 JSONL 資料集抽出一小份，變成 NanoMind 實際使用
的訓練資料。

## 使用情境
minimind 官方釋出的 `pretrain_t2t_mini.jsonl`（約 1.2GB）跟
`sft_t2t_mini.jsonl`（約 1.6GB）對 NanoMind 的目標（在一台 2014 年的
雙核 MacBook 上、用極少時間訓練）來說太大了。這支腳本負責從這些大檔案
裡，用均勻隨機的方式抽出一小份（目標大約 10MB / 3MB），同時把「怎麼抽
出來的」完整記錄下來（來源檔案是誰、雜湊值多少、用了什麼 seed），讓
這個抽樣過程可以被任何人重現。

真正的抽樣演算法、雜湊計算、記錄寫入邏輯都在 `nanomind/sampling.py`
裡（那邊每個函式都有自己的單元測試）。這支腳本只負責「串起來、變成一個
可以打指令執行的工具」，本身用整合測試驗證（見
`tests/test_sample_dataset_cli.py`）。

## 用法範例
```bash
# 直接指定要抽幾行：
python scripts/sample_dataset.py \\
    --input /path/to/pretrain_t2t_mini.jsonl \\
    --output dataset/nano_pretrain.jsonl \\
    --lines 20000

# 或者直接說「我要大約幾 MB」，讓腳本自己估算該抽幾行：
python scripts/sample_dataset.py \\
    --input /path/to/sft_t2t_mini.jsonl \\
    --output dataset/nano_sft.jsonl \\
    --target-mb 3
```

兩次執行都會把抽樣紀錄追加寫進 `dataset/raw/MANIFEST.md`（可以用
`--manifest` 改路徑），所以對 pretrain 和 sft 兩個檔案各執行一次之後，
MANIFEST.md 會同時保留兩筆完整記錄。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 這支腳本放在 scripts/ 資料夾底下，但需要 import 專案根目錄的
# nanomind package。用 `python scripts/sample_dataset.py` 直接執行時，
# Python 預設只會把「這支腳本所在的資料夾」（也就是 scripts/）加進
# sys.path，並不會自動加入專案根目錄，所以如果不手動處理，
# `import nanomind` 會找不到套件而失敗。這裡手動把專案根目錄（這支
# 腳本的上一層）加進 sys.path，確保不管從哪個目錄執行這支腳本都能正常
# import nanomind——這是「用 python 直接執行腳本」這種用法常見的做法，
# 如果之後把專案改成用 `pip install -e .` 安裝成套件，就不需要這段了。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nanomind.sampling import (  # noqa: E402  (需要在上面 sys.path 設定之後才能 import)
    compute_sha256,
    estimate_line_count_for_target_size,
    reservoir_sample_lines,
    write_manifest_entry,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """組出這支 CLI 工具的命令列參數規則。"""
    parser = argparse.ArgumentParser(
        description="從大型 JSONL 資料集均勻隨機抽出一小份，並記錄抽樣過程。"
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="來源 JSONL 檔案路徑（例如下載好的 minimind 資料集）",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="抽樣結果要寫到哪個檔案（例如 dataset/nano_pretrain.jsonl）",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="亂數種子，預設 42，同樣的 seed 保證可重現"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("dataset/raw/MANIFEST.md"),
        help="抽樣記錄要追加寫到哪個檔案，預設 dataset/raw/MANIFEST.md",
    )

    # --lines 跟 --target-mb 是「二選一」的關係：使用者要嘛直接說要抽
    # 幾行，要嘛說想要大約幾 MB（由腳本自動換算成行數），不能兩個都給
    # （會搞不清楚該聽誰的），也不能兩個都不給（不知道要抽多少）。
    # argparse 的 mutually_exclusive_group(required=True) 剛好完整表達
    # 這個「恰好選一個」的限制，比自己手動寫 if/else 檢查更不容易漏掉
    # 邊界情況（例如忘記檢查「兩個都沒給」）。
    size_group = parser.add_mutually_exclusive_group(required=True)
    size_group.add_argument("--lines", type=int, help="直接指定要抽出幾行")
    size_group.add_argument(
        "--target-mb", type=float, help="想要大約幾 MB 的輸出檔案，由腳本自動估算該抽幾行"
    )
    return parser


def run(args: argparse.Namespace) -> None:
    """實際執行抽樣流程，拿掉了 argparse 外殼，方便測試直接呼叫。

    流程：決定要抽幾行 → 算來源檔案雜湊 → 抽樣 → 寫出結果檔案 →
    把整個過程記錄進 manifest。
    """
    if args.lines is not None:
        line_count = args.lines
    else:
        target_bytes = int(args.target_mb * 1024 * 1024)
        line_count = estimate_line_count_for_target_size(args.input, target_bytes)
        print(f"根據探測結果，估算需要抽約 {line_count} 行才能達到約 {args.target_mb} MB")

    print("正在計算來源檔案的 SHA256（檔案較大時可能需要一點時間）...")
    source_sha256 = compute_sha256(args.input)

    print(f"正在抽樣 {line_count} 行（seed={args.seed}）...")
    sampled_lines = reservoir_sample_lines(args.input, k=line_count, seed=args.seed)

    # 確保輸出檔案所在的資料夾存在（例如第一次執行時 dataset/ 可能
    # 還沒建立），parents=True 代表連中間缺的資料夾也一併建立，
    # exist_ok=True 代表資料夾已存在時不要報錯。
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.writelines(sampled_lines)

    write_manifest_entry(
        args.manifest,
        source_path=str(args.input),
        source_sha256=source_sha256,
        sample_seed=args.seed,
        sample_line_count=len(sampled_lines),
        output_path=str(args.output),
    )

    output_size_mb = args.output.stat().st_size / (1024 * 1024)
    print(
        f"完成！抽出 {len(sampled_lines)} 行，寫入 {args.output}"
        f"（約 {output_size_mb:.2f} MB）"
    )
    print(f"抽樣記錄已追加寫入 {args.manifest}")


def main() -> None:
    """真正的命令列進入點：解析 sys.argv，再呼叫 run()。"""
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

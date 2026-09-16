"""在 NanoMind 的訓練語料上，訓練一個小型 Byte-Level BPE tokenizer。

真正的訓練邏輯（怎麼從 JSONL 抽出文字、怎麼呼叫 BPE 訓練器）都在
`nanomind/tokenizer.py`，並且已經用合成的假語料測過正確性（見
`tests/test_tokenizer.py`）。這支腳本只負責把它包成一個可以直接在終端機
執行的指令，串接邏輯本身用整合測試驗證（見
`tests/test_train_tokenizer_cli.py`）。

## 用法範例
```bash
python scripts/train_tokenizer.py \\
    --corpus dataset/nano_pretrain.jsonl dataset/nano_sft.jsonl \\
    --vocab-size 1536 \\
    --output-dir nanomind_artifacts/tokenizer_tier1
```

`--corpus` 可以一次給多個檔案（同時吃 pretrain 跟 sft 兩份資料），
訓練出來的 tokenizer 才會認得兩種資料裡出現的所有文字內容。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 跟 scripts/sample_dataset.py 同樣的理由：手動把專案根目錄加進
# sys.path，確保用 `python scripts/train_tokenizer.py` 直接執行時，
# 不管從哪個工作目錄執行，都能正確 import 到 nanomind package。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nanomind.tokenizer import train_bpe_tokenizer  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="在給定的 JSONL 語料上訓練一個 Byte-Level BPE tokenizer。"
    )
    parser.add_argument(
        "--corpus",
        required=True,
        nargs="+",
        type=Path,
        help="一或多個 JSONL 語料檔案路徑，例如 dataset/nano_pretrain.jsonl "
        "dataset/nano_sft.jsonl（可以一次給多個，用空白隔開）",
    )
    parser.add_argument(
        "--vocab-size",
        required=True,
        type=int,
        help="目標詞表大小（實際結果可能略小於這個數字，語料變化不夠豐富時"
        "BPE 訓練器會提早停止，這是正常現象）",
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="訓練好的 tokenizer 要存到哪個資料夾"
    )
    return parser


def run(args: argparse.Namespace) -> None:
    """實際執行訓練，拿掉了 argparse 外殼，方便測試直接呼叫。"""
    print(f"正在讀取 {len(args.corpus)} 個語料檔案：")
    for path in args.corpus:
        print(f"  - {path}")

    tokenizer = train_bpe_tokenizer(
        corpus_paths=args.corpus,
        vocab_size=args.vocab_size,
        output_dir=args.output_dir,
    )

    print(f"訓練完成，實際詞表大小：{tokenizer.vocab_size}（目標上限：{args.vocab_size}）")
    print(f"已存到：{args.output_dir}")

    # 印幾個簡單的編碼範例，讓使用者可以「肉眼」快速確認中英文都正常，
    # 不用另外再開一個 Python shell 手動測試。
    demo_texts = ["Hello, NanoMind!", "你好，NanoMind！"]
    print("\n簡單 round-trip 檢查（編碼再解碼，應該要跟原文一模一樣）：")
    for text in demo_texts:
        decoded = tokenizer.decode(tokenizer.encode(text))
        status = "OK" if decoded == text else "!! 不一致 !!"
        print(f'  [{status}] "{text}" -> "{decoded}"')


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

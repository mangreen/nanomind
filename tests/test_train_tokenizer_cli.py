"""測試 scripts/train_tokenizer.py：把 nanomind/tokenizer.py 的訓練邏輯
串成一個可以在終端機執行的指令。

底層的 BPE 訓練/round-trip 正確性已經在 tests/test_tokenizer.py 測過，
這裡只測「CLI 串接對不對」：參數有沒有正確傳下去、多個語料檔案能不能
一起吃進去、輸出資料夾裡有沒有產生預期的檔案。
"""

import argparse
import json
from pathlib import Path

from nanomind.tokenizer import load_tokenizer
from scripts.train_tokenizer import run


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _make_args(**overrides):
    defaults = dict(corpus=None, vocab_size=300, output_dir=None)
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _rich_records(n=50):
    """造出足夠豐富的假資料，讓 BPE 訓練有東西可以合併。"""
    sentences = [
        "今天天氣真好，適合出去走走。",
        "The quick brown fox jumps over the lazy dog.",
        "人工智慧正在改變我們的生活方式。",
        "Machine learning models require large amounts of data.",
    ]
    return [{"text": f"{sentences[i % len(sentences)]}（{i}）"} for i in range(n)]


class TestRun:
    def test_trains_tokenizer_from_single_corpus_file(self, tmp_path):
        corpus_path = tmp_path / "corpus.jsonl"
        _write_jsonl(corpus_path, _rich_records())
        output_dir = tmp_path / "out"

        run(_make_args(corpus=[corpus_path], output_dir=output_dir))

        # 訓練完應該可以直接用 load_tokenizer 載回來，這是最直接的
        # 「有沒有真的產生一個可用 tokenizer」的驗證方式。
        tokenizer = load_tokenizer(output_dir)
        assert tokenizer.vocab_size > 0

    def test_trains_tokenizer_from_multiple_corpus_files(self, tmp_path):
        # NanoMind 實際使用情境：同時吃 nano_pretrain.jsonl 跟
        # nano_sft.jsonl 兩個檔案，訓練同一份 tokenizer。
        corpus_a = tmp_path / "pretrain.jsonl"
        corpus_b = tmp_path / "sft.jsonl"
        _write_jsonl(corpus_a, _rich_records(30))
        _write_jsonl(
            corpus_b,
            [
                {
                    "conversations": [
                        {"role": "user", "content": f"問題{i}"},
                        {"role": "assistant", "content": f"回答{i}"},
                    ]
                }
                for i in range(30)
            ],
        )
        output_dir = tmp_path / "out"

        run(_make_args(corpus=[corpus_a, corpus_b], output_dir=output_dir))

        tokenizer = load_tokenizer(output_dir)
        # 兩個檔案的內容都應該有被納入訓練，用一個來自 sft 檔案的
        # 對話標記字串試著編碼、解碼，確認能正確處理。
        text = "<|im_start|>user\n問題0<|im_end|>\n"
        assert tokenizer.decode(tokenizer.encode(text)) == text

    def test_creates_output_directory_if_missing(self, tmp_path):
        corpus_path = tmp_path / "corpus.jsonl"
        _write_jsonl(corpus_path, _rich_records())
        # 特意用一個「還不存在、而且中間還缺一層」的路徑，確認 CLI
        # 會自動建立資料夾，而不是因為資料夾不存在而報錯。
        output_dir = tmp_path / "does" / "not" / "exist" / "yet"

        run(_make_args(corpus=[corpus_path], output_dir=output_dir))

        assert (Path(output_dir) / "tokenizer.json").exists()

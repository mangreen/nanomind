"""測試 scripts/sample_dataset.py：把 nanomind/sampling.py 的三個工具
串成一個可以在終端機執行的抽樣指令。

這裡測的是「串接邏輯」本身（讀參數 → 抽樣 → 寫檔 → 寫 manifest 這一整條
流程有沒有正確接起來），底層的抽樣演算法/雜湊計算/manifest 寫入邏輯已經
分別在 tests/test_sampling.py 測過了，這裡不重複測那些細節，屬於「整合
測試」而不是「單元測試」（差異見 CLAUDE.md 的「測試哲學」章節）。

因為 argparse 的 CLI 入口不好直接測（測試裡直接組出一堆命令列字串很
麻煩），這裡改成呼叫 `run(args)` 這個「拿掉 argparse 外殼」的核心函式，
用 `argparse.Namespace` 直接組出參數物件——這是測試「有 argparse 的
CLI 腳本」時常見的做法：把「解析參數」跟「實際做事」分成兩個函式，
測試只測後者。
"""

import argparse
import json

from scripts.sample_dataset import run


def _write_fixture_jsonl(tmp_path, n_lines: int):
    path = tmp_path / "source.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n_lines):
            f.write(json.dumps({"text": f"這是第 {i} 筆假資料 fake line {i}"}) + "\n")
    return path


def _make_args(**overrides):
    """組出一個 run() 需要的參數物件，預設值都給一個沒有意義但合法的值，
    每個測試只需要覆寫自己在意的欄位，可讀性比每次都寫滿所有欄位好。
    """
    defaults = dict(
        input=None,
        output=None,
        seed=42,
        manifest=None,
        lines=None,
        target_mb=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestRunWithExplicitLineCount:
    def test_creates_output_file_with_exact_line_count(self, tmp_path):
        source = _write_fixture_jsonl(tmp_path, 500)
        output = tmp_path / "nano_pretrain.jsonl"
        manifest = tmp_path / "MANIFEST.md"

        run(_make_args(input=source, output=output, lines=20, manifest=manifest))

        output_lines = output.read_text(encoding="utf-8").splitlines()
        assert len(output_lines) == 20

    def test_every_output_line_is_valid_json(self, tmp_path):
        # 確保抽樣過程沒有不小心截斷行內容、破壞 JSON 格式——如果抽樣
        # 邏輯有 bug 把某一行從中間切斷，這個測試會抓到。
        source = _write_fixture_jsonl(tmp_path, 300)
        output = tmp_path / "nano_pretrain.jsonl"
        manifest = tmp_path / "MANIFEST.md"

        run(_make_args(input=source, output=output, lines=15, manifest=manifest))

        for line in output.read_text(encoding="utf-8").splitlines():
            parsed = json.loads(line)  # 如果格式壞了，這裡會直接丟例外
            assert "text" in parsed

    def test_writes_manifest_entry(self, tmp_path):
        source = _write_fixture_jsonl(tmp_path, 100)
        output = tmp_path / "nano_pretrain.jsonl"
        manifest = tmp_path / "MANIFEST.md"

        run(_make_args(input=source, output=output, lines=10, manifest=manifest))

        manifest_content = manifest.read_text(encoding="utf-8")
        assert str(output) in manifest_content
        assert "10" in manifest_content  # 抽樣行數有被記錄下來


class TestRunWithTargetMegabytes:
    def test_creates_output_file_approximately_matching_target_size(self, tmp_path):
        # 造一個每行長度已知、固定的檔案，這樣可以驗證「目標 MB數」轉換
        # 成「抽幾行」的估算邏輯，最終產出的檔案大小是不是合理接近目標。
        source = tmp_path / "uniform.jsonl"
        line = json.dumps({"text": "x" * 90}) + "\n"  # 每行固定長度
        with open(source, "w", encoding="utf-8") as f:
            for _ in range(20000):
                f.write(line)

        output = tmp_path / "nano_pretrain.jsonl"
        manifest = tmp_path / "MANIFEST.md"
        # 目標 0.5 MB，用固定行長度的檔案，估算應該要相當準確。
        run(
            _make_args(
                input=source, output=output, target_mb=0.5, manifest=manifest
            )
        )

        actual_mb = output.stat().st_size / (1024 * 1024)
        # 允許 30% 誤差——這是「估算」不是精確控制，見
        # estimate_line_count_for_target_size 的說明。
        assert 0.35 <= actual_mb <= 0.65

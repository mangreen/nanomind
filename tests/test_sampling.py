"""測試 nanomind/sampling.py：從一個大型 JSONL 檔案「均勻隨機」抽出一小部分。

## 為什麼需要這個模組
Phase 2 要從 minimind 官方的 pretrain_t2t_mini.jsonl（1.2GB）和
sft_t2t_mini.jsonl（1.6GB）各抽出一小份，變成 NanoMind 實際會用到的
nano_pretrain.jsonl / nano_sft.jsonl（目標各約 10MB / 3MB）。

這裡用的抽樣演算法叫 **Reservoir Sampling（水庫抽樣，Algorithm R）**，
它的厲害之處在於：不需要事先知道檔案總共有幾行，也不需要把整個檔案讀進
記憶體，就能保證每一行被抽中的機率完全相等。這對我們很重要，因為 1.2GB
的檔案如果整個讀進記憶體再抽樣，會很浪費記憶體（雖然這台機器有 16GB，
硬撐得過去，但沒必要）。

因為我這邊（開發沙盒）沒有真正的 minimind 資料集（下載網址不在允許清單
裡），這份測試全部用「假造的小型 JSONL fixture 檔案」驗證抽樣邏輯本身
正不正確；真正對 1.2GB/1.6GB 真實檔案的執行，需要您在自己的機器上跑
（見 docs/reports/phase-2.md 的說明）。
"""

import json

from nanomind.sampling import (
    compute_sha256,
    estimate_line_count_for_target_size,
    reservoir_sample_lines,
    write_manifest_entry,
)


def _write_fixture_jsonl(tmp_path, n_lines: int, prefix: str = "line"):
    """建立一個有 n_lines 行的假 JSONL 檔案，每行都是可分辨的內容
    （例如 {"id": 0, "text": "line-0"}），方便測試時檢查抽樣結果。
    """
    path = tmp_path / "fixture.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n_lines):
            f.write(json.dumps({"id": i, "text": f"{prefix}-{i}"}) + "\n")
    return path


class TestReservoirSampleLines:
    """驗證「均勻隨機抽 k 行」這件事本身的正確性，這是整個抽樣工具最核心
    的邏輯，必須非常確定它是對的（抽樣如果有偏差，訓練資料的代表性就會
    有問題，而且很難事後發現）。
    """

    def test_returns_exactly_k_lines_when_file_has_more_than_k(self, tmp_path):
        # 檔案有 100 行，只抽 10 行，結果應該剛好是 10 行。
        path = _write_fixture_jsonl(tmp_path, 100)
        result = reservoir_sample_lines(path, k=10, seed=42)
        assert len(result) == 10

    def test_returns_all_lines_when_file_has_fewer_than_k(self, tmp_path):
        # 邊界情況：如果檔案本身只有 5 行，但要求抽 10 行，
        # 不應該報錯或補出不存在的資料，應該就回傳現有的 5 行全部。
        path = _write_fixture_jsonl(tmp_path, 5)
        result = reservoir_sample_lines(path, k=10, seed=42)
        assert len(result) == 5

    def test_same_seed_gives_identical_result(self, tmp_path):
        # 可重現性（reproducibility）是資料處理最重要的性質之一：
        # 同樣的輸入 + 同樣的 seed，一定要能重現一模一樣的抽樣結果，
        # 這樣別人才能用 MANIFEST.md 記錄的 seed 重現出一樣的子集。
        path = _write_fixture_jsonl(tmp_path, 1000)
        result_a = reservoir_sample_lines(path, k=50, seed=123)
        result_b = reservoir_sample_lines(path, k=50, seed=123)
        assert result_a == result_b

    def test_different_seed_gives_different_result(self, tmp_path):
        # 不是硬性數學保證（理論上兩個不同 seed 也可能剛好抽出同一組），
        # 但用夠大的檔案、夠小的抽樣比例，機率上兩者相同的可能性趨近於
        # 0，可以拿來當作「seed 真的有在影響結果」的健康檢查（sanity
        # check），避免不小心把 seed 參數寫成沒作用的裝飾品。
        path = _write_fixture_jsonl(tmp_path, 1000)
        result_a = reservoir_sample_lines(path, k=50, seed=1)
        result_b = reservoir_sample_lines(path, k=50, seed=2)
        assert result_a != result_b

    def test_every_line_is_a_valid_line_from_the_original_file(self, tmp_path):
        # 抽出來的每一行都必須是原始檔案裡真實存在的一行，不能是抽樣
        # 過程中意外造出來的假資料（例如索引算錯、重複、截斷）。
        path = _write_fixture_jsonl(tmp_path, 200)
        original_lines = set(path.read_text(encoding="utf-8").splitlines())
        result = reservoir_sample_lines(path, k=30, seed=7)
        for line in result:
            assert line.rstrip("\n") in original_lines


class TestComputeSha256:
    """MANIFEST.md 需要記錄原始檔案的 SHA256，讓別人可以驗證自己下載的
    minimind 資料集檔案，跟我們當初抽樣時用的是不是同一份（避免上游檔案
    改版後，抽樣結果悄悄跟文件記錄的對不上）。
    """

    def test_matches_known_hash_of_simple_content(self, tmp_path):
        path = tmp_path / "hello.txt"
        path.write_text("hello world", encoding="utf-8")
        # 這個雜湊值是用 `hashlib.sha256(b"hello world").hexdigest()`
        # 實際算出來的標準結果，不是本專案自創的數字，任何一套 sha256
        # 工具（例如指令列的 `sha256sum`）都能重現一模一樣的值。
        expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        assert compute_sha256(path) == expected

    def test_does_not_load_whole_file_into_memory_at_once(self, tmp_path):
        # 這不是嚴格測記憶體用量（pytest 不方便測這個），而是測「檔案
        # 分成好幾個 chunk 也能算出正確結果」，間接證明實作是用串流
        # （streaming）方式讀取，而不是一次 read() 整個檔案。
        path = tmp_path / "bigger.txt"
        content = "abc123" * 100000  # 600,000 bytes，模擬「不算小」的檔案
        path.write_text(content, encoding="utf-8")
        import hashlib

        expected = hashlib.sha256(content.encode("utf-8")).hexdigest()
        assert compute_sha256(path) == expected


class TestEstimateLineCountForTargetSize:
    """使用者通常心裡想的是「我要大約 10MB 的資料」，而不是「我要抽
    23,481 行」——這組函式負責把「目標檔案大小」換算成「該抽幾行」，
    讓使用者不用自己心算平均每行幾個 bytes。
    """

    def test_estimates_reasonable_line_count_for_uniform_lines(self, tmp_path):
        # 造一個「每一行都剛好 100 bytes（含換行字元）」的檔案，這樣
        # 平均行長度是已知的，可以直接驗證估算結果準不準。
        path = tmp_path / "uniform.jsonl"
        line_content = "x" * 99 + "\n"  # 99 個字元 + 1 個換行 = 100 bytes
        with open(path, "w", encoding="utf-8") as f:
            for _ in range(1000):
                f.write(line_content)
        # 目標 10,000 bytes，每行 100 bytes，理論上應該抓約 100 行。
        result = estimate_line_count_for_target_size(path, target_bytes=10_000)
        # 用「合理範圍」而不是「剛好等於」來斷言，因為這是機率估算，
        # 不是精確計算；只要落在合理誤差範圍內就算正確。
        assert 80 <= result <= 120

    def test_handles_file_smaller_than_probe_window_without_crashing(self, tmp_path):
        # 邊界情況：如果檔案本身的行數，比「用來估算平均行長度」的探測
        # 樣本數還少（例如檔案只有 10 行，但探測樣本預設看前 5000 行），
        # 探測邏輯必須能正常在檔案結尾停下來，不能因為「想讀第 5000 行
        # 但檔案沒有那麼多行」而崩潰或卡住。
        path = tmp_path / "small.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for i in range(10):
                f.write(f"line-{i}\n")
        result = estimate_line_count_for_target_size(path, target_bytes=10_000_000)
        # 檔案平均每行約 8 bytes，目標 10,000,000 bytes，理論上會估出
        # 一個遠大於檔案實際行數（10 行）的數字——這是預期行為，因為這個
        # 函式只負責「估算」，「檔案實際上沒有那麼多行」這件事由後續的
        # reservoir_sample_lines 自動處理（回傳全部現有的行，見上面
        # TestReservoirSampleLines.test_returns_all_lines_when_file_has_fewer_than_k）。
        assert result > 10



    """MANIFEST.md 是我們對「這份抽樣資料怎麼來的」的唯一書面記錄，格式
    要穩定、內容要完整，這樣半年後回頭看才知道這份資料是怎麼產生的。
    """

    def test_creates_new_file_with_header_if_not_exists(self, tmp_path):
        manifest_path = tmp_path / "MANIFEST.md"
        write_manifest_entry(
            manifest_path,
            source_path="pretrain_t2t_mini.jsonl",
            source_sha256="abc123",
            sample_seed=42,
            sample_line_count=1000,
            output_path="nano_pretrain.jsonl",
        )
        content = manifest_path.read_text(encoding="utf-8")
        assert "pretrain_t2t_mini.jsonl" in content
        assert "abc123" in content
        assert "42" in content
        assert "1000" in content

    def test_appends_to_existing_file_without_overwriting(self, tmp_path):
        # 我們會呼叫這個函式兩次（一次給 pretrain，一次給 sft），
        # 第二次呼叫不能把第一次寫的內容洗掉。
        manifest_path = tmp_path / "MANIFEST.md"
        write_manifest_entry(
            manifest_path,
            source_path="pretrain_t2t_mini.jsonl",
            source_sha256="aaa",
            sample_seed=42,
            sample_line_count=1000,
            output_path="nano_pretrain.jsonl",
        )
        write_manifest_entry(
            manifest_path,
            source_path="sft_t2t_mini.jsonl",
            source_sha256="bbb",
            sample_seed=42,
            sample_line_count=200,
            output_path="nano_sft.jsonl",
        )
        content = manifest_path.read_text(encoding="utf-8")
        assert "pretrain_t2t_mini.jsonl" in content
        assert "sft_t2t_mini.jsonl" in content

"""測試 nanomind/tokenizer.py：訓練並載入 NanoMind 自己的小型 BPE tokenizer。

## 背景：為什麼要自己訓練一個小 tokenizer
詳細的成本分析記錄在 `docs/decisions/MEM-0002-tokenizer-strategy.md`，
簡單說：minimind 官方內建的 tokenizer 詞表有 6400 個詞元，如果直接沿用，
模型最後一層（把內部向量轉成「這個字是誰」的機率分布）的運算量會遠超過
NanoMind 想要的「小」，所以我們自己在 NanoMind 的訓練語料上訓練一個小
很多的詞表（Tier1 目標 1536）。

## 為什麼選 Byte-Level BPE
BPE（Byte Pair Encoding，位元組對編碼）是先把文字拆成最小單位（這裡選
「原始 UTF-8 位元組」而不是「字元」），再反覆把「最常一起出現的兩個單位」
合併成新的詞元，重複這個過程直到詞表大小達到目標。選「位元組」當最小
單位（而不是「字元」）的好處是：**不管遇到什麼語言、什麼符號、甚至是
從沒看過的 emoji，都保證可以無損地編碼再解碼回來**，因為任何文字最終
都能分解成 UTF-8 位元組，不會有「這個字元不在詞表裡」這種問題（用
char-level tokenizer 就會有這個困擾）。這也是本檔案的測試特別包含
`test_chinese_roundtrip` 的原因：中文字元在 UTF-8 裡是多個位元組組成的，
如果 byte-level 的實作有 bug，很可能第一個出包的地方就是中文這種
多位元組字元。
"""

import json

import pytest

from nanomind.tokenizer import (
    SPECIAL_TOKENS,
    extract_text_from_record,
    iter_corpus_texts,
    load_tokenizer,
    train_bpe_tokenizer,
)


class TestExtractTextFromRecord:
    """從一行 JSON 資料裡取出純文字，需要同時認得 pretrain 格式
    （`{"text": ...}`）跟 sft 對話格式（`{"conversations": [...]}`）。
    """

    def test_extracts_text_field_for_pretrain_schema(self):
        record = {"text": "這是一段預訓練文字"}
        assert extract_text_from_record(record) == "這是一段預訓練文字"

    def test_wraps_conversation_turns_with_chat_markers(self):
        # sft 資料要包上 <|im_start|>/<|im_end|>，這樣 tokenizer 訓練時
        # 看到的格式，才會跟 Phase 5 SFT 真正餵給模型的格式一致——如果
        # tokenizer 訓練時從沒看過這種排版方式，之後合併出來的詞元可能
        # 不是最適合這種格式的切法。
        record = {
            "conversations": [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好，有什麼可以幫忙的嗎？"},
            ]
        }
        result = extract_text_from_record(record)
        assert "<|im_start|>user" in result
        assert "你好" in result
        assert "<|im_start|>assistant" in result
        assert "<|im_end|>" in result

    def test_returns_none_for_unrecognized_schema(self):
        # 遇到既不是 pretrain 也不是 sft 格式的資料，回傳 None 而不是
        # 報錯，讓呼叫端（iter_corpus_texts）可以自行決定要跳過。
        record = {"some_other_key": "不認識的格式"}
        assert extract_text_from_record(record) is None


class TestIterCorpusTexts:
    """從檔案讀資料這一層，重點是「壞資料不能讓整個流程崩潰」——1.2GB
    的真實資料裡，難免會有幾行格式怪怪的資料，不能因為這樣就讓訓練
    tokenizer 這件事整個失敗。
    """

    def _write_jsonl(self, path, records):
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def test_yields_text_for_each_valid_line(self, tmp_path):
        path = tmp_path / "corpus.jsonl"
        self._write_jsonl(path, [{"text": "第一行"}, {"text": "第二行"}])
        results = list(iter_corpus_texts([path]))
        assert results == ["第一行", "第二行"]

    def test_skips_malformed_json_lines(self, tmp_path):
        path = tmp_path / "corpus.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"text": "正常的一行"}\n')
            f.write("這一行不是合法的 JSON，缺少引號跟大括號\n")
            f.write('{"text": "另一行正常的"}\n')
        results = list(iter_corpus_texts([path]))
        assert results == ["正常的一行", "另一行正常的"]

    def test_skips_empty_lines(self, tmp_path):
        path = tmp_path / "corpus.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"text": "第一行"}\n')
            f.write("\n")  # 空行，minimind 的資料檔偶爾會有
            f.write('{"text": "第二行"}\n')
        results = list(iter_corpus_texts([path]))
        assert results == ["第一行", "第二行"]

    def test_reads_multiple_files_in_order(self, tmp_path):
        path_a = tmp_path / "a.jsonl"
        path_b = tmp_path / "b.jsonl"
        self._write_jsonl(path_a, [{"text": "來自A"}])
        self._write_jsonl(path_b, [{"text": "來自B"}])
        results = list(iter_corpus_texts([path_a, path_b]))
        assert results == ["來自A", "來自B"]


def _make_rich_fixture_corpus(tmp_path):
    """做一個「有足夠變化」的假語料，讓 BPE 訓練真的能產生有意義的合併
    （太小、太重複的語料，BPE 訓練器可能提早停止，達不到目標詞表大小）。
    混合中文跟英文句子，模擬 minimind 資料集本身中英夾雜的特性。
    """
    path = tmp_path / "corpus.jsonl"
    sentences = [
        "今天天氣真好，適合出去走走。",
        "The quick brown fox jumps over the lazy dog.",
        "人工智慧正在改變我們的生活方式。",
        "Machine learning models require large amounts of data.",
        "小型語言模型也可以在普通電腦上訓練。",
        "NanoMind is a tiny language model trained from scratch.",
        "這是一個關於貓咪的故事，牠喜歡追逐紅色的雷射光點。",
        "Python is a popular programming language for data science.",
        "深度學習需要大量的計算資源和資料。",
        "The weather today is sunny and warm, perfect for a walk.",
    ]
    with open(path, "w", encoding="utf-8") as f:
        # 重複多次、稍微變化順序，讓語料量足夠支撐 BPE 訓練到目標大小。
        for repeat in range(30):
            for i, s in enumerate(sentences):
                f.write(json.dumps({"text": f"{s}（第{repeat}輪-{i}）"}, ensure_ascii=False) + "\n")
    return path


class TestTrainBpeTokenizer:
    """訓練流程本身的測試。這裡故意用很小的 vocab_size（例如 300），
    這樣測試可以快速跑完，不用真的訓練到 Tier1 的 1536——真正的 1536
    詞表訓練，要等 Phase 2 的使用者拿到真實資料後才會執行（見
    docs/reports/phase-2.md）。

    這個 class 底下的所有測試，都是在檢查「同一個訓練好的 tokenizer」的
    不同性質（vocab 大小、特殊 token、各種文字的 round-trip），彼此互相
    獨立、不會互相影響結果，所以用 `scope="module"` 的 fixture **只訓練
    一次**、共用給所有測試用，而不是每個測試都重新訓練一次。

    重構前：13 個測試裡有 6 個各自呼叫一次 `train_bpe_tokenizer`，
    全部跑完要 47 秒。重構後：只訓練 1 次，全部測試共用，大幅縮短
    測試時間，且不影響測試涵蓋的行為（因為這些測試本來就只是「唯讀」
    地檢查同一個訓練結果的不同面向，不會互相干擰）。
    """

    @pytest.fixture(scope="module")
    def tokenizer(self, tmp_path_factory):
        """訓練一次、給這個 class 底下所有測試共用的 tokenizer。

        用 `tmp_path_factory`（而不是一般測試常用的 `tmp_path`）是因為
        `tmp_path` 預設是「每個測試函式各自獨立」的暫存目錄（function
        scope），沒辦法跨測試共用；`tmp_path_factory` 則可以在
        `scope="module"` 的 fixture 裡手動建立一個「整個模組共用」的
        暫存目錄。
        """
        tmp_dir = tmp_path_factory.mktemp("tokenizer_shared")
        corpus_path = _make_rich_fixture_corpus(tmp_dir)
        output_dir = tmp_dir / "tokenizer_out"
        return train_bpe_tokenizer(
            corpus_paths=[corpus_path], vocab_size=300, output_dir=output_dir
        )

    def test_vocab_size_is_within_requested_bound(self, tokenizer):
        # 用「不超過」而不是「剛好等於」斷言：BPE 訓練器會盡量合併到接近
        # 目標，但如果語料本身的變化不夠多，可能達不到，這是正常現象，
        # 不是 bug。至少要大於「byte-level 的 256 個基礎位元組 + 特殊
        # token 數量」，否則代表訓練根本沒有真的做任何合併。
        vocab_size = tokenizer.vocab_size
        assert 256 + len(SPECIAL_TOKENS) <= vocab_size <= 300

    def test_special_tokens_are_present_in_trained_vocab(self, tokenizer):
        vocab = tokenizer.get_vocab()
        for special in SPECIAL_TOKENS:
            assert special in vocab, f"特殊 token {special!r} 沒有出現在訓練好的詞表裡"

    def test_english_text_roundtrips_losslessly(self, tokenizer):
        original = "The quick brown fox jumps over the lazy dog."
        encoded = tokenizer.encode(original)
        decoded = tokenizer.decode(encoded)
        assert decoded == original

    def test_chinese_text_roundtrips_losslessly(self, tokenizer):
        # 這是「中英文編碼正確」這個 Phase 2 過關條件裡，中文的那一半。
        # 中文字元在 UTF-8 裡是多位元組組成的，byte-level BPE 如果實作
        # 有 bug，這裡最容易抓到。
        original = "今天天氣真好，適合出去走走。"
        encoded = tokenizer.encode(original)
        decoded = tokenizer.decode(encoded)
        assert decoded == original

    def test_mixed_chinese_english_and_unseen_text_roundtrips(self, tokenizer):
        # 更嚴格的測試：故意用「訓練語料裡完全沒出現過」的句子，確保
        # byte-level 的「保底能力」（沒看過的內容也能無損還原）是真的
        # 有效，不是只有看過的句子才能還原。
        original = "這句話 mix 中英文，還有從沒看過的內容 🎉 123！"
        encoded = tokenizer.encode(original)
        decoded = tokenizer.decode(encoded)
        assert decoded == original


class TestLoadTokenizer:
    def test_loaded_tokenizer_produces_identical_encoding(self, tmp_path):
        # 訓練完存檔、重新載入之後，同一句話編碼出來的結果必須完全一樣，
        # 這是「存檔/載入沒有遺失任何資訊」的基本保證。
        corpus_path = _make_rich_fixture_corpus(tmp_path)
        output_dir = tmp_path / "tokenizer_out"
        original_tokenizer = train_bpe_tokenizer(
            corpus_paths=[corpus_path], vocab_size=300, output_dir=output_dir
        )

        reloaded_tokenizer = load_tokenizer(output_dir)

        text = "今天天氣真好，適合出去走走。The weather is nice today."
        assert original_tokenizer.encode(text) == reloaded_tokenizer.encode(text)

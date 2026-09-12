"""TDD 第一輪練習：NanoMindConfig。

這個 config 類別之後會被 Phase 3（模型架構）拿來產生 tier0/1/2 的
config/*.json，不是用完即丟的練習品。intermediate_size 的自動計算公式
沿用 minimind 的慣例：ceil(hidden_size * pi / 64) * 64。
"""

import json
import math

import pytest

from nanomind.config import NanoMindConfig, compute_intermediate_size


class TestComputeIntermediateSize:
    def test_matches_minimind_formula_for_known_values(self):
        # 手算對照（見 docs/decisions/ 的規模估算表）
        assert compute_intermediate_size(64) == 256
        assert compute_intermediate_size(128) == 448
        assert compute_intermediate_size(256) == 832

    def test_result_is_always_multiple_of_64(self):
        for hidden in [32, 64, 100, 128, 200, 256, 300]:
            assert compute_intermediate_size(hidden) % 64 == 0

    def test_result_is_at_least_hidden_times_pi_over_64_rounded_up(self):
        hidden = 96
        expected = math.ceil(hidden * math.pi / 64) * 64
        assert compute_intermediate_size(hidden) == expected


class TestNanoMindConfigDefaultsAndAutoFields:
    def test_intermediate_size_auto_computed_when_not_given(self):
        cfg = NanoMindConfig(
            vocab_size=1536,
            hidden_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_seq_len=256,
        )
        assert cfg.intermediate_size == compute_intermediate_size(128)

    def test_intermediate_size_can_be_overridden_explicitly(self):
        cfg = NanoMindConfig(
            vocab_size=1536,
            hidden_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_seq_len=256,
            intermediate_size=512,
        )
        assert cfg.intermediate_size == 512

    def test_default_tie_word_embeddings_is_true(self):
        cfg = NanoMindConfig(
            vocab_size=512,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_seq_len=128,
        )
        assert cfg.tie_word_embeddings is True


class TestNanoMindConfigValidation:
    def test_hidden_size_not_divisible_by_heads_raises(self):
        with pytest.raises(ValueError, match="hidden_size"):
            NanoMindConfig(
                vocab_size=512,
                hidden_size=65,  # 65 % 4 != 0
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_seq_len=128,
            )

    def test_kv_heads_greater_than_attention_heads_raises(self):
        with pytest.raises(ValueError, match="num_key_value_heads"):
            NanoMindConfig(
                vocab_size=512,
                hidden_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=8,  # 比 attention heads 還多，不合理
                max_seq_len=128,
            )

    def test_attention_heads_not_divisible_by_kv_heads_raises(self):
        with pytest.raises(ValueError, match="num_key_value_heads"):
            NanoMindConfig(
                vocab_size=512,
                hidden_size=64,
                num_hidden_layers=2,
                num_attention_heads=5,
                num_key_value_heads=2,  # 5 % 2 != 0，GQA 分組除不盡
                max_seq_len=128,
            )


class TestNanoMindConfigJsonRoundtrip:
    """Phase 3 會把 tier0/1/2 存成 config/*.json，這裡先確保 round-trip 正確。"""

    def _make_tier1_like_config(self) -> NanoMindConfig:
        return NanoMindConfig(
            vocab_size=1536,
            hidden_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_seq_len=256,
        )

    def test_to_dict_contains_all_fields(self):
        cfg = self._make_tier1_like_config()
        d = cfg.to_dict()
        assert d["vocab_size"] == 1536
        assert d["hidden_size"] == 128
        assert d["intermediate_size"] == compute_intermediate_size(128)

    def test_roundtrip_through_json_file(self, tmp_path):
        cfg = self._make_tier1_like_config()
        path = tmp_path / "tier1.json"
        path.write_text(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2))

        loaded = NanoMindConfig.from_json(path)

        assert loaded == cfg

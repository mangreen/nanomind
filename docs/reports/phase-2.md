# Phase 2 報告：Tokenizer + 資料裁切

## ✅ 狀態：已關閉（2026-09-17）

在使用者機器上用真實 minimind 資料集驗證：

| 項目 | 結果 |
|---|---|
| nano_pretrain.jsonl | 13,927 行，12.95 MB（目標 10MB，估算誤差約 30%，可接受） |
| nano_sft.jsonl | 1,106 行，2.01 MB（目標 3MB） |
| Tier1 tokenizer vocab_size | **1536 / 1536（剛好命中目標上限）** |
| Round-trip 檢查 | 中文/英文皆通過 |
| pytest | 42 passed in 5.31s |
| ruff check | All checks passed |

完整數字與分析見 `docs/decisions/MEM-0002-tokenizer-strategy.md` 的
「實際執行結果」區塊。`dataset/raw/MANIFEST.md` 記錄了完整的來源檔案
雜湊與抽樣參數。`nanomind_artifacts/tokenizer_tier1/`（tokenizer 本身）
刻意不進 git，因為 BPE 訓練無隨機性、完全可從 MANIFEST.md 記錄的資訊
重現，詳見 MEM-0002。

---

（以下為原始 Phase 2 執行報告，保留作為記錄）

## 我這邊已經完成、並用合成資料測試驗證的部分

### 1. 資料抽樣工具（`nanomind/sampling.py` + `scripts/sample_dataset.py`）
- `reservoir_sample_lines`：single-pass、O(k) 記憶體的均勻隨機抽樣
  （Algorithm R），不需要事先知道檔案總行數，適合 1.2GB/1.6GB 這種大
  檔案。
- `estimate_line_count_for_target_size`：讓您可以直接說「我要大約
  10MB」，不用自己心算平均行長度。
- `compute_sha256`：串流計算大檔案雜湊值，記錄進 MANIFEST，方便日後
  驗證用的是不是同一份原始檔案。
- `write_manifest_entry`：把每次抽樣的來源/雜湊/seed/行數完整記錄進
  `dataset/raw/MANIFEST.md`（只會新增，不會覆蓋既有記錄）。
- 已用假造的小型 fixture 檔案測過：抽樣行數正確、seed 可重現、不同
  seed 結果不同、邊界情況（檔案行數不足）不會出錯。

### 2. Tokenizer 訓練工具（`nanomind/tokenizer.py` + `scripts/train_tokenizer.py`）
- Byte-Level BPE，特殊 token：`<unk>`, `<pad>`, `<|im_start|>`,
  `<|im_end|>`（沿用 minimind 命名慣例）。
- 自動判斷 pretrain（`text` 欄位）與 sft（`conversations` 欄位，自動包
  上 `<|im_start|>`/`<|im_end|>`）兩種資料格式。
- 已用合成的中英夾雜假語料測過：
  - vocab_size 落在合理範圍
  - 特殊 token 都在詞表裡
  - **英文、中文、中英混合＋從沒看過的內容，編碼再解碼都能無損還原**
    （這是 Phase 2 過關條件「中英文編碼正確」的具體驗證）
  - 存檔後重新載入，編碼結果完全一致
- 決策記錄：`docs/decisions/MEM-0002-tokenizer-strategy.md`

### 測試與 lint 現況
```
pytest: 42 passed
ruff check .: All checks passed!
```

## 需要您做的事

### 步驟 1：下載 minimind 官方資料集
二選一（優先 ModelScope，網速不理想的話用 HuggingFace）：
- ModelScope：https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files
- HuggingFace：https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main

只需要下載這兩個檔案（合計約 2.8GB，抽樣完成後可以刪除，不用留在 repo
裡）：
- `pretrain_t2t_mini.jsonl`（約 1.2GB）
- `sft_t2t_mini.jsonl`（約 1.6GB）

建議先放在 repo 外面（例如 `~/Downloads/`），不要放進 `dataset/raw/`，
避免不小心被 git 追蹤到（雖然 `.gitignore` 已經排除 `dataset/raw/*.jsonl`，
多一層小心比較保險）。

### 步驟 2：抽樣（在 repo 根目錄執行）
```bash
source .venv/bin/activate   # 確保用有裝 requirements-nano.txt 的環境
pip install -r requirements-nano.txt -r requirements-dev.txt   # 確保兩份都裝了

python scripts/sample_dataset.py \
    --input ~/Downloads/pretrain_t2t_mini.jsonl \
    --output dataset/nano_pretrain.jsonl \
    --target-mb 10

python scripts/sample_dataset.py \
    --input ~/Downloads/sft_t2t_mini.jsonl \
    --output dataset/nano_sft.jsonl \
    --target-mb 3
```
（`--target-mb` 是估算值，實際檔案大小可能有些落差，屬於正常現象，見
`estimate_line_count_for_target_size` 的說明；如果差太多可以之後再用
`--lines` 直接指定精確行數重跑一次。）

### 步驟 3：訓練 Tier1 tokenizer
```bash
python scripts/train_tokenizer.py \
    --corpus dataset/nano_pretrain.jsonl dataset/nano_sft.jsonl \
    --vocab-size 1536 \
    --output-dir nanomind_artifacts/tokenizer_tier1
```

### 步驟 4：跑一次完整測試確認環境沒問題
```bash
pytest && ruff check .
```

### 請回報給我
1. `sample_dataset.py` 兩次執行的完整終端機輸出（尤其實際抽出的檔案
   大小、行數）
2. `dataset/raw/MANIFEST.md` 的內容
3. `train_tokenizer.py` 的完整終端機輸出（尤其實際訓練出來的 vocab_size，
   跟兩句 round-trip 檢查的結果）
4. `pytest && ruff check .` 的結果

拿到這些之後，我會補完 MEM-0002 的「實際訓練結果」區塊、正式關閉
Phase 2，然後 merge 回 `main`，開始 Phase 3（模型架構）。

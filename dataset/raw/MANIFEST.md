# Dataset Sampling Manifest

本檔案記錄 NanoMind 資料集是「從哪個原始檔案、用什麼設定」抽樣出來的，任何人都可以照著這裡的 seed 重現一模一樣的子集。

## dataset/nano_pretrain.jsonl
- 來源檔案：`../minidataset/pretrain_t2t_mini.jsonl`
- 來源檔案 SHA256：`6dd6716c84ab36897bdbfc7f88e04f4441c48c1ab7ecee88ce0b0e7d4685560c`
- 抽樣 seed：42
- 抽樣行數：13927
- 輸出檔案：`dataset/nano_pretrain.jsonl`

## dataset/nano_sft.jsonl
- 來源檔案：`../minidataset/sft_t2t_mini.jsonl`
- 來源檔案 SHA256：`abb1e76b2056e14728beb78db96b7b3c491a0bef1ed3e34a9b381b28f29fa518`
- 抽樣 seed：42
- 抽樣行數：1106
- 輸出檔案：`dataset/nano_sft.jsonl`

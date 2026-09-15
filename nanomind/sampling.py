"""從大型 JSONL 資料檔案抽出一小份代表性子集，並記錄抽樣過程的工具函式。

## 這個檔案負責什麼
Phase 2 需要從 minimind 官方釋出的兩個大檔案：
- `pretrain_t2t_mini.jsonl`（約 1.2GB，每行是 `{"text": "..."}`）
- `sft_t2t_mini.jsonl`（約 1.6GB，每行是
  `{"conversations": [{"role": "...", "content": "..."}, ...]}`）

各抽出一小份（目標大約 10MB / 3MB），變成 NanoMind 真正會拿來訓練的
`nano_pretrain.jsonl` / `nano_sft.jsonl`。這個檔案提供三個獨立、各自
可以單獨測試的小工具：

1. `reservoir_sample_lines`：均勻隨機抽出 k 行文字（不用先知道檔案總行數）。
2. `compute_sha256`：對大檔案算雜湊值，用來在文件裡記錄「我們當初抽樣時
   用的是哪一份原始檔案」，方便日後驗證。
3. `write_manifest_entry`：把一次抽樣的來龍去脈（來源、雜湊、seed、
   抽了幾行、輸出到哪裡）追加寫進一份 `MANIFEST.md`，讓抽樣過程可以被
   任何人重現，而不是只存在對話紀錄或某個人的記憶裡。

實際串起這三個工具、變成一個可以在終端機執行的抽樣指令的地方，是
`scripts/sample_dataset.py`（那裡負責處理指令列參數，這裡只負責「邏輯
本身要正確」）。
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path


def reservoir_sample_lines(path: str | Path, k: int, seed: int) -> list[str]:
    """從一個文字檔裡「均勻隨機」抽出最多 k 行，用的是 Reservoir
    Sampling（水庫抽樣）演算法，正式名稱是 Algorithm R。

    ## 為什麼不能簡單地「讀出全部行數，再隨機挑 k 個索引」？
    那個做法完全可行，也比較直覺，但需要「先知道檔案總共有幾行」，代表
    要嘛先完整讀過一次檔案數行數（多一次 I/O），要嘛把整個檔案讀進記憶體
    （對 1.2GB 的檔案來說是不必要的浪費）。Reservoir Sampling 巧妙地
    只需要「讀過一次檔案、只保留 k 行在記憶體裡」就能做到完全均勻的
    隨機抽樣，數學上可以證明每一行被抽中的機率都恰好是 k/n（n 是檔案
    總行數），即使我們事前完全不知道 n 是多少。

    ## 演算法怎麼運作（白話版）
    想像你手上永遠只有一個「水庫」，容量剛好是 k：
    1. 檔案的前 k 行，直接全部放進水庫（水庫還沒滿，來者不拒）。
    2. 從第 k+1 行開始（用 0-indexed 來說是第 i 行，i >= k），
       用機率 k/(i+1) 決定「要不要把這一行換進水庫，並且隨機淘汰水庫
       裡原本的某一行」。
    3. 讀到檔案結尾時，水庫裡剩下的 k 行，就是均勻隨機抽樣的結果。

    直覺上為什麼這樣是「均勻」的：檔案愈後面的行，被「換進水庫」的機率
    雖然愈來愈低（因為 i 愈大，k/(i+1) 愈小），但一旦真的被抽中換進去，
    水庫裡「原本每一行被留下來」的機率也會相應地愈來愈高地被保護，兩者
    數學上剛好抵銷，最後每一行的中選機率完全相等。

    Args:
        path: 來源檔案路徑，一行一筆資料（例如 JSONL）。
        k: 要抽出的行數上限。
        seed: 亂數種子，同樣的 seed 一定會抽出一模一樣的結果，這是
            「可重現性」的關鍵——之後 MANIFEST.md 會把這個 seed 記下來，
            任何人都能重跑一次得到完全一樣的子集。

    Returns:
        抽出來的行內容組成的 list（保留行尾的換行字元）。如果檔案本身的
        行數少於 k，就回傳檔案裡的全部行，不會報錯也不會湊出不存在的
        資料。
    """
    rng = random.Random(seed)
    reservoir: list[str] = []

    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < k:
                # 水庫還沒滿，前 k 行來者不拒。
                reservoir.append(line)
            else:
                # 水庫已滿，用 k/(i+1) 的機率決定要不要「換血」。
                # random.randint(0, i) 會等機率地產生 0 到 i（含）之間的
                # 整數，一共有 i+1 種可能；如果剛好落在 [0, k) 這個範圍
                # 裡（機率是 k/(i+1)），就代表這次「中獎」，用第 i 行
                # 取代水庫裡第 j 個位置原本的內容。
                j = rng.randint(0, i)
                if j < k:
                    reservoir[j] = line

    return reservoir


def compute_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """算出一個檔案的 SHA256 雜湊值，用來確認/記錄「這是不是同一份檔案」。

    ## 為什麼要一塊一塊（chunk）讀，不直接 `f.read()` 整個檔案？
    對幾 KB 的小檔案來說直接整個讀進來當然沒差，但 minimind 的原始資料
    檔動輒 1GB 以上，如果 `f.read()` 整個讀進記憶體再算雜湊，會不必要地
    佔用等同檔案大小的記憶體。改成「每次只讀一小塊（預設 1MB），算完就
    丟掉，再讀下一塊」，不管檔案多大，記憶體用量都固定在 chunk_size
    左右——這是處理大檔案時的標準手法，不是 NanoMind 特有的技巧。

    Args:
        path: 要計算雜湊值的檔案路徑。
        chunk_size: 每次讀取的位元組數，預設 1MB，一般情況不需要調整。

    Returns:
        64 個字元的十六進位 SHA256 雜湊字串（小寫）。
    """
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def write_manifest_entry(
    manifest_path: str | Path,
    *,
    source_path: str,
    source_sha256: str,
    sample_seed: int,
    sample_line_count: int,
    output_path: str,
) -> None:
    """把一次抽樣的完整資訊，以人類看得懂的格式追加寫進 MANIFEST.md。

    這個函式故意設計成「只能追加、不能覆蓋」：如果檔案已經存在，新的
    紀錄會接在後面，不會把之前寫的內容洗掉。因為我們會對 pretrain 和
    sft 兩個資料檔各呼叫一次，兩筆記錄都要保留下來。

    所有參數都用關鍵字參數（keyword-only，也就是呼叫時必須寫成
    `write_manifest_entry(path, source_path=..., ...)` 而不能只靠位置），
    是為了避免日後參數順序改變或呼叫時參數對錯位置卻沒發現的低級錯誤——
    這幾個參數剛好都是字串或數字，型別檢查工具抓不出「參數位置放反了」
    這種錯誤，所以用關鍵字參數在語法層級直接杜絕這個風險。
    """
    manifest_path = Path(manifest_path)
    is_new_file = not manifest_path.exists()

    lines = []
    if is_new_file:
        lines.append("# Dataset Sampling Manifest\n")
        lines.append("\n")
        lines.append(
            "本檔案記錄 NanoMind 資料集是「從哪個原始檔案、用什麼設定」抽樣出來的，"
            "任何人都可以照著這裡的 seed 重現一模一樣的子集。\n"
        )
        lines.append("\n")

    lines.append(f"## {output_path}\n")
    lines.append(f"- 來源檔案：`{source_path}`\n")
    lines.append(f"- 來源檔案 SHA256：`{source_sha256}`\n")
    lines.append(f"- 抽樣 seed：{sample_seed}\n")
    lines.append(f"- 抽樣行數：{sample_line_count}\n")
    lines.append(f"- 輸出檔案：`{output_path}`\n")
    lines.append("\n")

    with open(manifest_path, "a", encoding="utf-8") as f:
        f.writelines(lines)

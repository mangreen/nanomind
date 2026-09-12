# ERR: `git add -A` 誤將整個 `.venv-dev/` 虛擬環境 commit 進版本控制
Date: 2026-09-12
Severity: Medium（本地版本控制汙染，未推送到任何遠端，修復成本低，但若
沒抓到會讓 repo 大小暴增且洩漏一堆不必要的第三方原始碼檔案）

## Description
Phase 1 建立 TDD 測試迴圈時，新建了 `.venv-dev/` 開發用虛擬環境。收尾時
用了 `git add -A` 想一次加入所有異動檔案，結果因為 `.gitignore` 當時只
排除了 `.venv/` 和 `.venv-test/`（Phase 0 用過的名字），沒有涵蓋這次新取
的名字 `.venv-dev/`，導致該次 commit 一次帶入 **14,212 個檔案、
321 萬行新增**（torch 的 C++ 標頭檔、transformers 原始碼等）。

## Root Cause
`.gitignore` 用逐一列舉虛擬環境資料夾名稱（`.venv/`、`.venv-test/`）而不
是萬用字元規則，任何新取的虛擬環境目錄名稱都不會被自動排除。加上用了
`git add -A`（等同 `git add --all`，不看任何白名單，只看 `.gitignore` 黑
名單）放大了這個漏洞的影響範圍。

## Solution
1. 把 `.gitignore` 的規則改成萬用字元 `.venv*/`，涵蓋任何以 `.venv` 開頭
   的目錄名稱，不用每次取新名字都要記得手動加規則。
2. 用 `git reset --hard <上一個乾淨的commit>` 清掉那個誤 commit（因為
   還沒推送到任何地方，本地修正完全安全，不影響任何協作者）。
3. 重新用明確檔名 `git add <file1> <file2> ...` 補做該次 commit。

## Prevention
- **一律用明確檔名 `git add`，不用 `-A` / `--all` / `.`**，除非已經先用
  `git status` 確認過異動清單完全在預期範圍內。
- `.gitignore` 的虛擬環境規則一律用萬用字元（`.venv*/`），不要逐一列舉。
- Commit 前養成習慣：先 `git status --short` 看一眼異動檔案數量，如果
  數字看起來不合理地大（例如個位數的程式碼改動卻顯示幾千個檔案），先
  停下來檢查，不要直接 commit。

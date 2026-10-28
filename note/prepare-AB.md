# Prepare AB：清洗、近似去重與補足 tokens

**狀態：2026-10-16 已還原 AA 流程。** `src/prepare.py` 內容恢復為 commit `5b34f02` 的 `src/prepare_data.py`，沿用目前檔名與 Slurm 入口，只保留一套實作。AB 程式保存在 commit `66ea6ae`，全量執行及訓練結果先提交於 `83cea7a`；以下保留 AB 的歷史紀錄。

## 清洗規則

只過濾訓練候選，保留文件原文；命中規則時刪除整份文件。

| 原因 | 條件 |
| --- | --- |
| `short_text` | 非空白字元少於 50；包括空文件與去重後只剩數字、標題的殘片 |
| `few_words` | 少於 10 個單詞；可辨識的 code、Stack、arXiv、math、PDF、OCR 來源豁免此規則 |
| `invalid_characters` | Unicode replacement character `�` 或控制字元占全文超過 2%；允許正常換行、tab、CR |
| `character_runs` | 連續至少 20 個相同非空白字元的區段，占非空白字元數超過 30% |
| `repeated_lines` | 完全相同行保留一份後，其餘重複行占行字元數超過 50%；比對時去掉行首尾空白、略過空行 |
| `repeated_5grams` | 重複的單詞 5-gram 出現次數（不含每組第一次）占全部 5-gram 超過 70%；上述結構化來源豁免此規則 |

來源優先讀 `source`，其次讀 JSON `metadata` 的 `source` 或 `dataset`。無法辨識時視為 `unknown`，仍套用一般規則。單詞依 Unicode word characters 切分，保留詞內撇號與連字號，轉小寫後計算。

這些閾值是保守起點，並非已驗證的最佳值。尤其短而有效的程式片段仍可能被 50 字元規則刪除；控制字元規則也不代表能辨識所有形式的亂碼。

## 去重

- 沿用 AA 的全文去重：小寫、合併空白後計算 64-bit BLAKE2b，移除重複文件及驗證文件副本。
- 新增全文單詞 5-gram MinHash：固定 seed 0，64 個 hash，8 bands × 8 hashes。LSH 先找候選，再要求全部 64 個 hash 的相等比例至少為 0.9，才判為近似重複。
- 0.9 是 **MinHash 估計的 Jaccard 相似度**，沒有再對原始 5-gram 集合做精確確認；可能漏判或誤判。少於 5 個單詞的文件不做 MinHash，仍做全文去重與品質檢查。
- 先索引全部 50,000 份 held-out 文件，訓練端移除其精確與近似副本。本機 eval 文件本身不刪除、不修改。
- 訓練近似去重只索引通過篩選的代表文件，避免沿相似關係鏈一路擴大刪除；全文集合與近似索引跨補抽批次保留。
- 這版採用文件級 MinHash，沒有額外裁切重複子串。文件內的極端重複由上述品質規則處理。

## 切分、補抽與 packing

沿用 `allenai/dolma3_mix-150B-1025`，先以 seed 42 排列文件，最後 50,000 份保留，取其中前 10,000 份作本機 eval。每批候選按原始資料索引讀取，降低隨機 I/O；驗證集選取、tokenization 與 packing 和 AA 相同。

先處理 7,000,000 份訓練候選。不足 `TARGET_TOKENS = 6_000_000_000` 時，沿同一排列每次補抽 1,000,000 份；只從訓練 split 補抽，不重複取樣。如果訓練 split 耗盡仍不足，直接報錯。

通過篩選後才 tokenize。沿用 Llama 3.2 1B tokenizer、BOS + text、不追加 EOS、每文件截斷至 8,192 tokens；每批 10,000 份文件以 best-fit packing，文件不跨 row，僅含 BOS 的文件略過。品質與 MinHash 計算的是截斷前全文。

最後以 seed 42 打亂 packed rows，再選取累計達到目標的完整 rows，避免直接截取按來源索引排序的前段。準備成功時，訓練資料量在 **6B 至 6B + 8,191 tokens** 之間，包含 BOS、不含 padding。實際訓練上限仍由 `training.py` 的 `MAX_TOKENS` 控制；其 callback 會在下一步可能超限前停止。

## 執行與輸出

```bash
sbatch script/prepare_data.sbatch
```

Slurm 入口已改為 `src/prepare.py`。輸出沿用 `data/bestfit/train`、`data/bestfit/eval`，欄位維持 `input_ids`、`doc_lens`、`n_tokens`，訓練端可直接讀取。

`data/bestfit/prepare_stats.json` 記錄候選數、預算裁切前接受的文件數、各原因移除的文件數、最終 rows／tokens，以及固定 eval 索引的 SHA-256。每份被刪文件只歸入第一個生效原因；這些是**文件數，不是刪除的 token 數**。前處理使用 12 個 worker 計算品質與 MinHash，之後以單一全域索引去重，再平行 tokenize；eval 維持單程序。

檢查使用離線合成資料與簡單 tokenizer，覆蓋品質規則、LSH 候選確認、跨批次去重、驗證集隔離、補抽、token 下限、packing 與資料耗盡。全量執行與訓練比較結果如下。

## 全量執行紀錄

2026-10-15 18:35（Asia/Taipei）啟動 Slurm job `552807`，節點 `25a-hgpn040`，使用 commit `66ea6ae` 的 `src/prepare.py`。配置沿用 dev partition、12 CPUs、200 GB RAM、1 GPU 配額及 4 小時上限；資料處理本身使用 CPU。

當時有四個訓練工作使用舊資料，因此從獨立工作目錄 `/work/u9603854/lab5-training-llama/prepare-66ea6ae` 執行，輸出為該目錄下的 `data/bestfit/{train,eval,prepare_stats.json}`，不覆寫現有訓練資料。沿用叢集已安裝的 `.venv` 及共用 Hugging Face 快取；本次 launcher 省略 `uv sync`，直接執行既有 Python。

日誌：`/work/u9603854/lab5-training-llama/logs/lab5-prepare-552807.out`。工作於 2026-10-15 20:35:32（Asia/Taipei）完成，Slurm 狀態 `COMPLETED`、exit code `0:0`，耗時 2 小時 26 秒。

依輸出的 `prepare_stats.json`，首批 7,000,000 份候選保留 6,128,417 份，無須補抽；移除全文重複 749,487 份、held-out 全文副本 27,793 份、品質規則命中 62,557 份、訓練近似重複 30,615 份、held-out 近似副本 1,131 份。各原因只計第一個命中，不能直接視為相對 AA 額外移除的文件數。

最終 train 為 732,756 rows、**6,000,008,014 tokens**；eval 為 1,544 rows、12,647,446 tokens。eval 索引 SHA-256：`a545671030c73da713545253f6cc95469409829fdd4ea3f93a9971458bea7543`。

## 1B tokens 訓練結果

依 BM／BN 實驗紀錄，兩組使用相同訓練設定、seed 42 與相同 eval 內容，差異為訓練資料及其順序。

| 實驗 | 資料 | Slurm job | 最終 eval perplexity |
| --- | --- | --- | --- |
| BM | 原本 best-fit 資料 | `552943` | 17.143 |
| BN | 本次 AB 資料 | `553156` | 17.151 |

AB 的 perplexity 約高 0.047%，這次沒有觀察到改善。只有一組比較，不能據此確認小幅差距具統計意義，也不能推論 6B tokens 的結果。新增過濾、近似去重、packing 與資料順序同時改變，這次比較無法單獨歸因到其中一項。

## 還原決定

依使用者決定還原 AA：固定抽取 7,000,000 份候選，保留正規化全文去重、held-out 全文副本排除及原有 best-fit packing；移除新增的品質過濾、MinHash、自動補抽及 6B token 裁切。準備資料不再保證固定 token 數，訓練上限仍由 `training.py` 的 `MAX_TOKENS` 控制。

這次只還原程式並記錄結果，沒有重跑 prepare，也沒有刪除、覆寫或搬動任何已產生的 AA／AB 資料，訓練資料路徑設定也沒有更動。

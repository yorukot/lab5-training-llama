# Prepare AA：原有資料準備基準

程式：`src/prepare_data.py`。這份紀錄保存新增品質過濾與模糊去重之前的實作，參數直接寫在程式頂端。

## 資料選取與去重

- 讀取 `allenai/dolma3_mix-150B-1025` 的 `train` split。
- 以 seed 42 產生與 `dataset.shuffle(seed=42)` 相同的排列；最後 50,000 份文件保留為驗證資料，在 tokenization 與 packing 前完成切分。
- 取排列最前面的 7,000,000 份文件作為訓練候選，按原始資料索引排序以減少隨機讀取。
- 將全文轉為小寫、合併連續空白，以 64-bit BLAKE2b 雜湊辨識重複；每組僅保留一份。這是正規化後的全文去重，不處理局部或近似重複；雜湊碰撞沒有額外的原文比對。
- 移除與全部 50,000 份驗證文件具有相同全文雜湊的訓練候選。
- 本機評估使用保留排列中的前 10,000 份文件，即官方驗證集的 20%，按原始索引讀取；不額外清洗或去重驗證集。

## Tokenization 與 packing

- 使用 `meta-llama/Llama-3.2-1B` tokenizer，依既有 OJ 對齊方式產生 BOS + text，不額外追加 EOS。
- 每份文件截斷到最多 8,192 tokens；只剩 BOS 的文件沒有預測目標，會被略過。
- 每批 10,000 份文件按長度由大到小，以 best-fit 放進容量 8,192 的 row；文件不跨 row，準備階段不補 padding。
- 保存 `input_ids`、`doc_lens`、`n_tokens`，讓訓練端能隔離文件 attention 並計算非 padding tokens。
- 訓練集使用 12 個 worker；本機驗證集在單一程序內處理。
- 輸出到 `data/bestfit/train` 與 `data/bestfit/eval`，沿用訓練程式的預設讀取位置。

## 執行與限制

```bash
sbatch script/prepare_data.sbatch
```

執行時會印出全文重複、驗證副本的移除數，以及各 split 的 rows、tokens、填充率與長度達 8,192 的文件數。最後一項也包含原本剛好 8,192 tokens 的文件，不能視為精確的截斷文件數。

此版固定抽取 7,000,000 個候選，沒有品質過濾、MinHash、子串去重或自動補抽。程式中的約 7B tokens 是估計，實際產量依本次資料與 tokenizer 而定；準備的資料量與訓練端最多 6B tokens 的限制是兩件事。

本次僅保存原有流程，沒有重新跑完整資料準備或訓練，也沒有新的刪除比例、產量或 perplexity 結果。

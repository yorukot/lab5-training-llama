# T2 精度與評測對齊：OJ 純 bf16 推論 vs in-training eval

研究員 3 號。日期：2026-10-17。只讀程式、文獻與 CPU 模擬，沒有送 job、沒有用 GPU。

## 0. 一段話結論
- OJ 比 in-training 高 1.15%（CL）到 1.19%（BP）。我把它拆成三個來源，不是原本以為的兩個：
  1. **drop_last 漏掉的 8 列（新發現，估 +0.2% 到 +0.9%，中位約 +0.45%）**
     - eval 的列是 best-fit 由長到短打包的，最後建立的 8 列裝的全是最短的文件（約 500–1,700 篇，每篇少於約 150 token）
     - 短文件每個 token 的 loss 偏高（文件開頭幾個 token 最難）；in-training eval 剛好把它們丟掉
  2. **10k vs 50k 文件集（估 +0.4% 到 +0.7%）**
     - 隨機抽 10k 篇本身的抽樣標準差就有約 0.5–1%，所以這個量級很正常
  3. **bf16 推論（估 +0.02% 到 +0.3%，中位約 +0.1%，約佔 1.15% 的一成）**
     - 線性層權重和 logits 在兩邊**本來就都是 bf16**（autocast 每次把 fp32 權重轉成 bf16 去乘）
     - 真正不同的只有：residual 加總、embedding 輸出、norm 權重、RoPE 的 cos/sin
- 所以 T2 的訓練端對策（最後一段用純 bf16 forward、改捨入、自己轉 bf16）在 OJ 上能拿回的，大概只有 0.03–0.1%，在 1B 的 0.3% 門檻以下。
- **真正值得做的是量測**：約 45–60 分鐘的 1 張 H200 就能把三個來源各自量到 0.01%，6B 的 OJ 預估也會更準。
- 6B 的換算建議先沿用 OJ ≈ in-training × 1.012。三個來源都是評測集的固定性質，或隨 tokens 變化很慢。

---

## 1. 搜尋紀錄

工具：`papers/tools/search.py`（不分大小寫，比對標題加摘要；六大會議 68,363 篇）。共 39 組 regex，合計 446 筆命中，去重後 390 篇。我看過全部標題，讀了約 60 篇摘要；下載 27 篇全文，用 grep 找段落精讀其中 20 篇；另外用網路補了 7 份非六大會議的來源。

| # | regex | 命中 |
|---|---|---|
| q1 | `bfloat16\|\bbf16\b` | 37 |
| q2 | `(bf16\|bfloat16).{0,60}(inference\|evaluat\|perplexity)` | 0 |
| q3 | `scaling laws? for precision` | 3 |
| q4 | `(training\|train)[- ](inference\|rollout)[- ]?(precision )?mismatch` | 10 |
| q5 | `precision mismatch` | 0 |
| q6 | `stochastic rounding` | 7 |
| q7 | `(kahan\|compensated) summation` | 1 |
| q8 | `(accumulat\w+).{0,40}(precision\|error\|bits)` | 99 |
| q9 | `low[- ]precision (inference\|evaluation)` | 2 |
| q10 | `quantization[- ]aware training` | 58 |
| q11 | `post[- ]training quantization.{0,120}(token\|data\|overtrain\|degrad)` | 17 |
| q12 | `(numerical\|floating[- ]point) (precision\|error\|nondetermin\|stability).{0,80}(language model\|llm\|transformer)` | 5 |
| q13 | `nondetermin\w+.{0,80}(inference\|llm\|language model)` | 2 |
| q14 | `(mixed[- ]precision).{0,80}(training\|residual)` | 17 |
| q15 | `residual (stream\|connection).{0,60}(precision\|bf16\|fp32\|quantiz)` | 3 |
| q16 | `(rope\|rotary).{0,80}(precision\|bf16\|bfloat16\|numerical)` | 43（幾乎都是 "Europe"、"property" 之類的誤中） |
| q17 | `(adaround\|adaptive rounding\|rounding (scheme\|policy\|optimization))` | 16 |
| q18 | `(fp8\|8-bit).{0,40}(training).{0,80}(language model\|llm)` | 5 |
| q19 | `(fp16\|float16).{0,60}(rl\|reinforcement\|policy)` | 3 |
| q20 | `massive activations\|outlier (features\|dimensions\|activations)` | 23 |
| q21 | `(lm head\|output layer\|logits?).{0,60}(fp32\|precision\|quantiz)` | 4 |
| q22 | `(round[- ]to[- ]nearest\|\bRTN\b)` | 6 |
| q23 | `(quantization\|precision).{0,40}scaling law` | 6 |
| q24 | `(low[- ]precision training).{0,80}(language model\|llm)` | 2 |
| q25 | `(fp32\|float32) (master\|weights\|accumulat)` | 0 |
| q26 | `(error\|noise) (accumulat\|propagat)\w*.{0,60}(layer\|depth\|transformer)` | 11 |
| q27 | `(overtrain\|over-train\|more tokens\|longer training).{0,100}quantiz` | 2 |
| q28 | `quantiz.{0,100}(overtrain\|over-train\|more (training )?tokens\|harder to quantize)` | 2 |
| q29 | `rotary position(al)? embedding.{0,120}(precision\|bfloat16\|bf16\|numerical)` | 0 |
| q30 | `(importance sampling\|off-policy).{0,80}(mismatch\|inference engine\|rollout engine\|vllm)` | 8 |
| q31 | `(inference engine\|vllm\|sglang).{0,80}(mismatch\|discrepan\|numerical)` | 1 |
| q32 | `(fake\|pseudo\|simulated)[- ]quantiz` | 0 |
| q33 | `(noise injection\|injected noise\|additive noise).{0,80}quantiz` | 1 |
| q34 | `(robust\w*\|sensitiv\w*) to (weight )?(quantization\|rounding\|low precision)` | 6 |
| q35 | `quantization.{0,60}(scale with\|grows with\|increases with).{0,40}(data\|tokens\|training)` | 0 |
| q36 | `(numerical precision\|floating[- ]point precision\|machine precision)` | 40 |
| q37 | `(fp16\|float16\|half precision).{0,60}(bf16\|bfloat16)` | 1 |
| q38 | `(QAT\|quantization-aware).{0,80}(end of\|final\|last).{0,40}(training\|phase\|stage)` | 0 |
| q39 | `(knowledge distillation\|self-distillation).{0,80}quantiz` | 5 |

- **最重要的空白：** q2、q5、q25、q29、q35 都是 0。六大會議裡沒有一篇直接量過「同一組權重，bf16 激活 vs fp32 激活的 ppl 差多少」，更沒有 1B 級的。
  - 最接近的是非六大會議的 llama.cpp 量測（Llama-3-8B，bf16 vs fp16 權重），見第 3 節
- **非六大會議（WebSearch/WebFetch，arXiv 共抓 5 份，每份間隔數秒）：**
  - Qi 等 2025（FP16 修 RL mismatch）
  - Wang 等 2024/TMLR 2025（bf16 RoPE）
  - MiniMax-M1 技術報告
  - llama.cpp PR #7150
  - Zamirai 等 2020（Revisiting BFloat16 Training）
  - "Greedy Decoding Is Not Precision-Invariant"（arXiv 2609.26621）
  - NVIDIA NeMo 的 FSDP2 混合精度文件
- PDF 放在 `/home/yorukot/ml/papers/pdf/T2/`（27 篇，含 `.txt`）。

---

## 2. 程式碼層級的精度分析

### 2.1 兩條路徑逐元件比較

- **in-training eval：** `training.py:83` fp32 模型；`:223` `bf16=True`（autocast）；`:224` Liger；FA3
- **OJ：** `lab5.py:21` `dtype="bfloat16"`；`perplexity.py:192-203` 用 bf16 載入、FA2 hub kernel、`torch.compile(fullgraph, dynamic)`

| 元件 | in-training eval | OJ | 差異 |
|---|---|---|---|
| 線性層權重（q/k/v/o/gate/up/down） | fp32 主權重，autocast 每次用 RNE 轉成 bf16 再乘 | 載入時轉 bf16：`transformers/core_model_loading.py:1240` 是 `tensor.to(device, dtype)`，也是 RNE | **完全相同的位元**。PTQ 那一類的「權重量化損失」兩邊都有，不是這 1.2% 的來源 |
| GEMM 輸入（norm 輸出） | fp32 的 norm 輸出被 autocast 轉成 bf16 | 本來就是 bf16 | 兩邊都捨入一次。OJ 的輸入是從「已經帶 bf16 誤差的 residual」算出來的 |
| GEMM 輸出 | bf16（fp32 累加） | bf16（fp32 累加） | 相同 |
| **residual stream** | `modeling_llama.py:306, 312` `residual + hidden_states`：fp32 + bf16 → **fp32** | bf16 + bf16 → **bf16**，每次加法捨入一次，16 層共 32 次 | **主要差異** |
| embedding 輸出 | `embed_tokens` 不在 autocast 名單，輸出 **fp32**（`modeling_llama.py:370`） | bf16 表 | 只差輸入端一次約 0.16% 的相對雜訊。tied 的 lm_head 兩邊都用 bf16 版本 |
| RMSNorm | Liger `casting_mode="llama"`（`liger_kernel/ops/rms_norm.py:107-150`）：輸入 fp32，整段 fp32，權重 fp32 | HF `modeling_llama.py:61-66`：內部 fp32 求 rstd，`.to(bf16)` 再乘 **bf16 權重** | 33 個 norm 權重向量各被捨入（每通道約 ±0.2%）。輸出在 eager 下多一次捨入，但 compile 很可能把中間那次省掉（`torch/_inductor/config.py:3156-3166`，`emulate_precision_casts` 預設 False，「elide the downcast-upcast pairs」） |
| RoPE cos/sin | dtype 跟著 hidden：`modeling_llama.py:116` `cos.to(dtype=x.dtype)` → **fp32**；Liger 把 q 轉成 cos 的 dtype，fp32 算完存回 bf16（`liger_kernel/ops/rope.py:79-95`），只捨入一次 | cos/sin 是 **bf16**（每個值最多 2^-9 的誤差）；`modeling_llama.py:147` 的乘加在 eager 下有 3 次捨入，compile 後多半融合成 1 次 | 每個 q、k 都多一份約 0.2% 的相位或幅度誤差。角度本身兩邊都用 fp32 算（`:111`），相同 |
| attention | FA3，fp32 的 q/k/v 先轉 bf16（`integrations/flash_attention.py:13-25`、`modeling_flash_attention_utils.py:586-588`） | FA2 | 兩者 softmax 都在 fp32；FA3 論文 Table 3：FA2 和 FA3 的 RMSE 同為 1.9e-4，比一般實作的 3.2e-4 小。**不是來源** |
| SwiGLU | Liger：silu 用 fp32 算，轉 bf16 後乘 up（`liger_kernel/ops/swiglu.py:72-74`） | HF `act_fn(gate)*up`（`modeling_llama.py:164`）；compile 後融合 | ≈ 相同，OJ 甚至可能少一次捨入 |
| logits | eval 也走 Liger 的 fused linear CE：`transformers/trainer.py:3231` 傳 `skip_logits=True`，`liger_kernel/transformers/model/llama.py:113-118`。matmul 在 autocast 內：`amp_custom_fwd` 不轉輸入（`ops/fused_linear_cross_entropy.py:469`），`_can_use_mm_out` 因 dtype 不同回 False（`:33-48`），所以 `:168` `_input_chunk @ weight.t()` 得到 **bf16 logits**，kernel 裡再升 fp32 | `modeling_llama.py:469` 得到 bf16 logits，`perplexity.py:72` `.float()` 後算 CE | **相同**：兩邊都吃 bf16 logits 的捨入 |
| 蒸餾 run 的 eval | `distill.py:100-101`：eval 列沒有 `kd_ids`，走原本的 Liger forward | — | 同上 |
| 文件處理 | BOS + text，截斷 8192，不加 EOS，只有 BOS 的文件丟掉（`prepare.py:64-77`） | 同（`perplexity.py:162-177`，少於 2 個 token 的跳過） | 相同 |
| **評到哪些 token** | `training.py:233` `dataloader_drop_last=True`，每輪 16 列（MICRO_BATCH 2 × 8 卡）。accelerate 的 `BatchSamplerShard` 丟掉最後不滿一輪的列（`accelerate/data_loader.py:218-235`），eval sampler 是依序的（`trainer.py:1174-1177`）。列的順序是 best-fit 由長到短打包時的建立順序（`prepare.py:47-61`，`sorted(docs, key=len, reverse=True)`） | 全部 49,999 篇 | **被丟掉的 8 列 = 最後建立的列 = 最短的文件**，見 2.3 |

另外一個小地方：Trainer 的 eval_loss 是「每個 batch 的平均」再平均（以列加權）。因為列幾乎全滿（padding 0.01%），跟以 token 加權幾乎一樣，可以忽略。

### 2.2 bf16 這一項有多大（推估）

- **每次加法的捨入：** bf16 有 8 位有效位數，RNE 的相對誤差標準差約 0.72 × 2^-7 / √12 ≈ **0.16%**（以 |h| 計）。
- **累積：** 32 次加法，加上 embedding 一次，到最後一層約 √33 × 0.16% ≈ **0.9%** 的相對雜訊。
  - autocast 的路徑裡，每個子層的輸入本來就有一份「新鮮」的 0.16%（GEMM 輸入捨入）
  - OJ 的子層輸入雜訊變異數平均約是它的 10–20 倍
- **直接作用在 logits：** 最後 hidden 有 0.9% 的等向雜訊，logit 的雜訊約為 0.9% × logit 在詞表上的散布（約 2–3），大約 0.02。
  - 期望 loss 增加量 ≈ ½·σ²·(1 − Σp²) ≈ 1.5e-4 nats，約 **0.015% ppl**
- **放大因子：** 早層的雜訊會經過 attention 和 MLP 傳下去。兩個可能把它放大的機制：
  - (i) attention sink：對 sink 的分數很大，相對誤差換成絕對誤差就變大。Wang 等 2024 發現 bf16 RoPE 的誤差主要來自第一個 token
  - (ii) Muon 訓練的模型對量化雜訊特別脆弱：Beyond Outliers（ICLR 2026）的 1.5B W4A4 結果，Muon 的 test loss 9.725，AdamW 3.769
  - 放大 2–20 倍的話是 **0.03–0.3%**
- **外部錨點一：** Yuan 等（NeurIPS 2025）在 7–8B 模型上，只改 GPU、卡數、batch，bf16 下 top-1 機率的標準差是 (10–60)×10⁻⁴。
  - 換算每個 token 的 KL 約 1e-5 到 1e-4，也就是 ≤0.01% ppl
  - 這是 bf16 對 bf16 的差，不是 bf16 對 fp32，但量級相同
- **外部錨點二（非六大會議）：** llama.cpp PR #7150，Llama-3-8B 的 bf16 權重 vs fp16 權重：Δln PPL = 0.040% ± 0.009%，平均 KLD 2.5e-5。
  - 那是 15T token、極度過度訓練的 8B。依 Scaling Laws for Precision 的 D^0.51/N^0.34 換到我們（1.24B、6B tokens）約小 28 倍
  - 而且權重捨入兩邊本來就一樣
- **cos/sin、norm 權重、embedding 三項：** 都是單次約 0.1–0.2% 的相對誤差，量級跟一次 residual 捨入差不多，合計不超過上面的範圍。
- **小結：** bf16 推論估 **+0.02% 到 +0.3%，中位約 +0.1%**。信心低到中：沒有直接量過 1B 的數字，Muon 的脆弱性是最大的不確定來源。

### 2.3 drop_last 這一項（新發現，CPU 模擬）

- **機制：**
  - `prepare.py:100-106`：10k 篇 eval 文件在一個 batch 裡做 best-fit decreasing 打包
  - 長文件先建列，中短文件填空隙；空隙用完以後新建的列，只會裝剩下最短的文件
  - in-training eval 丟掉最後 8 列（00-context.md 第 2 節已記錄「最後 8 列沒評到」）
- **模擬：**
  - 腳本：`/tmp/claude-1000/-home-yorukot-ml-lab5-training-llama/94460c4a-2e94-42fc-8d1d-1c1378fd5e26/scratchpad/sim/droplast8.py`
  - 照抄 `prepare.py` 的 `pack()`
  - 文件長度用 lognormal，平均約 1,000–1,300 token，對齊 OJ 的 62.2M/50k ≈ 1,244
  - 每個位置的 loss 用 c + a·t^(−b) 兩組參數：第 1 個 token 約 6–7.5 nats，長上下文約 2.0–2.2
- **結果：**

| 長度分布 | 最後 8 列的文件數 | 其中最長 | token 佔比 | ppl(全部)/ppl(去掉 8 列) − 1 |
|---|---|---|---|---|
| LN(450, 1.45) | 1,120–1,155 | 93 | 0.53% | +0.42% 到 +0.63% |
| LN(600, 1.25) | 773–852 | 124 | 0.51% | +0.33% 到 +0.50% |
| LN(300, 1.70) | 1,607–1,730 | 69 | 0.60% | +0.58% 到 +0.86% |
| LN(800, 1.00) | 453–516 | 177 | 0.47% | +0.22% 到 +0.34% |

- **結論：** in-training eval 少算了約 0.5% 的 token，但那些全是最難的「文件開頭」token，所以 in-training ppl 被低估約 **0.2–0.9%**。
- **影響範圍：**
  - 純粹是量測偏差，不影響 OJ 分數
  - 不影響 run 之間的比較：每個 run 的 eval 列相同，偏差幾乎是常數
  - 但會影響「in-training → OJ」的換算，以及「bf16 吃掉多少」的判斷
- **驗證：** 在叢集上用 CPU 讀 `data/bestfit/eval`（CL 用的 DATA_DIR），看最後 8 列的 doc_lens 就能確認，不需要 GPU（第 5 節步驟 0）。

### 2.4 文件集差異（10k vs 50k）

- 以 token 加權的 corpus ppl 是比值估計量。
- 隨機 10k 子集相對 50k 全集的標準差 ≈ √[(1/10⁴ − 1/(5×10⁴)) · E[n²(L_i − L)²]/n̄²]。
- 假設：
  - 文件長度的 E[n²]/n̄² ≈ 3–5
  - 文件平均 loss 的標準差 ≈ 0.3–0.5 nats
- 得到約 **0.5–1.0%**。扣掉 drop_last 和 bf16 後剩下的 +0.4–0.7%，只有 0.5–1σ，很正常。
- BP 和 CL 的比值幾乎一樣（1.0119 vs 1.0115）也說得通：哪些文件比較難，不同模型之間高度一致。

---

## 3. 論文表

精讀＝下載全文並 grep 相關段落；摘要＝只讀摘要。

| # | 論文 | 來源 | 跟 T2 的關係 | 關鍵數字或結論 | 讀法 |
|---|---|---|---|---|---|
| 1 | Scaling Laws for Precision（Kumar 等） | ICLR 2025 | 事後量化的損失隨 tokens 變大 | δPTQ = C_T·D^γD/N^γN·e^(−P/γpost)；Table 2（p.27）：C_T 0.0598、γD 0.507、γN 0.344、γpost 0.591。代入 N 1.24B、D 6B、P 8 只得約 1e-5 nats。用低精度訓練會讓模型對 PTQ 更穩（robustification） | 精讀 |
| 2 | Scaling Laws for Floating-Point Quantization Training | ICML 2025 | 指數位 vs 尾數位 | 只量化 GEMM 輸入，bf16 是基準，跟 residual 無關 | 精讀 |
| 3 | Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference（Yuan 等） | NeurIPS 2025 | bf16 推論的數值雜訊 | bf16 下 top-1 機率標準差 (10–60)×10⁻⁴（Fig. 4），fp32 ≈ 0；提出 LayerCast（bf16 權重、fp32 計算） | 精讀 |
| 4 | "Give Me BF16 or Give Me Death"? | ACL 2025 | bf16 是無損基準 | W8A8-FP 在 Llama-3.1 全系列無損 | 摘要 |
| 5 | The case for 4-bit precision（Dettmers 等） | ICML 2023 | 16-bit 被當作無損基準 | 16-bit 基準不量化 | 精讀（基準段） |
| 6 | Intriguing Properties of Quantization at Scale（Ahmadian 等） | NeurIPS 2023 | 訓練設定影響對量化的敏感度 | wd 0.1 vs 0.001：PTQ 退化 0.09% vs 1.36%；fp16 訓練比 bf16 敏感；grad clip 有幫助（我們的 τ 0.39 屬於重 wd，偏穩） | 精讀 |
| 7 | Beyond Outliers: A Study of Optimizers Under Quantization | ICLR 2026 | Muon 模型對量化雜訊的敏感度 | 1.5B（20 tok/param），RTN W4A4：Muon test loss 9.725 vs AdamW 3.769（Table 6），zero-shot 47.75 vs 62.51（Table 3）；誤差傳遞比 outlier 指標更重要 | 精讀 |
| 8 | Compute-Optimal Quantization-Aware Training | ICLR 2026 | 對策 (a) 的依據 | 最佳 QAT 比例隨 tokens/參數位元組增加（10²–10⁴ 時是 15%→55%）；QAT 跟 lr 冷卻一起做最好 | 精讀 |
| 9 | Scaling Law for Quantization-Aware Training | ICML 2026 | 量化誤差隨 tokens 變大 | 誤差隨 N 下降、隨 tokens 上升；W8A8 接近無損 | 摘要＋grep |
| 10 | Continual Quantization-Aware Pre-Training（16→1.58 bit） | ACL 2025 | 對策 (a)：何時切換 | 先 16-bit 再切 QAT，比全程 QAT 好；切換時有 loss 跳升，保留優化器狀態可以減輕 | 摘要 |
| 11 | Model-Preserving Adaptive Rounding（YAQA） | ICML 2026 | 對策 (b) | 以輸出端誤差為目標的自適應捨入，比逐層 proxy 好 | 摘要 |
| 12 | Optimize Weight Rounding via Signed Gradient Descent（SignRound） | EMNLP 2024 | 對策 (b) | 200 步 SignSGD 調捨入方向，在 2–4 bit 有大幅收益 | 摘要 |
| 13 | Quantization Error Propagation | NeurIPS 2025 | 誤差逐層累積 | 層間誤差成長是主要退化來源，低 bit 時最明顯 | 摘要 |
| 14 | Training with Fewer Bits: Stochastic Rounding | EMNLP 2025 | 隨機捨入 | 隨機捨入在訓練中無偏；較大 batch 可以補低精度。只談訓練 | 摘要 |
| 15 | ELMO | ICML 2025 | 純 bf16 訓練 | Kahan summation 加隨機捨入讓純 bf16/FP8 訓練可行（XMC） | grep |
| 16 | FlashAttention-3 | NeurIPS 2024 | FA2 vs FA3 | fp16 下 FA2、FA3 的 RMSE 同為 1.9e-4，一般實作是 3.2e-4（Table 3） | 精讀 |
| 17 | Cut Your Losses in Large-Vocabulary LMs | ICLR 2025 | bf16 logits 和 CE | bf16 的 7-bit 尾數讓小於 2^-12 的梯度被忽略；CE 用 Kahan 變體 | grep |
| 18 | Precision-Induced Miscalibration | ICML 2026 | logit 大小 × 捨入 | softmax 在捨入下不再對平移不變，扭曲 ∝ ε × logit norm（bf16 的 ε = 2^-7） | 精讀 |
| 19 | Scaling FP8 training to trillion-token LLMs | ICLR 2025 | 敏感度隨 tokens 成長 | SwiGLU 的 outlier 放大只在長時間訓練後出現 | 摘要 |
| 20 | Deterministic Inference across TP Sizes That Eliminates Training–Inference Mismatch | ICML 2026 | 訓練與推論數值不一致 | vanilla bf16 下 12 種設定幾乎各出不同結果（Table 1）；BIO+TBIK 讓輸出完全一致 | 精讀 |
| 21 | The Mirage of Optimizing Training Policies | NeurIPS 2026 | 同上（RL） | mismatch 造成 off-policy；引用 FP16 rollout 的修法 | grep |
| 22 | Jackpot | ICLR 2026 | 同上 | 系統面修法：FP32 LM head、確定性推論 | grep |
| 23 | Trust Region Masking for Long-Horizon LLM RL | ICML 2026 | 同上 | 把 BF16/FP32 累加的不同列為後端差異來源之一 | grep |
| 24 | Robust Training at Arbitrary Precision and Sparsity | ICLR 2026 | 雜訊感知訓練 | 把量化當成加性雜訊並讓梯度路徑完整，模型才學得到抗雜訊 | 摘要 |
| 25 | Towards Stable and Effective RL for MoE | ACL 2026 | 同上 | 次要 | grep |
| 26 | Rank-Aware Spectral Bounds on Attention Logits | ICML 2026 | attention 分數大小 | FP8 訓練的溢位界；跟 bf16 推論關係弱 | 摘要 |
| N1 | Defeating the Training-Inference Mismatch via FP16（Qi 等） | 非六大會議（arXiv 2510.26788） | bf16 雜訊量級 | 序列層級的 mismatch，fp16 比 bf16 小約 24 倍；但 bf16、fp16、fp32 推論的準確率「大致相當」（Table 2，AMC23 8K：50.38/50.60/51.54） | WebFetch |
| N2 | When Precision Meets Position: BFloat16 Breaks Down RoPE（Wang 等） | 非六大會議（TMLR 2025） | bf16 RoPE | bf16 破壞相對位置性質，誤差隨長度累積，第一個 token 貢獻最大；沒有 ppl 數字 | WebFetch |
| N3 | MiniMax-M1 技術報告 §3.2 | 非六大會議 | lm_head 精度 | 「high-magnitude activations in the LM head」是主要誤差；改 FP32 後訓練與推論機率的相關係數從約 0.9x 升到 0.99x | WebFetch |
| N4 | llama.cpp PR #7150 | 非六大會議 | bf16 權重的 ppl 損失 | Llama-3-8B，fp16 vs bf16 權重：Δln PPL 0.040% ± 0.009%，平均 KLD 2.5e-5，same top-p 99.74% | WebFetch |
| N5 | Revisiting BFloat16 Training（Zamirai 等 2020） | 非六大會議（arXiv 2010.06192） | 隨機捨入、Kahan | 純 16-bit 訓練時，最近捨入會吃掉小更新；兩招合用可救回 7% 準確率，跟 32-bit 差 −0.1% 到 +0.2% | WebFetch |
| N6 | Greedy Decoding Is Not Precision-Invariant | 非六大會議（arXiv 2609.26621） | bf16 vs fp16 | 49–100% 的 prompt 輸出會分歧；選擇性地用 FP32 算 LM head 可以改善 | WebFetch |
| N7 | NVIDIA NeMo AutoModel，FSDP2 MixedPrecisionPolicy | 非六大會議（文件） | 純 bf16 forward 是業界常態 | `param_dtype: bfloat16`、`reduce_dtype: float32`：forward/backward 用 bf16 計算，fp32 只用在梯度歸約 | WebSearch |

篩選過但沒有精讀：QuEST、Quartet、FP4 All the Way（沒有 PDF）、ECO（ICML 2026，不用 fp32 主權重的量化訓練）、Towards Cheaper Inference with Lower Bit-Width Accumulators（ICLR 2024）、Understanding and Minimising Outlier Features（NeurIPS 2024）、DFloat11、NestedFP、各種 4-bit 以下的 PTQ/QAT。它們談的是 ≤8 bit 或硬體格式，跟「bf16 對 fp32 激活」的距離太遠。

---

## 4. 候選對策

### (a) 最後 X% 的訓練用純 bf16 forward（fp32 主權重和優化器不變）
- **機制：** 訓練時的 forward 跟 OJ 的數值一致，residual、norm 輸出、RoPE 都是 bf16。模型會在衰減段學著對這種雜訊不敏感，等於很輕量的 QAT。順帶讓 in-training eval 跟 OJ 的數值一致。
- **證據：**
  - QAT 文獻：#8 最佳 QAT 比例在低 tokens/參數時小，跟冷卻一起做最好；#10 先高精度再切換比較好；#24 讓模型「看到」雜訊才會變穩
  - Kumar 等（#1）：低精度訓練的模型對事後量化更穩
  - 業界大多用 FSDP2/Megatron 的 bf16 compute（N7），residual 本來就是 bf16，模型品質沒問題
  - RL 文獻（#20–23、N1、N3）：訓練與推論的數值對齊確實改變行為。但那是比率問題，不是 loss 水準
- **預期效果與信心：**
  - 能拿回 bf16 損失的 30–70%
  - 若 bf16 損失是 0.1%，OJ 約 **−0.03% 到 −0.07%**；若量出來是 0.5%，約 −0.15% 到 −0.35%
  - 信心低：效果大小完全取決於第 5 節的量測
- **實作草圖：** 只要一個 hook，不改模型。

```python
# training.py：從某一步起，把 embedding 的輸出轉成 bf16，residual stream 之後就全是 bf16
class Bf16Stream(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step == int(0.8 * state.max_steps):  # 最後 20%，與衰減段重疊
            model.model.embed_tokens.register_forward_hook(lambda m, a, out: out.to(torch.bfloat16))
```

- **為什麼這樣就夠：**
  - Liger RMSNorm 的輸出 dtype 跟著輸入（`liger_kernel/ops/rms_norm.py:459`）
  - RoPE 的 cos/sin 跟著 hidden 變成 bf16（`modeling_llama.py:116`）
  - residual 的加法變成 bf16 + bf16
  - 最後的 norm 輸出 bf16，fused linear CE 仍在 autocast 下得到 bf16 logits
  - 蒸餾的 `MixedCrossEntropy` 本來就把 hidden 轉成 autocast dtype（`distill.py:65-67`）
  - 反向時，embedding 的梯度會自動轉回 fp32
- **跟 OJ 仍不一樣的地方：** norm 權重仍是 fp32（OJ 是 bf16），差別約 0.1%/通道，可以忽略。
- **成本：** 幾乎是 0，甚至更快：residual、norm、RoPE 的記憶體流量減半，估 +1–4% 吞吐，未量測。
- **風險：**
  - 切換那一步可能有很小的 loss 跳升（#10 在 1.58-bit 看到過；bf16 的擾動小得多）
  - eval 曲線在切換點有一個不連續（前面是 autocast 數值，後面是 OJ 數值）
  - 跟過去的 run 比較時，要用同一種數值重評
- **測試設計：**
  - 1B 梯子：CJ 的配方 vs CJ + 最後 20% 純 bf16 forward，同 seed
  - 兩個最終 checkpoint 都用第 5 節的 OJ 評估程式評（bf16、10k 和 50k）
  - seed 雜訊約 0.02%，所以 ≥0.06% 才算數
  - **只有當第 5 節量出 bf16 ≥ 0.15% 時才值得跑**

### (b) 上傳 bf16 權重時用比最近捨入更好的捨入
- **機制：** AdaRound、SignRound、YAQA 一類（#11、#12）逐個權重挑往上或往下捨入，讓輸出最接近。
- **證據：** 這些方法在 2–4 bit 很有效；在 8-bit 有效位數下，RTN 的損失本來就極小（N4：8B 也只有 0.04%；Kumar 等的公式代入我們約 1e-5 nats）。
- **反證：** autocast 在整個訓練中都是用 RTN(w) 做 forward，主權重在 fp32 上用 STE 式的梯度更新。所以最終的 RTN(w) 本身就是「對 bf16 forward 最佳化過」的點。換成別的捨入，反而偏離訓練看到的權重。
- **預期效果與信心：** ≤0.01%，信心中等（偏負面）。
- **實作草圖：** 拿最終權重，在幾百萬個訓練 token 上，以「OJ 數值下的 loss」為目標做 SignRound 式的捨入方向搜尋。
- **成本：**
  - 用到的訓練 token 要算進 6B 預算（規則是重看也算）
  - 約 1 GPU-h
- **風險：**
  - 在校準資料上過擬合
  - 不能用 eval 集校準，那等於作弊
- **測試設計：** 在 CL 上用 OJ 評估程式成對比較（每篇 loss 差），要看到 >0.02% 才有意義。
- **建議：不做。**

### (c) 自己轉好 bf16 再上傳，跟讓 OJ 轉有沒有差
- **機制：** OJ 的轉換是 `tensor.to(dtype)`（`transformers/core_model_loading.py:1240`），也就是 RNE。我們自己用 `model.to(torch.bfloat16).save_pretrained()` 也是 RNE，位元完全相同。
- **預期效果：** **0**（信心高）。
- **好處：**
  - 檔案從約 5 GB 變成約 2.5 GB
  - 上傳前就能確認 OJ 會看到的位元
  - 如果日後做 (b) 或下面的折疊，一定要自己轉
- **實作草圖：**

```python
m = AutoModelForCausalLM.from_pretrained(FINAL_DIR, dtype=torch.float32)
m.to(torch.bfloat16).save_pretrained(OUT_DIR)
```

- **測試設計（CPU）：**
  - 載入 fp32 checkpoint 時指定 `dtype=bfloat16`，跟載入自己存的 bf16 版本逐張量做 `torch.equal`，應該全部為真
  - 也檢查 `model.model.rotary_emb.inv_freq.dtype == torch.float32`（它在 init 時明確用 fp32 算：`modeling_llama.py:105`）
- **成本：** 0。**風險：** 0。可以順手做，但不影響分數。

### (d1) 修量測：最終 checkpoint 改用 OJ 評估程式評（不影響 OJ，但影響決策）
- **機制：** 2.3 的 drop_last 偏差、2.4 的文件集差異、2.2 的 bf16 差異，全部來自「in-training eval ≠ OJ」。
- **建議做法：**
  - 每個要比較或上傳的最終 checkpoint，都用 1 張 GPU 跑第 5 節的 `oj` 模式（10k 約 3 分鐘，50k 約 12 分鐘）
  - 訓練中的曲線照舊：偏差是常數，比較仍然公平
- **預期效果：** OJ 分數 0；6B 的 OJ 預估誤差從 ±0.5% 降到 ±0.05%。
- **風險：** 拿 50k 全集來調超參數，就是對測試集過擬合。只用來預估分數，不用來挑配方。
- **成本：** 每次 3–12 GPU 分鐘。

### (d2) z-loss 或讓 logits 置中，減少 bf16 logits 的捨入
- **機制：**
  - 兩邊的 logits 都是 bf16，|z| 在 16–32 時格距是 0.125
  - softmax 對平移不變，但捨入不是（#18）
  - 用 z-loss 把 log Z 拉向 0，高機率 token 的 logit 就落在 |z| < 4 這種格距細的區間
- **預期效果：** 0.01–0.05%，兩邊的 eval 都受益。信心低：我們不知道 CL 的 logit 實際大小，第 5 節的 `fp32_head` 模式可以量到上限。
- **成本與風險：** 改變訓練目標；PaLM 用 1e-4 的 z-loss 無害，但我們沒試過，T6 也列為盲點。
- **建議：** 只有 `fp32_head` 量出 >0.05% 才考慮。

### (d3) 不建議的
- **上傳時用隨機捨入：** 期望平方誤差是 RTN 的 2 倍（ulp²/6 對 ulp²/12），只會更差。
  - 隨機捨入的好處只在「沒有 fp32 主權重的純 bf16 訓練」（N5、#14、#15），我們不是這種情況
- **把 norm 權重折進後面的線性層（W·diag(g) 後 g 設成 1）：**
  - 數學上等價，可以少一次 norm 權重的捨入
  - 但 RTN(W·diag g) 不是訓練時看到的 RTN(W)，正負不確定，期望約 0

---

## 5. 量測設計（只寫設計和草稿，沒有執行）

### 5.1 目標
用 OJ 自己的 `PerplexityEvaluator`，在 CL（HF `yorukot/lab5-llama`，commit `46d18b0`，OJ 12.7803）上存下每篇的 loss，精確拆出：

- r_bf16 = ppl(OJ 數值, 50k) / ppl(autocast 數值, 50k)
- r_set  = ppl(autocast, 50k) / ppl(autocast, 前 10k)
- r_drop = ppl(autocast, 前 10k) / ppl(autocast, 前 10k 去掉最後 8 列的文件)
- 檢查一：r_bf16 × r_set × r_drop × [ppl(autocast, 10k 去掉 8 列) / 12.635] = 12.7803 / 12.635 = 1.0115
- 檢查二：ppl(OJ 數值, 50k) 要在 0.01% 內重現 12.7803
- 檢查三：ppl(autocast, 10k 去掉 8 列) 要在 0.05% 內重現 12.635
  - 剩下的差異來自 Liger vs HF、FA3 vs FA2；若超過 0.1%，代表還有沒找到的差異，先查清楚再下結論

### 5.2 步驟與成本（1 張 H200，dev 分區，總計約 45–60 分鐘，需要你同意）

| 步驟 | 內容 | 文件數 | 估時 |
|---|---|---|---|
| 0（CPU） | 讀 CL 的 `DATA_DIR/eval`：列數 mod 16、最後 8 列的 doc_lens、token 佔比；把這些文件對到 OJ 的文件編號 | — | 數分鐘 |
| 1 | `oj`：完全照 OJ（bf16 載入、FA2、compile） | 50k | 約 12–15 分 |
| 2 | `autocast`：fp32 載入、bf16 autocast、FA2、compile，等於 in-training 的數值 | 50k | 約 12–18 分 |
| 3 | 拆解 bf16 的來源：`bf16w`（權重落在 bf16 格點、激活 fp32）；`bf16w_rope`（再把 cos/sin 捨入成 bf16）；`fp32_head`（OJ 路徑但 logits 用 fp32） | 10k | 各約 3–4 分 |
| 4（選做） | `fp32`：真 fp32（sdpa、不開 TF32、一篇一篇評），看 autocast 本身離精確值多遠 | 2k | 約 5–10 分 |
| 5（選做） | BP 的 checkpoint 也跑步驟 1、2，看三個比值在不同模型之間是否穩定 | 50k | 約 30 分 |

拆解方式：`oj − bf16w` = residual、norm 輸出、RoPE 運算的激活精度；`bf16w − autocast` = norm 權重和 embedding 的捨入；`bf16w_rope − bf16w` = cos/sin 捨入；`oj − fp32_head` = bf16 logits 捨入（兩邊共有，是 (d2) 的上限）。

### 5.3 程式草稿

```python
# t2_measure.py（草稿，未執行）。用法：python -I t2_measure.py MODE N_DOCS OUT.npz
# MODE: oj | autocast | bf16w | bf16w_rope | fp32_head | fp32
import sys
from contextlib import nullcontext

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "<叢集上 study-group-online-judge 的路徑>/src")  # 只 import，不改 OJ
from judge.evaluators.perplexity import PerplexityEvaluator, eval_batch_ppl
from judge.tasks.lab5 import DATASET, SPLIT

MODEL, REVISION = "yorukot/lab5-llama", "46d18b0"  # CL，OJ 12.7803
mode, n_docs, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
ev = PerplexityEvaluator(tokenizer_id="meta-llama/Llama-3.2-1B", max_seq_length=8192, dtype="bfloat16")
data = load_dataset(DATASET, split=SPLIT).select(range(n_docs))
tokenizer = AutoTokenizer.from_pretrained(ev.tokenizer_id)

load_dtype = torch.float32 if mode in ("autocast", "fp32") else torch.bfloat16
attn = "sdpa" if mode == "fp32" else "kernels-community/flash-attn2"  # FA2 不收 fp32
model = AutoModelForCausalLM.from_pretrained(
    MODEL, revision=REVISION, dtype=load_dtype, attn_implementation=attn
)
assert model.model.rotary_emb.inv_freq.dtype == torch.float32
if mode in ("bf16w", "bf16w_rope"):
    model = model.float()  # 權重值落在 bf16 格點上，residual / norm / RoPE 用 fp32 算
if mode == "bf16w_rope":  # 只把 cos/sin 捨入成 bf16
    rope = model.model.rotary_emb.forward
    model.model.rotary_emb.forward = lambda x, pos: tuple(t.bfloat16().float() for t in rope(x, pos))
if mode == "fp32_head":  # OJ 的路徑，但 logits 用 fp32 算
    head = model.lm_head
    head.forward = lambda h: torch.nn.functional.linear(h.float(), head.weight.float())
model = model.cuda().eval()
uses_autocast = mode in ("autocast", "bf16w", "bf16w_rope")
ctx = torch.autocast("cuda", torch.bfloat16) if uses_autocast else nullcontext()

ids = tokenizer(data["text"], truncation=True, max_length=8192)["input_ids"]
doc = np.array([i for i, x in enumerate(ids) if len(x) >= 2])  # OJ 也跳過少於 2 個 token 的
loss, count = [], []
with ctx:
    if mode == "fp32":  # 一篇一篇評：不需要文件遮罩，sdpa 會忽略 cu_seq_lens
        torch.backends.cuda.matmul.allow_tf32 = False
        for i in doc:
            r = eval_batch_ppl(model, [ids[i]])
            loss += r["loss_sum"]
            count += r["token_count"]
    else:
        model = torch.compile(model, dynamic=True, fullgraph=True)  # 跟 OJ 一樣
        for r in ev.evaluation_batches(model, tokenizer, data):  # OJ 自己的打包與計分迴圈
            loss += r["loss_sum"]
            count += r["token_count"]
loss, count = np.array(loss), np.array(count)
print(f"{mode} {n_docs} docs: ppl {np.exp(loss.sum() / count.sum()):.4f}")
np.savez(out, doc=doc, loss=loss, count=count)
```

```python
# t2_dropped.py（CPU，草稿）：in-training eval 漏掉的最後 8 列，對到 OJ 的文件編號
import numpy as np
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

rows = Dataset.load_from_disk("<CL 的 DATA_DIR>/eval")
drop = len(rows) % 16  # 每輪 16 列（MICRO_BATCH 2 x 8 卡），預期 8
tail = rows.select(range(len(rows) - drop, len(rows)))
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
oj = load_dataset("cerulean-works/study-group-lab5-eval", split="train").select(range(10_000))
index = {tuple(x): i for i, x in enumerate(tok(oj["text"], truncation=True, max_length=8192)["input_ids"])}
dropped = []
for r in tail:
    start = 0
    for n in r["doc_lens"]:
        dropped.append(index[tuple(r["input_ids"][start : start + n])])
        start += n
n_tok = sum(tail["n_tokens"])
print(f"{drop} rows, {len(dropped)} docs, {n_tok} tokens ({n_tok / sum(rows['n_tokens']):.2%}), "
      f"longest {max(max(r) for r in tail['doc_lens'])}")
np.save("dropped_docs.npy", np.array(dropped))
```

```python
# t2_analyze.py（CPU，草稿）
import numpy as np

def ppl(r, keep=None):
    keep = np.ones(len(r["doc"]), bool) if keep is None else keep
    return np.exp(r["loss"][keep].sum() / r["count"][keep].sum())

oj, ac = np.load("oj_50k.npz"), np.load("autocast_50k.npz")
first = ac["doc"] < 10_000
kept = first & ~np.isin(ac["doc"], np.load("dropped_docs.npy"))
r_bf16, r_set = ppl(oj) / ppl(ac), ppl(ac) / ppl(ac, first)
r_drop = ppl(ac, first) / ppl(ac, kept)
print(f"OJ {ppl(oj):.4f}（應為 12.7803），in-training 等價 {ppl(ac, kept):.4f}（應約 12.635）")
print(f"bf16 {100*(r_bf16-1):+.3f}%、文件集 {100*(r_set-1):+.3f}%、drop_last {100*(r_drop-1):+.3f}%")
# 成對 bootstrap（以文件重抽）：bf16 效果的信賴區間；另外隨機抽 10k 篇算 r_set 的分布，
# 看「前 10k」在隨機 10k 子集裡落在哪個分位
rng = np.random.default_rng(0)
d = oj["loss"] - ac["loss"]
boot = [d[s].sum() / ac["count"][s].sum() for s in (rng.integers(0, len(d), len(d)) for _ in range(1000))]
print("bf16 95% CI（nats/token）", np.percentile(boot, [2.5, 97.5]))
```

- **選做：依位置分桶。** 把 `eval_batch_ppl` 照抄一份到量測腳本（模型呼叫不變），多回傳 [1,64)、[64,256)、[256,1k)、[1k,4k)、[4k,8k) 各桶的 loss 總和。用 10k 篇跑 `oj` 和 `autocast` 兩次，看 bf16 的損失是否集中在長位置（對應 Wang 等的 RoPE 結論）或文件開頭。

### 5.4 判讀規則
- **bf16 < 0.15%：** T2 的訓練端對策（a、b、d2）都不做；6B 預估改用量到的三個比值相乘。
- **0.15–0.5%：** 1B 梯子跑 (a)（最後 20%），一個 run，約 1.2 GPU-h × 8。
- **> 0.5%：** (a) 列為 6B 前必做；同時用 `bf16w` 和 `bf16w_rope` 看是哪一項主導。若是 RoPE，就考慮讓 cos/sin 的捨入在訓練中也發生。
- **drop_last 的部分：** 量到多少就修正多少，只影響預估，不影響分數。

---

## 6. 反證據

- **反對「bf16 是 1.2% 的主因」：**
  - 權重和 logits 的捨入兩邊完全相同（2.1 表）
  - Yuan 等的 bf16 雜訊、llama.cpp 的 8B 量測、Kumar 等的公式都指向 ≤0.05%
  - #4、#9 都說 W8A8 接近無損，bf16 激活比 int8 更精細
  - drop_last 和文件集兩項已經能解釋大部分
- **支持「bf16 可能比我估的大」：**
  - Beyond Outliers：Muon 訓練的 1.5B 在 W4A4 下崩成 9.7（AdamW 3.8），代表 Muon 模型的誤差傳遞增益可能偏大。但那是 4-bit，跟 bf16 的雜訊差了兩個數量級，外推不可靠
  - Wang 等：bf16 RoPE 的誤差在長序列、尤其是第一個 token 放大；我們的文件最長 8192
  - MiniMax-M1：lm_head 的大激活讓訓練與推論的機率相關只有 0.9x
  - Qi 等：序列層級的 bf16 mismatch 比 fp16 大 24 倍
  - 這些都是「雜訊存在」的證據，沒有一個給出 loss 水準的差距；Qi 等的 Table 2 反而顯示 bf16、fp16、fp32 推論的準確率差不多
- **反對 (a) 有用：**
  - 雜訊造成的損失是二階小量，訓練對它的梯度也很小，短期適應可能只拿回一小部分
  - #8 指出低 tokens/參數時最佳的 QAT 比例很小，但那是 1–6 bit 的設定
- **反對我的 drop_last 估計：**
  - 模擬用的是假設的長度分布和位置-loss 曲線，不是真資料
  - 如果短文件本身比較好預測（例如大量模板文字），偏差會變小
  - 步驟 0 在 CPU 上就能確認 token 佔比；loss 部分要靠步驟 2
- **反對「文件集差異可以忽略」：** 抽樣標準差 0.5–1% 的估計依賴文件 loss 的異質性假設；BP 和 CL 比值一致，只說明它穩定，不說明它小。

---

## 7. 結論

1. **1.15% 的拆解（推估）：**
   - drop_last 漏掉最短文件：+0.2% 到 +0.9%（中位約 0.45%）
   - 10k vs 50k：約 +0.4% 到 +0.7%
   - bf16 推論：+0.02% 到 +0.3%（中位約 0.1%，**約佔一成**）
   - drop_last 是這份研究的新發現，原本的兩個來源都沒有算到它
2. **bf16 這一項小的原因：** autocast 早就讓線性層權重、GEMM 輸入、logits 都是 bf16，跟 OJ 位元相同。真正的差異只在 residual 加總、embedding、norm 權重和 RoPE 的 cos/sin。
3. **前 3 名對策：**
   - 第 1 名，(d1) 修量測：最終 checkpoint 用 OJ 評估程式評。零風險，讓 6B 預估準到 ±0.05%
   - 第 2 名，(a) 最後 20% 用純 bf16 forward：一個 hook，近乎零成本，可能更快，預期 −0.03% 到 −0.1%；要等量測確認 bf16 ≥0.15% 才值得驗證
   - 第 3 名，(c) 自己轉 bf16 上傳：位元與 OJ 轉的完全相同，分數 0 差異，但檔案減半、可以事先驗證
   - (b) 更好的捨入、(d2) z-loss、隨機捨入、折疊 norm 權重：預期 ≤0.05% 或為負，不建議
4. **6B 預估：** 三個來源都是評測集固定的性質，或隨 tokens 變化很慢。在量測完成前，沿用 OJ ≈ in-training × 1.012。
5. **下一步：** 第 5 節的量測，約 45–60 分鐘的 1 張 H200（需要你同意）。步驟 0 只用 CPU，可以先做。

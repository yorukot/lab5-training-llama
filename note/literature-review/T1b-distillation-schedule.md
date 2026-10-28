# T1b 蒸餾 B：α 排程、token 加權、收益隨 tokens 的 scaling

研究員 2 號，2026-10-17。範圍見 `01-levers.md` 的 T1b 一節。PDF 和轉好的文字檔在 `/home/yorukot/ml/papers/pdf/T1b/`（非六大會議的放在 `web/` 子資料夾）。

## 0. 先講結論

- **α 排程有理論和文獻支持，但收益小。** 我們自己的數據顯示：如果只看「訓練停在 t 時哪個固定 α 最好」，答案從 5% 時的 ≥1.0，降到 50% 時約 0.96，最後約 0.86。下降集中在學習率衰減段。
  - 兩個獨立來源都指向「α 跟著 lr 一起降」：
    - Peng et al.（ACL 2025）的最佳設定是 WSD-α 搭配 WSD-LR，α 的形狀跟 lr 一樣
    - KD 的 variance-reduction 理論（NeurIPS 2023）：最佳 λ* 會隨步長 γ 變小而變小
  - 預期效果：1B 約 −0.1% 到 −0.3%，6B 差不多。值得用一個 1B run 來測（第 3 節 C1）。
- **依 token 加權：** 在「預訓練、評全部 token 的 ppl」這個條件下，找不到有效的證據。所有正面結果都來自 fine-tune 或生成類指標。排最後，不建議花 6B 前的 GPU 時間。
- **收益隨 tokens 的 scaling：**
  - 我們 1B→3B 每多一倍 tokens，KD（含 EMA）的收益剩 ×0.86
  - 文獻的範圍很寬：
    - Distillation Scaling Laws 套到 1.24B 學生約 ×0.70/倍
    - Strong-Teacher-Not-Needed 更快
    - Gemma 2 和 Peng et al. 在數百 B tokens 時收益幾乎沒衰減
  - 我們的 ×0.86 落在範圍中間，所以收益這一項可信
- **6B 推估：** 11.1–11.2 偏樂觀。主要風險不在 KD，而在「沒有 KD 的基準線」：
  - 現在用的 log-linear 外推（ppl ∝ D^−0.2）假設每多一倍 tokens 下降的 nats 數固定
  - 我們自己的 AO/AP/AL（0.25/0.5/1B）和 Chinchilla、Busbridge 的資料項都顯示「每倍下降量」會遞減
  - 改用飽和形式（L = E + B·D^−β），6B 是 11.4–11.5（OJ 約 11.55–11.65）
  - CM（4.5B）可以直接分辨兩種形式：log-linear 預測 11.77，飽和形式預測 11.89–11.93
- **最佳 α 是否該隨 tokens 降低：** 證據互相矛盾，淨效果是「略降或不變」。0.8–0.86 都可以，選錯的代價 ≤0.06%（由我們的二次擬合曲率算出）。

## 1. 搜尋紀錄

### 1.1 本地文獻庫（68,363 篇，NeurIPS/ICML/ICLR/ACL/EMNLP/NAACL 2023–2026）

- 工具：自己寫的多查詢腳本，一次讀索引，比對標題加摘要，不分大小寫；第 3 欄是額外的 AND 條件
- 共 50 組查詢，聯集 843 篇不重複標題，全部看過標題
- 命中數普遍偏低：多數 regex 有 AND 條件，而且 NeurIPS 2026 有一半沒摘要。命中最多的 q38 是 259，沒有超過 300 的

| id | regex | AND 條件 | 命中 |
|---|---|---|---|
| q01 | `distill\w* scaling law\|scaling laws? (of\|for) (knowledge )?distill\|compute-optimal distill` | — | 1 |
| q02 | `knowledge distillation.{0,60}pre-?train` | — | 25 |
| q03 | `pre-?train\w* distillation\|distillation (during\|in\|for) pre-?train` | — | 8 |
| q04 | `(mixing\|interpolat\w*\|balanc\w*\|trade-?off) (coefficient\|weight\|factor\|ratio\|parameter)` | `distill` | 3 |
| q05 | `(anneal\|schedul\|decay\|curriculum)\w*.{0,40}(distillation\|teacher\|kd) (weight\|loss\|coefficient\|signal\|guidance\|ratio)` | — | 2 |
| q06 | `(adaptive\|dynamic)\w* (distillation\|kd\|teacher) (weight\|loss\|temperature\|coefficient\|ratio)` | — | 2 |
| q07 | `token-?(level\|wise\|adaptive).{0,40}(distill\|kd)` | — | 7 |
| q08 | `(distill\|kd).{0,60}token-?(level\|wise\|selection\|weight\|selective)` | — | 15 |
| q09 | `teacher.{0,30}(entropy\|confidence\|uncertainty)` | `distill` | 11 |
| q10 | `excess loss\|rho-1\|selective language model\|not all tokens` | — | 19 |
| q11 | `capacity gap\|curse of capacity\|teacher-student gap\|student-teacher gap` | `distill\|teacher` | 14 |
| q12 | `(early\|initial\|first) (phase\|stage\|steps\|epochs).{0,80}(distill\|teacher)\|(distill\|teacher).{0,80}(early\|initial) (phase\|stage\|steps\|epochs)` | — | 12 |
| q13 | `minillm\|distillm\|generalized knowledge distillation\|\bGKD\b\|miniplm\|difference sampling` | — | 8 |
| q14 | `(logit\|logits\|teacher).{0,40}(cache\|cached\|offline\|precomputed\|top-?k\|sparse)` | `distill` | 4 |
| q15 | `(small\|compact\|tiny\|efficient\|sub-?billion) language models?.{0,120}distill\|distill.{0,120}(small\|compact\|tiny) language model` | — | 41 |
| q16 | `(curriculum\|progressive\|annealing\|annealed\|staged\|multi-?stage) (knowledge )?distillation` | — | 19 |
| q17 | `(hurt\|harm\|worse than\|fail\w*\|degrad\w*\|ineffective\|not always\|counterproductive).{0,60}(knowledge )?distillation\|distillation.{0,60}(hurt\|…)` | `language model\|LLM\|pre-?train` | 26 |
| q18 | `(distill\|teacher)\w*.{0,100}perplexity` | `language model\|LLM\|pre-?train` | 0 |
| q19 | `(student\|distillation).{0,40}(label smoothing\|soft labels?\|soft targets?\|hard (labels?\|targets?))` | `language model\|LLM\|token` | 2 |
| q20 | `(teacher-guided\|teacher guidance\|guided by a (larger\|smaller\|teacher)).{0,80}(pre-?train\|language model)` | — | 0 |
| q21 | `distill\w*.{0,80}(gemma\|llama\|minitron\|qwen\|phi-)` | — | 95 |
| q22 | `(small\|smaller\|weak)\w* (teacher\|model).{0,80}(larger\|stronger\|bigger) (student\|model).{0,60}(pre-?train\|distill)\|weak-to-strong distill\|reverse distill` | `language model\|LLM` | 2 |
| q23 | `(token\|sample\|data\|loss)[- ](level \|wise )?(weighting\|reweighting\|re-weighting).{0,80}(pre-?train\|language model\|LLM)` | — | 10 |
| q24 | `temperature` | `knowledge distillation.{0,80}(language model\|LLM)\|…` | 2 |
| q25 | `(hard\|difficult\|easy) tokens?` | `distill\|pre-?train` | 4 |
| q26 | `(interpolat\w*\|intermediate) (distribution\|teacher)` | `distill` | 2 |
| q27 | `co-?distill\|online distill\|codistill` | `language model\|LLM\|pre-?train` | 5 |
| q28 | `(bias-variance\|variance reduction\|statistical perspective\|why (does )?(knowledge )?distillation\|understanding (knowledge )?distillation\|does knowledge distillation really)` | `distill` | 12 |
| q29 | `(patient\|consistent\|function matching\|longer training\|training longer).{0,60}(distill\|teacher)` | — | 59 |
| q30 | `(distill\|teacher).{0,80}(data[- ]efficien\|sample[- ]efficien\|token[- ]efficien\|fewer tokens\|compute[- ]efficien)` | `language model\|LLM\|pre-?train` | 4 |
| q31 | `mixing hard and soft\|hard and soft labels\|bridge-garden\|soft and hard (labels\|targets)` | — | 2 |
| q32 | `(early stop\w*\|stop\w* (the )?distill\w*\|turn\w* off.{0,30}distill\|switch\w*.{0,30}(from )?distill)` | — | 112 |
| q33 | `distill` | `pre-?train\w* (language model\|LLM)\|LLM pre-?train\|language model pre-?train\|from scratch` | 118 |
| q34 | `(knowledge distillation\|distill\w*).{0,150}(next[- ]token\|autoregressive\|causal language model\|decoder-only)` | — | 43 |
| q35 | `(teacher\|distill)\w*.{0,120}(tokens? per parameter\|data budget\|training tokens\|token budget\|number of tokens)` | — | 5 |
| q36 | `(student\|teacher).{0,60}(top-?k\|top-p\|truncat\w*).{0,60}(logit\|distribution\|probabilit)` | — | 0 |
| q37 | `loss (weighting\|reweighting).{0,60}(token\|position)\|per-token (loss )?weight\|token importance` | `language model\|LLM\|pre-?train` | 44 |
| q38 | `\b(KD\|knowledge distillation)\b` | `(LLM\|language model)` | 259 |
| q39 | `little help goes a long way\|small (language )?models? (as\|to) (teach\|guide\|help).{0,60}(large\|bigger)\|…` | — | 0 |
| q40 | `(knowledge distillation\|distillation).{0,80}(warm-?up\|first phase\|two-phase\|two-stage\|then (standard\|regular\|plain))` | `language model\|LLM\|pre-?train` | 5 |
| q41 | `(teacher\|soft) (label\|target)s?.{0,80}(noise\|variance\|bias)` | `language model\|LLM\|next-token` | 1 |
| q42 | `self-distillation.{0,80}pre-?train` | `language model\|LLM` | 3 |
| q43 | `ensemble.{0,60}distill` | `pre-?train\|data-constrained\|limited data` | 4 |
| q44 | `(teacher\|kd\|distillation) (signal\|loss) (weight\|weighting\|scheduling\|schedule)` | — | 0 |
| q45 | `(confidence\|entropy\|uncertainty)-(aware\|guided\|based\|weighted).{0,60}(distill\|kd)` | — | 9 |
| q46 | `(selective\|token-selective\|selected) (knowledge )?distillation` | — | 6 |
| q47 | `student.{0,40}(surpass\|outperform)\w* (the \|its )?teacher` | `language model\|LLM\|pre-?train` | 4 |
| q48 | `distill\w*.{0,100}(data-constrained\|limited data\|data scarc\|small data\|data-efficient pre-?train)` | `language model\|LLM` | 3 |
| q49 | `\bKL\b.{0,60}(top-?k\|truncat\w*\|sparse) ` | `distill` | 0 |
| q50 | `(soft-?label\|soft target\|teacher distribution).{0,100}(language model\|LLM)` | — | 2 |

- **漏掉的檢查：** 另外直接比對標題，找到 `Distillation Scaling Laws`、`Strong Teacher Not Needed?`、`MiniPLM` 等，確認都已在命中裡。
- **注意：** 本機的 `grep` 是 shell function，會改變行為。所有全文 grep 都改用 `/usr/bin/grep -a`。

### 1.2 篩選與精讀

- **標題：** 看過 843 篇。多數是推理蒸餾、CoT、diffusion、視覺，跟本題無關。
- **摘要：** 讀了 55 篇，見 scratchpad 的 `cand_abs.txt`。
- **全文：** 下載 25 篇六大會議 PDF，再加 3 份 arXiv（非六大會議）：
  - SALT 2410.18779
  - LFM2 2511.23404
  - MobileLLM-Pro 2511.06719
  - arXiv 只下載 3 次，每次間隔 ≥3 秒，沒遇到 429
- **精讀**（用 grep 找段落、讀表格）共 20 篇：
  - Distillation Scaling Laws、Peng et al. design space、Strong Teacher Not Needed、MiniPLM、Sparse Logit Sampling、TAID、CPD、Bridge-Garden、DWT、KD partial variance reduction
  - Rho-1、Dynamic loss reweighting、ATKD、Minitron、capacity-gap law、high-dim KD、infinite compute、SALT、LFM2、MobileLLM-Pro
- **只讀摘要和少量段落的** 8 篇：Supervision Complexity、VocAgnoLM、Why-KD-works、ToDi、CaRE-KD、Progressive distillation、Token weighting long-range、TSLD
- **網頁**（WebFetch，非六大會議）5 份：
  - Gemma 2（arXiv HTML）
  - Gemma 3（arXiv HTML）
  - Apple AFM 2024（arXiv HTML）
  - Meta 的 Llama 3.2 blog
  - Meta 的 Llama 4 blog

### 1.3 引用追蹤

從 Busbridge et al. 和 Peng et al. 的參考文獻回查：
- Stanton 2021 不在庫中，屬於 NeurIPS 2021，在範圍外
- Hinton 2015 / Kim & Rush 2016 是背景文獻
- Muralidharan（Minitron）已讀
- SALT（Rawat 2024，非六大會議）已補
- Gemma 2/3、Llama 3.2、AFM 已補
- Llama 4 的「動態加權 soft / hard target」也已補

## 2. 論文表

**相符度**看的是跟我們的情境有多接近：從頭預訓練、評全部 token 的 ppl、強老師、1–6B tokens。

| 出處 | 標題 | 設定 | 主張 | 效果大小 | 相符度 | 頁碼 |
|---|---|---|---|---|---|---|
| ICML 2025 | Distillation Scaling Laws (Busbridge et al.) | 學生 143M–12.6B，老師 ≤12.6B，C4，μP，純 KD（λ=1）、τ=1 | (1) 給定 N、D 和老師 loss，有預測學生 loss 的公式；(2) 學生 tokens 夠多時，supervised 一定反超；(3) M=20 時 λ=1 跟 λ* 統計上差不多，λ>0.5 的區段很平；(4) top-k 截斷有傷害，混入 ground truth 可救回 | top-128 截斷 +0.11 nats；λ=0.7 加 top-128 的差距在 0.01 nats 內；τ=1 最好 | 高 | p4, p7, p31, p59–60, p63 |
| ACL 2025 | Pre-training Distillation for LLMs: A Design Space Exploration (Peng et al.) | GLM-4-9B → 1.9B，100B tokens，top-p 0.95 加 top-100，NLL loss，評下游 benchmark | 固定 α 時 0.9 最好；α 遞減優於遞增；WSD-α 搭配 WSD-LR 最好；收益到 500B tokens 仍在 | 相對 LM 的平均：α 0.9 +3.6%、α 1.0 +1.6%、linear dec +4.1%、linear inc +1.1%、1-α+WSD-LR +5.0%、WSD-α+WSD-LR +8.0%（單 seed，benchmark 指標） | 中高 | p3, p5–7 |
| NeurIPS 2026 | Strong Teacher Not Needed? On Distillation in LLM Pretraining | 1.7B 學生，50B tokens；老師 0.5–8B，訓練 10–300B tokens；α ∈ {0.2,…,1.0}；τ=1 | 最佳 α 隨老師強度上升；8B 老師訓練太多時，最佳 α 先升後降；學生 300B tokens 時，in-domain 收益剩很少 | in-domain ppl 最多 +4.3%；8B 老師的最佳 α（依老師 tokens）：0.4/0.6/0.8/1.0/1.0/0.8；300B tokens、α=0.2：+0.9% | 中（老師是自己訓練的，比 Llama 3.1 8B 弱很多） | p4, p7–8, p21–22 |
| ACL 2025 | Sparse Logit Sampling | 300M 學生、3B 老師、10B tokens（另有 ≤3B 學生） | top-k KD 有偏差；「Naive Fix」（tail 全給 ground truth，等於我們的做法）可以救回大部分 | Naive Fix 在 k=20/50/100 時，分別拿到 full KD 收益的 73%/83%/97%；CE 2.81 → full KD 2.74 nats | 高（同一種 target） | p4, p6 |
| NeurIPS 2023 | KD Performs Partial Variance Reduction | 線性與深度線性模型的理論，加上小型實驗 | KD 等於部分的 variance reduction，最佳權重 λ* = E⟨∇f_ξ(x*),∇f_ξ(θ)⟩/(E‖∇f_ξ(θ)‖² + ‖∇f(θ)‖²/γ) | γ（步長）越小，老師的偏差項越重要，λ* 越小 → 支持 α 跟 lr 連動 | 中（理論） | p7 |
| NeurIPS 2024 | Compact LMs via Pruning and KD（Minitron） | 剪枝後再訓練，約 94B tokens | 只用 logit KD（forward KL）比 CE+KD 好；τ=1 最好；top-K ≤100 會掉 | LM loss：只用 KD 2.107，CE+KD 2.117 | 中低（學生從老師剪出來，起點很接近老師） | p9, p17–18 |
| ACL 2023 Findings | KD from Weak Teacher for Scaling Up PLMs | BERT 類 MLM 預訓練，弱老師 | soft loss 權重大，前期收斂快；權重小，後期比較好；建議訓練中把權重從 4 降到 1 | 定性 | 中低 | p4 |
| arXiv 2024（非六大會議） | SALT: A Little Help Goes a Long Way（Rawat et al.） | 1.5B 小老師 → 2.8B 學生，Pile | 只在前 36K/208K 步做 KD（ω=0.667），之後純 LM；一步切換優於線性降；整段都做 KD 比 baseline 差 | 用 70% 的步數達到 baseline | 低（弱老師） | p8–9, p36 |
| ICLR 2025 | TAID | 持續預訓練與 instruction tuning；中間分布從學生插值到老師，t 隨時間上升 | 為了處理 capacity gap，老師的占比隨時間「上升」 | 定性加 benchmark | 低（反向的證據，見第 5 節） | p4, p7 |
| ICML 2026 | Bridge-Garden Dilemma | 在老師生成的序列上 fine-tune；hard 指老師 sample，不是資料 | 混合 hard 與 soft 比純 soft 好，原因是 exposure bias；依熵或信心給每個 token 不同權重有幫助 | 生成類 benchmark，平均 +1–2 分 | 低（機制來自生成；teacher-forced 的 ppl 沒有 exposure bias） | p3, p6 |
| ACL 2024 | Revisiting KD for Autoregressive LMs（ATKD） | instruction-tuning KD | 老師不確定的那一半 token 帶有大部分 dark knowledge，只訓練這一半比全部好 | 最多 +3.04 平均分 | 低 | p4 |
| NeurIPS 2024 | Rho-1 | 用參考模型的 excess loss 選 token，持續預訓練 | 只訓練 60–70% 的 token，下游更好；作者也說「不需要降低所有 token 的 loss」 | 數學 +30%（few-shot）；一般 +6.8% | 低（OJ 評所有 token） | p6, p8 |
| ICLR 2025 | Dynamic Loss-Based Sample Reweighting | 120M–7B 預訓練，以樣本為單位，依 loss 重新加權 | 降低簡單或重複樣本的權重最好 | 小幅 | 低 | — |
| ACL 2025 | Towards the Law of Capacity Gap | 各種規模 ≤7B 的蒸餾 | 最佳老師大小跟學生大小成線性關係，約 2.5 倍 | 定性 | 中（表示 8B 對 1.24B 偏大） | p6 |
| ICLR 2025 | High-dim Analysis of KD（W2S） | ridgeless regression 理論 | KD 能降低 test risk，但不能改變 data scaling 的指數 | 理論 | 中（支持「收益 ∝ 可降低的 loss」） | p3 |
| ICLR 2025 | MiniPLM | 老師離線做 difference sampling，挑選資料 | 線上的 vanilla KD 若跟 compute 對齊，收益會消失 | 定性 | 低（不是我們的情境） | p2 |
| ACL 2026 Findings | Calibrated Progressive Distillation | instruction KD；在老師與學生分布之間混合 | 混合係數從老師主導，逐步移向學生 | 小幅 | 低（混的是學生分布，不是資料） | p5 |
| 技術報告（非六大會議） | Gemma 2 | 2B/9B 用 2T/8T tokens 純 KD，取代 NTP | 消融：1B 的 ppl 17 → 15（7B 老師）；2B 在 500B tokens 時 benchmark 60.3 → 67.7；「收益隨模型變大仍在」 | 1B 約 −12% ppl | 中 | Table 6–7 |
| 技術報告（非六大會議） | Gemma 3 | 所有模型都用 KD；每 token 依老師機率 sample 256 個 logits 再重新正規化 | 訓練短時小老師較好，訓練長時大老師較好 | 圖上沒有數字 | 中 | §2.2, §5.4 |
| 技術報告（非六大會議） | Llama 3.2 1B/3B（Meta blog） | 從 8B 剪枝，用 8B 和 70B 的 logits 當 token-level target | α、τ、top-k 都沒公開 | — | 中 | blog |
| 技術報告（非六大會議） | Llama 4（Meta blog） | Maverick 從 Behemoth 做 codistillation | 「novel distillation loss function that dynamically weights the soft and hard targets through training」 | 沒有細節 | 中（表示大廠確實有做 α 排程） | blog |
| 技術報告（非六大會議） | Apple AFM-on-device 2024 | 6.3T tokens | target = 0.9 × 老師 top-1 + 0.1 × 真實標籤 | MMLU 約 +5%、GSM8K 約 +3% | 中 | §3 |
| 技術報告（非六大會議） | LFM2 | LFM1-7B 老師，top-32 logits（跟我們一樣），10–12T tokens | decoupled tempered top-K（top-K 總質量用 Bernoulli 項，內部用 conditional KL），再跟 NTP CE 加權平衡 | 權重沒公開 | 中（T1a 相關） | p9–10 |
| 技術報告（非六大會議） | MobileLLM-Pro | 1B，老師 Llama 4 Scout，所有預訓練階段都用 forward KL 取代 CE | 跟 FLOP 對齊的 CE 比，平均 +4.4 分 | — | 中 | p3, p14 |

## 3. 我們自己的數據：最佳 α 隨訓練下降

把 CD、CE、CH、CI 在各時間點相對 BU 的 log-ppl，對 α 做二次擬合。下表是「如果訓練停在 t，最好的固定 α」。這是累積的效果，不是當下的最佳 α。程式在 scratchpad 的 `T1b/tools/alpha_fit.py`。

| 時點（lr 階段） | α 0.5 | α 0.7 | α 0.9 | α 1.0 | 擬合最佳固定 α | 曲率（nats/α²） |
|---|---|---|---|---|---|---|
| 5%（stable） | −0.123 | −0.143 | −0.154 | −0.159 | ≥1（外推 1.23） | 0.07 |
| 20%（stable 結束、衰減開始） | −0.103 | −0.130 | −0.146 | −0.144 | 約 0.93–0.97 | 0.19 |
| 50%（lr 0.63） | −0.094 | −0.118 | −0.129 | −0.129 | 0.96 | 0.16 |
| 70%（lr 0.38） | −0.088 | −0.108 | −0.116 | −0.114 | 0.92 | 0.16 |
| 90%（lr 0.13） | −0.077 | −0.097 | −0.102 | −0.099 | 0.88 | 0.17 |
| 結束 | −0.072 | −0.091 | −0.095 | −0.092 | 0.86 | 0.18 |

觀察：

1. **CI 輸給 CH 的時點：** CI（α 1.0）從 10% 到 20% 之間開始輸 CH，剛好在 stable 段結束、衰減開始的地方。之後的差距大致隨 lr 下降而擴大：
   - 20%：+0.20
   - 50%：0.00
   - 70%：+0.14
   - 90%：+0.21
   - 結束：+0.26
   - 中間 20→50% 有一段回頭，所以不是乾淨的單調關係
2. **跟 variance-reduction 理論一致**（NeurIPS 2023，p7）：
   - KD 用老師的偏差換掉一部分梯度雜訊
   - 雜訊的影響跟步長成正比
   - lr 越小，老師偏差的相對代價越大，最佳 α 越小
3. **跟大廠的做法一致：** Peng et al. 的最佳設定 WSD-α + WSD-LR，本質上就是「α 的形狀等於 lr 的形狀」。
4. **也可能有另一種解釋：top-32 截斷的偏差：**
   - Busbridge 的資料：截斷越嚴重，越需要 λ<1
   - 學生越強，tail 被歸零造成的偏差在 loss 中占比越高
   - 這可能是 T1a 的問題（tail 怎麼分），不是 α 排程的問題。兩者要一起考慮，見 C1 的交互作用
5. **當下最佳 α 的推估：**
   - 只靠累積擬合推不出當下的最佳 α，因為較晚的效果權重較大
   - 如果結束時的 0.86 是前半（約 0.96）和後半的加權平均，後半權重 50–70%，那後半的當下最佳 α 約 0.75–0.82
   - 這只是推估，要用實驗確認
6. **代價很小：** 曲率約 0.18 nats/α²。α 偏離最佳值 0.1，ppl 只差約 0.18%；偏 0.06 只差約 0.06%。

## 4. 候選改動

### C1（第 1 名）α 跟 lr 連動：stable 段 1.0，衰減段線性降到 0.7

- **機制：**
  - α_t = α_end + (1.0 − α_end) × lr_t / lr_peak
  - 暖身期間也用 1.0，不跟 lr 從 0 升起：CI 在前 10% 最好，Peng 的 α 暖身沒有單獨驗證
  - 1B 測試用 α_end = 0.7。整段的平均 α ≈ 0.2 × 1 + 0.8 × 0.85 = 0.88，跟目前 1B 的最佳固定值 0.86 接近
  - 所以這個改動是在重新分配 α，不是改變 α 的平均
- **證據：**
  - 第 3 節我們自己的數據
  - Peng et al.（ACL 2025，p6）：
    - WSD-α+WSD-LR 比 1-α+WSD-LR 多 +3.0 分
    - Linear Dec 比 Linear Inc 多 +3.0 分
    - 單 seed，指標是 benchmark
  - 理論（NeurIPS 2023，p7）：λ* 隨步長變小而下降
  - DWT（ACL 2023 Findings，p4）：soft 權重前大後小比較好
  - Llama 4 有動態加權 soft / hard target（細節沒公開）
- **預期效果：**
  - 1B：−0.1% 到 −0.3%，中位約 −0.15%
    - 前期 α=1 的領先（CI 在 5–10% 有 −0.3 到 −0.5%）到最後大概只剩 1/4
    - 衰減段改用較低 α 可能再多 0.1–0.2%
  - 6B：差不多，或略大
    - 這個排程跟 lr 綁在一起，跟長度無關，可以直接沿用
    - 6B 衰減段的學生更接近老師，偏差的代價更高
  - 信心：中低。有 >0 收益的機率約 65%，超過 0.3% 的機率約 15%
- **跟 EMA-Nesterov 和 NorMuon 的交互：**
  - EMA 的視窗是 15–74%。在這段 α 從 1.0 降到約 0.80（74% 時 lr 比例 0.325），平均約 0.92，跟 CJ 的 0.9 幾乎一樣
  - 所以 EMA 開啟期間的梯度一致性大致不變，交互應該很小
  - EMA 關掉之後的衰減段，α 才降到 0.8 以下
  - NorMuon：更新的 RMS 固定在 0.2，lr 不用跟著 α 改；每列二階 EMA（β2 0.95，約 20 步）跟得上 α 的慢速變化
  - 跟 T1a（tail 怎麼分）有交互：如果 T1a 改成把 tail 依學生分布分配，截斷偏差變小，最佳 α 和它的下降幅度可能都會變小。建議先定 T1a，再測 C1；或者至少在 C1 的結果裡註明用的是哪一種 tail 處理
- **有沒有試過：** 沒有。只試過固定 α：CD、CE、CH、CI、CL。
- **實作草圖：**
  - `src/training.py` 第 63 行之後：
    - 加 `KD_ALPHA_END = float(os.environ.get("KD_ALPHA_END", KD_ALPHA))`
    - 預設值等於 KD_ALPHA，也就是固定 α，跟現在一樣
  - `src/training.py` 第 171–174 行的 `LabTrainer.training_step`：在呼叫 super 之前加一行
    ```python
    self.model.kd_alpha = KD_ALPHA_END + (KD_ALPHA - KD_ALPHA_END) * min(1.0, (1 - self.state.global_step / max_steps) / DECAY_FRACTION)
    ```
    - 這正是線性衰減段的 lr_t / lr_peak，暖身和 stable 段等於 1
  - `src/training.py` 第 167–168 行：在 `use_mixed_loss` 之前，先設 `self.model.kd_alpha = KD_ALPHA`
  - `src/distill.py` 第 94–119 行：`forward` 閉包裡把傳給 `MixedCrossEntropy.apply` 的 `alpha`（第 113 行）改成 `model.kd_alpha`
  - `log`（約第 190 行）：記錄 `kd_alpha`
  - 總共約 6 行
- **成本與風險：**
  - 不增加計算時間
  - 風險一：α 設在被 accelerate 包起來的模型上，而不是 `self.model`，這樣 α 會不生效
    - 檢查方法：KD_ALPHA_END = KD_ALPHA 的 CPU 小測要跟 CH 完全一樣
    - 再印前幾步和衰減段的 α
  - 風險二：α_end 太低。CD（0.5）整段都明顯差，所以 α_end 不要低於 0.5
- **1B 測試設計：**
  - 跟 CJ 比（KD 0.9 加 EMA，15.324），只改 α 排程（1.0 → 0.7，跟 lr 連動），seed 相同
  - 預期 15.28–15.31
  - 成功：結束時 ≤ −0.10%（≤ 15.309），而且 5–10% 時領先 CJ 約 0.3%，這是 CI 型的前期優勢
  - 失敗：> −0.05%，或衰減段後半落後 CJ。這時 6B 用固定 α
  - 如果成功，再跑 α_end 0.5 夾住最佳值
  - 一次約 73 分、10 H200-h

### C2（第 2 名）分段：前 20% 用 α=1.0，之後固定 0.85

- **機制：** 只拿 C1 的前半，也就是 stable 段的純老師目標。衰減段用接近最佳的固定值。比 C1 更簡單，也能把「前期 α=1」的效果單獨分出來。
- **證據：**
  - CI 前 10% 贏 CH：5% 時 −0.56%、10% 時 −0.33%
  - Peng：Linear Dec > Linear Inc
  - SALT：KD 在前期最有用（弱老師的情境）
- **預期效果：**
  - 1B：−0.05% 到 −0.12%；前期領先保留得少，這是主要的不確定
  - 6B：一樣或更小
  - 信心：中低
- **交互：** EMA 從 15% 開始，跟 α 切換點 20% 很近，兩個跳變疊在一起，可能讓 CJ 那種開啟時的落後更深。若要測，切換點可以改到 15%，或用 3% 的線性過渡。
- **有沒有試過：** 沒有。
- **實作：** 同 C1 的位置，把公式換成 `KD_ALPHA if step/max_steps < 0.2 else KD_ALPHA_END`。
- **成本與風險：** 同 C1。
- **1B 測試設計：**
  - 跟 CJ 比，成功 ≤ −0.08%，失敗 > −0.03%
  - 如果 C1 會測，C2 就不必測（C1 涵蓋它）

### C3（第 3 名）6B 固定 α 的選擇：0.85，不用 0.8（不需要 GPU）

- **機制：** 這是在「不做排程」時要選哪個固定值。
- **證據：**
  - 支持降低 α：
    - 我們在一次 run 內看到最佳 α 隨學生進步而下降
    - Strong-Teacher：學生相對老師越強，最佳 α 越低
  - 支持不降低：
    - Busbridge：λ>0.5 很平，M=20 時 λ=1 跟 λ* 差不多
    - Gemma 3：訓練越長，越偏好大老師
    - Peng：1.9B 在 100B tokens（M 約 50）時最佳仍是 0.9
    - AFM：6.3T tokens 時用 0.9
  - 淨效果：6B 的最佳 α 在 0.8–0.88
- **預期效果：** 0.85 和 0.8 的差距 ≤0.05%。信心中（方向不確定，但幅度確定很小）。
- **交互：** 無。
- **有沒有試過：** 3B 的 CL 用 0.8，沒有做 α 掃描。
- **實作：** 改 sbatch 的 KD_ALPHA。
- **成本與風險：** 零。
- **測試：** 不需要。如果 C1 在 1B 贏，就直接用 C1 的排程。

### C4（第 4 名）自適應 α：依「學生與老師的差距」調整整步的 α

- **機制：**
  - 用最近 100 步的學生 CE 對老師 CE 的差距（cache 裡有老師的 logsumexp，可以算出老師對 label 的 CE），把 α 從 1.0 映射到 0.7
  - 差距大時多信老師，差距小時多信資料
  - 動機：
    - Strong-Teacher：最佳 α 隨老師的相對強度變
    - TAID 和 CPD 的自適應排程
    - 第 3 節觀察到的現象
- **證據：** 只有間接證據。沒有任何一篇在「從頭預訓練、評 ppl」下測過這種形式。
- **預期效果：** 跟 C1 差不多，或差 0.05%。好處是 6B 不用猜 α_end。信心：低。
- **交互：** 跟 lr 衰減有相關，跟 C1 幾乎是同一件事的另一種參數化。
- **有沒有試過：** 沒有。
- **實作：**
  - `src/distill.py` 的 forward 回傳 KD loss 和 CE 兩部分
  - `training.py` 維護 EMA 並設定 α
  - 約 15 行
- **成本與風險：** 要多算一次 label 的 log-prob（幾乎不花時間）。映射常數要定，等於多了兩個超參數。
- **1B 測試設計：** 只有在 C1 贏的情況下才值得測，看能不能更好；失敗條件同 C1。

### C5（第 5 名）依老師熵給每個 token 的 α（平均不變）

- **機制：**
  - 換個角度看 ppl：最佳 target 是真實的條件分布 p*
  - 老師是有偏差但低變異的估計；one-hot 是無偏但高變異的估計，變異數 = 1 − Σp*²
  - shrinkage 的觀點：老師熵高的位置，one-hot 變異大，應該更信老師（α 高）
  - 老師很有把握的位置，老師分布接近 one-hot，α 影響很小
  - 形式：α_i = clip(ᾱ · (H_i/E[H])^p, 0, 1)，H 用 top-32 重新正規化後的熵
- **證據：**
  - Bridge-Garden 的 entropy-based weighting（ICML 2026，p6）
  - ATKD 的「老師不確定的 token 帶 dark knowledge」（ACL 2024，p4）
  - CaRE-KD、Entropy-aware OPD
  - 這些全部是 fine-tune 加生成指標，機制也不同（exposure bias）
  - 預訓練、評 ppl 的證據：0 篇
- **預期效果：** 1B 0 ± 0.1%，6B 同。信心：低。
- **交互：** 會改變每個位置的 target。跟 T1a 的 tail 處理高度相關，正確答案不在 top-32 的 7% 位置大多是高熵位置。
- **有沒有試過：** 沒有。
- **實作：** `src/distill.py` 第 72 行：
  ```python
  teacher = (alpha * w)[:, None] * kd_probs[s:s+CHUNK]
  ```
  w 由 kd_probs 算出熵後正規化；約 5 行。
- **成本與風險：**
  - 計算量可忽略
  - 風險：p 要調；平均 α 會被 clip 改變
- **1B 測試設計：** 不建議在 6B 前測。如果要測：跟 CJ 比，成功 ≤ −0.1%。

### 不建議做的（文獻或我們的數據顯示有害）

- **丟掉 token（Rho-1 或 ATKD 只訓練一部分 token）：**
  - OJ 評的是全部 token
  - Rho-1 的作者自己說，這個方法的目的是下游表現，不是降低所有 token 的 loss（p8）
- **最後階段把 α 降到 0**（Peng 的 WSD-α 最後 1% cosine 降到 0）：
  - 他們的衰減段只有 1%；我們是 80%，照抄等於平均 α 降到約 0.6
  - 由 CD（0.5）−6.94% 對 CH（0.9）−9.04% 來看，會明顯變差
  - 如果只在最後 1–2% 做，影響也小到量不出來
- **α 從 0 暖身**（Peng 和 Bridge-Garden 的 curriculum）：跟我們 CI 在 5% 最好的數據相反。

## 5. 6B 推估的交叉檢驗

### 5.1 KD 收益隨 tokens 的衰減

| 來源 | 設定 | 每多一倍學生 tokens 的收益保留率（nats） |
|---|---|---|
| 我們：CJ/BU −10.0% → CL/BY −8.0% | 1.24B，8B@15T 老師，1B→3B | ×0.86（KD+EMA）；只算 KD 的估計約 ×0.85 |
| Distillation Scaling Laws（Table 6 的係數，代入 N=1.24B、老師 L(8B, 15T)） | 公式外推 | 1→3B：×0.73/倍；3→6B：×0.70 |
| Strong Teacher Not Needed | 1.7B，50B → 300B tokens（α 沒重調，弱老師） | in-domain 約 4% → 0.9%，約 ×0.55/倍 |
| Peng et al. | 1.9B/3.8B，到 500B tokens | 「先增後收斂，稍微下降」，約 ×0.95–1.0 |
| Gemma 2 | 1B 學生，7B 老師（tokens 未標，消融一節用 500B） | 仍有 −12% ppl，表示高 M 下收益仍大 |

- **我們的 ×0.86 在範圍中間。** Busbridge 公式在 M<1 的區間是外推（他們的資料大多 M≥?；1B tokens 對 1.24B 是 M≈0.8），而且他們的 supervised 基準沒有針對短訓練調參。所以絕對值不可信（公式給 1B 的收益是 −0.59 nats，我們是 −0.10），只取它的形狀：衰減比我們快。
- **為什麼收益會這樣衰減：** High-dim KD（ICLR 2025）的理論說，KD 不改變 data scaling 的指數，只乘上一個常數倍。所以收益大致 ∝ 可降低的 loss，隨 D 一起變小。跟我們觀察到的方向一致。
- **6B 的 KD+EMA 收益：**
  - 用我們的 ×0.86：−6.9%
  - 用 Busbridge 的 ×0.70：−5.7%
  - 用 Peng/Gemma 的平坦假設：−8.0%

### 5.2 沒有 KD 的基準線怎麼外推（主要的風險）

- **現在的做法：** 6B 的 11.96 和 11.19 都假設「每多一倍 tokens 下降的 nats 數固定」（ppl ∝ D^−0.2 或 D^−0.176），也就是 log-linear。
- **反證：**
  - 我們自己的 AO → AP → AL（0.25 → 0.5 → 1B），每倍下降 0.357 → 0.239 nats，下降量明顯遞減（雖然 0.25B 那點太早期）
  - Chinchilla（Hoffmann 2022）和 Busbridge 的 supervised 公式都是 E + B/D^β 的形式。代入 N=1.24B，(3→6B 的下降)/(1→3B 的下降) = 0.50（Busbridge），而 log-linear 是 0.63
- **飽和形式的推估：** 改用 L = E′ + B′·D^−β，用 1B 和 3B 兩點定 B′，β 取 0.2–0.43：

| 情境 | 6B 沒有 KD | 6B KD+EMA（in-training） | 對應 OJ（×1.012） | 對 CM 4.5B 的預測 |
|---|---|---|---|---|
| log-linear 加我們的收益衰減（筆記的做法） | 11.96 | 11.13（KD 曲線直接外推 11.18） | 11.26 | 11.77 |
| log-linear 加 Busbridge 的收益衰減 | 11.96 | 11.28 | 11.41 | — |
| 飽和 β=0.2（KD 曲線直接外推） | 12.26 | 11.41 | 11.55 | 11.89 |
| 飽和 β=0.28（Chinchilla 的 β） | 12.36 | 11.50 | 11.64 | 11.93 |
| 飽和 β=0.28 加 Busbridge 的收益衰減 | 12.36 | 11.66 | 11.80 | — |

- **結論：**
  - 11.1–11.2 是樂觀的一端，前提是基準線完全不飽和
  - 比較合理的中心值是 in-training 約 11.3–11.5、OJ 約 11.45–11.65
  - 範圍 11.1–11.7
- **CM 是決定性的檢查：**
  - CM 在 11.80 以下：保留 11.1–11.2
  - CM 在 11.88 以上：6B 改估 11.4–11.5
  - 拿到 CM 後，可以用 CJ、CL、CM 三點擬合 E + B·D^−β
- **注意：**
  - 這一節的主要不確定性來自 T3a（基準線的 scaling 和 lr 遷移），不是 KD
  - KD 收益衰減造成的差異只有 ±0.15 ppl；基準線形式造成的差異約 0.3 ppl

### 5.3 最佳 α 是否該隨 tokens 降低

- **方向是「略降或不變」。** 6B 用 0.8–0.86 都行，選錯的代價 ≤0.06%（第 3 節的曲率）。
- **比固定值更重要的：** 用跟 lr 連動的排程（C1）。它跟長度無關，可以直接遷移，不需要猜 6B 的固定值。

## 6. 反證據

1. **純 KD（α=1）長期有效：**
   - 例子：Gemma 2/3、Minitron、MobileLLM-Pro 都用純 KD 訓練很長
   - 還有 Busbridge 的 λ=1 ≈ λ*
   - 這表示在完整詞表或無偏抽樣的 logits 下，α<1 和 α 排程可能都不需要
   - 我們需要 α<1，很可能是 top-32 截斷造成的（Busbridge 的 top-128 也要靠 λ=0.7 才救得回來，p63）
   - 推論：α 排程的部分收益，可能被 T1a 的更好 tail 處理取代
2. **α 隨時間上升的方法也有效：**
   - TAID、CPD：老師占比隨時間上升
   - Bridge-Garden：soft 權重從 0 暖身
   - Peng 的 WSD-α：前 10% 從 0 升到 1
   - 但這些是 fine-tune、生成指標，或者沒有單獨驗證；TAID/CPD 混的是學生分布，不是資料
   - 我們的 CI 數據直接反駁「早期 α 要低」
3. **Gemma 3：訓練越長，越偏好大老師。** 表示長訓練時「更信老師」才對，跟「6B 要降低 α」相反。
4. **KD 收益幾乎不衰減的報告：** Gemma 2 和 Peng（到 500B tokens）的收益幾乎不衰減，比我們的 ×0.86 還慢。如果是這樣，6B 的 KD 收益會更大（−8%）。
5. **KD 收益衰減很快的報告：**
   - Busbridge：supervised 遲早會反超
   - Strong-Teacher：300B tokens 時收益剩很少
   - 但我們的老師很強（8B@15T），Busbridge 公式給出的反超點在 1.24B 學生 ≫100B tokens，所以 6B 不會反超
6. **一步切換優於漸進：** SALT 發現一步切換（KD → 純 LM）比線性降好。不過那是弱老師，而且是把 KD 完全關掉，跟我們的「強老師、α 從 1.0 到 0.7」不同。
7. **token 加權沒有任何預訓練 ppl 的正面證據。** Rho-1 類方法明說不以全部 token 的 loss 為目標。

## 7. 結論

- **最值得做：** C1，α 跟 lr 連動，1.0 → 0.7。
  - 約 6 行程式，不增加計算時間
  - 有理論支持，也有大廠的類似做法
  - 1B 預期 −0.1% 到 −0.3%
  - 用 1 個 1B run（約 10 H200-h）就能判斷，可以在 6B 前完成
  - 建議先決定 T1a 的 tail 處理，再測 C1
- **如果不測：** 6B 用固定 0.8–0.85，差異 ≤0.05%。
- **token 加權（C5）：** 證據不足，6B 前不做。
- **6B 推估：**
  - 11.1–11.2 是樂觀的一端，比較穩的估計是 11.3–11.5（OJ 11.45–11.65）
  - 主要風險是沒有 KD 的基準線會飽和，不是 KD 收益
  - 用 CM 的結果（11.77 或 11.9）就能分辨
- **α 隨 tokens：** 只會略降或不變。排程比固定值的選擇重要。

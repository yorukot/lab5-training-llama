# T5 優化器：新的 Muon 變體、動量方法、embedding 與 norm 的優化器

研究員 7 號，2026-10-17。
- 範圍：ICLR 2025/2026、ICML 2025/2026、NeurIPS 2025/2026 為主，ACL 系列補充
- 標「非六大」的：arXiv 論文、技術報告、modded-nanogpt Track 3 排行榜（本地副本 `src/exp/vendor/nanogpt_optimization/UPSTREAM_README.md`）
- 頁碼一律指 PDF 頁（`=== page N ===`），檔案在 `/home/yorukot/ml/papers/pdf/T5/`
- 名詞：TPP = tokens/參數；「我們」= 00-context 的配方（NorMuon + Polar Express、τ 0.39、131k 批次、衰減到 0、EMA-Nesterov、KD）

---

## 1. 搜尋紀錄

### 1.1 查詢（56 組，命中數是標題加摘要的 regex 命中）

| id | regex | 年份 | 命中 |
|---|---|---|---|
| q01 | `\bMuon\b` | 2025,2026 | 81 |
| q02 | `orthogonali[sz]ed (momentum\|update\|gradient)\|orthogonal(ization\|ized)? optimi[sz]er` | 2024–2026 | 6 |
| q03 | `Newton[- ]Schulz\|polar decomposition.*(optimi\|train)\|polar factor` | 2024–2026 | 15 |
| q04 | `spectral (norm )?(descent\|steepest\|condition)\|steepest descent.*(norm\|spectral)\|modular (norm\|duality)` | 2024–2026 | 19 |
| q05 | `\bShampoo\b\|\bSOAP\b\|Kronecker[- ]factored precondition\|\bPSGD\b\|\bKron\b` | 2024–2026 | 38 |
| q06 | `AdEMAMix\|two (EMA\|momentum)s?\|mixture of (two )?EMA\|slow (EMA\|momentum)` | 2024–2026 | 5 |
| q07 | `\bMARS\b\|variance[- ]reduc.*(LLM\|language model\|large\|pretrain)\|STORM.*(adam\|LLM)` | 2024–2026 | 75 |
| q08 | `cautious (optimi\|update\|mask)\|C-?AdamW\|sign[- ]consistent update mask\|update mask` | 2024–2026 | 3 |
| q09 | `\bDion\b\|distributed orthonormal\|low[- ]rank orthonormal` | 2025,2026 | 0 |
| q10 | `\bScion\b\|linear minimi[sz]ation oracle\|\bLMO\b\|unconstrained (LMO\|conditional gradient)\|Gluon` | 2025,2026 | 22 |
| q11 | `PolarGrad\|AdaMuon\|NorMuon\|MuonClip\|Moonlight\|\bMuon\+\|Muon (variant\|with)` | 2025,2026 | 9 |
| q12 | `(benchmark\|fair comparison\|systematic (study\|comparison)\|revisit).{0,60}optimi[sz]ers?.{0,80}(pretrain\|language model\|LLM)` | 2024–2026 | 2 |
| q13 | `optimi[sz]ers?.{0,80}(speedup\|speed-up).{0,80}(vanish\|disappear\|shrink\|diminish\|overstat)` | 2024–2026 | 0 |
| q14 | `(embedding\|output layer\|lm head\|unembedding).{0,80}(optimi[sz]er\|learning rate\|Adam)` | 2024–2026 | 4 |
| q15 | `(layer[- ]wise\|per[- ]layer\|per[- ]parameter[- ]group\|layer[- ]type).{0,60}(learning rate\|step size\|optimi[sz]er).{0,80}(transformer\|LLM\|language)` | 2024–2026 | 0 |
| q16 | `(Lion\|Sophia\|Adafactor\|Adam-mini\|SignSGD\|Signum).{0,80}(pretrain\|language model\|LLM)` | 2024–2026 | 117 |
| q17 | `matrix[- ](whitening\|preconditioner\|optimi[sz]er\|aware)\|non[- ]diagonal precondition\|full[- ]matrix (adagrad\|precondition)` | 2024–2026 | 7 |
| q18 | `(optimi[sz]er\|optimization).{0,80}(data[- ](constrained\|limited\|scarce)\|token budget\|tokens per parameter\|over[- ]?train\|chinchilla)` | 2024–2026 | 4 |
| q19 | `Nesterov.{0,80}(LLM\|language model\|pretrain\|transformer)\|lookahead optimi[sz]er\|schedule[- ]free` | 2024–2026 | 9 |
| q20 | `(momentum\|beta1\|β1).{0,60}(schedule\|warmup\|decay\|tuning).{0,80}(LLM\|language model\|pretrain)` | 2024–2026 | 3 |
| q21 | `(normali[sz]ed\|sign\|spectral).{0,40}(gradient descent\|update).{0,80}(language model\|LLM\|transformer pretrain)` | 2025,2026 | 1 |
| q22 | `(Adam\|AdamW).{0,60}(beta2\|β2\|epsilon\|second moment).{0,80}(language model\|LLM\|batch size\|pretrain)` | 2024–2026 | 0 |
| q23 | `weight decay.{0,80}(Muon\|orthogonal\|spectral\|norm constraint\|hypersphere)\|norm[- ]constrained (training\|optimi)` | 2025,2026 | 3 |
| q24 | `(hyperparameter\|learning rate).{0,60}transfer.{0,80}(Muon\|optimi[sz]er\|token\|data)` | 2025,2026 | 4 |
| q25 | `low[- ]rank (optimi[sz]er\|projection\|gradient).{0,80}(pretrain\|LLM)\|GaLore\|subspace optimi` | 2024–2026 | 15 |
| q26 | `(second[- ]order\|curvature\|Hessian).{0,60}optimi[sz]er.{0,80}(LLM\|language model\|pretrain)` | 2025,2026 | 2 |
| q27 | `(efficient\|faster\|accelerat).{0,40}(LLM\|language model) (pre-?training).{0,80}optimi[sz]er` | 2025,2026 | 1 |
| q28 | `(Muon\|orthogonal).{0,80}(scal\|large\|billion\|1B\|3B\|7B)` | 2025,2026 | 77 |
| q29 | `sharpness.{0,60}(LLM pretrain\|language model pretrain)\|edge of stability.{0,60}(Adam\|transformer\|LLM)` | 2025,2026 | 1 |
| q30 | `(implicit bias\|why does\|understanding\|theory of\|convergence of).{0,40}(Muon\|orthogonal\|spectral descent\|Shampoo\|SOAP)` | 2025,2026 | 12 |
| q31 | `\boptimi[sz]ers?\b.{0,120}(pre-?train\|language model\|LLM)\|(pre-?train\|language model\|LLM).{0,120}\boptimi[sz]ers?\b` | 2025,2026 | 122 |
| q32 | `\bbenchmark.{0,100}\boptimi[sz]ers?\b\|\boptimi[sz]ers?\b.{0,100}\bbenchmark` | 2024–2026 | 21 |
| q33 | `\b(Adam\|AdamW)\b.{0,100}(pre-?train\|language model\|LLM)` | 2025,2026 | 40 |
| q34 | `\bmomentum\b.{0,120}(language model\|LLM\|transformer\|pre-?train)` | 2025,2026 | 25 |
| q35 | `\bweight decay\b` | 2025,2026 | 57 |
| q36 | `\bprecondition(er\|ing\|ed)?\b.{0,120}(language model\|LLM\|transformer\|pre-?train\|deep learning)` | 2025,2026 | 9 |
| q37 | `sign(ed)?[- ](descent\|SGD\|momentum)\|\bLion\b` | 2025,2026 | 25 |
| q38 | `(non-?Euclidean\|ℓ∞\|l_?\\?infty\|max-norm\|operator norm).{0,60}(descent\|geometry\|optimi)` | 2025,2026 | 20 |
| q39 | `Adam-?mini\|block-?wise (learning rate\|adaptiv)\|(fewer\|reduc).{0,30}(learning rates\|optimizer states?)` | 2024–2026 | 10 |
| q40 | `maximal update\|\bmuP\b\|μP\|spectral condition\|feature learning.{0,40}(scal\|width)` | 2025,2026 | 32 |
| q41 | `\bAdEMAMix\b\|\bAdemamix\b\|\bSOAP\b\|\bShampoo\b\|\bPSGD\b` | 2025,2026 | 35 |
| q42 | `\bscaling law.{0,100}\boptimi[sz]er\|\boptimi[sz]er.{0,100}scaling law` | 2025,2026 | 6 |
| q43 | `\bembedding.{0,120}(learning rate\|update\|gradient\|optimi)` | 2025,2026 | 182 |
| q44 | `\b(tied\|untied\|weight tying\|tie[ds]? (the )?(input\|output) embedding)` | 2025,2026 | 167 |
| q45 | `(learning rate\|optimi[sz]er).{0,80}(LayerNorm\|RMSNorm\|normali[sz]ation layer\|gain parameter)` | 2025,2026 | 1 |
| q46 | `AdaMuon\|PolarGrad\|\bDion\b\|\bScion\b\|MuonClip\|\bKron\b\|\bSPlus\b\|\bConda\b\|AdaGO\|Drop-?Muon\|Turbo-?Muon\|\bMano\b\|SinkGD\|Sinkhorn.{0,30}(gradient\|optimi)\|\bSNOO\b\|\bREG\b.{0,20}optimi` | 2025,2026 | 22 |
| q47 | `(practical\|real\|wall-?clock).{0,60}(efficien\|speedup).{0,60}(Muon\|optimi[sz]er)\|Muon is scalable\|Muon.{0,40}(compute\|token\|data) efficien` | 2025,2026 | 1 |
| q48 | `\b(Adam\|AdamW\|Muon)\b.{0,100}(learning rate\|hyperparameter).{0,60}(scal\|transfer\|token)` | 2025,2026 | 2 |
| q49 | `(simplif\|ablat\|deconstruct\|what (really )?matters).{0,80}(Adam\|optimi[sz]er\|Muon)` | 2025,2026 | 5 |
| q50 | `(noise\|stochastic).{0,60}(Muon\|orthogonal\|spectral descent\|sign descent)` | 2025,2026 | 14 |
| q51 | `(EMA\|exponential moving average).{0,60}(momentum\|weights?\|parameter).{0,80}(language model\|LLM\|pretrain)` | 2025,2026 | 32 |
| q52 | `(Polyak\|Nesterov\|lookahead\|outer (optimi\|momentum)\|extrapolat).{0,80}(optimi\|momentum\|train)` | 2025,2026 | 95 |
| q53 | `(update\|step)[- ](scale\|size\|norm).{0,60}(layer\|module\|parameter group\|matrix)` | 2025,2026 | 3 |
| q54 | `(hypersphere\|normali[sz]ed transformer\|nGPT\|weight normali[sz]ation\|norm[- ]constrain)` | 2025,2026 | 86 |
| q55 | `(heavy[- ]tail\|outlier).{0,80}(optimi[sz]er\|gradient\|Adam\|Muon).{0,80}(LLM\|language\|pretrain)` | 2025,2026 | 3 |
| q56 | `\b(gradient clipping\|clip)\b.{0,80}(LLM\|language model\|pretrain\|optimi)` | 2025,2026 | 79 |

- 年份與會議細分：大的命中組依會議和年份拆開看過。例如 q01（Muon）：ICLR 2026 17、ICML 2026 32、NeurIPS 2026 26、NeurIPS 2025 3、ICML 2025 1、ACL 2 篇
- 覆蓋缺口：NeurIPS 2026 有 3,166 篇沒有摘要，只能比對標題
- Muon 系列多數不在六大會議：q09（Dion）命中 0，AdaMuon、PolarGrad、MuonClip 也不在。這些用 WebSearch 補（下面 1.3）

### 1.2 篩選與精讀
- 命中總數 1,629，去重後 1,253 篇
- 標題檢視：
  - 去掉四個大組後的聯集 483 篇，全部看過標題
  - 四個大組（q16、q35、q43、q44，共 523 筆）用關鍵字過濾後再看標題
- 摘要：108 篇
- 全文精讀 46 份 PDF：會議站 25 份、arXiv 20 份（上限 20，每份間隔 3 秒，沒遇到 429）、網頁 PDF 1 份（SOAP-Muon 技術報告）
  - 精讀方式：grep 對照組設定、模型大小、tokens、結果表
  - 其中 37 篇由 3 個唯讀 subagent 依固定格式抽取，我抽查了關鍵數字：Secret Sauce 的 β2 那句、HTMuon 表 5、VAM 表 5
- 參考文獻回查：
  - 從 Fantastic、Benchmarking、EMA-Nesterov、DynMuon、Fantastic II 的相關工作章節回查
  - 加上 Track 3 排行榜 46 筆紀錄的連結

### 1.3 WebSearch（14 次）與 WebFetch（1 次），都標「非六大」
查了這些：AdaMuon、PolarGrad、Dion、Practical Efficiency of Muon、MARS-M、NAMO、embedding 用 Muon 的消融、Adam's Secret Sauce 的 ID、低 TPP 的優化器比較、How to Scale Second-Order 的 ID、AdEMAMix + Muon、tied embedding 的優化器、Spectral Allocation（SAMuon）、Curvature-Conditioned Multiscale Momentum。WebFetch 用來抓 SOAP-Muon 的 PDF。

新找到、而且影響結論的有：
- Fantastic II / Hyperball（arXiv 2606.16899）
- Variance-Adaptive Muon（arXiv 2601.14603）
- DynMuon（arXiv 2605.17109）
- NVIDIA 的 SOAP, Muon, and Beyond（arXiv 2607.20548）
- SOAP-Muon 技術報告

---

## 2. 論文表

「調參」只看對照組（Muon/NorMuon/AdamW），問的是它有沒有被公平地調過。「相符度」看跟我們的規模、TPP、批次、衰減方式有多接近。

### 2.1 benchmark 與公平比較

| 論文 | 設定 | 對照組調參 | 效果 | 相符度 | 頁 |
|---|---|---|---|---|---|
| Fantastic Pretraining Optimizers（ICLR 2026） | 0.13–1.2B，1–8× Chinchilla（以非 embedding 參數算 20 TPP），批次 ≥0.5M，WSD 線性到 0，11 種優化器 | 是：每個優化器都對全部超參數做座標下降，再對會隨規模改變的超參數擬合 scaling law | 矩陣類在 ≤0.5B 加速 1.3–1.4×，到 1.2B 只剩約 1.1×；1× 時 Muon 最好，≥8× 時 SOAP、Kron 超過它；1.2B 時 NAdamW ≈ Muon | 高（規模、衰減到 0）；TPP 比我們高 3 倍 | 1–3、8–9、22、29、72 |
| Benchmarking Optimizers for LLM Pretraining（NeurIPS 2026） | 124M–720M，1–48B tokens，序列 512，批次 16k–1M，主要用 cosine | 部分：124M 細調後沿用；720M 的 D-Muon wd 固定 0.1、cosine | 720M、1M 批次：AdEMAMix、MARS 最好；124M 最短的 run：D-Muon 最好；Muon 沒加 wd 會很差 | 中（序列短，Muon 調得弱，這點 Fantastic 有批評） | 2、5–6、13、70–71 |
| Fantastic II / Hyperball（arXiv，非六大） | 1.2B Qwen3 型，1–8× Chinchilla；Track 3 | 1.2B 的 MuonW：wd 0.1 不隨規模調；Track 3 有調 | 1.2B：MuonH 比 AdamW 快 20–30%，MuonW 約 10%；Track 3：NorMuonH = NorMuonW = 3250 步 | 中 | 3–5 |
| How to Scale Second-Order Optimization（NeurIPS 2025） | 190M–1.4B，20 TPP，**批次 131k**，線性衰減 | 是（µP，分階段調參） | SOAP ≈ Muon ≈ Shampoo（190M：3.1751／3.1780／3.1780，差距在雜訊內），都比 AdamW 快約 1.4×；embedding 用 Muon 比較差 | 高（批次相同） | 10、33–37 |
| SOAP, Muon, and Beyond（NVIDIA，arXiv，非六大） | 8B dense、30B/72B MoE，批次 25M+ | 以 update RMS 對齊 lr | KL-SOAP 略勝 Muon；AdamW 在超大批次時變差 | 低（批次差 200 倍） | 2、17–18 |
| modded-nanogpt Track 3（非六大） | 124M，約 1.7B tokens（約 14 TPP），批次 524k，每項 4–30 個 seed | 是（#36 是全部調過的 Muon） | 見第 3 節 | 中高（TPP 接近；規模小 10 倍） | 本地 README 56–103 行 |
| Deconstructing a Good Optimizer（ICLR 2025） | 1.2B，26B tokens，131k 批次，cosine 到 10% | 是（lr 掃描） | Adam ≈ Adafactor ≈ Lion ≈ Signum；最後一層和 LayerNorm 一定要用自適應的方法 | 中 | 2、8–9 |

### 2.2 Muon 變體

| 論文 | 設定 | 對照組調參 | 效果 | 相符度 | 頁 |
|---|---|---|---|---|---|
| NorMuon（ICML 2026）（我們在用的） | 1.1B/50B、5.4B/100B，2M 批次，線性到 0 | 全部用同一個 lr，wd 0.1 | 比 AdamW 少 21.7%／13.2% 步數，Muon 是 10.4%／5.6% | 中 | 6 |
| Variance-Adaptive Muon（Muon-VS，arXiv，非六大） | 1.2B，1× Chinchilla，Fantastic 的流程，WSD | 是（用 Fantastic 調好的 Muon 超參數；NorMuon 用同一份） | ppl：Muon 15.58、NorMuon 15.44、**Muon-VS 15.37（比 NorMuon −0.45%）**；看起來只跑一個 seed | 高（規模、流程） | 4、14 |
| DynMuon（arXiv，非六大） | 1.1B，10B/20B tokens（9/18 TPP），524k 批次，cosine，**wd 0.01** | lr 只在 127M 掃過 | 10B：loss 2.788 → 2.776（約 −1.2% ppl）；20B：2.722 → 2.713；127M 時也勝過 NorMuon | 中（wd 很弱） | 8、14 |
| HTMuon（ACL 2026） | LLaMA-1B，9B tokens（約 7–9 TPP），**131k 批次** | Muon 只試 2 個 lr，wd 固定 0.1 | ppl 14.33 → 14.17（−1.1%）；embedding 也用 Muon；每 5 步做一次 SVD | 高（設定），但對照組弱 | 7 |
| AdaMuon（arXiv，非六大） | 7B/235B、1.5B/100B，**不衰減** | 沒調（兩者共用 lr） | 7B：比 AdamW 少 29.6% tokens，Muon 是 24.8% | 低 | 5、7 |
| NAMO / NAMO-D（arXiv，非六大） | 355M，約 14 TPP，不衰減 | 掃了 5 個 lr，wd 0.01 | 2.9684 → 2.9507 | 低 | 13 |
| PolarGrad（arXiv，非六大） | 玩具規模，約 0.1B tokens | 沒掃 lr | 只有圖；GPT-2 上把 embedding 從 Adam 換成 PolarSGDM，曲線差不多 | 低 | 38、66 |
| DeVA（ICML 2026） | 274M，1.4B tokens | 主表 lr 共用 | 主表 −1%；Muon 掃 lr 後只差約 0.01；步時 +54% | 低 | 9、21 |
| Delving into Muon（ICML 2026） | 124M，98B tokens | 公平掃描 | Muon 沒有穩定贏過調好的 Adam；UΣ^p Vᵀ 的 p=¼ ≈ Muon | 低 | 7–8 |
| OLion（ICML 2026） | 7B/32B，cosine 到 10% | 沒調 | 只有曲線 | 低 | 6、12–13 |
| MuonMax + Momo（ICML 2026） | 774M，6B tokens（約 8 TPP），WSD 到 10% | 在 1B tokens 做雙網格 | ppl 17.29 → 17.21（−0.5%）；主要好處是 lr 範圍變寬；124M 時輸 | 中 | 28、34 |
| Scion（ICML 2025） | 3B，2.7B tokens（0.9 TPP），衰減到 0 | 1B 寬度有掃，3B 沿用 124M 的設定 | 3B：Muon 2.909 → Scion 2.882；1B 寬度：2.988 vs 2.984（平手）；embedding 用 sign 更新 | 中 | 9、35 |
| Gluon（ICML 2026） | 理論 | — | 預測 embedding 的步長應大 20–30 倍 | 低 | 8 |
| Spectral Sphere（SSO，ICML 2026） | 1.7B/100B，4M 批次 | Muon lr 來自 µP 掃描 | 比 Muon 快約 1.09×；步時 +11% | 低（TPP 59） | 7–8 |
| Spectra（頻譜異向，ICML 2026） | 8B/50B（6 TPP） | 沒說明 Muon 怎麼調 | loss 2.60 vs 2.59 | 中（TPP） | 8–9 |
| Spectral Clipping（ICML 2026） | 780M/16.6B；1.5B/8B（約 5 TPP） | D-Muon 只試 1–2 個 lr | −0.02 loss | 中 | 9、36 |
| HyperP（NeurIPS 2026） | 到 3.8B，50 TPP | 只在 208M 調，之後轉移 | 208M 打平；3.8B 有 1.58× 的算力效益，主要來自 lr 轉移 | 低 | 26 |
| Muon 贏在尾端記憶（ICLR 2026） | 160M/0.7B | 消融的設定沿用 Muon 的 lr | Muon 的好處主要來自 VO 和 FFN；embedding 沒做消融 | 中 | 4、21 |
| SOAP-Muon（技術報告，非六大） | modded-nanogpt 舊版，批次 4× | Muon 有調 | **批次 512k 時差距只有約 0.005**，大批次才明顯 | 中 | 3 |

### 2.3 動量與變異數縮減

| 論文 | 設定 | 對照組調參 | 效果 | 相符度 | 頁 |
|---|---|---|---|---|---|
| EMA-Nesterov（NeurIPS 2026）（我們在用的） | 60M–1B，1B 用 4M 批次跑 5000 步 | 跟 Muon、SNOO、Lookahead 比，都掃過 | 所有前瞻類方法中最好；建議 γ 0.99–0.995 | 中 | 9–10 |
| AdEMAMix（ICLR 2025） | 1.3B，**131k 批次**，≥26 TPP，cosine | AdamW 有掃 | 101B tokens ≈ AdamW 的 197B；run 越短優勢越小；沒有 Muon 版本 | 中（批次同；TPP 高） | 2、9、32 |
| MARS（ICML 2025） | 770M，約 64 TPP | 有掃，但 Muon 比 AdamW 還差 | MARS-AdamW 2.511、AdamW 2.568、Muon 2.606 | 低 | 8、34 |
| MARS-M（arXiv，非六大） | 1.5B，33 TPP | 沿用 Moonlight 的 lr，沒重調 | 只有曲線；下游 57.33 → 57.7 | 低 | 9–10 |
| Cautious Optimizers（ICLR 2026） | 1.2B，20 TPP | 是（座標下降） | C-AdamW ppl 11.36 → 11.32（−0.3%）；C-Muon 只在 100M 試過 | 中（只有 AdamW） | 8、22–25 |
| Cautious Weight Decay（ICLR 2026） | 2B，20 TPP，cosine | 是（Muon 的 momentum 也掃過，最佳 0.95） | Muon：2.4803 → 2.4703；**我們 BA 打平** | 中 | 7、29 |
| SODA（NeurIPS 2026） | 124M modded-nanogpt | 是 | 3.2815 → 3.2665；做法是取代 wd 的 1/k 平均 | 中 | 8、20–21 |
| MGUP（NeurIPS 2025） | 71M/150M | 沒調 | 約 1.1–1.2× | 低 | 7–8 |
| Denoise First（NeurIPS 2026） | 理論，350M 驗證 | — | momentum 要放在 polar 前；最佳 β 0.95；沒有給排程建議 | 中 | 12、39 |
| Deriving HP Scaling Laws（NeurIPS 2026） | 理論，代理實驗用固定 lr | — | 批次固定時 1−β* ∝ T^−½ | 低 | 4–5 |
| GradPower（ICML 2026） | 2B，20 TPP，131k 批次 | AdamW 有掃；Muon 沒掃 | AdamW −0.022 loss；Muon 版本只到 0.4B | 低 | 5–6、16 |
| In Search of Adam's Secret Sauce（NeurIPS 2025） | 160M/410M，全部用 Adam，1,500+ 個 run | 是 | β1=β2 幾乎總是最佳；「β1=0.9 時 β2=0.9875 很差，β1=0.975 時很好」 | 中（我們只有 embedding 和 norm 用 AdamW） | 5–6 |

### 2.4 embedding、norm 與分層 lr

| 論文 | 設定 | 對照組調參 | 效果 | 相符度 | 頁 |
|---|---|---|---|---|---|
| Coupled Adam（ACL 2025） | 到 2.6B，**tied** | GPT-3 的 lr，沒調 | 20 TPP 以下沒有幫助或更差（1.3B：2.448 vs 2.433）；760M/5B 平手 | 中 | 6–7 |
| SAGE（ACL 2026 Findings） | 1.3B，6.6B tokens（約 5 TPP），130k 批次，推測是 tied | 弱，沒有 Muon 對照 | 24.33 vs 28.71（對照組是 SinkGD + AdamW embedding） | 高（設定）／低（對照組） | 6 |
| SCALE（ICML 2026） | 1B/20B，untied | 有掃 | 輸出層（lm_head）梯度變異最大，只對它加 momentum 就拿到大部分好處 | 中 | 5–6、8、13 |
| Sharpness Disparity（ICML 2025） | 2B AdamW | 是 | 分塊 lr：Emb 10、QK 8、FFN 6、VO 4；norm 的 lr 加倍會變差；換到 Adam-mini 要重調 | 低（只有 AdamW） | 7–8、16 |
| SPlus（NeurIPS 2025） | 160M，**固定 lr、不衰減** | 有掃 | 好處主要來自權重平均，等於替代 lr 衰減 | 低 | 8–10 |
| Practical Efficiency of Muon（arXiv，非六大） | 到 4B/50B，序列 8192，cosine 到 10% | 有掃 | 比 AdamW 少 10–15% tokens；embedding 用 Adam，lr 跟 Muon 相同（RMS 對齊） | 中 | 3、11 |
| Muon River-Valley Limits（NeurIPS 2026） | 250M，4k 步 | 沒說明 | 後段把 Muon 換成 AdamW 有幫助；作者自稱是初步結果 | 低 | 9、42–44 |

---

## 3. benchmark 論文的結論整理

### 3.1 低 TPP 區間誰最好
- **Muon 類在資料最少的區間最強**
  - Fantastic：1× Chinchilla 時 Muon 最好；SOAP、Kron 要到 ≥8× 才超過它（p8–9）
  - 我們以非 embedding 參數算：6B / 0.97B ≈ 6 TPP ≈ 0.3× Chinchilla，比他們測過的最低點還低
  - Fantastic 的 scaling law 外推：Muon 的資料項指數較小（0.207 vs AdamW 0.280，p29），所以 D 越小，Muon 領先 AdamW 越多
  - 拿 N=1.2B 粗算：1× 時 Muon 的 loss 低約 0.015，0.25× 時低約 0.039。這是外推，只能當方向參考
- Benchmarking：124M 最短的 run（1B tokens，約 8 TPP）D-Muon 最好；訓練越長，AdEMAMix 越領先（p5–6）
  - 意思是慢 EMA 類方法的好處隨訓練長度增加，在我們這種短 run 最小
- SOAP / Shampoo 類：
  - 大批次才明顯。SOAP-Muon 報告：512k 批次時跟 Muon 只差約 0.005（p3）
  - How to Scale Second-Order 在跟我們一樣的 131k 批次下，SOAP ≈ Muon（p35）
- 1.2B 時，加了 Nesterov 的 AdamW 跟 Muon 差不多（Fantastic p9）。Fantastic 的推測是：模型越大，雜訊對訓練的影響越重要，所以矩陣類的優勢縮小

### 3.2 多數加速為什麼會消失（Fantastic、Benchmarking、Track 3 的共同結論）
1. **對照組 lr 沒調**：GPT-3 配方只調 lr 就能快 2×（Fantastic 圖 1）
2. **不同優化器的最佳 wd 不同**：例如 Lion 約 0.6、AdamW 約 0.1，共用超參數就不公平
   - Muon 不加 wd 會差很多（Benchmarking：D-Muon 比 Muon 好很多）
3. **只看訓練中途**：lr 衰減時曲線會交叉。我們自己的 BA、BB、BX、AQ 也都是中途領先、最後打平
4. **規模**：多數論文 ≤350M；1.2B 時加速只剩約 1.1×
5. **對照組本身有缺陷，新方法剛好補上**：例如 wd = 0、cosine 只降到 10%、或不衰減（AdaMuon、NAMO、SPlus）
   - 我們的配方已經有調好的 τ 和衰減到 0，所以這類好處多半已經拿到了
6. **批次大小**：變異數縮減類（MARS、AdEMAMix）在小批次佔優；矩陣類在大批次佔優。不同論文的排名因此互相矛盾（Fantastic p4、p22）

### 3.3 Track 3 排行榜：對調好的 Muon/NorMuon 公平比較（124M，n 是 seed 數）

| 改動 | 步數（越少越好） | 對照 | 判讀 |
|---|---|---|---|
| NorMuon（#10）、調好的 Muon（#36） | 3250、3250 | — | NorMuon ≈ 調好的 Muon |
| Hyperball：NorMuonH（#8）、MuonH 加調好的輔助 Adam（#37） | 3250、3250 | 3250 | **打平** |
| EMA-Nesterov（#39、#40） | 3325→3125、2930→2890 | 同配方 | 贏（我們已採用） |
| SOAP-Muon：只用在 MLP（#14）、用在全部矩陣（#44） | 3225→3150、2875→2750 | 同配方 | 贏，但 #44 同時改了別的東西 |
| KL-SOAP-H（#19）、SinkSOAP（#26） | 3125、3090 | NorMuonH 3250、NorMuon 3250 | 贏約 4–5% |
| DynMuon（#28，n=25） | 3175 | 調好的 Muon 3250 | 贏約 2.3% |
| AdamW 組獨立衰減 40%、矩陣組 80%（#24，n=10） | 3225→3175 | 同配方 | 贏約 1.6% |
| MuLoCo 外層 Nesterov（#13） | 3250→3210 | NorMuonH | 小贏，跟 EMA-Nesterov 同一類 |
| 最後一步混入權重 EMA（#45） | 2750→2720 | — | 我們在 150M 測過，沒有好處 |
| Shampoo、PSGD-Kron、單邊 Shampoo、Newton-Muon | 4100、3375、3375、3275 | 3250 | 沒有好處 |

注意 Track 3 的規模和步數：
- 124M，約 3000 步 × 524k 批次
- 我們是 1.24B，6B 時約 45.8k 步 × 131k 批次
- Track 3 的 wd 調過（τ 約 0.3，跟我們接近），所以第 5 類「對照組有缺陷」的問題比論文少；但模型小 10 倍

---

## 4. 候選改動

收益以 1B（比照 CJ：KD α 0.9 + EMA-Nesterov）的最終 in-training ppl 計。依 00-context，≥0.3% 視為真的；0.1–0.3% 要補一個 seed。

### C1. AdamW 組（tied embedding + norm）單獨用較短的衰減
- **機制：**
  - 現在四個組都在最後 80% 線性降到 0
  - Muon 組每步位移大小固定，需要長的衰減來壓掉震盪
  - AdamW 組的 embedding 一列只在該 token 出現時才有大梯度，多待在高 lr 可能讓少見 token 學更多
  - 所以讓 AdamW 組晚點才開始降，只在最後 40–60% 降
- **證據：**
  - Track 3 #24：輔助 Adam 衰減 0.4、矩陣 0.8，3225 → 3175 步（n=10，約 −1.6%）
  - Track 3 目前紀錄 #46：Adam 組平到 51% 才降，Muon 組 18% 就開始降
  - 我們的 CA（norm lr ×0.5 +0.68%）顯示 AdamW 這一側對 lr 很敏感
  - Benchmarking Takeaway 6：最佳排程隨優化器而不同
  - 沒有六大會議論文直接測過
- **預期：**
  - 1B：−0.1% 到 −0.3%，信心低到中
  - 6B：比例沿用，預期相近，信心低（Track 3 只到 124M）
- **交互：**
  - EMA-Nesterov：β 是用每組自己的 lr 比例算的（`optimizer.py` 的 `lookahead(..., lr / initial_lr)`）
    - embedding 組的 β 會在 0.6 待到它開始衰減，之後跟著 lr 下降；窗口在 74% 結束，不受影響
    - 這等於 embedding 的前瞻也變長了，要一起看
  - NorMuon：無
  - KD：KD 主要改的是輸出端的梯度，而 tied embedding 就是輸出層，可能會交互，所以要在 CJ（有 KD）上測
- **有沒有試過：** 沒有。00-context §7 列為「AdamW 組用較短的獨立衰減」，還沒測
- **實作草圖：** 把 HF 的 WSD 排程換成每組各一個函式的 `LambdaLR`，約 10 行

  ```python
  DECAY = {"attn": 0.8, "mlp": 0.8, "embedding": 0.4, "norms": 0.4}
  def wsd(frac):  # frac: 衰減段占全程的比例
      return lambda s: min(1, s / warm) if s < (1 - frac) * T else (T - s) / (frac * T)
  LambdaLR(opt, [wsd(DECAY[g["name"]]) for g in opt.param_groups])
  ```
- **成本與風險：**
  - 成本：沒有額外計算
  - 風險：Track 3 的 embedding 和 head 是分開的（lr 0.7 vs 0.004），好處可能只來自 embedding 那一側；我們是 tied，效果可能被稀釋
- **1B 測試：**
  - 在 CJ 配方上跑兩支：AdamW 組衰減 40% 和 60%，約 20 H200-h
  - 失敗條件：兩支都沒有比 CJ 好 0.1% 以上
  - 好 0.1–0.3% 的那支，換 seed 補跑一支

### C2. 正交化之前先按元素除以雜訊大小（Muon-VS）
- **機制：**
  - Muon 只用 momentum 的方向，不管每個元素有多吵
  - Muon-VS 另存一個 EMA：記錄「梯度偏離 momentum 多少」的平方
  - 進入 polar 之前，先把 Nesterov 輸入按元素除以它的平方根，讓雜訊大的元素先被壓小
  - NorMuon 是在 polar 之後按列調整；這是在 polar 之前按元素調整，兩者放的位置不同
  - 也可以看成 SOAP-Muon 的特例：基底固定成單位矩陣時的版本
- **證據：**
  - VAM（非六大）：1.2B、Fantastic 的流程、沿用調好的 Muon 超參數（p14 表 5）
    - ppl：Muon 15.58、NorMuon 15.44、Muon-VS 15.37
    - 也就是比 NorMuon 再 −0.45%；看起來只有一個 seed
  - SOAP-Muon 報告：polar 前做一次按元素開根號的白化，能修好不穩定，而且「沒有壞處」（p3）
  - 同一類但證據較弱：AdaMuon（放在 polar 後，對照組沒調、不衰減）、NAMO（只到 355M）
- **預期：**
  - 1B：0 到 −0.3%，信心低到中
  - 6B：批次不變，每步的雜訊結構相同，預期相近，信心低
- **交互：**
  - NorMuon：VAM 有比較兩者，但沒有疊在一起測
    - NorMuon 的好處會隨 wd 變大而縮小（−1.7% → −0.76%）。VS 也是按元素自適應，疊加後可能只剩一部分
  - KD：軟目標的梯度雜訊比較小，VS 的好處可能變小
  - EMA-Nesterov：機制重疊低（一個作用在每一步的輸入，一個是權重空間的慢前瞻）
- **有沒有試過：** 沒有。Sophia 是對角曲率，不一樣；NorMuon 是放在 polar 之後
- **實作草圖：** 在 `muon.py:muon_update` 加約 8 行。polar 會把整體大小正規化掉，所以 lr 不用改

  ```python
  var = state.setdefault("var", torch.zeros_like(grad))
  var.lerp_((grad - buf).square(), 1 - momentum)   # 要在 buf.lerp_ 之前（用上一步的 m）
  nesterov = grad.lerp(buf, momentum) / (var.sqrt() + 1e-8)
  ```

  bias correction 可以省：polar 不受整體縮放影響
- **成本與風險：**
  - 記憶體：每個矩陣多一個 fp32 buffer，1B 時每張卡多約 4 GB
  - 計算：只有逐元素運算
  - 風險：eps 太小時，雜訊極小的元素會被放大
- **測試（先 150M 再 1B）：**
  - 150M：NorMuon vs NorMuon + VS，3 個 seed，約 1.5 H200-h
  - 過關條件：平均 ≤ −0.15%，而且 3 個 seed 都是負的
  - 過關後在 1B 跟 CJ 比，lr 相同
  - 失敗條件：150M 沒過關，或 1B ≥ −0.1%

### C3. 後段改用溫和的負次方頻譜（DynMuon 後段 p ≈ −0.25）
- **機制：**
  - Muon 把動量矩陣所有的奇異值都設成 1
  - p < 0 時，原本弱的方向反而比強的方向走得更多
  - DynMuon 的理論：訓練後段，剩下的有用訊號集中在平坦（曲率低）的方向
  - 這跟研究筆記 02 裡 BulkBoost 的發現一致：雜訊帶以下的方向仍帶 10–64% 的訊號
- **證據：**
  - DynMuon（非六大）：1.1B、10B tokens，−1.2% ppl；20B 時 −0.9%（p8）
    - 但 wd 只有 0.01、cosine 衰減，Muon 的 lr 只在 127M 掃過
  - Track 3 #28（n=25）：3175 vs 調好的 Muon 3250
  - DynMuon 附錄 E：127M 時也勝過 NorMuon
- **反證據：**
  - HTMuon（ACL 2026，1B、9B tokens、131k 批次，跟我們最像）用的是 **p = +0.125**（方向相反），也贏 Muon（14.33 → 14.17，但 Muon 只試了 2 個 lr）
  - Delving：p = ¼ ≈ Muon
  - 我們的 head damping（壓頂端方向，等於相對放大其他方向，方向跟 p<0 相同的 rank-1 版本）在 150M 調好 lr 後打平
  - p 的正負在不同論文互相矛盾，代表效果可能小，或跟 lr 混在一起
- **預期：** 1B：0 到 −0.3%，信心低；6B：不確定
- **交互：**
  - NorMuon 在後面做按列正規化，可能抵銷一部分頻譜形狀
  - 跟 head damping 高度重疊（head damping 已擱置）
  - 跟 EMA-Nesterov 重疊低
- **有沒有試過：** 部分。head damping 是同方向的 rank-1 版本，150M 打平
- **實作草圖：**
  - 照 DynMuon 的 Algorithm 2：Polar Express 算出 Y ≈ UVᵀ 後，用 A = XXᵀ（小邊）做二階泰勒修正 C = I + δE + ½δ(δ−1)E²（E = A − I，δ = p/2），更新 = C·Y
  - p 只在前約 4% 之後才生效；前段保持 p = 0，不動 warmup
  - 約 12 行；他們報告步時 1.003–1.025×
- **成本與風險：** 泰勒近似只在 p 接近 0 時準；跟 NorMuon 疊加是新組合，沒有人測過
- **測試：**
  - 150M：NorMuon vs NorMuon + 後段 p = −0.25，3 個 seed，lr ×{1, 1.4}（p<0 會改變有效步長）
  - 過關條件同 C2；沒過就停

### C4. AdamW 組的動量：β1 = β2，或加 Nesterov
- **機制：** tied embedding 佔約 21% 的參數，而它的 AdamW 設定從來沒調過 β1
  - (a) Secret Sauce：β1 = β2 時，Adam 等於「依變異數調整大小的 sign 更新」，幾乎總是最佳。它也明講「β1 0.9 配 β2 0.9875 很差，β1 0.975 配同一個 β2 很好」（p5），而我們剛好是 (0.9, 0.9873)
  - (b) Nesterov AdamW 是成本最低的變異數縮減
    - Fantastic：調好參數後一律勝過 AdamW，1.2B 時 ≈ Muon（p9）
    - C-AdamW 在 1.2B 也有 −0.3%（Cautious p8），可以當第三個選項
- **證據的限制：**
  - 都是整個模型都用 Adam 的實驗
  - Secret Sauce 的差距尺度是 0–0.34 ppl，模型是 160M；只用在 embedding 和 norm 時，效果應該更小
  - Deconstructing、SCALE：最後一層和 embedding 是梯度變異最大的地方，最需要自適應和 momentum
- **預期：**
  - 1B：−0.05% 到 −0.2%，信心低
  - 6B：β 以每步計，批次不變，所以照用
- **交互：**
  - EMA-Nesterov 也作用在 embedding 上；Nesterov 是一步的前瞻，跟它有中度重疊
  - KD：同 C1，所以要在 CJ 上測
- **有沒有試過：** 沒有（只試過 embedding wd 0、norm lr）
- **實作草圖：** `adamw.py` 改 1–2 行

  ```python
  BETA1 = 0.975                       # (a)
  m_hat = (beta1 * exp_avg + (1 - beta1) * grad) / (1 - beta1 ** step)   # (b) Nesterov
  ```
- **成本與風險：** 沒有額外成本。β1 變大會讓動量變慢，延遲多約 25 步
- **1B 測試：**
  - 在 CJ 上跑 (a)、(b) 各一支，約 20 H200-h；可以跟 C1 同一批送
  - 失敗條件：≥ −0.1%

### C5. SOAP-Muon（polar 前先做單邊 SOAP 白化）：只在 C2 贏了之後才做，而且先在 150M
- **機制：**
  - Muon 只對接近正方形、而且只有一邊條件差的矩陣白化得好
  - gate、up、down 是 4:1 的長方形
  - 做法：在小的那一邊用梯度共變異數的特徵基底跑 Adam，再做 polar，兩邊就都被處理到
  - C2 是它在基底固定成單位矩陣時的特例
- **證據：**
  - Track 3 #14、#44；KL-SOAP-H、SinkSOAP 比 NorMuon 少約 4–5% 步數
  - Track 3 用了 SOAP-Muon 之後，把 NorMuon 拿掉了
- **反證據：**
  - SOAP-Muon 報告：512k 批次時只差約 0.005
  - How to Scale Second-Order（131k 批次、到 1.4B）：SOAP ≈ Muon
  - Fantastic：SOAP 要到 ≥8× Chinchilla 才超過 Muon，而我們是 0.3×
  - SOAP 原論文：批次小時好處變小
- **預期：**
  - 1B：0 到 −0.4%，信心低
  - 6B：TPP 太低，不太可能變大
- **交互：**
  - 會取代 NorMuon
  - 跟 EMA-Nesterov 可以並存（Track 3 的紀錄就是兩者都有）
- **有沒有試過：** 沒有。Mousse（單邊 Shampoo）擱置、K-FAC 放棄，都跟它不同
- **實作草圖：** 小邊 d 上維護 L = βL + GGᵀ，每 10 步做一次 QR 更新基底 Q；G' = QᵀG 上做 Adam，轉回去後再進 Polar Express。約 40 行
- **成本與風險：**
  - 記憶體：每個矩陣多 2d²，1B 共約 3.6 GB
  - 計算：每步約多 12 TFLOP
    - 如果用 fp32、而且每張卡都重複算一遍，步時 +20–35%
    - 用 bf16 旋轉並分到 8 張卡，約 +3–5%
  - 跟 64 H200-h 的上限有衝突，不適合截止前臨時加進 6B
- **150M 測試：** 3 個 seed、2 個 lr；比 NorMuon 好 ≥ 0.3% 才上 1B

### 小項（不另開 1B 測試）
- **Muon momentum 的升溫和降溫：**
  - Track 3 的做法：前 300 步 0.85 → 0.95，最後 200 步 0.95 → 0.85，但是跟其他改動綁在一起，沒單獨驗證
  - 我們的 BX（0.9 前段領先、衰減段被追過）支持「前期低、後期 0.95」
  - 這已列在 00-context §7，文獻沒有新證據，維持原計畫
- **6B 的 Muon momentum：**
  - 理論（Deriving HP Scaling Laws）說批次固定時，訓練越長最佳 β 越高：6× 長 → 1−β 約小 √6 倍，也就是 0.95 → 約 0.98
  - 但 Fantastic 實測：1–8× Chinchilla 之間，Muon 的 momentum「對規模不敏感」，只有 lr 要重調（p6、p29 表 5）
  - 加上我們 BD（0.98）+0.88%，**6B 維持 0.95**

---

## 5. 反證據與不建議的方法

| 方法 | 不建議的原因 |
|---|---|
| Hyperball / MuonH / HyperP / Spectral Sphere（用範數限制取代 wd） | Track 3 公平比較打平（NorMuonH = NorMuon = 3250；MuonH 加調好的輔助 Adam = 調好的 Muon = 3250）。1.2B 的大收益來自對照組 wd 沒隨規模調，而我們 τ 已調好（0.27–0.55 曲線平坦）。SSO 步時 +11%。研究筆記早已放棄 Frobenius hyperball，新證據沒有改變這個判斷 |
| AdEMAMix（或把慢 EMA 加進 Muon） | 跟 EMA-Nesterov 機制高度重疊（都是長時間的平均；SODA 把它們歸成同一家族）；run 越短優勢越小；沒有跟 Muon 疊加的公平證據；Benchmarking 裡的 Muon 調得弱。真要試，β3 用 0.999 |
| MARS / MARS-M | MARS 論文的 Muon 比 AdamW 還差（對照組弱）；MARS-M 沒重調 lr、33 TPP、只有曲線。它的近似版其實只是重新分配 Nesterov 的係數，我們的 momentum 掃描（BD、BX）已部分涵蓋。Fantastic：MARS 只比 AdamW 好一點，大批次時輸給矩陣類 |
| Cautious 更新遮罩用在 Muon 上 | 按元素遮罩會破壞正交性；C-Muon 只在 100M 試過；我們 BA（cautious wd）打平。要試的話只用在 AdamW 組（C4 的第三個選項） |
| AdaMuon、NAMO、OLion、PolarGrad、DeVA、Delving、MGUP、GradPower | 對照組沒調、或不衰減、或 ≤355M、或只有曲線。DeVA 的 Muon 掃 lr 後只差約 0.01，而且步時 +54%。Delving 發現 Muon 不一定贏調好的 Adam。AdaMuon 跟 NorMuon 同類（VAM 在 300M：AdaMuon ≈ NorMuon） |
| HTMuon（p = +0.125） | 設定最接近我們，但 Muon 只試 2 個 lr；要做 SVD（每 5 步，步時 +16%）；跟 DynMuon 的方向相反；embedding 也用 Muon，跟其他證據衝突 |
| Shampoo、PSGD-Kron、單邊 Shampoo、Newton-Muon、完整 SOAP | Track 3 沒有好處；How to Scale：131k 批次時 ≈ Muon；Kron/SOAP 要高 TPP 才贏；成本高 |
| embedding 改用 Muon 或 sign / Scion | How to Scale：embedding 用 Muon 比較差（p34–35）；Scion 的 sign 更新沒有單獨消融，3B 的對照組也是用代理設定；PolarGrad 在 GPT-2 上把 embedding 從 Adam 換掉，曲線差不多；SAGE 的對照組很弱 |
| Coupled Adam | 20 TPP 以下沒有幫助或更差，即使是 tied |
| 分塊 lr（Sharpness Disparity） | 倍數是 AdamW 專用的，換優化器要重調；我們 150M 的分類型 rotation 已顯示 Muon 組兩類都在最佳點 |
| 每種矩陣不同的更新尺度 | 研究筆記 02 §2 已說明：decoupled wd 下，尺度常數只決定權重範數，不決定步長；150M 的分類型 rotation 已放棄。文獻（µP 的 √(out/in)、SPlus 的 2/(m+n)）在調好的 Muon 上沒有超過現有約定的證據 |
| SODA | 把 wd 換成 1/k 平均，會跟我們調好的 τ 和 EMA-Nesterov 衝突；只在 124M 測過 |
| SPlus、權重平均類 | 好處主要在不衰減的設定；我們 150M 的尾段 EMA 也沒有幫助 |
| 後段把 Muon 換成 AdamW（River-Valley） | 250M、作者自稱初步；而我們的 AF（全 AdamW）+9.1% |
| SAMuon / Spectral Allocation | 就是我們 150M 測過的 head damping，打平 |
| Dion、MuonBP、LiMuon、低秩類 | 解決的是通訊和記憶體，不是 ppl |
| MuonClip（QK-clip） | 處理超大模型的 logit 爆炸，我們沒有這個問題 |

---

## 6. 結論
1. **benchmark 對我們區間的結論：**
   - 在 0.3× Chinchilla、131k 批次、1.2B 這種區間，Muon 類本身就是最好的一類
   - SOAP、Kron 這些更重的矩陣方法要高 TPP 或大批次才超過它
   - 調好參數之後，各種新優化器的加速多半縮到 ≤1.1×
   - 所以不要期待優化器核心有大突破。對 1B 的現實預期是：所有通過測試的改動加起來 −0.2% 到 −0.5%
2. **最值得先測的是最便宜的兩項：**
   - C1（AdamW 組較短的衰減）和 C4（AdamW 組 β1 = β2 或 Nesterov），都在 tied embedding 和 norm 那一側
   - 兩者都是零計算成本、幾行程式，可以同一批在 CJ 上跑 4 支（約 40 H200-h）
3. **Muon 核心的改動先在 150M 篩：**
   - C2（Muon-VS）證據最直接：1.2B、調好的對照組、勝過 NorMuon
   - C3（DynMuon 後段負次方）有 Track 3 n=25 的支持，但正負號在文獻裡互相矛盾
   - 兩者都要先在 150M 過關（約 3 H200-h）才上 1B
   - C5（SOAP-Muon）只在 C2 贏了之後才考慮，而且太貴，不適合放進截止前的 6B
4. **6B：** 除非某項先在 1B 過了，否則不要動優化器核心。Muon momentum 維持 0.95，τ 和 lr 照 T3 的規則
5. **跟 EMA-Nesterov 重疊、不要再加的：** AdEMAMix、SODA、MuLoCo 外層 Nesterov、權重平均類

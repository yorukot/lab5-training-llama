# T4 學習率排程與權重平均（第 6 號研究員）

日期：2026-10-17。範圍：
- 衰減比例、衰減形狀
- 權重平均
- EMA-Nesterov 的開啟方式
- schedule-free
- warmup

**一句話結論：**
- **衰減比例：** 6B 的衰減比例應該略長，0.85（理由在 §3.1）。
- **EMA-Nesterov 的開啟：**
  - 最有依據的改法是「提早開啟（以步數計）」。
  - β 漸進開啟（ramp）預期對最終值沒有影響，只能當保險。
- **權重平均、schedule-free：** 在我們的設定下都不值得做。

---

## 1. 搜尋紀錄

**工具與規模：**
- 索引：`papers.jsonl` 共 68,363 篇，跑了 **50 組 regex**
- 命中：合計 748 次，去重後 **643 個標題**，全部看過
- 篩選：約 55 篇讀摘要，**精讀 47 篇全文**，其中 42 篇是會議論文、5 篇是 arXiv 上的非六大會議論文
- 讀法：其中 39 篇分給三個分身並行精讀，每篇回報頁碼與數字；其餘 8 篇和關鍵段落由我自己讀並交叉核對
- arXiv：共下載 18 份（13 份是索引內 NeurIPS 2025/2026 的 arXiv 版，5 份是非六大會議），每份間隔 ≥2 秒，沒有遇到 429
- WebSearch：3 次，補 re-warming、Muon momentum warmup、GPA
- PDF 與 txt：`/home/yorukot/ml/papers/pdf/T4/`

**引用追蹤：**
- EMA-Nesterov → SNOO、GPA、DiLoCo、Lookahead
- Straight to Zero → LAWA、Defazio 的線性衰減
- Hägele → Dremov 的 cooldown 研究（TMLR）、re-warming（Gupta、Ibrahim）

**注意：** 這台機器的 `grep` 被 shell function 包住，讀 txt 時會無聲回傳空結果。要用 `/usr/bin/grep -a` 或 python。

**50 組 regex 與命中數**（在標題加摘要上比對，不分大小寫）：
```
q1  29  warmup.{0,5}stable.{0,5}decay|\bWSD\b
q2   8  learning.rate (annealing|anneal)|annealing (phase|stage|ratio)
q3   7  cool.?down (phase|length|stage|schedule)|\bcooldown\b
q4  62  learning.rate schedul
q5   4  decay(ing)? (the )?learning.rate to zero|linear decay to zero|straight to zero|\bD2Z\b
q6   3  river.valley|valley.river
q7   1  multi.power law|loss curve prediction|predict(ing)? (the )?(full )?loss curve
q8   7  schedule.free
q9  26  weight averag|averag(e|ing) (of )?(the )?weights|checkpoint averag
q10 15  \bLAWA\b|latest weight averag|stochastic weight averag|\bSWA\b
q11 11  exponential moving average of (the )?(model )?(weights|parameters)|\bEMA\b.{0,40}(weights|parameters|model)
q12 16  (model|checkpoint) merging.{0,60}pre.?train
q13  3  lookahead optimi|look.ahead (optimizer|step)|slow weights
q14  8  outer (optimizer|momentum|nesterov)|\bDiLoCo\b|local SGD.{0,40}nesterov
q15  2  nesterov (momentum|acceleration).{0,80}(language model|transformer|deep)
q16  0  momentum (warm.?up|schedul)|warm.?up (of )?(the )?momentum
q17 20  learning.rate warm.?up|warm.?up (the )?learning rate|\bwarm.?up\b.{0,40}(adam|transformer|language model|LLM)
q18  7  (optimal|best) learning.rate.{0,60}(token|horizon|training (length|duration))|token horizon
q19 15  training duration|beyond fixed training|any.?time (training|schedule)|infinite learning rate
q20 10  (polyak|primal|iterate) averag|tail averag|suffix averag
q21  3  AdEMAMix|slow (EMA|momentum)|mixture of (two )?(EMA|momenta)
q22 60  last.iterate.{0,60}(schedule|convergence)|convex.{0,40}(learning.rate schedul|cooldown)
q23 12  scaling law.{0,60}(schedule|learning rate)|learning rate.{0,40}scaling law
q24 42  grokfast|amplif(y|ying) (slow|low.frequency)|low.frequency (component|gradient)
q25 11  re.?warm(ing)?|warm restart|cyclic(al)? learning rate|learning rate (re.?warm|restart)
q26 18  loss spike|training instabilit.{0,60}(learning rate|momentum|warm)
q27  8  extrapolat.{0,40}(weights|iterates|parameters|checkpoint)|weight extrapolation
q28  6  \bMuon\b.{0,80}(schedul|decay|warm.?up|averag)
q29  1  continual pre.?train.{0,80}(learning rate|warm|schedule)
q30  0  averag.{0,40}(iterates|checkpoints).{0,80}(language model|pre.?train|LLM)
q31 14  optimal (learning.rate )?schedule|schedule (design|optimization) for (training|SGD)
q32  9  (pseudo|outer).?gradient|two.?(time.?scale|loop) optimi
q33 25  heavy.?ball|polyak momentum|quasi.?hyperbolic
q34  1  \bEMA\b.{0,60}(language model|LLM|pre.?train)|(language model|LLM|pre.?train).{0,60}\bEMA\b
q35  2  learning rate.{0,30}(decay|annealing).{0,60}(language model|LLM|pre.?train|transformer)
q36  2  (edge of stability|sharpness).{0,80}(warm.?up|learning rate schedul|decay)
q37  2  power.?law (schedule|learning rate decay)|power scheduler|inverse.?(square.?root|sqrt) (decay|schedule)
q38  3  weight decay.{0,60}(time.?scale|EMA|exponential moving)|EMA view
q39  0  (nesterov|momentum).{0,60}(pre.?train|language model|LLM).{0,60}(improv|accelerat)
q40 81  anytime|budget.?(agnostic|aware) (training|schedule)|budgeted.?iteration
q41  0  (lr|learning.rate) sensitivity|robust(ness)? to (the )?learning.rate
q42 12  SGDR|cosine (decay|schedule|annealing)
q43  2  momentum (coefficient|parameter).{0,60}(increas|schedul|time.?varying|dynamic)|(increasing|time.?varying|dynamic) momentum
q44 91  \blookahead\b
q45 52  \bnesterov\b
q46  7  (warm.?up|warmup).{0,60}(duration|length|steps|fraction)|(duration|length) of (the )?warm.?up
q47 22  local SGD|federated averaging.{0,40}(momentum|extrapolat)|server (learning rate|momentum)
q48  2  (EMA|exponential moving average).{0,40}(warm.?up|decay rate|momentum schedule|bias correction)
q49  4  loss (landscape|curve).{0,60}(learning rate (decay|drop|schedule))|sudden (drop|decrease) in (the )?loss
q50  2  (compute|token).?optimal.{0,60}(schedule|cooldown|decay)
```

**意外收穫：** q44 / q45 找到 EMA-Nesterov 原論文其實已收進 **NeurIPS 2026**（索引裡沒有 PDF，所以用 arXiv 2605.25395）。

---

## 2. 論文表

標「非六大會議」的不在索引的六個會議內。

### A. 衰減比例與形狀

| # | 論文 | 會議 | 規模、優化器 | 跟我們有關的結論（頁碼） |
|---|---|---|---|---|
| 1 | Hägele et al., Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations | NeurIPS 2024 | 33M–8B，AdamW | 20% cooldown 追平 cosine，加長到 20% 以上收益變平（Fig.5，124M、15–50k 步）。訓練越長，需要的衰減步數占比越小（p.5）。1-sqrt 在 10–20% 衰減時勝 linear，但「衰減長很多時順序可能改變」（App. B.1）。re-warming 會造成 spike，且恢復很慢（p.3）。SWA 補不上衰減；schedule-free 在調好 β 後也只是追平或略輸 cooldown（p.7–8）。 |
| 2 | Bergsma et al., Straight to Zero | ICLR 2025 | 111M–1.7B，AdamW µP，2–200 TPP | 線性衰減到 0（D2Z）在 ≥20 TPP 最好，收益隨 TPP 變大。**但 2 TPP 時 10× 衰減勝 D2Z**（p.7）。批次越小，D2Z 越有利（Fig.7）。D2Z 的最佳 lr 較穩、lr 的誤差曲線較平。80 TPP 時 D2Z 比 22.5% cooldown 的 WSD 好 0.84%（p.9）。linear 微勝 cosine。 |
| 3 | Luo et al., A Multi-Power Law for Loss Curve Prediction | ICLR 2025 | 25M–1B，AdamW | 最佳化出來的排程接近 WSD，衰減段約 (1−τ)^1.5，降到接近 0（p.10）。400M、24k 步時 WSD 最佳衰減約 6000 步（25%）。附表 7 給出擬合參數。 |
| 4 | Tissue et al., Scaling Law with LR Annealing | NeurIPS 2025 | 594M，AdamW | L = L0 + A·S1^−α − C·S2，WSD 最佳衰減比例 10–20%。**1-sqrt 在 10% 時勝 cosine，50% 時輸**，有實驗驗證（§4.6）——跟我們的 AQ 一致。 |
| 5 | Wen et al., River Valley | ICLR 2025 | 0.1–1.2B，AdamW | 山壁損失約 ∝ η·σ²，衰減段把它降掉。8–12% 的衰減比例差異不大（高 TPP、最低 lr 0.1×）。短衰減後把 lr 調回去，不傷最終 loss（p.19）。 |
| 6 | Schaipp et al., Surprising Agreement Between Convex Theory and LR Scheduling | ICML 2025 | 凸理論，124M/210M AdamW | **每個衰減比例都配自己的最佳 lr 時，最佳是全程線性到 0；lr 固定時是拋物線；lr 夠大時越長越好**（§4.1，Fig.21）。γ\*(0.994) ≈ e^0.7·γ\*(0.2)，實驗吻合（Fig.12）。horizon 變 2 倍時，lr 降為 0.525×，或衰減比例拉到約 0.6（p.7）。 |
| 7 | Li et al., Functional Scaling Laws | NeurIPS 2025 | 核回歸，1B MoE | 容易的任務最佳衰減比例 ≈ 1；困難的任務最佳比例隨資料量變多而縮短（定理 5.4）。 |
| 8 | Bordelon & Mori, Theory of Optimal LR Schedules (random feature) | NeurIPS 2026 | 理論，GPT-2 125M | 容易區：全程多項式衰減，沒有穩定段。困難區：WSD，衰減比例隨 T 縮短。GPT-2 / C4 屬於容易區（p.8–9）。最佳 momentum 在接近結尾時上升。 |
| 9 | Attia & Koren, LR Annealing Improves Tuning Robustness | ICML 2026 | 凸理論 | 衰減到 0 讓 lr 設錯 ρ 倍的代價變成 ρ^{1/3}（線性衰減），固定 lr 則是 ρ。 |
| 10 | Convex Dominance in Deep Learning I | ICLR 2026 | GPT-2 0.1–7B，AdamW / Muon | 合格的排程下，最佳峰值 lr ∝ 1/√T（wd 固定，不是 τ 固定）。 |
| 11 | Bjorck et al., Scaling Optimal LR Across Token Horizons | ICLR 2025 | ≤7B，AdamW cosine | 最佳 lr ∝ D^−0.32（wd 固定）。 |
| 12 | How to Set the LR for Large-Scale Pre-training? | ACL 2026 Findings | Qwen3-MoE，AdamW WSD | lr ∝ N^−0.222·D^−0.351。 |
| 13 | Muon 的 river valley 分析 | NeurIPS 2026 | 理論，250M | Muon 接近終點容易 overshoot，一定要衰減。 |
| 14 | Deriving HP Scaling Laws via Modern Optimization Theory | NeurIPS 2026 | LMO / Muon 理論 | 1−momentum ∝ T^−1/2，預測 6B 的 momentum 約 0.98，但我們的 BD 在 1B 輸了。 |
| 15 | Gemstones | NeurIPS 2025 | 多種規模 | 只做 10% cooldown 的消融，沒有直接資訊。 |
| 16 | Bergsma et al., Power Lines | NeurIPS 2025 | AdamW，線性 D2Z | τ_opt = 1.084·TPP^−0.527，屬 T3a 的範圍。 |
| 17 | Wang & Aitchison, AdamW weight decay | ICML 2025 | AdamW | EMA 觀點：衰減段 lr 下降，等於把平均窗口拉長。 |
| 18 | Dremov, Hägele, Kosson, Jaggi, Cooldown Stage in WSD | TMLR 2025（非六大會議） | 約 124M，AdamW，20% cooldown | 不同形狀之間是 bias 與 variance 的取捨，sqrt 與「lowered linear 0.7」最佳。cooldown 時 AdamW β2 調大一致有效。多次 cooldown 再平均，輸給一次較長的訓練。 |
| 19 | Pre-training LLM without LR Decay Enhances SFT | ICLR 2026 | Llama-3.2-1B 架構，350B，AdamW | 預訓練 loss：WSD 25% 到 0 最好；全程線性到 0 多 0.016（lr 沒重調）；不衰減多 0.071。 |
| 20 | Benchmarking Optimizers for LLM Pretraining | NeurIPS 2026 | 124M–720M | 對 Muon 來說 WSD 最好。最終 lr 0.001–0.01× 略勝正好 0（210M）。 |

### B. 權重平均與 schedule-free

| # | 論文 | 會議 | 規模 | 結論 |
|---|---|---|---|---|
| 21 | Ajroldi et al., When, Where and Why to Average Weights? | ICML 2025 | AlgoPerf，124M LM 5B tokens | 最佳情況下 EMA 約 −0.12% ppl、LAWA 約 −0.04%（Table 4）。衰減到 0 時收益變小（Fig.7b）。平均取代不了衰減。最佳窗口約占訓練的 1%。 |
| 22 | WSM | ICLR 2026 | 16B MoE（1.4B active） | 用 merge 取代衰減，語言模型 loss 反而較差：0.697 vs WSD 0.675。「先衰減再 merge」沒有額外收益。EMA 式 merge 最差。 |
| 23 | Model Merging in Pre-training（PMA） | NeurIPS 2025 | 0.4–70B | stable 段 merge 約等於 annealing。用 spike 前 3 個 checkpoint 平均後重啟（PMA-init），能救回 loss spike。 |
| 24 | How LR Decay Wastes Your Best Data | ICLR 2026 | 1.5B，30B tokens | 資料隨機排序時，常數 lr + EMA 追不上 WSD。 |
| 25 | Extra-Merge | ICML 2026 | 124M–2B（含 Muon），Pythia-12B | checkpoint 幾乎落在一條直線上，事後沿這條方向外插有效。但終點 lr 是 0.05–0.1×，外插係數用驗證集線搜尋。 |
| 26 | Busbridge et al., How to Scale Your EMA | NeurIPS 2023 | 視覺 / 語音 | 批次 ×κ 時 EMA 係數改成 ρ^κ。我們的批次不變，γ 以每步計是對的。 |
| 27 | Defazio et al., The Road Less Scheduled | NeurIPS 2024 | NanoGPT 124M 等 | schedule-free 比 cosine 到 10% 低 2.2% ppl（約 2,400 TPP）。LLM 要 β 0.98，仍需 warmup。 |
| 28 | Through the River（schedule-free） | NeurIPS 2025 | 124M | schedule-free 本身走在河道上。2M 批次時原版輸 cosine，改良版只打平。 |
| 29 | AlgoPerf 分析 | ICLR 2025 | 多工作負載 | schedule-free 贏的是自調參賽道；有調參的 Shampoo 更快。 |
| 30 | Sanyal et al., LAWA | arXiv 2306.03241（非六大會議，workshop） | 125M–770M，Pythia | lr 越高、訓練前中期，平均的收益越大；到 cosine 末段收益遞減。 |

### C. EMA-Nesterov、外插、慢 momentum

| # | 論文 | 會議 | 規模 | 結論 |
|---|---|---|---|---|
| 31 | Yau et al., EMA-Nesterov | NeurIPS 2026（arXiv 2605.25395） | NanoGPT 124M/162M，Llama 60M–1B；Muon、NorMuon、Adam、SOAP | **β 分三段：前段 0 → 中段 β·α_t/max α（一步開到位）→ 後段 0**，作者說後段的休息「很關鍵」（§3.3，Fig.8）。Llama 的設定是 Tw = 30%、Tr = 80%（App. C.4）。**最佳 γ 隨總步數上升**：2500 → 5000 步時 0.99 → 0.995（App. C.3）；模型越大，最佳 γ 越大、β 越小（Fig.6）。**沒有測過 ramp。** |
| 32 | Pagliardini et al., AdEMAMix | ICLR 2025 | 110M–1.3B，AdamW | 從頭使用慢 EMA 必須讓 α、β3 warmup，否則發散（Fig.28）。**從中途的 checkpoint 開啟時不需要 warmup：T 取 {0, 25k, 50k, 100k}，越長只是收斂越慢，最終 loss 幾乎一樣**（Fig.27）。**開啟時 loss 先小升再超越 baseline，越早開，最終越好**（Fig.5）。 |
| 33 | Kallusky et al., SNOO | arXiv 2510.15830（非六大會議） | 至 1e23 FLOPs | 每 K 步對 pseudo-gradient 做 Nesterov，全程開，沒有 warmup。compute 增益 1.5–2.5×（對照組 AdamW 的 lr 固定 3e-4）。 |
| 34 | Douillard et al., DiLoCo | arXiv 2311.08105（非六大會議） | 150M | outer Nesterov 用 lr 0.7、momentum 0.9。從頭開始只比從 24k 步的 checkpoint 開始差約 0.1 PPL（Fig.3）。 |
| 35 | Charles et al., Scaling Laws for DiLoCo | NeurIPS 2025 | 35M–10B | 單 worker（M = 1）在各規模都贏 data-parallel 0.1–0.4%。外層參數可跨規模沿用，沒有外層 warmup。 |
| 36 | MuLoCo | ICML 2026 | Muon 內層，≤3.1B | K = 1 時 Muon 的最佳外層 momentum µ 約 0.6（AdamW 是 0.8），跟我們的 β 0.6 一致。 |
| 37 | High-dimensional theory of two-phase optimizers | NeurIPS 2026 | 線性回歸理論 | 開外層 momentum 時，最佳內層 lr 會下降。 |
| 38 | MT-DAO | ICLR 2026 | 16M–1.3B | 慢 momentum 在 warmup 結束時一步開啟，之後重調 lr。 |
| 39 | Momentum 的頻域分析（FSGDM） | ICLR 2025 | CNN / CIFAR 為主 | 逐步放大低頻成分最好：u_t = t/(t+µ)，µ ∝ 總步數。 |
| 40 | Nesterov for Async Pipeline Parallel | ICML 2025 | ≤1B | 沿用 NAdam 的 momentum warmup；固定 0.99 跟自適應版差不多。 |
| 41 | Defazio et al., GPA（Smoothing DiLoCo with Primal Averaging） | arXiv 2512.17131（非六大會議，WebSearch 找到） | 160M–8B | 單迴圈的平均加外插，比 AdamW 省 8.7–10% 步數。沒有關於 warmup 的資訊。 |
| 42 | Gupta et al., Continual Pre-training: How to (re)warm | arXiv 2308.04014（非六大會議，workshop） | Pythia 410M | re-warming 先讓 loss 上升，warmup 長度對最終值影響不大（arXiv 版；workshop 版結論不同）。 |
| 43 | Hop, Skip, Jump（lr 切換） | EMNLP 2024 Findings | Phi-3 3.8B | lr 突然改變後，loss 會收斂到新 lr 的軌跡，前面的軌跡幾乎沒有留下記憶。 |
| 44 | Learning Dynamics in Continual Pre-Training | ICML 2025 | AdamW | lr 上升只帶來暫時的 S2 負值，之後衰減會補回。 |

### D. Warmup

| # | 論文 | 會議 | 結論 |
|---|---|---|---|
| 45 | Kalra & Barkeshli, Why Warmup the LR? | NeurIPS 2024 | warmup 的作用是把模型帶到能承受目標 lr 的區域，長度由初期 sharpness 決定（以步數計）。再加長只有邊際好處。 |
| 46 | Kosson et al., Analyzing & Reducing the Need for Warmup | NeurIPS 2024 | 固定大小更新的優化器（類似 Muon）仍需要短 warmup，原因是初期的相對更新太大、臨界批次太小。 |
| 47 | Where Does Warm-Up Come From?（Muon / Lion） | NeurIPS 2026 | warmup 在 loss 跌破某個門檻時結束，所以長度由 loss 水準決定，不是 T 的比例。自適應版不輸手調的最佳值。 |
| 48 | Why Do We Need Warm-up? | ICML 2026 | 理論上 warmup 長度同樣由 loss 決定；但 Adam 70M / 160M 的粗網格裡 10–20% 最好。 |
| 49 | Porian et al., Resolving Discrepancies | NeurIPS 2024 | warmup tokens 約等於參數量 N（AdamW 小模型），跟我們 BR 的結果（5% → 1% 贏）矛盾。 |
| 50 | modded-nanogpt 的 Muon momentum warmup（0.85 → 0.95，300 步） | 非六大會議（WebSearch 找到） | 實務上常見的 momentum warmup，與 T5 有關。 |

---

## 3. 用我們的數據檢驗

### 3.1 「1B 最佳衰減 76%」被 lr 拉低了

AZ（0.5）、BM（0.8）、BL（0.95）都在 lr ×1.0（1.2e-3）下跑，之後 BQ / BU / BV 才證明 80% 衰減的最佳 lr 是 ×1.23。

Schaipp et al.（#6）與 Straight to Zero（#2）都指出：
- 衰減越長，最佳峰值 lr 越高
- lr 固定在偏低值時，衰減比例的拋物線最佳值會偏短

**怎麼校正：**
- lr 的誤差曲線：用 BR（×1.0，17.061）、BU（×1.25，17.027）、BV（×1.5，17.058）擬合 log-lr 拋物線。最佳在 ×1.230，曲率 4.67% /（ln lr）²，所以 ×1.0 時要付 0.20%。
- 每個衰減比例的最佳 lr 位移，用三種模型估計：
  - 凸理論的 ∝ 1/√(Σs²)
  - 面積法 ∝ 1/Σs
  - Schaipp 實測（位移是 √ 規則的 1.46 倍，對應 γ\*(1)/γ\*(0.2) = e^0.7）
- 程式：`/tmp/claude-1000/-home-yorukot-ml-lab5-training-llama/94460c4a-2e94-42fc-8d1d-1c1378fd5e26/scratchpad/scripts/lrcorr.py`

**結果：**

| lr 位移模型 | 0.5 | 0.8 | 0.95 | 拋物線頂點 |
|---|---|---|---|---|
| 未校正（原始） | +0.68% | 0 | +0.37% | 0.76 |
| √(Σs²) | +0.88% | 0 | +0.03% | **0.86** |
| 1/Σs | +0.88% | 0 | +0.01% | **0.87** |
| Schaipp 實測 | +0.86% | 0 | −0.17% | **≥0.95**（越長越好） |

- **結論：** lr 調好後，1B 的最佳衰減比例 ≥0.86，可能接近全程。
- 但 0.8 到 0.95 之間的差距只有約 ±0.2%，曲線很平。
- 限制：曲率是在 80% 衰減下量的；Straight to Zero 說 D2Z 的 lr 誤差曲線更平，所以這是近似值。

### 3.2 損失曲線定律的定量預測在我們的區間不成立

我用論文給的擬合參數模擬了兩個定律，在 1% warmup 下比較線性衰減的比例 0.1–0.95：
- MPL：400M 的擬合參數
- Tissue：594M 的擬合參數
- 程式：`/tmp/claude-1000/-home-yorukot-ml-lab5-training-llama/94460c4a-2e94-42fc-8d1d-1c1378fd5e26/scratchpad/scripts/sched_sim.py`

結果：

| 定律 | 最佳衰減比例 | 80% 比最佳差多少 | T 變長時 |
|---|---|---|---|
| MPL | 0.30（T = 8k–144k 都一樣） | +1.65%（8k 步）→ +0.58%（144k） | 最佳比例不變，懲罰變小 |
| Tissue | 0.20 → 0.15 | +2.0% → +0.66% | 最佳比例變短 |

- **這跟我們的 1B 實測完全相反：** 0.8 比 0.5 好 0.70%，校正 lr 後是 0.88%。
- 原因可能是這些參數來自 AdamW、低 lr（2–3e-4）、大批次的設定。我們用的是 Muon、高 lr、τ 0.39（wd 很強）、131k 小批次。Straight to Zero 說小批次、雜訊主導時長衰減最有利。
- **可借用的只有方向：** 這兩個定律都不支持「T 變長時要更短的衰減比例」帶來大收益——到 T 很大時，懲罰本來就已經變小。

### 3.3 1-sqrt 與「20% 1-sqrt」

- AQ 的結果（1-sqrt 在 50% 衰減時輸 +0.60%）跟 Tissue §4.6 一模一樣：1-sqrt 在 10% 勝、50% 敗。Hägele 也說長衰減時順序可能反轉。
- **§7 待辦清單裡的「20% 1-sqrt」：**
  - 用 §3.1 校正後的拋物線（A ≈ 7%/單位²）外插，20% 線性衰減約比 80% 差 +2–3%（粗略外插）。
  - 1-sqrt 在 20% 時對 linear 的優勢，Hägele 的 124M 約 0.5–1%，補不回來。
  - **建議從待辦清單移除。**
- (1−x)^1.5（MPL 的最佳形狀）的平均 lr 是 0.4，低於 linear 的 0.5，在 80% 衰減時同樣會少掉面積，不建議。

### 3.4 EMA-Nesterov 的跳升：3B 其實沒有比 1B 大多少

**機制**（`src/ema_nesterov.py` 與 `optimizer.py`）：
- 每步的位移是 step + β·m，m ← (γ + (1−γ)β)·m + (1−γ)·step
- 穩態時，持續的方向被放大 1/(1−β) = 2.5 倍
- 開啟瞬間是 1.6 倍，之後以時間常數 1/((1−γ)(1−β)) = 250 步爬到 2.5 倍
- 換算成占整段的比例：1B 3.3%、3B 1.1%、6B 0.55%
- 所以系統本身就有約 250 步的「自然 ramp」

**3B 跳升的拆解：**
- CL 跟 BY 比：15% 時領先 −11.9%，20% 時 −4.5%。但這個領先本來就含有蒸餾收益，而蒸餾收益會隨訓練變小。
- 5% → 15% 間領先每 1% 縮 0.18 pt；45–80% 穩定在約 −10%（其中含 EMA 的貢獻）。據此外插，沒有 EMA 時 20% 的領先約 −11.0 到 −11.4%。
- 所以 EMA 造成的跳升約 **+6.5 到 +6.9 pt**。
- 1B 是 +5.4%（CJ vs CH）到 +6.0%（CF vs BU），而且是在開啟後才 381 步量的，放大倍率還沒到穩態。3B 是開啟後 1,145 步量的，已經到穩態。
- **結論：同樣的放大倍率下，兩個規模的跳升差不多；「3B 跳升更大」主要是拿 BY 當對照的讀法造成的。**

**恢復時間跟著「比例」走，不跟步數：**
- 1B（CJ）在 40% 時追平；3B 從 −4.5 回到 −9.5% 也是在 40% 左右。
- 這表示跳升主要是 lr·β 偏高時抬高的「山壁」成分（River valley #5、MPL 的 LD 項），會隨 lr / β 下降而消失。它不是固定步數的暫態。
- 推論：
  - 幾百步的 ramp 只會改 15–20% 那一段的曲線形狀，最終值幾乎不變。
  - 這跟 AdEMAMix Fig.27（ramp 只會變慢，最終值一樣）以及 re-warming 的文獻（#42–44）一致。
- CG（γ 0.997）沒有跳升，但最後輸 CF 0.09%。代表跳升時「多付的山壁」換來了更多沿河前進，衰減段把山壁收掉之後，這筆帳是划算的。

### 3.5 Tail EMA

- 150M 的 +0.06% / +0.23%，落在文獻的範圍內：衰減到 0 時 EMA 最多 −0.12%，WSM 甚至更差。
- 6B 學到 0 時，最後 1% 的 lr 只有峰值的 0–1.25%，平均的權重跟最終權重幾乎一樣。
- 不必再測。

### 3.6 Warmup

- BR（1% = 76 步）最好，BW（19 步）在第 20 步出現 grad norm 27 的 spike。
- 理論（#45–47）都說 warmup 長度由初期動態或 loss 水準決定，以步數計，不隨總長變。
- 6B 的 1% = 458 步，是 1B 的 6 倍，lr 又是 ×0.8，所以是安全的那一邊。縮短的空間只有約 0.67% 的整段。

---

## 4. 候選改動

收益以 in-training ppl 計，負值代表較好。1B 的雜訊約 0.02%。

### C1. 6B 衰減比例 0.80 → 0.85（lr 維持 ×0.8 時）

- **機制：**
  - 衰減越長，「平均更多步、降低雜訊」的效果越強（EMA 觀點）。
  - 同一個峰值 lr 下，衰減越長，這個 lr 越不容易偏高。
- **證據：**
  - §3.1：1B 校正 lr 後的最佳值 ≥0.86。
  - Schaipp（#6）：lr 調好時最佳是全程；lr 大時越長越好。
  - Straight to Zero（#2）：收益隨 TPP 變大、小批次更有利。
  - Bordelon & Mori（#8）：語言模型屬於全程衰減的那一區。
  - Attia & Koren（#9）：衰減到 0 能降低 lr 設錯的代價。
  - 文獻和 BY 都暗示 6B 的 lr 可能偏高（T3a 的議題），這時衰減越長越有利。
- **預期：**

  | | 預期 | 信心 |
  |---|---|---|
  | 1B | 0 到 −0.05% | 中 |
  | 6B | 0 到 −0.1% | 中低 |

- **交互作用：**
  - **跟 lr 綁在一起。** T3a 若把 6B 的 lr 再降（例如 ×0.65 以下），就維持 0.8。
  - 衰減改從 15% 開始，EMA-Nesterov 開啟（15%）時 β 會立刻開始隨 lr 下降。最初 5% 的放大略少，影響很小。
  - 跟蒸餾沒有直接交互作用。
- **已試過：** BE（0.8）、BL（0.95，但在偏低的 lr 下跑）。0.85–0.9 沒試過。
- **實作：** `DECAY_FRACTION=0.85`（env 已經有，不用改程式）。
- **成本與風險：** 0。最壞情況是 +0.05%（BL 那種情況，但 6B 的 lr 不太可能偏低）。
- **1B 測試**（可選，約 73 分、約 10 H200-h）：
  - CJ 的配方加 `DECAY_FRACTION=0.9`，對照 CJ（15.324）
  - ≤ +0.05%：6B 用 0.85（若 ≤ −0.05%，6B 可用 0.9）
  - ≥ +0.15%：維持 0.8
  - 因為 1B 的 TPP 只有 0.8，文獻預期 6B 會比 1B 更偏向長衰減，所以 1B 打平就足以支持改。

### C2. EMA-Nesterov 以步數決定開啟點：6B 從 15% 提早到約 5%（約 2,290 步）

- **機制：**
  - 開啟點的用意是避開「初期不穩定」，而初期不穩定是以步數計的現象，跟 warmup 同理。
  - 1B 在第 1,145 步開啟是安全的；6B 的 15% 卻要等到第 6,870 步，多浪費了 10% 的整段——而這段正是 β 0.6、lr 在峰值、放大最強的時候。
  - 粗估：β·lr 加權的「額外前進面積」，15–74% 視窗約 0.38，提早到 5% 會再加約 0.15（+39%）。
- **證據：**
  - AdEMAMix Fig.5（#32）：中途開啟慢 EMA 時，越早開最終越好；開啟時的小幅上升會被超越。
  - DiLoCo（#34）：從頭開啟只比從 24k 步開啟差約 0.1 PPL；24k 以前開啟結果都差不多，所以開啟點並不敏感。
  - EMA-Nesterov 原論文（#31）：Tw 取「訓練進入穩定段」的時間點，跟 lr 排程掛鉤，不是固定比例。
  - SNOO（#33）、DiLoCo scaling（#35）：外層 momentum 全程開也有效。
- **預期：**

  | | 預期 | 信心 |
  |---|---|---|
  | 1B（測 8%） | −0.05 到 −0.2% | 低 |
  | 6B | −0.05 到 −0.15% | 低 |

- **交互作用：**
  - 蒸餾前期收益最大，而 CJ 顯示有蒸餾時 EMA 的收益更大。
  - 開啟點越早，跳升也越早出現。
  - 若同時採用 C1，5–15% 是 β 0.6 的穩定段。
- **已試過：** 沒有。150M、1B、3B 都是 15–74%。
- **實作**（`src/ema_nesterov.py`）：
  ```python
  WINDOW = (0.05, 0.74)  # 6B: 5% = 2,290 steps (1B switched on at step 1,145)
  ```
- **成本與風險：**
  - 不增加時間。
  - 開啟期間持續方向的有效步長是 2.5 倍，但只放大低頻、低曲率的方向，不碰高曲率方向，所以發散風險低。
  - 前期的 m 比較吵，跳升可能更大。
- **1B 測試：**
  - CJ 的配方，`WINDOW = (0.08, 0.74)`（第 610 步開啟，比 6B 預計的 2,290 步還早，算是壓力測試），對照 CJ（15.324）
  - ≤ −0.05%：6B 用 5%
  - 在 ±0.05% 內：維持 15%，因為沒有收益就不冒險
  - 出現發散、grad norm 持續大於 5、或跳升超過 +12%：放棄

### C3. β 漸進開啟（線性 ramp，500 步）——當保險，不當收益

- **機制：** 把開啟瞬間的 0 → 1.6 倍改成慢慢爬升，削掉開啟時的暫態。
- **證據（都是間接的）：**
  - 支持 ramp 的：
    - FSGDM（#39）：逐步放大低頻成分最好。
    - NAdam 的 momentum warmup（#40）。
    - AdEMAMix 從頭使用時必須 warmup（#32 Fig.28）。
  - 說 ramp 沒用的：
    - **AdEMAMix 從中途開啟時，ramp 0–100k 步，最終值一樣**（#32 Fig.27）。
    - Kalra（#45）：再加長 warmup 只有邊際好處。
    - §3.4：跳升的大小與恢復時間跟著 lr·β 的比例走，不是暫態。
  - 沒有任何論文直接比較 ramp 與一步開到位。
- **預期：**
  - 1B：最終值 ±0.05%，20% 時的跳升約小 1–3 pt（信心中）
  - 6B：最終值 ±0.03%（6B 的自然 ramp 已經只占 0.55%）
- **交互作用：** 跟 C2 一起用最合理——開得越早，m 越吵，保險越有價值。
- **已試過：** 沒有。CG（γ 0.997）是用另一種方式消除跳升，但輸了 0.09%。
- **實作：**
  ```python
  RAMP = 500  # steps: 2x the EMA's own ~250-step response time

  def lookahead(beta_max, progress, lr_ratio, steps_on):
      """steps_on: steps since the window opened."""
      if not WINDOW[0] < progress <= WINDOW[1]:
          return 0.0
      return beta_max * lr_ratio * min(1.0, steps_on / RAMP)
  ```
  `optimizer.py` 用 `group["step"] − WINDOW[0]·total_steps` 算出 `steps_on`。
- **成本：** 0。**風險：** 少了約 250 步的放大，約 +0.01%。
- **1B 測試：** 只在 C2 也要測時才併測——C2 的配方加上 ramp，對照 C2。最終值差 ≤0.05% 就保留當保險。單獨測不值得一個 slot。

### C4. 6B warmup 1% → 0.5%（229 步）——可選

- **機制：** 所需的 warmup 以步數計。229 步是 1B 的 76 步的 3 倍，而 6B 的 lr 更低。
- **證據：** #45–47；我們的 BR / BW（76 步最好，19 步會 spike）。
- **預期：** 6B −0.0 到 −0.05%，信心低。1B 不能測，1B 已經是 76 步。
- **風險：** spike（BW 的情況）。6B 只能跑一次，代價不對稱。
- **建議：** **維持 1%**，除非 T3a 或其他研究員有理由要改 warmup。

### 不建議（附原因）

| 想法 | 原因 |
|---|---|
| 衰減段或結尾的 LAWA / EMA / checkpoint merge | 衰減到 0 時收益 ≤0.12%（#21），WSM 更差（#22），150M 也輸 |
| schedule-free | 贏的對照組都是 cosine 到 10% 或沒調好的；碰到大批次或調好的 AdamW 就輸或打平（#27–29、#20），還要全部重調 |
| 1-sqrt、(1−x)^1.5、20% 1-sqrt | 只在 ≤25% 的衰減有利；長衰減時面積不夠（§3.3） |
| 衰減到 0.01× 而不是 0 | 只有 210M 有微小收益（#20），跟 Straight to Zero 相反，不值得 |
| AdamW 組用較短的獨立衰減 | 理論（#6）指向全程衰減，沒有支持的證據 |
| γ 改成以整段比例計（6B 約 0.9995） | CF > CG（1B）。原論文說最佳 γ 隨步數上升（#31），但我們的資料相反；維持 0.99，CL 在 3B 用 0.99 也沒問題 |
| 開 β 時降低 lr（#37、#38） | 150M 時 EMA-Nesterov 配較低的 lr 反而變差 |
| cooldown 時調大 AdamW β2（#18） | AdamW 只管 embedding 和 norm，收益可能很小；屬 T3a / T5，列為觀察 |

---

## 5. 6B 排程建議

前提：沿用 CL 的配方與 T3a 決定的 lr（目前是 ×0.8：Muon 1.2e-3、AdamW 6.8e-3，τ 0.39）。

| 項目 | 現在的計畫 | 建議 | 條件 |
|---|---|---|---|
| warmup | 1%（458 步），線性 | **不變** | — |
| 衰減比例 | 最後 80% 線性到 0 | **最後 85% 線性到 0** | lr 維持 ×0.8；lr 再降則維持 0.80；C1 的 1B 測試若 ≥ +0.15% 也維持 0.80 |
| 衰減形狀 | linear | **不變** | — |
| 最終 lr | 0 | 不變 | — |
| EMA-Nesterov β | 0.6 × lr_t/lr_peak | 不變 | — |
| EMA-Nesterov γ | 0.99（以每步計） | 不變 | — |
| EMA-Nesterov 視窗 | 15–74% | **5–74%**（C2 的 1B 測試 ≤ −0.05% 時）；否則 15–74% | 1B 測 8% 開啟 |
| 開啟方式 | 一步開到位 | 可選加 500 步線性 ramp（保險）；搭配 C2 時建議加 | — |
| 權重平均 / schedule-free | 無 | 無 | — |

**監控與保險：**
- 6B 開啟後 1,000 步內，in-training loss 預期比沒開時高約 5–7%（1B / 3B 的經驗），40% 前會追回。
- 若跳升超過 +12%，或 grad norm 持續大於 5，用 PMA-init（#23：平均開啟前的 3 個 checkpoint，然後重啟）當緊急方案。
- 需要每 5% 存一次 checkpoint；目前是否有存，未確認。

**截止前的實驗優先序**（每個 1B 約 73 分、約 10 H200-h）：
1. C2：8% 開啟
2. C1：衰減 0.9（可跟 C2 平行跑）
3. C2 + C3 併測（只在 C2 贏時）

---

## 6. 反證據

**衰減比例偏短的一方：**
- Hägele 的收益在 20% 就平了，訓練越長需要的衰減占比越小（#1）。
- Wen 的 8–12% 差不多（#5）。
- MPL / Tissue 的擬合最佳在 20–30%（#3、#4，§3.2）。
- FSL / Bordelon 的困難區，衰減比例隨 T 縮短（#7、#8）。
- 業界的技術報告多用 10–20% 衰減（非六大會議）。
- 我的回應：這些多半是 AdamW、低 lr、大批次、高 TPP、最低 lr 0.1× 的條件。我們的 1B 實測（0.8 勝 0.5 0.70%）已經直接否定「短衰減」在我們的區間成立。所以不改成更短；但這些證據讓我不建議一步跳到 0.95。

**Straight to Zero 在 2 TPP 時 10× 勝 D2Z（#2）：** 低 TPP 偏好少衰減，跟我們 0.8 TPP 時偏好長衰減相反，代表我們的區間不同（小批次、Muon、τ 很強）。這也提醒：校正 lr 的推論依賴於「曲率在不同衰減下相同」這個近似。

**衰減到 0：** Benchmarking（#20）在 210M 時 0.001–0.01× 略勝正好 0。量級很小。

**C2（提早開啟）：**
- DiLoCo 從頭開始比從 24k 步開始差約 0.1 PPL（#34），這是「開太早會略差」的證據。
- EMA-Nesterov 原論文在 Llama 上反而用了更晚的 30%（#31）。
- 所以 C2 的信心低，必須先在 1B 測。

**γ 以每步計：** 原論文說最佳 γ 隨總步數上升（2500 → 5000：0.99 → 0.995，#31）。照這個趨勢，6B 也許要 0.995。我們的 CF > CG 跟它衝突；兩者的批次和排程都不同。這是 6B 的未定項，但 CL 在 3B 用 0.99 沒出問題。

**ramp：** FSGDM、NAdam 的經驗支持逐步開啟（#39、#40）。只是規模小或不是 LLM，而且沒有人量過最終值的差別。

**Warmup：** Porian（#49）建議 warmup tokens 約等於 N，換到我們是約 20%；Why Do We Need Warm-up 的 Adam 網格裡 10–20% 最好（#48）。兩者都跟 BR 矛盾，是 AdamW 小模型的結果。

**權重平均：**
- PMA、Extra-Merge 在大模型上有收益（#23、#25），但終點 lr 不是 0，或是在 stable 段做。
- LAWA 在高 lr 時收益最大（#30）。
- 這些都不適用於學到 0 的線性衰減。

---

## 7. 結論

1. **衰減比例：**
   - 1B 擬合的 76% 是在偏低的 lr 下得到的；校正後，最佳值 ≥0.86（§3.1）。
   - 文獻（Schaipp、Straight to Zero、Bordelon & Mori）一致指出：lr 調好、雜訊主導、TPP 變高時，衰減應該更長。
   - **6B 建議 0.85**（lr 維持 ×0.8 時；lr 再降則維持 0.8），預期 0 到 −0.1%。
   - 形狀維持 linear。1-sqrt 和 (1−x)^1.5 只適合 ≤25% 的衰減；「20% 1-sqrt」預期輸 2–3%，應從待辦清單移除。
2. **權重平均：** 在學到 0 的衰減下收益 ≤0.12%，有些情況反而更差，跟 150M 一致。不做。
3. **schedule-free：** 沒有證據顯示它勝過調好的 WSD 加 Muon，再加 EMA-Nesterov 就更沒有。不做。
4. **EMA-Nesterov 的開啟：**
   - 跳升在 1B 與 3B 其實差不多（+5.4–6% vs 約 +6.5–6.9%）。恢復時間跟著整段比例走，是「有效 lr 偏高時的山壁成分」，衰減段會收掉。
   - 所以 ramp 預期不會改變最終值（AdEMAMix Fig.27），只能當保險。
   - 有依據、可能真的有收益的是「以步數決定開啟點」：6B 從 15% 提早到約 5%（AdEMAMix Fig.5），但信心低，需要先在 1B 測 8% 開啟。
   - β = 0.6·lr_t/lr_peak、γ 0.99、74% 關閉都維持不變。
5. **Warmup：** 理論說以步數計，6B 的 458 步已經在安全的那一邊。縮到 0.5% 的收益 ≤0.05%，6B 只能跑一次，建議維持 1%。

# T3b 6B 遷移 B：批次大小隨資料量與訓練進度的變化、批次漸增、吞吐量

研究員 5 號，2026-10-17。範圍：01-levers.md 的 T3b（假設 9、10）與吞吐。PDF 與轉出的文字在 `/home/yorukot/ml/papers/pdf/T3b/`（50 份 PDF）。

**一句話結論：** 6B 預設維持固定 131k。BO 的 +1.53% 大部分可以用「批次加倍時 lr 和 Muon momentum 沒換算」來解釋，所以「262k 在 6B 打平」這件事還沒被否定，但也沒被證實；預期差距約 0 ± 0.3%，不值得在沒驗證的情況下換。吞吐量方面，Newton-Schulz 分散到 8 卡是唯一低風險、值得做的項目（−10~16%）；FP8 和 torch.compile 在截止前不划算。

---

## 1. 搜尋紀錄

索引 `/home/yorukot/ml/papers/papers.jsonl`（68,363 篇），`search.py` 不分大小寫，比對標題加摘要。共 47 組 regex，另加 1 次「標題含 batch」的掃描（2024 年以後的 NeurIPS/ICML/ICLR，105 篇，用來補 NeurIPS 2026 缺摘要的缺口）。

| # | regex | 命中 | # | regex | 命中 |
|---|---|---|---|---|---|
| q01 | `critical batch` | 8 | q25 | `\bMFU\b\|model flops utilization\|training throughput.{0,40}(LLM\|language model)` | 11 |
| q02 | `optimal batch.?size` | 11 | q26 | `(warmup\|warm-up).{0,60}(batch size\|critical)` | 1 |
| q03 | `batch.?size (warm.?up\|ramp\|schedul\|increas)` | 13 | q27 | `sequence length (warmup\|...\|schedul)` | 1 |
| q04 | `(increas\|grow)\w* (the )?batch` | 15 | q28 | `batch.?size.{0,120}(language model\|LLM\|pre-?train)` | 22 |
| q05 | `gradient noise scale` | 4 | q29 | `(learning rate\|\blr\b).{0,60}(transfer\|scal).{0,80}batch` | 0 |
| q06 | `surge phenomenon\|batch.{0,40}surge` | 1 | q30 | `hyperparameter transfer\|transfer.{0,20}hyperparameter` | 29 |
| q07 | `small batch.{0,40}(language model\|pretrain\|LLM)` | 1 | q31 | `gradient accumulation` | 12 |
| q08 | `batch size.{0,60}(learning rate\|lr).{0,40}(scal\|rule)` | 2 | q32 | `(SDE\|stochastic differential).{0,80}(adaptive\|adam\|rmsprop)` | 3 |
| q09 | `square.?root (scaling\|rule)\|sqrt.?scaling` | 1 | q33 | `scaling (laws?\|rules?) for (the )?(optimal )?(learning rate\|batch\|hyperparam)` | 4 |
| q10 | `stochastic differential equation.{0,80}(adam\|...\|batch)` | 1 | q34 | `Muon.{0,120}(batch\|token\|compute\|efficien\|scal)` | 26 |
| q11 | `linear scaling rule` | 1 | q35 | `compute.?optimal.{0,80}(batch\|hyperparam\|learning rate)` | 2 |
| q12 | `\bmuon\b` | 81 | q36 | `(stage\|multi.?stage\|progressive\|two.?phase).{0,40}batch` | 6 |
| q13 | `power lines\|weight decay.{0,40}batch\|batch.{0,40}weight decay` | 9 | q37 | `(AdamW\|weight decay).{0,80}(timescale\|time.?scale\|EMA)` | 10 |
| q14 | `\bseesaw\b` | 12 | q38 | `Newton.?Schulz\|polar (decomposition\|factor).{0,80}(gpu\|fast\|efficien)` | 14 |
| q15 | `(large\|larger).?batch.{0,60}(train\|optimi)` | 23 | q39 | `(data.?parallel\|distributed).{0,60}(optimizer\|preconditioner).{0,60}(shard\|...)` | 0 |
| q16 | `(beta2\|β2\|β_2\|second.?moment).{0,80}batch` | 1 | q40 | `learning rate.{0,40}(warm.?up\|warmup)` | 16 |
| q17 | `(fp8\|8-bit float\|float8).{0,80}(train\|pretrain)` | 15 | q41 | `(noise\|signal).?to.?(noise\|signal).{0,60}(gradient\|batch)` | 4 |
| q18 | `(distributed\|shard)\w*.{0,60}(shampoo\|muon\|orthogonal\|newton.?schulz\|dion)` | 5 | q42 | `\btokens? per parameter\|data.?constrained\|over.?training` | 100 |
| q19 | `\bdion\b\|orthonormal.{0,30}update.{0,60}distribut` | 0 | q44 | `distill\w*.{0,80}(gradient variance\|gradient noise\|batch size\|variance reduction)` | 4 |
| q20 | `token horizon\|(data\|dataset) size.{0,60}(hyperparameter\|learning rate\|batch)` | 12 | q45 | `soft (label\|target)s?.{0,80}(variance\|noise\|batch)` | 2 |
| q21 | `batch size.{0,80}(loss\|data\|token).{0,40}(scal\|law)` | 1 | q46 | `(muon\|spectral\|orthogonali).{0,100}(large batch\|batch size\|critical batch)` | 1 |
| q22 | `step law\|hyperparameter scaling law\|scaling law.{0,40}hyperparameter` | 3 | q47 | `(ZeRO\|shard\w*).{0,60}optimizer.{0,60}(state\|step\|update)` | 3 |
| q23 | `adaptive batch\|dynamic batch size\|batch size adapt` | 11 | q48 | `(learning rate\|lr).{0,40}(batch size).{0,40}(square.root\|sqrt\|√\|linear)` | 0 |
| q24 | `(momentum\|ema).{0,60}batch size.{0,60}(scal\|rule)` | 0 | q43 | 標題掃描 `batch`（2024+） | 105 |

- **合計：** 47 組 regex 共 502 次命中，去重後 396 篇標題，加上標題掃描的 105 篇；全部標題都看過。
- **篩選：** 讀了約 70 篇的摘要，下載 43 篇（索引內 42 篇 + 1 篇 KD 變異數），另外從 arXiv 抓了 8 篇六大會議以外的（arXiv 合計 18 次，每次間隔 ≥2 秒，沒遇到 429）。
- **精讀：** 32 篇，用 grep 找擬合式、指數、表格。另有 17 篇只讀摘要和結論段。
- **六大會議以外的（WebSearch / arXiv）：**
  - Practical Efficiency of Muon（Essential AI，2505.02222）
  - Moonlight（2502.16982）
  - Dion（2504.05295）
  - Step Law（2503.04715）
  - DeepSeek LLM（2401.02954）
  - DeepSeek-V3（2412.19437）
  - Optimal Scaling Needs Optimal Norm（2510.03871；網路顯示是 ICML 2026，但不在索引裡）
  - Malladi et al. SDE（NeurIPS 2022，在索引年份之外）
- **引用回查：** 從 Power Lines、CBS Revisited、Seesaw、Deriving HP Scaling 的參考文獻回查到 McCandlish 2018、Malladi 2022、Busbridge 2023、Step Law、von Rütte、Filatov、Gray 2024、Safaryan 2023（KD 是部分降變異數）。
- **沒找到的：**
  - 在約 5 TPP、Muon、衰減到 0 的條件下做批次漸增的研究：沒有；最接近的是 Naganuma（Muon，20 TPP，cosine）
  - 蒸餾對臨界批次的直接量測：沒有，只有 Safaryan 的理論

---

## 2. 論文表

標記：【非】= 六大會議以外；【精】= 精讀；【摘】= 只讀摘要和結論。

### 2a. 最佳或臨界批次隨資料量的變化
| 論文 | 會議 | 設定 | 關鍵數字 | 對我們的意義 |
|---|---|---|---|---|
| How Does Critical Batch Size Scale in Pre-training?（Zhang et al.）【精】 | ICLR 2025 | 85M–1.2B，C4，ctx 512，Adam，固定 lr + 權重 EWA，沒有 wd | 臨界批次 = 「比線性多 20% 資料」的點；Chinchilla 設定 B* = 93.2·N^0.47（序列）；**固定 D 時 B* ∝ N^0.087**（幾乎不變）；302M 只加 D：B* = 22.91·D^0.47 tokens；小批次要調高 β2 | 臨界批次主要跟 D 走、跟 N 無關，正好是我們「固定模型、tokens ×6」的軸；×6 → ×2.3 |
| Power Lines（Bergsma et al.）【精】 | NeurIPS 2025 | 111M–3.3B，ctx 2048，AdamW + µP，每個 B 都調 λ | **B_opt = 0.0306·D^0.383**、**B_crit = 0.0471·D_min^0.462**（序列 × 2048）；τ = B/(ηλD) 在 B_opt ≤ B ≤ B_crit 之間固定（λ ∝ B），**η 不隨 B 放大**（受穩定性限制）；τ_opt = 1.084·TPP^−0.527 | 我們的 τ 規則就是他們的 τ；BO 的「lr 不變、τ 不變」正是 Power Lines 量 B_opt 的做法（在 AdamW 上）；在 D=1B 時 B_opt ≈ 175k tokens、6B ≈ 348k；B_crit 1B ≈ 1.39M、6B ≈ 3.2M |
| Critical Batch Size Revisited（Merrill et al.，AI2）【精】 | NeurIPS 2025 | OLMo 1B / 7B 的 checkpoint 分支訓練 2B tokens 量局部臨界批次 | 臨界批次一開始接近 0，前期快速上升，約 4096 序列 × 4096 tokens（約 16.8M tokens）時趨平，1B 和 7B 一樣；批次漸增：1024 → 2048（168B tokens）→ 4096（503B），每次加倍時 lr ×√2；**loss 比小批次對照好 0.0166（預訓練後）/0.0053（中段訓練後），少 43% 步數**；固定大批次的對照最差 | 漸增在 OLMo 上能減少步數而且不傷 loss；但門檻落在數百 B tokens，我們整段只有 6B |
| Surge Phenomenon（Li et al.）【精】 | NeurIPS 2024 | 理論 + CV / NLP 實驗，Adam 類（看梯度符號） | 最佳 lr 對 B：B ≪ B_noise 時 ∝ √B，B = B_noise 時達到峰值，之後下降；**峰值位置隨訓練往大批次移** | 在 B ≪ B_noise 的區間，批次加倍時 lr 應 ×√2 |
| Optimal Scaling Needs Optimal Norm（Filatov et al.）【非】【精】 | arXiv（ICML 2026） | 最大 1.3B / 138B tokens，Scion（譜範數 LMO，跟 Muon 同類），Llama-3 tokenizer，ctx 4096 | **η* ∝ B^0.62·D^−0.56**，**B* ∝ D^0.45±0.07**；加 momentum 會讓批次敏感度大幅降低 | Muon 類優化器在批次加倍時 lr 也要上調（×1.5 左右） |
| Step Law（Li et al.）【非】【精】 | arXiv 2025 | 大規模網格，ctx 2048，AdamW | **B_opt = 0.58·D^0.571**（tokens）；η = 1.79·N^−0.713·D^0.307 | 1B → 80k、6B → 222k；×6 → ×2.8 |
| DeepSeek LLM【非】【精】 | 技術報告 2024 | 跨模型大小的 compute fit | B_opt = 0.292·C^0.3271，η = 0.3118·C^−0.125；**loss 在 0.25% 內的超參數帶很寬** | 套到我們的模型：1B 約 415k、6B 約 746k；這是跨規模的 fit，固定 N 時不適用；但「最佳點附近很平」值得注意 |
| Deriving Hyperparameter Scaling Laws via Modern Optimization Theory【精】 | NeurIPS 2026 | LMO 收斂界（含 Muon、signSGD）理論 | momentum 固定時 **lr ∝ √B**；b* ∝ T^1/2；**最佳 momentum 1−β ≈ b·T^−1/2**（批次 ×κ ⇒ 1−β ×κ，也就是 token 半衰期固定）；同時調三者時有多種近似最佳的組合；表 1 整理：DeepSeek T^0.327、StepLaw T^0.571、von Rütte T^0.82、Filatov T^0.45 | 理論上支持 √ 規則和 momentum 的 token 半衰期規則；1−β ∝ T^−1/2 也暗示在固定批次下 6B 想要更高的 momentum（但 BD 在 1B 用 0.98 輸 0.88%，沒測過 6B） |
| Normalization Layer Per-Example Gradients … GNS（Gray et al.）【摘】 | NeurIPS 2024 | Cerebras，只用 norm 層估梯度雜訊尺度 | 依雜訊尺度排批次，Chinchilla 模型訓練時間 −18% | 雜訊尺度可以幾乎零成本量到 |
| Scaling Optimal LR Across Token Horizons（Bjorck et al.）【摘】 | ICLR 2025 | 到 1.3B，批次 0.5M / 1M | 批次大時最佳 lr 較高；lr 對 D 的指數在兩個批次下大致相同 | lr 與批次的耦合，在不同訓練長度下大致可分開處理 |

### 2b. 批次漸增
| 論文 | 會議 | 設定 | 關鍵數字 | 意義 |
|---|---|---|---|---|
| Seesaw（Meterez et al.）【精】 | ICLR 2026 | 150M/300M/600M，20 TPP，AdamW wd = 0，cosine，在臨界批次訓練 | 對 normalized SGD（Adam 的代理）：**lr 降 α 倍、批次 ×β，只要 α√β 固定就等價**；Seesaw = 原本要把 lr 減半的地方改成 lr ×1/√2、批次 ×2；跟 cosine 同 FLOPs，**步數 −36%**（理論上限 1−2/π）；最激進只能到 α = √β | 在 lr 衰減段可以用加倍批次取代一部分衰減：(2B, η/√2) ≡ (B, η/2)；相當於 (2B, √2·η) ≡ (B, η) |
| Fast Catch-Up, Late Switching（Wang et al.）【精】 | ICLR 2026 | 50M–0.5B dense（20 TPP）+ 1B/1.1B MoE（0.4T/1T tokens），AdamW，**固定 lr** | 固定資料量下的最佳排程：難的任務前期維持小批次，**晚期才切到大批次**；切換後 loss 很快追上「全程大批次」的軌跡；最佳切換點約在 **70% tokens**；D − P* ∝ D^γ；cosine 衰減下現象仍在；「固定 lr + 批次排程」只跟 cosine / WSD **打平** | 晚切換 > 早切換；但在有 lr 衰減到 0 的情況下，跟「全程小批次」比沒有明確收益 |
| Adaptive Batch Sizes Using Non-Euclidean GNS for Sign and Spectral Descent（Naganuma et al.）【精】 | ICML 2026 | 160M（3.2B tokens）/ 1B（22B tokens）Llama-3，ctx 2048，cosine，15% warmup，wd 0.1，**含 Muon** | 160M Muon：B = 64（131k tokens）**3.3041**，B = 256（524k）**3.3411**，依雜訊尺度漸增（從 64 開始，lr ∝ √B）**3.3061**；達到同 loss 的步數 −66.8%；1B 時 AdamW 漸增追不上對照 | **最接近我們的設定**：Muon 在 131k 也贏 524k（+0.037 nats ≈ +3.8% ppl，跟我們 AS vs AU 的 +4.0% 幾乎一樣）；漸增追平小批次的 loss，步數大減 |
| DeepSeek-V3【非】【精】 | 技術報告 | 14.8T tokens | 前 469B tokens（3%）內批次從 3072 漸增到 15360，之後固定 | 業界做法：漸增只放在最前面的一小段 |
| GPT-3 / Llama 3【非】 | 技術報告 | — | GPT-3 用 McCandlish 的雜訊尺度排批次（見 Merrill 的摘要）；Llama 3 在前期加倍兩次（細節是記憶，沒有重新查證） | 同上 |
| McCandlish et al. 2018（透過 Power Lines 轉述）【非】 | arXiv | — | S/S_min − 1 = (D/D_min − 1)^−1；D = D_min(1 + B/B_crit)；動態批次的理論收益不大 | 用來把我們的 ppl 差距換成「臨界批次」 |

### 2c. 批次改變時其他超參數怎麼換算
| 論文 | 會議 | 規則 | 備註 |
|---|---|---|---|
| Malladi et al., SDEs and Scaling Rules for Adaptive Methods【非（NeurIPS 2022，索引外）】【精】 | NeurIPS 2022 | Adam 批次 ×κ：**η ×√κ，1−β1 ×κ，1−β2 ×κ，ε ÷√κ** | 需要雜訊主導；沒有 wd |
| Adaptive Methods through the Lens of SDEs（Compagnoni et al.）【精】 | ICLR 2025 | AdamW：批次 ×δ ⇒ **η ×√δ、wd θ ×√δ**（η·θ ×δ，也就是 τ 固定）、1−β ×√δ；作者說在 LLM 上比 Malladi 的 β 規則好 | wd 要跟著換，否則漸近 loss 會變 |
| How to Scale Your EMA（Busbridge et al.）【精】 | NeurIPS 2023 | 模型 EMA 的動量：批次 ×κ ⇒ **ρ → ρ^κ** | 用在 EMA-Nesterov 的 γ |
| Small Batch Size Training（Marek et al.）【精】 | NeurIPS 2025 | **β2 的 token 半衰期固定**（β2 → β2^κ）；β1 = 0.9 不用改；**lr 隨批次放大得比 √ 慢**（批次 1 → 1024 時 lr 只要 ×3，約 B^0.16）；30M 模型上，小批次時各優化器的 loss 相近，Muon 的優勢出現在大批次 | AT/AU/AV 的 β2 規則出處 |
| Straight to Zero（Bergsma et al.）【精】 | ICLR 2025 | 610M、20 TPP，AdamW：**用線性衰減到 0 時，最佳 lr 對批次比 10× 衰減穩定得多**；√ 規則是在衰減到 0 的條件下觀察到的，線性規則是在固定 lr 或 WSD 下 | 我們最後 80% 是線性衰減到 0，lr 對批次的敏感度可能比 SDE 預測的小 |
| Resolving Discrepancies in Compute-Optimal Scaling（Porian et al.）【精】 | NeurIPS 2024 | 小批次時一定要調高 AdamW β2（0.99 / 0.999） | 跟 token 半衰期規則一致 |
| Power Lines（同上） | NeurIPS 2025 | AdamW：**調 λ、不要調 η**（λ ∝ B） | 反方：認為 η 不該放大 |

### 2d. Muon 與大批次
| 論文 | 會議 | 結論 |
|---|---|---|
| Practical Efficiency of Muon for Pretraining（Essential AI）【非】【精】 | arXiv 2025 | 100M–4B，批次 128K–16M tokens：AdamW/Muon 達到同 loss 所需 token 比 ≥ 1，而且隨批次**不降反升**；Muon 能用的最佳批次通常 ≥ AdamW，**而且隨訓練進行而變大**；DCLM loss 3.25 時兩者的最佳批次都是 3.5M |
| Moonlight【非】【精】 | 技術報告 2025 | 分散式 Muon（ZeRO-1 式）：通訊量是分散式 AdamW 的 1–1.25 倍；在大規模上優化器延遲只占前向 + 反向的 1–3% |
| Phases of Muon【摘】 | NeurIPS 2026 | 理論：**大批次時 SignSVD（Muon）對資料共變異做 √ 預條件；小批次時小特徵方向退化成 SGD**，收斂變慢 |
| Sharp Capacity Scaling of Spectral Optimizers【摘】 | NeurIPS 2026 | 理論：Muon 飽和的臨界批次比 SGD 大 |
| Compute Efficiency … Stochastic Momentum【摘】 | NeurIPS 2026 | 理論：heavy-ball momentum 能保持 SGD 等級效率的批次範圍比 SGD 大 √κ 倍 |
| MuonBP【摘】 | ICLR 2026 | 模型平行時，Muon 比 AdamW 慢 5–10%；分塊加定期全矩陣正交化，8B 吞吐 +8% |
| NorMuon【摘】 | ICML 2026 | 在 FSDP2 下把正交化分散到各卡並平衡負載 |
| Dion【非】【摘】 | arXiv 2025 | 低秩正交化，主要是大規模下的通訊和計算優勢 |
| Knowledge Distillation Performs Partial Variance Reduction（Safaryan et al.）【精】 | NeurIPS 2023 | 蒸餾的 target = (1−λ)·one-hot + λ·老師，跟我們的形式一樣；**理論上等於部分降低梯度雜訊**（類似 SVRG） |

### 2e. 吞吐量
| 論文 | 會議 | 數字 |
|---|---|---|
| TorchTitan【精】 | ICLR 2025 | Llama 3.1 8B、8×H100：FSDP 6,258 → +compile 6,674（+6.6%）→ +Float8 9,409 tok/s/GPU（比 compile 多 +41%） |
| Towards Fully FP8 GEMM LLM Training at Scale（FOG）【精】 | NeurIPS 2025 | **1.5B：+15–18%**（GH200，ctx 4096）；8B：+34–43%；模型越大收益越大；FOG 要改架構（我們不能改） |
| µnit Scaling【精】 | ICML 2025 | FP8 比 bf16 快 25–33%（64×H100，較大的模型）；要改參數化 |
| Scaling FP8 training to trillion-token LLMs（Fishman et al.）【精】 | ICLR 2025 | 約 200B tokens 之後 SwiGLU 的離群值才會讓 FP8 發散；要改 SwiGLU；+34% 吞吐 |
| Dataset Decomposition【摘】 | NeurIPS 2024 | 依序列長度做課程；我們已經用文件遮罩的 FA3，attention 已經按文件計，收益小 |

---

## 3. 用我們的批次數據檢驗

### 3a. 我們的數據放進文獻的座標
在 D = 1B tokens 時，各篇論文預測的最佳批次（換成 tokens）：
- Step Law：80k
- Power Lines：175k（AdamW、ctx 2048）
- Zhang 的臨界批次：389k
- DeepSeek LLM：415k（跨規模的 fit）
- **我們實測：131k 最好**（在 lr 固定的前提下）

**→ 我們的 131k 落在文獻範圍的中間，沒有異常。**

推到 6B：
| 指數來源 | ×6 的倍數 | 6B 最佳批次 |
|---|---|---|
| DeepSeek C^0.327 | ×1.80 | 236k |
| Power Lines D^0.383 | ×1.99 | 260k |
| Filatov D^0.45 | ×2.24 | 294k |
| Zhang / Power Lines 的臨界批次 D^0.46–0.47 | ×2.29–2.32 | 300k |
| Step Law D^0.571 | ×2.78 | 364k |

**→ 6B 的最佳批次推估約 240–360k，中心值約 280k，最接近的 2 的次方是 262k。** 這是「固定模型、只加 tokens」的軸（Zhang、Power Lines 明確指出跟 N 無關），正是我們的情況。

### 3b. 把 ppl 差距換成臨界批次（McCandlish：D = D_min(1 + B/B_crit)）
用固定模型時 ppl ∝ D^−0.20，把 ppl 差距換成「等效多用的 tokens」：
| 比較 | ppl 差 | 等效多用的資料 | 推出的 B_crit（1B tokens 時） |
|---|---|---|---|
| AT（262k）vs AU（131k），舊配方 | +0.97% | +4.9% | 2.5M |
| AS（524k）vs AU（131k），舊配方 | +4.05% | +21.9% | 1.66M |
| BO（262k）vs BM（131k），調好的配方 | +1.53% | +7.9% | 1.53M |

- **→ 三組一致：B_crit 約 1.5–2.5M tokens。** Power Lines 的公式在 1B tokens 時給 1.39M，**跟我們幾乎一樣。**
  - 這三組都沒有換算 lr 和 momentum，差距可能被高估，所以 1.5–2.5M 是下限。
- 我們的批次（131k–262k）只有 B_crit 的 1/6 到 1/12，所以處在雜訊主導、Seesaw 和 SDE 等價大致成立的區間。
- **BO 的差距在前期最大：** 20% 時 +5.2%，50% 時 +2.3%，75% 時 +1.6%，最後 +1.53%（AT vs AS 也一樣），符合「臨界批次隨訓練變大」。
  - 但「lr 太低造成的時間不足」也會產生同樣的前期落後，兩種解釋分不開（見 3c）。

### 3c. BO 的 +1.53% 有多少是沒換算超參數造成的
SDE 和 LMO 理論的等價關係：在雜訊主導的區間，**(2B, √2·η) ≡ (B, η)**（Malladi、Seesaw、Surge、Deriving HP 都同意這個方向）。
- 也就是說 **BO 的 (262k, η) ≈ (131k, η/√2)**。

用我們自己在 131k 的 lr 曲線估計：
- **1B 的 lr 曲線：**
  - BQ：lr ×1.25，−0.31%
  - BV：lr ×1.5，比最佳差 +0.18%
  - 擬合最佳在 BM lr 的 ×1.23
  - 以 log-lr 的二次式擬合，k ≈ 4.6–7（% / ln²）
- **BO 的等效 lr 只有 131k 最佳 lr 的 0.81/√2 = 0.57 倍**：
  - 預測代價 k·ln(0.57)² ≈ 1.4–2.2%
  - BM 自己在 0.81 倍也有 0.31% 的代價，扣掉後**預測 BO 落後 0.9–1.8%**
  - **實測 +1.53%，落在範圍裡。**
- **Muon momentum 也沒換算：**
  - 每步 0.95 在 262k 時，token 半衰期約 3.5M，等於 131k 時用 0.975
  - BD（131k、0.98，半衰期 4.5M）+0.88%；BX（0.9，半衰期 0.86M）打平
  - 所以 BO 的 momentum 可能又多吃了 0.3–0.5%
- **舊配方的對照（AT）：**
  - AT（262k、lr 1.7e-3）按等價應該接近 AW（131k、Muon lr 1.2e-3）的 17.42
  - AdamW 那組的等效 lr 也偏低，加約 0.3%，預期約 17.47
  - 實測 17.71，**剩下約 +0.8% 沒辦法用超參數解釋**
- **反方（Straight to Zero、Power Lines）：**
  - 衰減到 0 時，最佳 lr 對批次比較不敏感
  - 如果是這樣，lr 的混淆只有約 0–0.3%，BO 的差距大部分是真的

**→ 1B 時「262k 搭配正確換算」vs 131k 的真實差距估計約 0–0.8%，中心值約 +0.4%。BO 不能否定「262k 打平」，也不能證實。**

### 3d. 推到 6B
- **McCandlish 的看法：** 越大越差，但罰則隨 B_crit 變大而縮小
  - B_crit ∝ D^0.46，×6 後 ×2.3
  - 1B 的 +0.4% 到 6B 約剩 **+0.15%**（範圍 0 到 +0.35%）
  - 如果 BO 的 1.53% 全是真的，6B 約 +0.6%（ppl 指數用 KD 的 0.176）
- **最佳批次右移的看法：** 6B 的最佳批次約 280k
  - 131k 在 6B 等於「最佳值的一半」，262k 反而可能比較好，差 0 到 −0.5%
- **蒸餾：** 目標分布較軟，梯度雜訊較小（Safaryan），臨界批次較大，大批次的罰則較小
  - 這點偏向 262k，但沒有直接量測
- **免費的檢驗：** CM（4.5B、131k，預期約 11.75）
  - 如果 CM 落在預測值附近或更好，表示到 4.5B 為止 131k 沒有變成瓶頸
  - 如果比預測差 0.5% 以上，表示 131k 可能已經太小（或者 lr 太高）

**→ 6B 時 131k 和 262k（換算正確）的差距約 0 ± 0.3%，正負號不確定。**

### 3e. 一個重要的耦合：批次和 lr 是同一個旋鈕
- 在雜訊主導的區間，有效的「溫度」是 η/√B，而且以 token 計的漂移和雜訊都只跟 η/√B 有關。
- 所以 6B「262k、lr 不變」≈「131k、lr × 0.71」。
- 如果 T3a 結論是 6B 的 lr 應該比計畫（×0.8）更低，那麼「改 262k、lr 不放大」等於同時做了這件事，而且快 21%。
- **建議 T3a 和 T3b 的結論用 η/√B 表達，避免重複修正。**

### 3f. AV（65k）的崩壞
- **舊的解釋：** lr 太高。但用 3c 的 lr 曲線估，65k 時 lr 只比最佳高 1.4–1.7 倍，應該只有約 +1–2%，**解釋不了 +22.5%。**
- **其他可能的原因：**
  - Muon momentum 每步 0.95，在 65k 時半衰期只剩 0.9M tokens
  - Phases of Muon：小批次時 Muon 的預條件會退化成 SGD
  - 每步的 NorMuon 統計量太吵
- **對 6B 的意義：** 方向是「不要再往小批次走」。這跟 6B 想要更大批次的推論一致。

---

## 4. 6B 批次建議

| 方案 | 預期 ppl（相對固定 131k） | 6B 時間 | 風險 |
|---|---|---|---|
| **固定 131k（建議預設）** | 0（基準） | 6.85 h、54.8 H200-h | 最低：lr、τ、β2 規則都在 131k 驗證到 3B（CL），4.5B 在跑 |
| 固定 262k，全套換算（見 §5） | −0.5% 到 +0.6%，中心約 0 | 約 5.4 h、43 H200-h（BO：−21%） | 一次改 6 個超參數，6B 沒測過；AdamW 組的 lr 放大可能不穩 |
| 固定 196k（24 序列，MICRO_BATCH 3） | −0.3% 到 +0.3% | 約 5.8 h、47 H200-h（約 −15%） | 介於兩者之間的避險；沒跑過 MICRO_BATCH 3 |
| 漸增 131k → 262k（60% tokens 時切換） | −0.2% 到 +0.2% | 約 6.2 h、50 H200-h（約 −9%） | HF Trainer 要改：批次、lr 排程要以 token 計，資料位置要對；EMA-Nesterov 的開啟視窗裡要處理 β 的跳動 |

**建議：6B 用固定 131k（信心約 65%）。理由：**
1. 6B 的時間 54.8 H200-h 在上限 64 以內，換批次唯一確定的好處是時間，但時間不是瓶頸。
2. 1B 時「換算正確的 262k」的真實差距不知道（0–0.8%），6B 的預期差距約 0 ± 0.3%，正負號不確定。
3. 131k 的整套遷移規則已經在 3B 驗證過；262k 要同時改 lr、wd、momentum、兩個 β2、γ，任何一項換算錯，代價都可能超過潛在收益。
4. **只有當 §7 的驗證在 3B 顯示 262k 已經不輸（g3 ≤ 0），才改用 262k。** 在這種情況下 6B 預期會贏約 0.1–0.4%，而且快 21%。

漸增不建議用在 6B 正式跑：預期收益很小（文獻都是「步數減少、loss 打平」，Merrill 的 OLMo 只好 0.005 nats），而且要改 HF Trainer。

---

## 5. 批次漸增或改成 262k 的具體換算

基準是 6B 計畫（131k）：Muon lr 1.2e-3、AdamW lr 6.8e-3、τ 0.39、AdamW β2 0.9873、Muon momentum 0.95、NorMuon β2 0.95、EMA-Nesterov γ 0.99、β_max 0.6。

| 超參數 | 131k | 262k（×2） | 196k（×1.5） | 依據 |
|---|---|---|---|---|
| Muon lr | 1.2e-3 | **×1.19 → 1.43e-3**（保守）或 ×1.41 → 1.70e-3 | ×1.11 → 1.33e-3 | √ 規則（Malladi、Surge、Seesaw、Merrill、Naganuma、LMO 理論；Filatov 甚至是 B^0.62）vs 不放大（Power Lines、Straight to Zero 衰減到 0 時較穩）；取 2^0.25 作為中心值 |
| AdamW lr（embedding、norm） | 6.8e-3 | 同 Muon 的倍數（維持 AdamW/Muon = 5.67） | 同 | AW–AZ 顯示兩組的比例很重要；Power Lines 警告 AdamW 的 lr 受穩定性限制，所以用保守的 ×1.19 |
| wd（每組） | 1/(lr·0.39·steps) | **τ 固定**：步數 ÷2、lr ×s ⇒ wd ×2/s（s=1.19 時 ×1.68） | ×1.5/s | Power Lines（λ ∝ B）、Compagnoni（η·θ ×δ），兩者都等於 τ 固定 |
| AdamW β2 | 0.9873 | **0.9747** | 0.981 | Marek、Malladi（token 半衰期）；AT、BO 已經用過 |
| Muon momentum | 0.95 | **0.90**（0.95²） | 0.926 | Malladi、Deriving HP（1−β ∝ b）；BD（131k 用 0.98 輸 0.88%）表示 token 半衰期變長會傷；BX（0.9 打平）表示變短沒關係 |
| NorMuon β2（每列 EMA） | 0.95 | **0.9025** | 0.926 | token 半衰期 |
| EMA-Nesterov γ | 0.99 | **0.98** | 0.985 | How to Scale Your EMA：ρ → ρ^κ |
| EMA-Nesterov β_max、視窗 | 0.6；15–74% | 不變（放大倍數 1/(1−β) 沒有單位；視窗以比例計） | 不變 | |
| warmup、衰減比例、KD α | 1%、80%、0.8 | 不變（以比例計，token 數一樣） | 不變 | |
| MICRO_BATCH | 2 | 4 | 3 | |

**漸增版本（如果之後要做）：**
- **切換點：** 在 60% tokens，這時線性衰減的 lr 剛好是峰值的 0.5，也就是 Seesaw「原本要把 lr 減半」的位置。
  - 文獻：Wang 在固定 lr 下最佳約 70%；Merrill 則是在臨界批次達到 2 倍時切換。
- **切換時的換算：** 批次 ×2，之後 lr = √2 × 原排程（×1.19 是比較保守的版本），其餘照上表。
  - 60% 時原排程是峰值的 0.5，乘 √2 後是 0.71，低於峰值，不會有穩定性問題。
- **EMA-Nesterov：** 計算 β 用的 lr_t/lr_peak 要用「沒乘 √2 的原排程」，不然 β 會從 0.30 跳到 0.42。
- **實作：** HF Trainer 原生不支援中途換批次。需要：
  - 自己寫 sampler，讓切換後每卡給 4 列
  - lr 排程改成以 token 計
  - checkpoint 和資料位置要以 token 對齊
  - KD 快取用列編號查找，只要列的順序不變就沒問題
- **預期：** ppl 在固定 131k 的 ±0.2% 內，時間 −9%。

---

## 6. 吞吐量選項

**現況分解**（從 AU/AT/AV、BM/BO 擬合每步時間 = a + b·(B/131k)）：
- 擬合：a ≈ 0.21–0.23 s（每步固定成本），b ≈ 0.27–0.28 s（每 131k tokens 的前向 + 反向）
- **→ 在 131k 時，每步固定成本約占步時間的 42%。** 6B 有約 45.8k 步，光固定成本就約 2.8 h。
- **固定成本的組成：**
  - Newton-Schulz：
    - Polar Express 5 次迭代，每次 4m²n + 2m³ 的 FLOPs
    - 1B 的 112 個矩陣合計約 46 TFLOP/步/卡，在 500–700 TFLOPS 下約 65–90 ms，每張卡都算全部矩陣
    - 筆記從 AF 和 BF 推算約占 19–21%
  - 其餘約 0.12–0.15 s：
    - 反向結束後沒有和計算重疊到的梯度 all-reduce（fp32 約 5 GB）
    - 優化器對約 150 個參數逐一執行 fp32 小 kernel 的 Python 迴圈，加上 EMA-Nesterov
    - grad-norm clip、Trainer 的額外開銷

| 選項 | 6B 時間 / H200-h | 節省 | 風險 | 建議 |
|---|---|---|---|---|
| 基準（131k，KD + EMA，從 CL 的 3:25:17 ×2 推算） | 6.85 h / 54.8 | — | — | |
| **Newton-Schulz 分散到 8 卡** | 5.8–6.2 h / 46–49 | **−10~16%** | 低（見下面的作法） | **建議做**，先用 1B 驗證 |
| 先做 100 步的 torch.profiler | — | 找出剩下約 0.13 s 的固定成本 | 無 | 建議（約 5 分鐘、8 卡） |
| 優化器改用 `torch._foreach_*` 或合併 kernel | 可能 −3~8% | | 低到中 | 看 profiler 結果再決定 |
| 批次 262k | 5.4 h / 43 | −21%（BO 實測） | ppl 風險見 §4 | 只在驗證通過後 |
| 漸增 131k → 262k（60%） | 6.2 h / 50 | −9% | 要改 Trainer | 不建議在截止前做 |
| FP8（torchao，只用在 block 的 Linear；lm_head、embedding 保持 bf16） | 約 6.3 h / 50 | −5~10% | 中到高：要配合 torch.compile 才有速度；跟 Liger、FA3 varlen、HF Trainer、KD 的整合沒測過；1.5B 的文獻收益只有 +15–18%，而且我們的 lm_head 占線性層 FLOPs 的 21%、每步固定成本占 42%，這兩部分都吃不到 FP8；品質上 6B tokens 遠低於 FP8 出問題的 200B 等級 | **不建議**（工程時間不夠） |
| torch.compile | −3~7% | | 中（Liger 已經做了大部分 kernel 融合） | 不建議 |
| eval 從 20 次減到 10 次；checkpoint 從 10 次減到 5 次 | −1~2% | | 低 | 可以順手做 |
| NS 分散 + 262k | 約 4.9 h / 39 | −28% | | |

**Newton-Schulz 分散的作法**（只分散 NS，其他不變，對 checkpoint 沒有影響）：
- **對每一種矩陣**（q、k、v、o、gate、up、down，每種 16 個）：
  - 第 r 張卡只算 layer 2r 和 2r+1 那兩個（每卡兩個，剛好平均）
  - `zeropower_via_newtonschulz5` 用 batch 形式一次算 [2, m, n]
  - 再用 `all_gather_into_tensor` 收成 [16, m, n] 的 bf16
  - 每步共 7 次 all-gather，總量約 1.95 GB bf16，NVLink 約 5–8 ms
- **其餘照舊在每張卡上做：** momentum、NorMuon、RMS 縮放、wd、EMA-Nesterov
  - 這樣優化器狀態仍然完全複製在每張卡上
  - HF 只存 rank 0 的 checkpoint 也不會壞
  - Moonlight 的 ZeRO-1 版會把狀態切開，checkpoint 就會有問題
- **驗證：** 同一步的更新跟原本的版本比，差距應該只在 bf16 的誤差範圍內（batch 形式的 matmul 加總順序可能不同）
  - 再跑一個 1B 的對照，ppl 應該在雜訊 0.02% 以內

**換來的空間能做什麼：** token 上限固定、模型固定，時間沒辦法直接換成 ppl。
- 唯一能用時間換的是更小的批次（65k）：
  - 時間：在 NS 分散加上壓低固定成本的情況下，6B 約 52 H200-h，放得進上限
  - **但文獻和 AV 都指向「6B 要更大的批次，而不是更小」，所以不建議。**
- 所以吞吐量的價值是：
  - 給 6B 正式跑留餘量（節點慢、重啟）
  - 讓截止前能多排一次跑

---

## 7. 便宜的驗證實驗（10-18 前能做完）

目的：決定 6B 要不要改 262k。對照組用現有的結果，不用重跑。

| 編號 | 設定 | 時間 | 對照 | 預期 | 失敗條件和下一步 |
|---|---|---|---|---|---|
| **V1** | CJ 的配方（KD 0.9 + EMA），1B tokens、262k；Muon lr ×1.41（2.12e-3）、AdamW ×1.41（1.2e-2）；wd 依 τ 0.39 換算（×1.41）；AdamW β2 0.9747、`MUON_MOMENTUM=0.90`、`NORMUON_BETA2=0.9025`、`EMA_NESTEROV_GAMMA=0.98`、MICRO_BATCH 4 | 約 0.85 h、7 H200-h | CJ 15.324 | +0.1 到 +0.6% | 出現 loss spike（AdamW 組不穩）→ 以 V2 為準 |
| **V2** | 同 V1，但 lr ×1.19（Muon 1.79e-3、AdamW 1.01e-2）、wd ×1.68、momentum 0.93、NorMuon β2 0.929 | 約 0.85 h、7 H200-h | CJ | +0.2 到 +0.8% | V1 和 V2 都 ≥ +1.0% → 1B 的批次罰則是真的 → **6B 維持 131k，停止，不跑 V3** |
| **V3**（V1、V2 中最好的 ≤ +0.6% 才跑） | CL 的配方（KD 0.8 + EMA），3B、262k，用 V1/V2 中較好的換算 | 約 2.7 h、22 H200-h | CL 12.635 | 差距 g3 ≈ 0.5 × g1（約 0 到 +0.3%），或 ≤ 0（最佳批次右移的看法） | 外推 6B：**g6 = g3·(g3/g1)^0.63**（以 log D 線性外推）。**g6 ≤ −0.1% → 6B 改 262k**（用同一套換算）；否則維持 131k |
| V0（免費） | CM（4.5B、131k）的結果 | 進行中 | 預測 11.75 | 落在預測值附近 | 比預測差 ≥ 0.5% → 131k 可能已經太小（或 lr 太高），提高 V3 的優先順序 |
| T1（吞吐，要寫程式） | NS 分散後跑 300 步計時，再加一次 1B 完整的對照 | 約 5 分鐘加 1.1 h | BU 或 CJ | 步時間 −10~16%，ppl 相差 ≤ 0.05% | ppl 相差 > 0.1% → 有 bug，不要用在 6B |

- **成本：** V1 + V2 平行跑，約 1 小時。V3 接在後面，約 2.7 小時。全部 4 小時內完成，約 36 H200-h。
- **如果 GPU 時間很緊：** 全部跳過，6B 照 131k 跑。這是 §4 的預設，預期損失約 0 ± 0.3%。
- **順手可以加：** V1、V2、V3 都在 log 裡記錄梯度雜訊尺度。
  - 作法：每張卡的局部梯度範數平方，對照 all-reduce 後的全域範數平方（McCandlish 的兩批次估計；Naganuma 的分散式技巧）
  - 用途：直接量 B_noise，取代 §3b 的間接推算

---

## 8. 反證據，以及結論為什麼不同

1. **Marek et al.：越小越好，lr 對批次只要 ~B^0.16**
   - 說法：只要 β2 用 token 半衰期，小批次一直到 1 都不會比較差，而且 lr 對批次的放大比 √ 慢得多
   - 對我們的意義：如果這是對的，「最佳批次右移」只是 Power Lines 沒調 β2 造成的假象；臨界批次的罰則是單調的，6B 時 131k 仍然小幅勝出（+0.15~0.6%）
   - 為什麼不同：Marek 主要是 AdamW、SGD，模型 30M–1.3B，批次到很小；Power Lines 承認小批次的 β 沒調好
   - 我們的 AV 在 Muon 上反而大輸，可能是 Muon 特有的問題（見下一點）
2. **Muon 在小批次的處境不同**
   - 說法：Phases of Muon、Essential AI、Marek 的 Fig 1a 都說 Muon 的優勢在大批次，小批次時會退化
   - 對我們的意義：Muon 的最佳批次可能比 AdamW 大
   - 為什麼不同：Essential AI 的區間是數十到數百 B tokens，最佳批次約 3.5M；我們是 1–6B tokens
   - 注意：這點跟「131k 在 1B 最好」不矛盾（Naganuma 在 160M 的 Muon 也是 131k 贏 524k），但支持「6B 不要更小」
3. **Straight to Zero、Power Lines：衰減到 0 時 lr 不太需要隨批次調**
   - 對我們的意義：如果是這樣，§3c 的 lr 混淆只剩約 0.3%，BO 的 +1.53% 大部分是真的，6B 時 262k 約 +0.6%
   - 為什麼不同：
     - 他們用 AdamW + µP，lr 受穩定性限制
     - SDE 系列假設 lr 遠離穩定邊界、而且雜訊主導
     - 我們的 Muon lr 曲線在高側很平（BV ×1.5 只差 +0.18%），所以兩邊都說得通
   - V1 和 V2 就是要分開這兩種情況
4. **AT 的殘差 +0.8%**
   - 舊配方的 262k 在扣掉 lr 和 momentum 的混淆後，仍然落後約 0.8%
   - 這說明 1B 時有一部分真實的批次罰則，是 §3c 中心值 +0.4% 的上半部
5. **Fast Catch-Up 與 Seesaw 的前提**
   - Wang 的最佳化是在固定 lr 下做的；Seesaw 的 wd 是 0，而且以 cosine 為基準；兩者的 TPP 都是 20
   - 加上衰減到 0、τ 固定、約 5 TPP 之後，「晚切換贏全程小批次」沒有直接證據
   - Seesaw 說最激進只能 α = √β，也就是 lr 不能比原排程高；Merrill 和 Naganuma 則在加倍時把 lr ×√2 也成功了
   - 兩者衝突的原因：Seesaw 的證明假設基準跑在穩定邊界
6. **McCandlish（透過 Power Lines 轉述）：動態批次的理論收益很小**
   - 業界（DeepSeek-V3）的漸增只放在前 3% tokens，主要是為了系統效率，不是為了 loss
7. **臨界批次的定義不同造成數字相差 5–10 倍**
   - Zhang：+20% 資料、固定 lr + EWA、沒有 wd、ctx 512
   - Power Lines：2 倍 D_min、每個批次都調 λ
   - Merrill：局部分支、loss 容許 0.01
   - B_opt（最低 loss）和 B_crit（時間換資料的轉折點）是不同的量
   - 我們關心的是固定 token 下的最低 loss，所以 B_opt 和「罰則項 B/B_crit」比較相關
8. **梯度範數的弱證據互相矛盾**
   - AV（65k）的梯度範數 0.25–0.3，AS（524k）0.15–0.3，只差約 1.3 倍，不是雜訊主導預期的 2.8 倍
   - 這組數字推出 B_noise ≈ 100k，跟 loss 推出的約 1.5M 矛盾
   - 但兩次跑的狀態和有效 lr 都不同，這個比較不可靠，所以要用 §7 的方法直接量

---

## 9. 結論

- **6B 的最佳批次：** 文獻推估是 1B 的 2–2.8 倍，約 240–360k。
  - 但我們 1B 的證據（BO +1.53%）主要被沒換算的 lr（√ 規則）和 Muon momentum（token 半衰期）污染
  - 扣掉之後，真實差距約 0–0.8%
  - 推到 6B 後，131k 和 262k 相差約 0 ± 0.3%
- **建議：** 6B 用**固定 131k**（信心約 65%）。
  - 只有在 V1/V2（1B）≤ +0.6%，而且 V3（3B）外推的 g6 ≤ −0.1% 時，才改用 262k
  - 改的時候必須用 §5 的整套換算：lr ×1.19–1.41、τ 固定、AdamW β2 0.9747、momentum 0.90、NorMuon β2 0.9025、γ 0.98
- **批次漸增：** 在我們的區間沒有能降 ppl 的證據。
  - 文獻一致的結果是「步數少 9–67%、loss 打平」
  - 6B 時只省約 9%，又要改 HF Trainer，所以截止前不做
- **批次與 lr 是同一件事：** 有效溫度是 η/√B，所以 T3a 的 lr 結論要跟這裡一起看。
  - 「262k、lr 不變」≈「131k、lr × 0.71」
- **吞吐量：**
  - 建議做 Newton-Schulz 分散到 8 卡（−10~16%，低風險，不影響 checkpoint）
  - 先用 profiler 找出每步約 0.13 s 的其他固定成本
  - FP8 和 torch.compile 在截止前不划算
  - 6B 照 131k 跑是 54.8 H200-h，上限內還有餘量
# T3b 6B 遷移 B：批次大小隨資料量與訓練進度的變化、批次漸增、吞吐量

研究員 5 號，2026-10-17。範圍：01-levers.md 的 T3b（假設 9、10）與吞吐。PDF 與轉出的文字在 `/home/yorukot/ml/papers/pdf/T3b/`（50 份 PDF）。

**一句話結論：** 6B 預設維持固定 131k。BO 的 +1.53% 大部分可以用「批次加倍時 lr 和 Muon momentum 沒換算」來解釋，所以「262k 在 6B 打平」這件事還沒被否定，但也沒被證實；預期差距約 0 ± 0.3%，不值得在沒驗證的情況下換。吞吐量方面，Newton-Schulz 分散到 8 卡是唯一低風險、值得做的項目（−10~16%）；FP8 和 torch.compile 在截止前不划算。

---

## 1. 搜尋紀錄

索引 `/home/yorukot/ml/papers/papers.jsonl`（68,363 篇），`search.py` 不分大小寫，比對標題加摘要。共 47 組 regex，另加 1 次「標題含 batch」的掃描（2024 年以後的 NeurIPS/ICML/ICLR，105 篇，用來補 NeurIPS 2026 缺摘要的缺口）。

| # | regex | 命中 | # | regex | 命中 |
|---|---|---|---|---|---|
| q01 | `critical batch` | 8 | q25 | `\bMFU\b\|model flops utilization\|training throughput.{0,40}(LLM\|language model)` | 11 |
| q02 | `optimal batch.?size` | 11 | q26 | `(warmup\|warm-up).{0,60}(batch size\|critical)` | 1 |
| q03 | `batch.?size (warm.?up\|ramp\|schedul\|increas)` | 13 | q27 | `sequence length (warmup\|...\|schedul)` | 1 |
| q04 | `(increas\|grow)\w* (the )?batch` | 15 | q28 | `batch.?size.{0,120}(language model\|LLM\|pre-?train)` | 22 |
| q05 | `gradient noise scale` | 4 | q29 | `(learning rate\|\blr\b).{0,60}(transfer\|scal).{0,80}batch` | 0 |
| q06 | `surge phenomenon\|batch.{0,40}surge` | 1 | q30 | `hyperparameter transfer\|transfer.{0,20}hyperparameter` | 29 |
| q07 | `small batch.{0,40}(language model\|pretrain\|LLM)` | 1 | q31 | `gradient accumulation` | 12 |
| q08 | `batch size.{0,60}(learning rate\|lr).{0,40}(scal\|rule)` | 2 | q32 | `(SDE\|stochastic differential).{0,80}(adaptive\|adam\|rmsprop)` | 3 |
| q09 | `square.?root (scaling\|rule)\|sqrt.?scaling` | 1 | q33 | `scaling (laws?\|rules?) for (the )?(optimal )?(learning rate\|batch\|hyperparam)` | 4 |
| q10 | `stochastic differential equation.{0,80}(adam\|...\|batch)` | 1 | q34 | `Muon.{0,120}(batch\|token\|compute\|efficien\|scal)` | 26 |
| q11 | `linear scaling rule` | 1 | q35 | `compute.?optimal.{0,80}(batch\|hyperparam\|learning rate)` | 2 |
| q12 | `\bmuon\b` | 81 | q36 | `(stage\|multi.?stage\|progressive\|two.?phase).{0,40}batch` | 6 |
| q13 | `power lines\|weight decay.{0,40}batch\|batch.{0,40}weight decay` | 9 | q37 | `(AdamW\|weight decay).{0,80}(timescale\|time.?scale\|EMA)` | 10 |
| q14 | `\bseesaw\b` | 12 | q38 | `Newton.?Schulz\|polar (decomposition\|factor).{0,80}(gpu\|fast\|efficien)` | 14 |
| q15 | `(large\|larger).?batch.{0,60}(train\|optimi)` | 23 | q39 | `(data.?parallel\|distributed).{0,60}(optimizer\|preconditioner).{0,60}(shard\|...)` | 0 |
| q16 | `(beta2\|β2\|β_2\|second.?moment).{0,80}batch` | 1 | q40 | `learning rate.{0,40}(warm.?up\|warmup)` | 16 |
| q17 | `(fp8\|8-bit float\|float8).{0,80}(train\|pretrain)` | 15 | q41 | `(noise\|signal).?to.?(noise\|signal).{0,60}(gradient\|batch)` | 4 |
| q18 | `(distributed\|shard)\w*.{0,60}(shampoo\|muon\|orthogonal\|newton.?schulz\|dion)` | 5 | q42 | `\btokens? per parameter\|data.?constrained\|over.?training` | 100 |
| q19 | `\bdion\b\|orthonormal.{0,30}update.{0,60}distribut` | 0 | q44 | `distill\w*.{0,80}(gradient variance\|gradient noise\|batch size\|variance reduction)` | 4 |
| q20 | `token horizon\|(data\|dataset) size.{0,60}(hyperparameter\|learning rate\|batch)` | 12 | q45 | `soft (label\|target)s?.{0,80}(variance\|noise\|batch)` | 2 |
| q21 | `batch size.{0,80}(loss\|data\|token).{0,40}(scal\|law)` | 1 | q46 | `(muon\|spectral\|orthogonali).{0,100}(large batch\|batch size\|critical batch)` | 1 |
| q22 | `step law\|hyperparameter scaling law\|scaling law.{0,40}hyperparameter` | 3 | q47 | `(ZeRO\|shard\w*).{0,60}optimizer.{0,60}(state\|step\|update)` | 3 |
| q23 | `adaptive batch\|dynamic batch size\|batch size adapt` | 11 | q48 | `(learning rate\|lr).{0,40}(batch size).{0,40}(square.root\|sqrt\|√\|linear)` | 0 |
| q24 | `(momentum\|ema).{0,60}batch size.{0,60}(scal\|rule)` | 0 | q43 | 標題掃描 `batch`（2024+） | 105 |

- **合計：** 47 組 regex 共 593 次命中，去重後 396 篇標題，加上標題掃描的 105 篇；全部標題都看過。
- **篩選：** 讀了約 70 篇的摘要，下載 43 篇（索引內 42 篇 + 1 篇 KD 變異數），另外從 arXiv 抓了 8 篇六大會議以外的（arXiv 合計 18 次，每次間隔 ≥2 秒，沒遇到 429）。
- **精讀：** 32 篇，用 grep 找擬合式、指數、表格。另有 17 篇只讀摘要和結論段。
- **六大會議以外的（WebSearch / arXiv）：**
  - Practical Efficiency of Muon（Essential AI，2505.02222）
  - Moonlight（2502.16982）
  - Dion（2504.05295）
  - Step Law（2503.04715）
  - DeepSeek LLM（2401.02954）
  - DeepSeek-V3（2412.19437）
  - Optimal Scaling Needs Optimal Norm（2510.03871；網路顯示是 ICML 2026，但不在索引裡）
  - Malladi et al. SDE（NeurIPS 2022，在索引年份之外）
- **引用回查：** 從 Power Lines、CBS Revisited、Seesaw、Deriving HP Scaling 的參考文獻回查到 McCandlish 2018、Malladi 2022、Busbridge 2023、Step Law、von Rütte、Filatov、Gray 2024、Safaryan 2023（KD 是部分降變異數）。
- **沒找到的：**
  - 在約 5 TPP、Muon、衰減到 0 的條件下做批次漸增的研究：沒有；最接近的是 Naganuma（Muon，20 TPP，cosine）
  - 蒸餾對臨界批次的直接量測：沒有，只有 Safaryan 的理論

---

## 2. 論文表

標記：【非】= 六大會議以外；【精】= 精讀；【摘】= 只讀摘要和結論。

### 2a. 最佳或臨界批次隨資料量的變化
| 論文 | 會議 | 設定 | 關鍵數字 | 對我們的意義 |
|---|---|---|---|---|
| How Does Critical Batch Size Scale in Pre-training?（Zhang et al.）【精】 | ICLR 2025 | 85M–1.2B，C4，ctx 512，Adam，固定 lr + 權重 EWA，沒有 wd | 臨界批次 = 「比線性多 20% 資料」的點；Chinchilla 設定 B* = 93.2·N^0.47（序列）；**固定 D 時 B* ∝ N^0.087**（幾乎不變）；302M 只加 D：B* = 22.91·D^0.47 tokens；小批次要調高 β2 | 臨界批次主要跟 D 走、跟 N 無關，正好是我們「固定模型、tokens ×6」的軸；×6 → ×2.3 |
| Power Lines（Bergsma et al.）【精】 | NeurIPS 2025 | 111M–3.3B，ctx 2048，AdamW + µP，每個 B 都調 λ | **B_opt = 0.0306·D^0.383**、**B_crit = 0.0471·D_min^0.462**（序列 × 2048）；τ = B/(ηλD) 在 B_opt ≤ B ≤ B_crit 之間固定（λ ∝ B），**η 不隨 B 放大**（受穩定性限制）；τ_opt = 1.084·TPP^−0.527 | 我們的 τ 規則就是他們的 τ；BO 的「lr 不變、τ 不變」正是 Power Lines 量 B_opt 的做法（在 AdamW 上）；在 D=1B 時 B_opt ≈ 175k tokens、6B ≈ 348k；B_crit 1B ≈ 1.39M、6B ≈ 3.2M |
| Critical Batch Size Revisited（Merrill et al.，AI2）【精】 | NeurIPS 2025 | OLMo 1B / 7B 的 checkpoint 分支訓練 2B tokens 量局部臨界批次 | 臨界批次一開始接近 0，前期快速上升，約 4096 序列 × 4096 tokens（約 16.8M tokens）時趨平，1B 和 7B 一樣；批次漸增：1024 → 2048（168B tokens）→ 4096（503B），每次加倍時 lr ×√2；**loss 比小批次對照好 0.0166（預訓練後）/0.0053（中段訓練後），少 43% 步數**；固定大批次的對照最差 | 漸增在 OLMo 上能減少步數而且不傷 loss；但門檻落在數百 B tokens，我們整段只有 6B |
| Surge Phenomenon（Li et al.）【精】 | NeurIPS 2024 | 理論 + CV / NLP 實驗，Adam 類（看梯度符號） | 最佳 lr 對 B：B ≪ B_noise 時 ∝ √B，B = B_noise 時達到峰值，之後下降；**峰值位置隨訓練往大批次移** | 在 B ≪ B_noise 的區間，批次加倍時 lr 應 ×√2 |
| Optimal Scaling Needs Optimal Norm（Filatov et al.）【非】【精】 | arXiv（ICML 2026） | 最大 1.3B / 138B tokens，Scion（譜範數 LMO，跟 Muon 同類），Llama-3 tokenizer，ctx 4096 | **η* ∝ B^0.62·D^−0.56**，**B* ∝ D^0.45±0.07**；加 momentum 會讓批次敏感度大幅降低 | Muon 類優化器在批次加倍時 lr 也要上調（×1.5 左右） |
| Step Law（Li et al.）【非】【精】 | arXiv 2025 | 大規模網格，ctx 2048，AdamW | **B_opt = 0.58·D^0.571**（tokens）；η = 1.79·N^−0.713·D^0.307 | 1B → 80k、6B → 222k；×6 → ×2.8 |
| DeepSeek LLM【非】【精】 | 技術報告 2024 | 跨模型大小的 compute fit | B_opt = 0.292·C^0.3271，η = 0.3118·C^−0.125；**loss 在 0.25% 內的超參數帶很寬** | 套到我們的模型：1B 約 415k、6B 約 746k；這是跨規模的 fit，固定 N 時不適用；但「最佳點附近很平」值得注意 |
| Deriving Hyperparameter Scaling Laws via Modern Optimization Theory【精】 | NeurIPS 2026 | LMO 收斂界（含 Muon、signSGD）理論 | momentum 固定時 **lr ∝ √B**；b* ∝ T^1/2；**最佳 momentum 1−β ≈ b·T^−1/2**（批次 ×κ ⇒ 1−β ×κ，也就是 token 半衰期固定）；同時調三者時有多種近似最佳的組合；表 1 整理：DeepSeek T^0.327、StepLaw T^0.571、von Rütte T^0.82、Filatov T^0.45 | 理論上支持 √ 規則和 momentum 的 token 半衰期規則；1−β ∝ T^−1/2 也暗示在固定批次下 6B 想要更高的 momentum（但 BD 在 1B 用 0.98 輸 0.88%，沒測過 6B） |
| Normalization Layer Per-Example Gradients … GNS（Gray et al.）【摘】 | NeurIPS 2024 | Cerebras，只用 norm 層估梯度雜訊尺度 | 依雜訊尺度排批次，Chinchilla 模型訓練時間 −18% | 雜訊尺度可以幾乎零成本量到 |
| Scaling Optimal LR Across Token Horizons（Bjorck et al.）【摘】 | ICLR 2025 | 到 1.3B，批次 0.5M / 1M | 批次大時最佳 lr 較高；lr 對 D 的指數在兩個批次下大致相同 | lr 與批次的耦合，在不同訓練長度下大致可分開處理 |

### 2b. 批次漸增
| 論文 | 會議 | 設定 | 關鍵數字 | 意義 |
|---|---|---|---|---|
| Seesaw（Meterez et al.）【精】 | ICLR 2026 | 150M/300M/600M，20 TPP，AdamW wd = 0，cosine，在臨界批次訓練 | 對 normalized SGD（Adam 的代理）：**lr 降 α 倍、批次 ×β，只要 α√β 固定就等價**；Seesaw = 原本要把 lr 減半的地方改成 lr ×1/√2、批次 ×2；跟 cosine 同 FLOPs，**步數 −36%**（理論上限 1−2/π）；最激進只能到 α = √β | 在 lr 衰減段可以用加倍批次取代一部分衰減：(2B, η/√2) ≡ (B, η/2)；相當於 (2B, √2·η) ≡ (B, η) |
| Fast Catch-Up, Late Switching（Wang et al.）【精】 | ICLR 2026 | 50M–0.5B dense（20 TPP）+ 1B/1.1B MoE（0.4T/1T tokens），AdamW，**固定 lr** | 固定資料量下的最佳排程：難的任務前期維持小批次，**晚期才切到大批次**；切換後 loss 很快追上「全程大批次」的軌跡；最佳切換點約在 **70% tokens**；D − P* ∝ D^γ；cosine 衰減下現象仍在；「固定 lr + 批次排程」只跟 cosine / WSD **打平** | 晚切換 > 早切換；但在有 lr 衰減到 0 的情況下，跟「全程小批次」比沒有明確收益 |
| Adaptive Batch Sizes Using Non-Euclidean GNS for Sign and Spectral Descent（Naganuma et al.）【精】 | ICML 2026 | 160M（3.2B tokens）/ 1B（22B tokens）Llama-3，ctx 2048，cosine，15% warmup，wd 0.1，**含 Muon** | 160M Muon：B = 64（131k tokens）**3.3041**，B = 256（524k）**3.3411**，依雜訊尺度漸增（從 64 開始，lr ∝ √B）**3.3061**；達到同 loss 的步數 −66.8%；1B 時 AdamW 漸增追不上對照 | **最接近我們的設定**：Muon 在 131k 也贏 524k（+0.037 nats ≈ +3.8% ppl，跟我們 AS vs AU 的 +4.0% 幾乎一樣）；漸增追平小批次的 loss，步數大減 |
| DeepSeek-V3【非】【精】 | 技術報告 | 14.8T tokens | 前 469B tokens（3%）內批次從 3072 漸增到 15360，之後固定 | 業界做法：漸增只放在最前面的一小段 |
| GPT-3 / Llama 3【非】 | 技術報告 | — | GPT-3 用 McCandlish 的雜訊尺度排批次（見 Merrill 的摘要）；Llama 3 在前期加倍兩次（細節是記憶，沒有重新查證） | 同上 |
| McCandlish et al. 2018（透過 Power Lines 轉述）【非】 | arXiv | — | S/S_min − 1 = (D/D_min − 1)^−1；D = D_min(1 + B/B_crit)；動態批次的理論收益不大 | 用來把我們的 ppl 差距換成「臨界批次」 |

### 2c. 批次改變時其他超參數怎麼換算
| 論文 | 會議 | 規則 | 備註 |
|---|---|---|---|
| Malladi et al., SDEs and Scaling Rules for Adaptive Methods【非（NeurIPS 2022，索引外）】【精】 | NeurIPS 2022 | Adam 批次 ×κ：**η ×√κ，1−β1 ×κ，1−β2 ×κ，ε ÷√κ** | 需要雜訊主導；沒有 wd |
| Adaptive Methods through the Lens of SDEs（Compagnoni et al.）【精】 | ICLR 2025 | AdamW：批次 ×δ ⇒ **η ×√δ、wd θ ×√δ**（η·θ ×δ，也就是 τ 固定）、1−β ×√δ；作者說在 LLM 上比 Malladi 的 β 規則好 | wd 要跟著換，否則漸近 loss 會變 |
| How to Scale Your EMA（Busbridge et al.）【精】 | NeurIPS 2023 | 模型 EMA 的動量：批次 ×κ ⇒ **ρ → ρ^κ** | 用在 EMA-Nesterov 的 γ |
| Small Batch Size Training（Marek et al.）【精】 | NeurIPS 2025 | **β2 的 token 半衰期固定**（β2 → β2^κ）；β1 = 0.9 不用改；**lr 隨批次放大得比 √ 慢**（批次 1 → 1024 時 lr 只要 ×3，約 B^0.16）；30M 模型上，小批次時各優化器的 loss 相近，Muon 的優勢出現在大批次 | AT/AU/AV 的 β2 規則出處 |
| Straight to Zero（Bergsma et al.）【精】 | ICLR 2025 | 610M、20 TPP，AdamW：**用線性衰減到 0 時，最佳 lr 對批次比 10× 衰減穩定得多**；√ 規則是在衰減到 0 的條件下觀察到的，線性規則是在固定 lr 或 WSD 下 | 我們最後 80% 是線性衰減到 0，lr 對批次的敏感度可能比 SDE 預測的小 |
| Resolving Discrepancies in Compute-Optimal Scaling（Porian et al.）【精】 | NeurIPS 2024 | 小批次時一定要調高 AdamW β2（0.99 / 0.999） | 跟 token 半衰期規則一致 |
| Power Lines（同上） | NeurIPS 2025 | AdamW：**調 λ、不要調 η**（λ ∝ B） | 反方：認為 η 不該放大 |

### 2d. Muon 與大批次
| 論文 | 會議 | 結論 |
|---|---|---|
| Practical Efficiency of Muon for Pretraining（Essential AI）【非】【精】 | arXiv 2025 | 100M–4B，批次 128K–16M tokens：AdamW/Muon 達到同 loss 所需 token 比 ≥ 1，而且隨批次**不降反升**；Muon 能用的最佳批次通常 ≥ AdamW，**而且隨訓練進行而變大**；DCLM loss 3.25 時兩者的最佳批次都是 3.5M |
| Moonlight【非】【精】 | 技術報告 2025 | 分散式 Muon（ZeRO-1 式）：通訊量是分散式 AdamW 的 1–1.25 倍；在大規模上優化器延遲只占前向 + 反向的 1–3% |
| Phases of Muon【摘】 | NeurIPS 2026 | 理論：**大批次時 SignSVD（Muon）對資料共變異做 √ 預條件；小批次時小特徵方向退化成 SGD**，收斂變慢 |
| Sharp Capacity Scaling of Spectral Optimizers【摘】 | NeurIPS 2026 | 理論：Muon 飽和的臨界批次比 SGD 大 |
| Compute Efficiency … Stochastic Momentum【摘】 | NeurIPS 2026 | 理論：heavy-ball momentum 能保持 SGD 等級效率的批次範圍比 SGD 大 √κ 倍 |
| MuonBP【摘】 | ICLR 2026 | 模型平行時，Muon 比 AdamW 慢 5–10%；分塊加定期全矩陣正交化，8B 吞吐 +8% |
| NorMuon【摘】 | ICML 2026 | 在 FSDP2 下把正交化分散到各卡並平衡負載 |
| Dion【非】【摘】 | arXiv 2025 | 低秩正交化，主要是大規模下的通訊和計算優勢 |
| Knowledge Distillation Performs Partial Variance Reduction（Safaryan et al.）【精】 | NeurIPS 2023 | 蒸餾的 target = (1−λ)·one-hot + λ·老師，跟我們的形式一樣；**理論上等於部分降低梯度雜訊**（類似 SVRG） |

### 2e. 吞吐量
| 論文 | 會議 | 數字 |
|---|---|---|
| TorchTitan【精】 | ICLR 2025 | Llama 3.1 8B、8×H100：FSDP 6,258 → +compile 6,674（+6.6%）→ +Float8 9,409 tok/s/GPU（比 compile 多 +41%） |
| Towards Fully FP8 GEMM LLM Training at Scale（FOG）【精】 | NeurIPS 2025 | **1.5B：+15–18%**（GH200，ctx 4096）；8B：+34–43%；模型越大收益越大；FOG 要改架構（我們不能改） |
| µnit Scaling【精】 | ICML 2025 | FP8 比 bf16 快 25–33%（64×H100，較大的模型）；要改參數化 |
| Scaling FP8 training to trillion-token LLMs（Fishman et al.）【精】 | ICLR 2025 | 約 200B tokens 之後 SwiGLU 的離群值才會讓 FP8 發散；要改 SwiGLU；+34% 吞吐 |
| Dataset Decomposition【摘】 | NeurIPS 2024 | 依序列長度做課程；我們已經用文件遮罩的 FA3，attention 已經按文件計，收益小 |

---

## 3. 用我們的批次數據檢驗

### 3a. 我們的數據放進文獻的座標
在 D = 1B tokens 時，各篇論文預測的最佳批次（換成 tokens）：
- Step Law：80k
- Power Lines：175k（AdamW、ctx 2048）
- Zhang 的臨界批次：389k
- DeepSeek LLM：415k（跨規模的 fit）
- **我們實測：131k 最好**（在 lr 固定的前提下）

**→ 我們的 131k 落在文獻範圍的中間，沒有異常。**

推到 6B：
| 指數來源 | ×6 的倍數 | 6B 最佳批次 |
|---|---|---|
| DeepSeek C^0.327 | ×1.80 | 236k |
| Power Lines D^0.383 | ×1.99 | 260k |
| Filatov D^0.45 | ×2.24 | 294k |
| Zhang / Power Lines 的臨界批次 D^0.46–0.47 | ×2.29–2.32 | 300k |
| Step Law D^0.571 | ×2.78 | 364k |

**→ 6B 的最佳批次推估約 240–360k，中心值約 280k，最接近的 2 的次方是 262k。** 這是「固定模型、只加 tokens」的軸（Zhang、Power Lines 明確指出跟 N 無關），正是我們的情況。

### 3b. 把 ppl 差距換成臨界批次（McCandlish：D = D_min(1 + B/B_crit)）
用固定模型時 ppl ∝ D^−0.20，把 ppl 差距換成「等效多用的 tokens」：
| 比較 | ppl 差 | 等效多用的資料 | 推出的 B_crit（1B tokens 時） |
|---|---|---|---|
| AT（262k）vs AU（131k），舊配方 | +0.97% | +4.9% | 2.5M |
| AS（524k）vs AU（131k），舊配方 | +4.05% | +21.9% | 1.66M |
| BO（262k）vs BM（131k），調好的配方 | +1.53% | +7.9% | 1.53M |

- **→ 三組一致：B_crit 約 1.5–2.5M tokens。** Power Lines 的公式在 1B tokens 時給 1.39M，**跟我們幾乎一樣。**
  - 這三組都沒有換算 lr 和 momentum，差距可能被高估，所以 1.5–2.5M 是下限。
- 我們的批次（131k–262k）只有 B_crit 的 1/6 到 1/12，所以處在雜訊主導、Seesaw 和 SDE 等價大致成立的區間。
- **BO 的差距在前期最大：** 20% 時 +5.2%，50% 時 +2.3%，75% 時 +1.6%，最後 +1.53%（AT vs AS 也一樣），符合「臨界批次隨訓練變大」。
  - 但「lr 太低造成的時間不足」也會產生同樣的前期落後，兩種解釋分不開（見 3c）。

### 3c. BO 的 +1.53% 有多少是沒換算超參數造成的
SDE 和 LMO 理論的等價關係：在雜訊主導的區間，**(2B, √2·η) ≡ (B, η)**（Malladi、Seesaw、Surge、Deriving HP 都同意這個方向）。
- 也就是說 **BO 的 (262k, η) ≈ (131k, η/√2)**。

用我們自己在 131k 的 lr 曲線估計：
- **1B 的 lr 曲線：**
  - BQ：lr ×1.25，−0.31%
  - BV：lr ×1.5，比最佳差 +0.18%
  - 擬合最佳在 BM lr 的 ×1.23
  - 以 log-lr 的二次式擬合，k ≈ 4.6–7（% / ln²）
- **BO 的等效 lr 只有 131k 最佳 lr 的 0.81/√2 = 0.57 倍**：
  - 預測代價 k·ln(0.57)² ≈ 1.4–2.2%
  - BM 自己在 0.81 倍也有 0.31% 的代價，扣掉後**預測 BO 落後 0.9–1.8%**
  - **實測 +1.53%，落在範圍裡。**
- **Muon momentum 也沒換算：**
  - 每步 0.95 在 262k 時，token 半衰期約 3.5M，等於 131k 時用 0.975
  - BD（131k、0.98，半衰期 4.5M）+0.88%；BX（0.9，半衰期 0.86M）打平
  - 所以 BO 的 momentum 可能又多吃了 0.3–0.5%
- **舊配方的對照（AT）：**
  - AT（262k、lr 1.7e-3）按等價應該接近 AW（131k、Muon lr 1.2e-3）的 17.42
  - AdamW 那組的等效 lr 也偏低，加約 0.3%，預期約 17.47
  - 實測 17.71，**剩下約 +0.8% 沒辦法用超參數解釋**
- **反方（Straight to Zero、Power Lines）：**
  - 衰減到 0 時，最佳 lr 對批次比較不敏感
  - 如果是這樣，lr 的混淆只有約 0–0.3%，BO 的差距大部分是真的

**→ 1B 時「262k 搭配正確換算」vs 131k 的真實差距估計約 0–0.8%，中心值約 +0.4%。BO 不能否定「262k 打平」，也不能證實。**

### 3d. 推到 6B
- **McCandlish 的看法：** 越大越差，但罰則隨 B_crit 變大而縮小
  - B_crit ∝ D^0.46，×6 後 ×2.3
  - 1B 的 +0.4% 到 6B 約剩 **+0.15%**（範圍 0 到 +0.35%）
  - 如果 BO 的 1.53% 全是真的，6B 約 +0.6%（ppl 指數用 KD 的 0.176）
- **最佳批次右移的看法：** 6B 的最佳批次約 280k
  - 131k 在 6B 等於「最佳值的一半」，262k 反而可能比較好，差 0 到 −0.5%
- **蒸餾：** 目標分布較軟，梯度雜訊較小（Safaryan），臨界批次較大，大批次的罰則較小
  - 這點偏向 262k，但沒有直接量測
- **免費的檢驗：** CM（4.5B、131k，預期約 11.75）
  - 如果 CM 落在預測值附近或更好，表示到 4.5B 為止 131k 沒有變成瓶頸
  - 如果比預測差 0.5% 以上，表示 131k 可能已經太小（或者 lr 太高）

**→ 6B 時 131k 和 262k（換算正確）的差距約 0 ± 0.3%，正負號不確定。**

### 3e. 一個重要的耦合：批次和 lr 是同一個旋鈕
- 在雜訊主導的區間，有效的「溫度」是 η/√B，而且以 token 計的漂移和雜訊都只跟 η/√B 有關。
- 所以 6B「262k、lr 不變」≈「131k、lr × 0.71」。
- 如果 T3a 結論是 6B 的 lr 應該比計畫（×0.8）更低，那麼「改 262k、lr 不放大」等於同時做了這件事，而且快 21%。
- **建議 T3a 和 T3b 的結論用 η/√B 表達，避免重複修正。**

### 3f. AV（65k）的崩壞
- **舊的解釋：** lr 太高。但用 3c 的 lr 曲線估，65k 時 lr 只比最佳高 1.4–1.7 倍，應該只有約 +1–2%，**解釋不了 +22.5%。**
- **其他可能的原因：**
  - Muon momentum 每步 0.95，在 65k 時半衰期只剩 0.9M tokens
  - Phases of Muon：小批次時 Muon 的預條件會退化成 SGD
  - 每步的 NorMuon 統計量太吵
- **對 6B 的意義：** 方向是「不要再往小批次走」。這跟 6B 想要更大批次的推論一致。

---

## 4. 6B 批次建議

| 方案 | 預期 ppl（相對固定 131k） | 6B 時間 | 風險 |
|---|---|---|---|
| **固定 131k（建議預設）** | 0（基準） | 6.85 h、54.8 H200-h | 最低：lr、τ、β2 規則都在 131k 驗證到 3B（CL），4.5B 在跑 |
| 固定 262k，全套換算（見 §5） | −0.5% 到 +0.6%，中心約 0 | 約 5.4 h、43 H200-h（BO：−21%） | 一次改 6 個超參數，6B 沒測過；AdamW 組的 lr 放大可能不穩 |
| 固定 196k（24 序列，MICRO_BATCH 3） | −0.3% 到 +0.3% | 約 5.8 h、47 H200-h（約 −15%） | 介於兩者之間的避險；沒跑過 MICRO_BATCH 3 |
| 漸增 131k → 262k（60% tokens 時切換） | −0.2% 到 +0.2% | 約 6.2 h、50 H200-h（約 −9%） | HF Trainer 要改：批次、lr 排程要以 token 計，資料位置要對；EMA-Nesterov 的開啟視窗裡要處理 β 的跳動 |

**建議：6B 用固定 131k（信心約 65%）。理由：**
1. 6B 的時間 54.8 H200-h 在上限 64 以內，換批次唯一確定的好處是時間，但時間不是瓶頸。
2. 1B 時「換算正確的 262k」的真實差距不知道（0–0.8%），6B 的預期差距約 0 ± 0.3%，正負號不確定。
3. 131k 的整套遷移規則已經在 3B 驗證過；262k 要同時改 lr、wd、momentum、兩個 β2、γ，任何一項換算錯，代價都可能超過潛在收益。
4. **只有當 §7 的驗證在 3B 顯示 262k 已經不輸（g3 ≤ 0），才改用 262k。** 在這種情況下 6B 預期會贏約 0.1–0.4%，而且快 21%。

漸增不建議用在 6B 正式跑：預期收益很小（文獻都是「步數減少、loss 打平」，Merrill 的 OLMo 只好 0.005 nats），而且要改 HF Trainer。

---

## 5. 批次漸增或改成 262k 的具體換算

基準是 6B 計畫（131k）：Muon lr 1.2e-3、AdamW lr 6.8e-3、τ 0.39、AdamW β2 0.9873、Muon momentum 0.95、NorMuon β2 0.95、EMA-Nesterov γ 0.99、β_max 0.6。

| 超參數 | 131k | 262k（×2） | 196k（×1.5） | 依據 |
|---|---|---|---|---|
| Muon lr | 1.2e-3 | **×1.19 → 1.43e-3**（保守）或 ×1.41 → 1.70e-3 | ×1.11 → 1.33e-3 | √ 規則（Malladi、Surge、Seesaw、Merrill、Naganuma、LMO 理論；Filatov 甚至是 B^0.62）vs 不放大（Power Lines、Straight to Zero 衰減到 0 時較穩）；取 2^0.25 作為中心值 |
| AdamW lr（embedding、norm） | 6.8e-3 | 同 Muon 的倍數（維持 AdamW/Muon = 5.67） | 同 | AW–AZ 顯示兩組的比例很重要；Power Lines 警告 AdamW 的 lr 受穩定性限制，所以用保守的 ×1.19 |
| wd（每組） | 1/(lr·0.39·steps) | **τ 固定**：步數 ÷2、lr ×s ⇒ wd ×2/s（s=1.19 時 ×1.68） | ×1.5/s | Power Lines（λ ∝ B）、Compagnoni（η·θ ×δ），兩者都等於 τ 固定 |
| AdamW β2 | 0.9873 | **0.9747** | 0.981 | Marek、Malladi（token 半衰期）；AT、BO 已經用過 |
| Muon momentum | 0.95 | **0.90**（0.95²） | 0.926 | Malladi、Deriving HP（1−β ∝ b）；BD（131k 用 0.98 輸 0.88%）表示 token 半衰期變長會傷；BX（0.9 打平）表示變短沒關係 |
| NorMuon β2（每列 EMA） | 0.95 | **0.9025** | 0.926 | token 半衰期 |
| EMA-Nesterov γ | 0.99 | **0.98** | 0.985 | How to Scale Your EMA：ρ → ρ^κ |
| EMA-Nesterov β_max、視窗 | 0.6；15–74% | 不變（放大倍數 1/(1−β) 沒有單位；視窗以比例計） | 不變 | |
| warmup、衰減比例、KD α | 1%、80%、0.8 | 不變（以比例計，token 數一樣） | 不變 | |
| MICRO_BATCH | 2 | 4 | 3 | |

**漸增版本（如果之後要做）：**
- **切換點：** 在 60% tokens，這時線性衰減的 lr 剛好是峰值的 0.5，也就是 Seesaw「原本要把 lr 減半」的位置。
  - 文獻：Wang 在固定 lr 下最佳約 70%；Merrill 則是在臨界批次達到 2 倍時切換。
- **切換時的換算：** 批次 ×2，之後 lr = √2 × 原排程（×1.19 是比較保守的版本），其餘照上表。
  - 60% 時原排程是峰值的 0.5，乘 √2 後是 0.71，低於峰值，不會有穩定性問題。
- **EMA-Nesterov：** 計算 β 用的 lr_t/lr_peak 要用「沒乘 √2 的原排程」，不然 β 會從 0.30 跳到 0.42。
- **實作：** HF Trainer 原生不支援中途換批次。需要：
  - 自己寫 sampler，讓切換後每卡給 4 列
  - lr 排程改成以 token 計
  - checkpoint 和資料位置要以 token 對齊
  - KD 快取用列編號查找，只要列的順序不變就沒問題
- **預期：** ppl 在固定 131k 的 ±0.2% 內，時間 −9%。

---

## 6. 吞吐量選項

**現況分解**（從 AU/AT/AV、BM/BO 擬合每步時間 = a + b·(B/131k)）：
- 擬合：a ≈ 0.21–0.23 s（每步固定成本），b ≈ 0.27–0.28 s（每 131k tokens 的前向 + 反向）
- **→ 在 131k 時，每步固定成本約占步時間的 42%。** 6B 有約 45.8k 步，光固定成本就約 2.8 h。
- **固定成本的組成：**
  - Newton-Schulz：
    - Polar Express 5 次迭代，每次 4m²n + 2m³ 的 FLOPs
    - 1B 的 112 個矩陣合計約 46 TFLOP/步/卡，在 500–700 TFLOPS 下約 65–90 ms，每張卡都算全部矩陣
    - 筆記從 AF 和 BF 推算約占 19–21%
  - 其餘約 0.12–0.15 s：
    - 反向結束後沒有和計算重疊到的梯度 all-reduce（fp32 約 5 GB）
    - 優化器對約 150 個參數逐一執行 fp32 小 kernel 的 Python 迴圈，加上 EMA-Nesterov
    - grad-norm clip、Trainer 的額外開銷

| 選項 | 6B 時間 / H200-h | 節省 | 風險 | 建議 |
|---|---|---|---|---|
| 基準（131k，KD + EMA，從 CL 的 3:25:17 ×2 推算） | 6.85 h / 54.8 | — | — | |
| **Newton-Schulz 分散到 8 卡** | 5.8–6.2 h / 46–49 | **−10~16%** | 低（見下面的作法） | **建議做**，先用 1B 驗證 |
| 先做 100 步的 torch.profiler | — | 找出剩下約 0.13 s 的固定成本 | 無 | 建議（約 5 分鐘、8 卡） |
| 優化器改用 `torch._foreach_*` 或合併 kernel | 可能 −3~8% | | 低到中 | 看 profiler 結果再決定 |
| 批次 262k | 5.4 h / 43 | −21%（BO 實測） | ppl 風險見 §4 | 只在驗證通過後 |
| 漸增 131k → 262k（60%） | 6.2 h / 50 | −9% | 要改 Trainer | 不建議在截止前做 |
| FP8（torchao，只用在 block 的 Linear；lm_head、embedding 保持 bf16） | 約 6.3 h / 50 | −5~10% | 中到高：要配合 torch.compile 才有速度；跟 Liger、FA3 varlen、HF Trainer、KD 的整合沒測過；1.5B 的文獻收益只有 +15–18%，而且我們的 lm_head 占線性層 FLOPs 的 21%、每步固定成本占 42%，這兩部分都吃不到 FP8；品質上 6B tokens 遠低於 FP8 出問題的 200B 等級 | **不建議**（工程時間不夠） |
| torch.compile | −3~7% | | 中（Liger 已經做了大部分 kernel 融合） | 不建議 |
| eval 從 20 次減到 10 次；checkpoint 從 10 次減到 5 次 | −1~2% | | 低 | 可以順手做 |
| NS 分散 + 262k | 約 4.9 h / 39 | −28% | | |

**Newton-Schulz 分散的作法**（只分散 NS，其他不變，對 checkpoint 沒有影響）：
- **對每一種矩陣**（q、k、v、o、gate、up、down，每種 16 個）：
  - 第 r 張卡只算 layer 2r 和 2r+1 那兩個（每卡兩個，剛好平均）
  - `zeropower_via_newtonschulz5` 用 batch 形式一次算 [2, m, n]
  - 再用 `all_gather_into_tensor` 收成 [16, m, n] 的 bf16
  - 每步共 7 次 all-gather，總量約 1.95 GB bf16，NVLink 約 5–8 ms
- **其餘照舊在每張卡上做：** momentum、NorMuon、RMS 縮放、wd、EMA-Nesterov
  - 這樣優化器狀態仍然完全複製在每張卡上
  - HF 只存 rank 0 的 checkpoint 也不會壞
  - Moonlight 的 ZeRO-1 版會把狀態切開，checkpoint 就會有問題
- **驗證：** 同一步的更新跟原本的版本比，差距應該只在 bf16 的誤差範圍內（batch 形式的 matmul 加總順序可能不同）
  - 再跑一個 1B 的對照，ppl 應該在雜訊 0.02% 以內

**換來的空間能做什麼：** token 上限固定、模型固定，時間沒辦法直接換成 ppl。
- 唯一能用時間換的是更小的批次（65k）：
  - 時間：在 NS 分散加上壓低固定成本的情況下，6B 約 52 H200-h，放得進上限
  - **但文獻和 AV 都指向「6B 要更大的批次，而不是更小」，所以不建議。**
- 所以吞吐量的價值是：
  - 給 6B 正式跑留餘量（節點慢、重啟）
  - 讓截止前能多排一次跑

---

## 7. 便宜的驗證實驗（10-18 前能做完）

目的：決定 6B 要不要改 262k。對照組用現有的結果，不用重跑。

| 編號 | 設定 | 時間 | 對照 | 預期 | 失敗條件和下一步 |
|---|---|---|---|---|---|
| **V1** | CJ 的配方（KD 0.9 + EMA），1B tokens、262k；Muon lr ×1.41（2.12e-3）、AdamW ×1.41（1.2e-2）；wd 依 τ 0.39 換算（×1.41）；AdamW β2 0.9747、`MUON_MOMENTUM=0.90`、`NORMUON_BETA2=0.9025`、`EMA_NESTEROV_GAMMA=0.98`、MICRO_BATCH 4 | 約 0.85 h、7 H200-h | CJ 15.324 | +0.1 到 +0.6% | 出現 loss spike（AdamW 組不穩）→ 以 V2 為準 |
| **V2** | 同 V1，但 lr ×1.19（Muon 1.79e-3、AdamW 1.01e-2）、wd ×1.68、momentum 0.93、NorMuon β2 0.929 | 約 0.85 h、7 H200-h | CJ | +0.2 到 +0.8% | V1 和 V2 都 ≥ +1.0% → 1B 的批次罰則是真的 → **6B 維持 131k，停止，不跑 V3** |
| **V3**（V1、V2 中最好的 ≤ +0.6% 才跑） | CL 的配方（KD 0.8 + EMA），3B、262k，用 V1/V2 中較好的換算 | 約 2.7 h、22 H200-h | CL 12.635 | 差距 g3 ≈ 0.5 × g1（約 0 到 +0.3%），或 ≤ 0（最佳批次右移的看法） | 外推 6B：**g6 = g3·(g3/g1)^0.63**（以 log D 線性外推）。**g6 ≤ −0.1% → 6B 改 262k**（用同一套換算）；否則維持 131k |
| V0（免費） | CM（4.5B、131k）的結果 | 進行中 | 預測 11.75 | 落在預測值附近 | 比預測差 ≥ 0.5% → 131k 可能已經太小（或 lr 太高），提高 V3 的優先順序 |
| T1（吞吐，要寫程式） | NS 分散後跑 300 步計時，再加一次 1B 完整的對照 | 約 5 分鐘加 1.1 h | BU 或 CJ | 步時間 −10~16%，ppl 相差 ≤ 0.05% | ppl 相差 > 0.1% → 有 bug，不要用在 6B |

- **成本：** V1 + V2 平行跑，約 1 小時。V3 接在後面，約 2.7 小時。全部 4 小時內完成，約 36 H200-h。
- **如果 GPU 時間很緊：** 全部跳過，6B 照 131k 跑。這是 §4 的預設，預期損失約 0 ± 0.3%。
- **順手可以加：** V1、V2、V3 都在 log 裡記錄梯度雜訊尺度。
  - 作法：每張卡的局部梯度範數平方，對照 all-reduce 後的全域範數平方（McCandlish 的兩批次估計；Naganuma 的分散式技巧）
  - 用途：直接量 B_noise，取代 §3b 的間接推算

---

## 8. 反證據，以及結論為什麼不同

1. **Marek et al.：越小越好，lr 對批次只要 ~B^0.16**
   - 說法：只要 β2 用 token 半衰期，小批次一直到 1 都不會比較差，而且 lr 對批次的放大比 √ 慢得多
   - 對我們的意義：如果這是對的，「最佳批次右移」只是 Power Lines 沒調 β2 造成的假象；臨界批次的罰則是單調的，6B 時 131k 仍然小幅勝出（+0.15~0.6%）
   - 為什麼不同：Marek 主要是 AdamW、SGD，模型 30M–1.3B，批次到很小；Power Lines 承認小批次的 β 沒調好
   - 我們的 AV 在 Muon 上反而大輸，可能是 Muon 特有的問題（見下一點）
2. **Muon 在小批次的處境不同**
   - 說法：Phases of Muon、Essential AI、Marek 的 Fig 1a 都說 Muon 的優勢在大批次，小批次時會退化
   - 對我們的意義：Muon 的最佳批次可能比 AdamW 大
   - 為什麼不同：Essential AI 的區間是數十到數百 B tokens，最佳批次約 3.5M；我們是 1–6B tokens
   - 注意：這點跟「131k 在 1B 最好」不矛盾（Naganuma 在 160M 的 Muon 也是 131k 贏 524k），但支持「6B 不要更小」
3. **Straight to Zero、Power Lines：衰減到 0 時 lr 不太需要隨批次調**
   - 對我們的意義：如果是這樣，§3c 的 lr 混淆只剩約 0.3%，BO 的 +1.53% 大部分是真的，6B 時 262k 約 +0.6%
   - 為什麼不同：
     - 他們用 AdamW + µP，lr 受穩定性限制
     - SDE 系列假設 lr 遠離穩定邊界、而且雜訊主導
     - 我們的 Muon lr 曲線在高側很平（BV ×1.5 只差 +0.18%），所以兩邊都說得通
   - V1 和 V2 就是要分開這兩種情況
4. **AT 的殘差 +0.8%**
   - 舊配方的 262k 在扣掉 lr 和 momentum 的混淆後，仍然落後約 0.8%
   - 這說明 1B 時有一部分真實的批次罰則，是 §3c 中心值 +0.4% 的上半部
5. **Fast Catch-Up 與 Seesaw 的前提**
   - Wang 的最佳化是在固定 lr 下做的；Seesaw 的 wd 是 0，而且以 cosine 為基準；兩者的 TPP 都是 20
   - 加上衰減到 0、τ 固定、約 5 TPP 之後，「晚切換贏全程小批次」沒有直接證據
   - Seesaw 說最激進只能 α = √β，也就是 lr 不能比原排程高；Merrill 和 Naganuma 則在加倍時把 lr ×√2 也成功了
   - 兩者衝突的原因：Seesaw 的證明假設基準跑在穩定邊界
6. **McCandlish（透過 Power Lines 轉述）：動態批次的理論收益很小**
   - 業界（DeepSeek-V3）的漸增只放在前 3% tokens，主要是為了系統效率，不是為了 loss
7. **臨界批次的定義不同造成數字相差 5–10 倍**
   - Zhang：+20% 資料、固定 lr + EWA、沒有 wd、ctx 512
   - Power Lines：2 倍 D_min、每個批次都調 λ
   - Merrill：局部分支、loss 容許 0.01
   - B_opt（最低 loss）和 B_crit（時間換資料的轉折點）是不同的量
   - 我們關心的是固定 token 下的最低 loss，所以 B_opt 和「罰則項 B/B_crit」比較相關
8. **梯度範數的弱證據互相矛盾**
   - AV（65k）的梯度範數 0.25–0.3，AS（524k）0.15–0.3，只差約 1.3 倍，不是雜訊主導預期的 2.8 倍
   - 這組數字推出 B_noise ≈ 100k，跟 loss 推出的約 1.5M 矛盾
   - 但兩次跑的狀態和有效 lr 都不同，這個比較不可靠，所以要用 §7 的方法直接量

---

## 9. 結論

- **6B 的最佳批次：** 文獻推估是 1B 的 2–2.8 倍，約 240–360k。
  - 但我們 1B 的證據（BO +1.53%）主要被沒換算的 lr（√ 規則）和 Muon momentum（token 半衰期）污染
  - 扣掉之後，真實差距約 0–0.8%
  - 推到 6B 後，131k 和 262k 相差約 0 ± 0.3%
- **建議：** 6B 用**固定 131k**（信心約 65%）。
  - 只有在 V1/V2（1B）≤ +0.6%，而且 V3（3B）外推的 g6 ≤ −0.1% 時，才改用 262k
  - 改的時候必須用 §5 的整套換算：lr ×1.19–1.41、τ 固定、AdamW β2 0.9747、momentum 0.90、NorMuon β2 0.9025、γ 0.98
- **批次漸增：** 在我們的區間沒有能降 ppl 的證據。
  - 文獻一致的結果是「步數少 9–67%、loss 打平」
  - 6B 時只省約 9%，又要改 HF Trainer，所以截止前不做
- **批次與 lr 是同一件事：** 有效溫度是 η/√B，所以 T3a 的 lr 結論要跟這裡一起看。
  - 「262k、lr 不變」≈「131k、lr × 0.71」
- **吞吐量：**
  - 建議做 Newton-Schulz 分散到 8 卡（−10~16%，低風險，不影響 checkpoint）
  - 先用 profiler 找出每步約 0.13 s 的其他固定成本
  - FP8 和 torch.compile 在截止前不划算
  - 6B 照 131k 跑是 54.8 H200-h，上限內還有餘量

# 20 引用查核（第 2 波）

- 日期：2026-10-17
- 範圍：00-context 的任務清單第 1–20 項全部做完，另外補了幾條報告裡的附帶主張
- 依據：
  - 本地全文 `~/ml/papers/pdf/<軌>/*.txt`，頁碼以檔案裡的 `=== page N ===` 為準
  - 另外下載的檔案放在 `~/ml/papers/pdf/verify/`：
    - Shen *Power Scheduler*（arXiv 2408.13359）的 pdf 和 txt
    - modded-nanogpt Track 3 的 #11、#24、#46 紀錄 log（`track3/r11.txt`、`r24.txt`、`r46.txt`，從 GitHub raw 下載）
- 判定的定義：
  - 相符：數字和條件都對
  - 部分相符：數字對，但條件、指標或歸因有落差
  - 誇大：原文的說法比報告弱
  - 不符：原文說的跟報告不一樣
  - 找不到：原文裡沒有這句話

## 1. 核對表

### 1.1 蒸餾（T1a、T1b）

| # | 主張 | 出處 | 報告頁碼 | 我找到的頁碼與原文 | 判定 | 備註 |
|---|---|---|---|---|---|---|
| 1a | Naive Fix 在 K=20/50/100 拿到 FullKD 收益的 73%/83%/97% | SLS（ACL 2025） | Table 2 p.4 | p.4 Table 2：「Naive Fix: Remaining Probability to Ground Truth … Top-k 20 2.83 2.76 73 … Top-k 50 2.80 2.76 83 … Top-k 100 2.77 2.75 97」 | 相符 | 設定：300M 學生、3B 老師、10B tokens。<br>CE 到 FullKD 的差距只有 0.07 nats（2.81→2.74），K=20 和 K=50 的 loss 都是 2.76；以兩位小數計，每格約 ±7 個百分點，73% 和 83% 的差距落在捨入範圍內。<br>沒有寫 seed 數 |
| 1b | ghost token 59%；label smoothing −73% | SLS | Table 2 p.4 | p.4 Table 2：「Smoothing 2.80 2.85 −73 … Ghost Token 2.80 2.77 59」；說明：「Smoothing … and Ghost Token use 50 tokens.」 | 相符 | 兩者都是 K=50 |
| 1c | RS-KD 12 個樣本 ≈ FullKD | SLS | Table 6 p.6 | p.6 Table 6：12.1 個 unique token 時 2.75，FullKD 2.75。p.18 App. C：「For example, N = 50, we find K = 12.」 | 部分相符 | 12 是 **unique token 數**，實際抽樣 50 次。以快取容量來比時要用 12 |
| 1d | 梯度夾角 top-50 48°、RS 4° | SLS | Table 3 p.5 | p.5 Table 3：「Top-K 50 48° 1.8 … Random Sampling 12 4° 1.0」 | 相符 | 量的是 **vanilla top-K**（tail 不處理），只量一個 batch，不是 Naive Fix。T1a 第 174 行拿它描述我們現在的做法，不能直接套用 |
| 1e | 8B→3B（Table 8）：vanilla top-50 2.40 > CE 2.37 > RS 2.35 ≈ Full 2.34 | SLS | Table 8 p.7 | p.7 Table 8：「CE 2.37 … Top-K 12 2.50 … Top-K 50 2.40 … Ours (12) 2.35 … FullKD 2.34」 | 相符 | 原文只寫「Top-K KD」，沒說是 vanilla 還是 Naive Fix。從 ECE 看比較像 vanilla，但這是推斷 |
| 1f | 「Naive Fix」等於我們的 target | SLS | §3.3 p.4 | p.4：「the probability of the target token is adjusted to ensure that the target probabilities sum up to 1.」p.22 的程式：`naive_fix_dist[:top_k] = data_dist[:top_k]; naive_fix_dist += data_dist * (1 - np.sum(naive_fix_dist))` | 部分相符 | **差異：**<br>- Naive Fix 的老師 top-K 機率不重新正規化，殘差 1−Σ 全給正確答案，而且是純 KD，沒有另外的 CE 項，等於我們的 **α=1**（也就是 CI）<br>- 我們的 α·p_T(top-K) + (1−α·m)·e_y = **α·NaiveFix + (1−α)·e_y**，也就是 Naive Fix 再加上權重 1−α 的 CE<br>- SLS 沒有測過「Naive Fix + CE」；CE 混入只在 RS-KD 上測過（Table 10，LR ratio 1.0 那一列：CE 0.3/0.2/0.1/0 分別是 101/111/95/98%，看不出明確方向） |
| 1g | 「現在做法的偏差有證明：SLS App. A.4 式 (5)」（T1a 第 172 行）；10-candidates 寫「SLS 證明現在的做法（Naive Fix）有偏差」 | SLS | A.4 p.16 | p.16 A.4 的標題是「Vanilla Top-K KD provides scaled teacher as target」，式 (5) 是 vanilla top-K 的梯度。Naive Fix 只在 p.4 有一句定性描述：「The methods above are either biased estimators of the teacher probability distribution, and/or lack adequate supervision in the tail.」 | 誇大 | 式 (5) 證明的是 vanilla top-K，不是 Naive Fix。Naive Fix 的 tail 偏差可以由式 (4) 自己推出來（T1a 算的「tail 只拿到資料機率的 0.28 倍」代數是對的），但那是我們的推導，不是論文證明的 |
| 1h | 「hard token」lr ×2 加 CE 0.1，比 FullKD 多拿 25% | SLS | Table 10 p.8 | p.8 Table 10：LR Ratio 2.0、CE 0.1 → 125 | 相符 | 是 RS-KD 加這兩個技巧，不是 top-K。表的標題寫 8B→300M，內文寫 3B 老師，論文自己前後不一致 |
| 2a | top-k=128 截斷 +0.11 nats | Distillation Scaling Laws（ICML 2025） | G.4 Fig.54 p.63 | p.63：「at k = 128, validation loss increases by 0.11 nats compared to full distillation (k = 32768)」 | 相符 | top-k 有**重新正規化**（式 55），λ=1 |
| 2b | k=128 加 λ=0.7 時差距 <0.01 nats | 同上 | p.63 | p.63：「Using λ = 0.7 with k = 128 maintains performance within 0.01 nats」；「statistically indistinguishable from those obtained using the complete teacher distribution」 | 相符 | 跟我們的設定最接近的證據：截斷後混入 30% 的 CE 就能救回大部分 |
| 2c | reverse KL 比 forward KL 差 0.28 nats | 同上 | G.5 p.63–64 | p.64：「The forward KL achieves a lower data cross-entropy compared to the reverse KL (Table 7), with an average improvement of 0.28 nats.」Table 7：2.42 vs 2.70 | 相符 | 只有一組設定（1.82B→546M） |
| 2d | τ=1 最好 | 同上 | G.2 p.61 | p.61：「We find optimal performance at τ = 1 across all model scales」 | 相符 | |
| 2e | λ>0.5 時曲線平坦 | 同上 | G.1 p.60–61 | p.61：「the student cross-entropy LS remains mostly flat for choices of λ > 0.5」 | 相符 | 弱老師時偏好較小的 λ |
| 2f | M=20 時 λ=1 ≈ λ* | 同上 | p.4 | p.4：「We verify the choice λ = 1 produces results statistically similar to the optimal λ∗」 | 相符 | 完整分布、M=20；Fig.51b 的 λ* 有些明顯小於 1 |
| 3a | KLD 2.107、CLM+KLD 2.117、RKLD 2.140 | Minitron（NeurIPS 2024） | A.5 p.17–18 | p.18 Table 15：「LCLM + Llogits(RKLD) 2.140 … LCLM + Llogits(KLD) 2.117 … Llogits(KLD) 2.107」 | 相符 | RKLD 那列其實是 CLM+RKLD，同條件下該跟 2.117 比（差 0.023）。<br>情境是剪枝後重新訓練（Nemotron-3 8B），不是從頭訓練 |
| 3b | top-K ≤100 掉分 | Minitron | A.5 p.18 | p.18：「a low value of top-K (<=100) results in a significant drop in accuracy.」 | 相符 | 沒有數字。它的 top-K 是老師和學生的 logits 都只留 top-K，跟我們「tail 給答案」不同 |
| 4a | 靜態 α 0.9 最好 | Peng et al.（ACL 2025） | Table 4 p.5 | p.5 Table 4：α 0.5/0.6/0.7/0.8/0.9/0.95/1.0 → 1.5/1.4/2.9/2.0/3.6/2.5/1.6（%）；「with the best performance at α = 0.9」 | 相符 | 數列不單調（0.7 > 0.8 < 0.9），雜訊約 ±1 個百分點 |
| 4b | WSD-α + WSD-LR +8.0%、Linear Dec +4.1% | Peng | Table 5 p.6 | p.6 Table 5：「Linear Dec … ↑4.1% … 1-α+WSD-LR … ↑5.0% … WSD-α+WSD-LR … ↑8.0%」 | 相符 | **WSD-α 的定義（p.6）：**「linearly increase α from 0 to 1.0 during the warm up stage … cosine decay … from 1.0 to 0. We set the warmup ratio at 10% and the decay ratio at 1%.」<br>也就是說 α 只在**最後 1%** 降到 0，lr 也只在最後 1% 衰減。<br>baseline（LLM-LM）是 cosine 降到 10%。WSD-LR 本身就有 +1.2%（0-α+WSD-LR） |
| 4c | 單一 seed、指標是 benchmark | Peng | — | p.3：「we conduct supervised fine-tuning (SFT; Ouyang et al., 2022) with additional 10B high-quality instruct-tuning data after pre-training.」全文沒有 seed 的字樣 | 相符 | 指標是 **10B tokens SFT 之後**的 8 個 benchmark 平均，不是 ppl。老師是 top-p 0.95 加 top-100，有重新正規化 |
| 5a | 純 KD 比最佳 α 差 | Strong Teacher Not Needed（NeurIPS 2026） | Table 5–6 p.18 | p.18：「In almost all configurations, there exists a better loss mixing than pure distillation.」 | 相符 | 8B 老師的 in-domain ppl 差距：老師 tokens 10B −11.5%、30B −1.8%、50B −0.9%、80B −0.6%、100B −0.4%、300B −0.1% |
| 5b | 8B 老師的最佳 α 依老師 tokens 是 0.4/0.6/0.8/1.0/1.0/0.8 | 同上 | p.8 | p.8：「In downstream accuracy, for example, the 8.0B teacher’s optimal α across token budgets (10B, 30B, 50B, 80B, 100B, 300B) is 0.4, 0.6, 0.8, 1.0, 1.0, 0.8」 | 部分相符 | 這是 **downstream 準確率**的最佳 α；T1b 沒寫指標。<br>以 in-domain ppl 來看（Table 6），每個老師 tokens 都是 α<1 比較好，只是差距縮到 0.1–0.9% |
| 5c | 300B tokens、α=0.2：in-domain +0.9% | 同上 | p.21–22 | p.21 Table 9：8.0B 老師的 in-domain ppl +0.9% | 相符 | 學生和老師都訓練 300B tokens，而且只測 α=0.2 |
| 6a | λ* = E⟨∇f_ξ(x*),∇f_ξ(θ)⟩ / (E‖∇f_ξ(θ)‖² + ‖∇f(θ)‖²/γ) | KD Performs Partial Variance Reduction（NeurIPS 2023） | p.7 | p.7 式 (14)，跟報告寫的一樣；p.20 Lemma 1 式 (33) 是一般化的版本（cγ） | 相符 | |
| 6b | 「步長越小 λ* 越小」 | 同上 | p.7 | 論文沒有這句話。論文對 λ* 的解讀（p.8）是：「the better the teacher’s model θ is, the bigger λ∗≤1 gets」 | 部分相符 | 從式 (14) 確實推得出來：只要老師不完美、分子 >0，γ 變小時分母變大。<br>但有三個限制：<br>- 這是固定步長下的穩態鄰域，條件是 self-distillation、凸或 PL，實驗是線性模型<br>- 論文沒討論排程<br>- 照字面套用，lr→0 時 λ*→0，也就是 D2Z 結束時 α 該降到 0，跟 K3 的下限 0.7 以及「不要把 α 降到 0」都不一致 |

### 1.2 超參數遷移與批次（T3a、T3b）

| # | 主張 | 出處 | 報告頁碼 | 我找到的頁碼與原文 | 判定 | 備註 |
|---|---|---|---|---|---|---|
| 7a | τ_opt = 1.084·TPP^−0.527 | Power Lines（NeurIPS 2025） | p.1、4–5 | p.1 Fig.1：「τ = 1.084(TPP)−0.527」；p.5：「A precise power law emerges (R2=0.975) … 10th and 90th percentiles of fitted mτ over all points are (-0.529, -0.507)」 | 相符 | p.24 有一處寫 mτ = −0.520，論文自己前後差一點 |
| 7b | B_opt ∝ D^0.383 | 同上 | — | p.1 Fig.1、p.7 Fig.5：「Bopt(D) = 0.0306D0.383」；p.8：10–90% 區間 (0.367, 0.391) | 相符 | 同一張圖的 B_crit = 0.0471·D^0.462 |
| 7c | 適用範圍：20–1280 TPP、AdamW、D2Z | 同上 | p.3、17 | p.2：「We study TPPs from 20 to 1280.」p.3：「a linear LR schedule, with a 10% warmup followed by decay-to-zero」p.17：「Our work focuses on AdamW」「We present results with a single (standard) learning rate schedule.」 | 相符 | 另外：μP，AdamW β2 0.95，只用一份資料集。我們的 1B→6B 是 0.8–4.8 TPP，全部在範圍外 |
| 8a | Wang & Aitchison 的「τ 固定」來自多 epoch 實驗 | ICML 2025 | p.4–5 | p.5：「We trained the model using the first 1/4, 1/2, and the whole dataset for approximately 4 epochs」；ImageNet 子集跑 100 epoch（p.4–5） | 相符 | |
| 8b | 作者自己在 p.8 改引 Power Lines | 同上 | p.8 | p.8：「Bergsma et al. (2025a) show that optimal τepoch should not stay constant in one epoch setting, but rather follows a power law τ optimal epoch ≈(TPP)−0.527 … and is always smaller than 1 when TPP > 1.」 | 相符 | p.8 也提到 Dey 2025、Narayan 2025 說 τ 能**跨寬度**遷移。那是固定 TPP、換模型大小，不是我們要的「固定模型、加 tokens」 |
| 9a | lr 指數 −0.32 | Bjorck（ICLR 2025） | p.7、14 | p.7 式 (4)：「C ∼1.55 × 10−3, α ∼0.23, β ∼0.32」；「valid in the regime of large models (≥ 760M parameters)」 | 相符 | 各尺寸（p.14）：760M 0.316、1.3B 0.317、2.7B 0.418。條件（p.13）：wd 固定 0.1，cosine 降到 0.1×，批次 0.5M |
| 9b | D2Z 下 lr 幾乎不用動 | Straight to Zero（ICLR 2025） | p.8 | p.8 Finding 3：「optimal LR shifts substantially lower for Constant, somewhat lower for 10×, and hardly at all for D2Z.」 | 相符 | 這些 sweep 的 wd 是**固定**的，所以 τ ∝ 1/D 會自動變小。T3a 第 18 行說「又已經用 τ 調過 wd 的情況下」，不是這篇的條件（Power Lines p.5 才有「只調 λ」的實驗） |
| 9c | Shen 的 −0.51 | Power Scheduler（arXiv 2408.13359，非六大會議） | p.5 | p.5 式 (5)：「γ = 4.6T −0.51」，γ = ηopt/β；p.3：「We use the WSD scheduler with 10% tokens in the decay phase」 | 部分相符 | 指數對。但 T3a 說「WSD 的固定 lr 段」不對，實際是完整的 WSD，最後 10% 有衰減。模型只有 12M/36M/121M |
| 10a | 臨界批次 ∝ D^0.47 | Zhang et al.（ICLR 2025） | — | p.24 Table 7 說明：「B∗= 22.91 × D0.47 where … data size D is in billions.」 | 相符 | 只有 302M 一種模型；臨界的定義是比線性多用 20% 資料。改成 10%/50% 時指數是 0.48/0.44 |
| 10b | Filatov 的 D^0.45 | arXiv 2510.03871（非六大會議） | p.2、8 | p.2：「B∗(D) ∝D0.45±0.07」 | 部分相符 | 數字對，但這是**最佳批次**，不是臨界批次。條件：Scion，沒有 wd、沒有 momentum，固定 lr |
| 10c | Step Law 的 D^0.571 | arXiv 2503.04715（非六大會議） | — | p.2–3：「B(D) = 0.58D0.571」 | 部分相符 | 同 10b，這也是最佳批次。T3b 的表 B 寫的是 B_opt，比較正確；任務清單把它歸成「臨界批次」 |
| 11a | Naganuma：Muon 160M 131k 3.3041 vs 漸增 3.3061、步數 −67% | ICML 2026 | — | p.8 Table 2（10 seeds）：「MUON 3.3041 ± 0.0021 3.3411 ± 0.0019 3.3181 ± 0.0072 3.3061 ± 0.0027 66.77」 | 相符 | −66.77% 是「達到 baseline 最低 loss 所需步數」的中位數。1B 時 AdamW 漸增追不上對照（p.8） |
| 11b | Merrill（OLMo）：好 0.005 nats、步數 −43% | NeurIPS 2025 | — | p.9：「outperforming the small-batch control in loss by a margin of 0.0166. After mid-training … a smaller margin of 0.0053 … using 43% fewer gradient steps.」 | 相符 | |

### 1.3 排程與優化器（T4、T5）

| # | 主張 | 出處 | 報告頁碼 | 我找到的頁碼與原文 | 判定 | 備註 |
|---|---|---|---|---|---|---|
| 12a | AdEMAMix 中途開啟時越早越好 | ICLR 2025 | Fig.5 | p.8 Fig.5 說明：「In both cases, the earlier AdEMAMix is used, the better the final loss.」p.9：「the earlier the switch, the larger the gain, with diminishing returns.」 | 相符 | 機制不同：AdEMAMix 是梯度的慢 EMA（β3 0.9999），EMA-Nesterov 是權重位移的 EMA。拿來支持 K7 只能算類比 |
| 12b | ramp 0–100k 步，最終值一樣 | 同上 | Fig.27 | p.38：「We observe how increasing Tβ3 = Tα only slows down the convergence.」 | 部分相符 | 文字只說「只是收斂變慢」，「最終一樣」是看圖得來的。條件：110M，在 300k 步（約 60%）才切換 |
| 13a | 用 Schaipp 把 1B 最佳衰減比例從 0.76 校正到 ≥0.86 | Schaipp et al.（ICML 2025） | §4.1、Fig.12、Fig.21 | p.5：「if the base learning-rate γ is fully tuned, then the optimal cooldown fraction is 1 (linear decay). For fixed γ, the optimal cooldown fraction can be smaller than one.」p.5：「for large enough γ, the parabola turns into a monotonically decreasing curve.」p.8：「γ(0.994) ≈e0.7γ⋆(0.2)」 | 部分相符 | **方向**有論文支持：lr 偏低時，衰減比例的拋物線頂點會偏短。<br>**0.76→≥0.86 這個數字**是 T4 自己的計算，論文沒有這種「修正擬合」的做法；結果還取決於用哪種 lr 位移模型（0.86 / 0.87 / ≥0.95）。<br>e^0.7 是 AdamW 124M、線性 cooldown 的結果，T4 套到 Muon 上 |
| 13b | 「horizon 變 2 倍時，lr 降為 0.525×，或衰減比例拉到約 0.6（p.7）」 | Schaipp | p.7 | p.6：「our analysis suggests to decrease the schedule by ρ = 0.525 for T2 = 2T1」；「When doubling the training length (T2 = 2T1), the adapted cooldown fraction is roughly c1 = 0.6.」 | 部分相符 | 頁碼是 p.6。0.525 是分段排程 (B2) 裡**延長那一段**的 lr 倍率（前段維持原 lr），而且來自凸理論模擬，不是「整個 lr ×0.525」。T4 沒有拿它做計算，影響小 |
| 14a | 「Secret Sauce」的完整標題和出處 | — | — | *In Search of Adam’s Secret Sauce*，Orvieto & Gower，**NeurIPS 2025**（arXiv 2505.21829，有在索引裡） | 相符 | T5 用的是 arXiv 版 |
| 14b | β1 0.9 配 β2 0.9875 很差，β1 0.975 配同一個 β2 很好 | 同上 | p.5 | p.5 Fig.3 旁的文字：「We notice high correlation between beta values (e.g., β2 = 0.9875 is a terrible option at β1 = 0.9, but a good one at β1 = 0.975). While results are noisy …」 | 相符 | 「很差」的量級：Fig.3 的色階上限約 0.34 ppl，對應 ppl 約 21.9，大約 1.5% |
| 14c | 實驗設定（模型大小、全模型都用 Adam） | 同上 | p.5–6 | p.2 Table 1：「160M parameters, 3.2B SlimPajama tokens, sequence length 2048, batch size 256 – Chinchilla optimal」；p.6 Fig.5 驗證了 410M、8.2B tokens | 相符 | **全模型都用 Adam**，wd 0.1，cosine。我們只有 embedding 和 norm 用 AdamW |
| 15a | Nesterov AdamW 一律勝過 AdamW | Fantastic（ICLR 2026） | p.9 | p.9：「all variance-reduced Adam variants (NAdamW, Mars, Cautious) surpass vanilla AdamW—except for a small lag at the smallest experiment.」 | 誇大 | 「一律」不對，原文有例外，雖然是小例外 |
| 15b | 1.2B 時 Nesterov AdamW ≈ Muon | 同上 | p.9 | p.9：「leading to similar performance between Muon and Nesterov AdamW.」Fig.5：「no longer significantly speed up over NAdamW」 | 相符 | |
| 15c | 低 TPP 時 Muon 最好；SOAP、Kron 要 ≥8× Chinchilla 才超過 | 同上 | p.8–9 | p.3：「while Muon is consistently the best optimizer in smaller Chinchilla ratio regimes, it is outperformed by Kron and Soap when the data-to-model ratio increases to 8x or larger」；p.8 也有同樣的話 | 相符 | 它的 1× Chinchilla 是以非 embedding 參數算 20 TPP |
| 15d | 加速到 1.2B 只剩約 1.1× | 同上 | p.1–3、9 | p.1：「decreases with model size to only 1.1× for 1.2B parameter models」 | 相符 | |
| 16a | Track 3 #24：輔助 Adam 衰減 0.4、矩陣 0.8，3225 → 3175 步，n=10 | modded-nanogpt（非六大會議） | 本地 README 56–103 行 | README #24：「Setup from #11, with split LR cooldown: aux Adam cooldown_frac=0.4 and matrix cooldown_frac=0.8」3175 步，3.2782（n=10）。<br>**#11 的 log：**`def set_hparams(step, cooldown_frac=0.7)`，兩組都是 0.7。#24 對 #11 的 diff 只有 train_steps 和這兩行 cooldown_frac | 部分相符 | 數字對，但 #24 **同時改了兩件事**：Adam 0.7→0.4（變短）、矩陣 0.7→0.8（變長）。<br>省下的 50 步（−1.6%）不能全歸給「Adam 衰減變短」。我們的矩陣組已經是 0.8。<br>沒有對 #11 的 p 值；#11 是 n=16 |
| 16b | Track 3 #46：Adam 組平到 51% 才降，Muon 組 18% 就開始降 | 同上 | — | 從 #46 log 的 PowerCool 常數算（power 1.2，t_end 2900）：embed 和其他 Adam 組 t_cd ≈ 1487（51%），Muon t_cd ≈ 514（18%） | 相符 | #46 本身是「#45 + RowUpdateFloor + CWD」，不是分開衰減的消融。另外 #37（MuonH）把輔助 Adam 的衰減調到 0.85、矩陣 1.0 |

### 1.4 資料、精度（T6、T2、T5）

| # | 主張 | 出處 | 報告頁碼 | 我找到的頁碼與原文 | 判定 | 備註 |
|---|---|---|---|---|---|---|
| 17 | SkyLadder：上下文窗口由短到長的收益、模型大小、是否搭配 doc masking | NeurIPS 2025 | — | p.1：「we pretrain 1B-parameter models (up to 32K context) and 3B-parameter models (8K context) on 100B tokens … consistent gains of up to 3.7% on common benchmarks」。p.6 Table 1：IntraDoc 47.4 → 加 SkyLadder 49.3（+1.9）。p.7 Table 6（32K）：IntraDoc 54.0 → 54.9（+0.9） | 相符 | 3.7% 是對 Random（不遮罩）的收益，疊在 IntraDoc 上只剩 +0.9 到 +1.9 個百分點（平均準確率）。<br>p.4：30B tokens 的池子訓練到 100B（約 3.3 epoch）。<br>主要指標是 benchmark，IntraDoc + SkyLadder 沒有 ppl。T6 的「長文 13.0→13.2」是長文 benchmark 的準確率，不是 ppl |
| 18 | Muon-VS 1.2B 比 NorMuon −0.45%，是否單 seed | arXiv 2601.14603（非六大會議） | 4、14 | p.14 Table 5：「NorMuon 2.7367 15.44 … Muon-VS 2.7323 15.37」；p.17：「we use same-nominal-seed reruns as a repeatability check rather than an independent-seed robustness analysis.」 | 相符 | 確定是單 seed；1.2B 連同 seed 的重跑都沒做（重跑只做了 210M、720M、GPT-2） |
| 19 | Beyond Outliers：Muon 1.5B W4A4 loss 9.7、AdamW 3.8 | ICLR 2026 | Table 6 | p.13 Table 6（row-wise W4A4 PTQ）：1.5B「AdamW … 3.769」「Muon … 9.725」 | 相符 | 50M/125M 時 Muon 反而比較好（4.305 vs 5.445、4.145 vs 4.233）。是 4-bit RTN，跟 bf16 差很多 |
| 20 | 事後量化的損失隨訓練 tokens 增加 | Scaling Laws for Precision（ICLR 2025） | Table 2 p.27 | p.1：「the degradation introduced by post-training quantization increases as models are trained on more data」；p.27 Table 2：C_T 0.0598、γD 0.5068、γN 0.3439、γ 0.5907 | 相符 | T2 引用的係數都對 |

**統計（共 54 列）：**
- 相符 41
- 部分相符 11：1c、1f、5b、6b、9c、10b、10c、12b、13a、13b、16a
- 誇大 2：1g、15a
- 不符 0
- 找不到 0

## 2. 改變結論的發現

### 2.1 該降級的候選

**K8（AdamW 組用較短的衰減）：降級**
- 唯一的直接 A/B 是 Track 3 #24 對 #11
  - #11 兩組的衰減比例都是 0.7
  - #24 同時把 Adam 改成 0.4、矩陣改成 0.8
- 50 步的收益有多少來自「矩陣衰減變長」，無法分開
- 我們的矩陣組已經是 0.8，能拿到的部分可能接近 0
- 之後的紀錄（#46：Adam 約 0.49、Muon 約 0.82；#37：Adam 0.85、矩陣 1.0）都維持「Adam 比矩陣短」，但都沒有消融
- 建議：信心從「低到中」降到「低」，預期改成 0 到 −0.15%；在 10-18 的 1B 批次裡排到後面

**K3（α 跟 lr 連動）：文獻支持變弱，只剩我們自己的數據**
- Peng 的 WSD-α：
  - α 在最後 **1%** 才降到 0，前 10% 從 0 暖身
  - lr 的衰減也只有 1%
  - 跟「α 隨長達 80% 的衰減一起下降」不是同一件事
  - 指標是 SFT 之後的 benchmark、單 seed，Table 4 的數列不單調（雜訊約 ±1 個百分點）
- KD 變異數縮減理論：
  - 式子正確，但條件是固定步長、self-distillation、凸問題
  - 照字面套用，lr→0 時 α 該降到 0，跟 K3 的下限 0.7 不一致
- Strong Teacher 的「α 先升後降」：
  - 是對**老師**的訓練 tokens，不是學生訓練中的時間
  - 而且是 downstream 指標
- 結論：K3 只靠 CD/CE/CH/CI 的擬合（最佳 α 隨時間變小）撐著。信心從「中低」降到「低」，預期 −0.1 到 −0.3% 偏樂觀
- 順帶一提：10-candidates D 段用「CD 的 α 0.5 明顯差」排除「最後把 α 降到 0」，理由不成立
  - CD 是全程固定 0.5，沒有測「只在最後一小段降到 0」
  - Peng 的最佳設定正好包含這件事，只是跟 α 暖身混在一起
  - 這個方向沒有證據說有害，但也沒有證據說有益

**K1（無偏 tail target）：前提的引用要修正，預期收益偏樂觀**
- SLS 的 73%/83% 是 **α=1 的 Naive Fix**（等於我們的 CI），不是我們的 α=0.8
- 把 CE 混進截斷 target 的直接證據是 Busbridge G.4：
  - 重新正規化的 top-128 加 30% CE，跟完整分布差 <0.01 nats
  - 我們 α=0.8 的 target 在精神上接近這個設定，所以剩下的 tail 偏差可能已經很小
  - 這跟 T6 的「真實偏差只有理想值的 1/20」一致
- 「SLS 證明現在的做法有偏差」要改寫：式 (5) 證明的是 vanilla top-K；Naive Fix 的偏差是我們自己用式 (4) 推的
- SLS 的 K=20 和 K=50 都是 2.76，73% 和 83% 的差距落在兩位小數的捨入範圍內，用來內插 K=32 的「78%」精度不高
- 建議：把「45% 機率超過 0.3%」下修到約 30%；先做 K2 的 D0（CPU 從快取算 tail 少給了多少），再決定要不要用 1B 的時間

**K9（AdamW β1 0.9 → 0.975）：維持低，傾向不測**
- Secret Sauce 的句子屬實，但條件是全模型用 Adam（160M/410M）
- Fantastic 的「一律勝過」有例外
- 新發現的反向證據：Track 3 調得最細的設定（Muon + 輔助 Adam，跟我們的結構最像）
  - #11、#36 的輔助 Adam 用 β1 **0.8**、β2 0.95
  - #46 用 β1 0.8、β2 0.99
  - 方向是**更低**的 β1，不是 0.975
- 我們的 (0.9, 0.9873) 剛好是 Secret Sauce 點名很差的組合（只差 0.0002），但那個「很差」是全模型 Adam 時約 1.5% ppl；只作用在 embedding 和 norm 時，效果應該小很多

### 2.2 不受影響或略為加強的

**K4（τ 0.39 → 0.2）：引用全部屬實，結論不變**
- Power Lines 的 −0.527、適用範圍（20–1280 TPP、AdamW、D2Z、μP）都對
- Wang & Aitchison 自己在 p.8 承認單趟訓練時 τ 不該固定
- 剩下的風險跟原本一樣：TPP<20 是外推，而且是 Muon。這一點引用核對解決不了，只能靠 E1（3B 的 τ 測試）或 CM

**K5（Muon lr 1.2e-3 → 1.05e-3）：不變**
- Bjorck −0.32（wd 固定、cosine 到 10%）、Straight to Zero（D2Z 時幾乎不變，wd 固定）、Shen −0.51 的數字都對
- Shen 的條件描述要修正：是 WSD 加 10% 衰減，不是固定段
- Straight to Zero 的「lr 不用動」是在 wd 固定、τ ∝ 1/D 自動變小的情況下測的。這反而說明：τ 固定時，lr 可能需要降得比 D2Z 文獻顯示的多

**K6（衰減比例 0.80 → 0.85）：方向有支持，略為加強**
- Schaipp 的定性結論屬實：lr 調好時越長越好，lr 夠大時也是單調變好
- 0.76→≥0.86 是 T4 自己的計算，結果取決於模型
- Track 3 #24 也把矩陣衰減從 0.7 拉到 0.8，而且有收益（雖然混了變因）；之後的紀錄矩陣衰減都在 0.82–1.0。這是「矩陣組衰減長一點」的旁證

**K7（EMA-Nesterov 提早開啟）：不變，維持低**
- 「越早越好」屬實，但原文還說收益遞減
- 「ramp 只變慢、最終一樣」只有看圖
- 機制不同，只能算類比

**K12（批次維持 131k）：不變**
- Naganuma（10 seeds）、Zhang、Merrill 的數字都對
- Filatov 和 Step Law 是最佳批次，不是臨界批次，但不影響「不改 262k」的結論

**K13（上下文窗口由短到長）：維持低，略降**
- 疊在 IntraDoc 上只剩 +0.9 到 +1.9 個百分點的 benchmark 收益
- 訓練是多 epoch，主要指標是 benchmark，在我們的條件（ppl、單趟）下沒有證據

**D 段（不建議清單）：結論不變**
- reverse KL 差 0.28 nats（雖然只有一組設定）、label smoothing −73%、ghost 59%、τ=1 最好都屬實

**T2 的精度結論：不變**
- Beyond Outliers 的 9.725 vs 3.769 是 4-bit RTN；Scaling Laws for Precision 的係數都對
- T2 已經說明不能外推到 bf16，這個判斷正確

## 3. 總結

- 報告裡的**數字幾乎都抄對了**：54 列裡沒有不符，也沒有找不到
- 問題集中在**條件和歸因**，有四處會影響候選：
  1. SLS 的 Naive Fix 等於我們 α=1 的版本，A.4 的證明是 vanilla top-K。K1 的前提要改寫；把 CE 混入截斷 target 的效果，Busbridge 已經顯示接近完整分布，所以 K1 的空間可能比估的小
  2. Peng 的 WSD-α 只在最後 1% 衰減，而且看 SFT 後的 benchmark、單 seed，不能拿來支持 K3 的長時間 α 衰減；K3 只剩我們自己的擬合
  3. Track 3 #24 同時改了 Adam 和矩陣兩組的衰減，K8 的證據被混在一起，對我們（矩陣已經是 0.8）可能接近 0
  4. Track 3 的輔助 Adam 用 β1 0.8，方向跟 K9 相反
- 跟 6B 設定有關的 K4、K5、K6，引用全部成立，風險仍然是「在文獻的 TPP 範圍外、而且是 Muon」，這要靠實驗回答，不是靠查文獻
- 對 10-18 的 1B 批次的含意：
  - K8、K9 往後排
  - K1 先做 D0 診斷再決定
  - K3 若要測，要了解它只有內部數據支持
# 20 引用查核（第 2 波）

- 日期：2026-10-17
- 範圍：00-context 的任務清單第 1–20 項全部做完，另外補了幾條報告裡的附帶主張
- 依據：
  - 本地全文 `~/ml/papers/pdf/<軌>/*.txt`，頁碼以檔案裡的 `=== page N ===` 為準
  - 另外下載的檔案放在 `~/ml/papers/pdf/verify/`：
    - Shen *Power Scheduler*（arXiv 2408.13359）的 pdf 和 txt
    - modded-nanogpt Track 3 的 #11、#24、#46 紀錄 log（`track3/r11.txt`、`r24.txt`、`r46.txt`，從 GitHub raw 下載）
- 判定的定義：
  - 相符：數字和條件都對
  - 部分相符：數字對，但條件、指標或歸因有落差
  - 誇大：原文的說法比報告弱
  - 不符：原文說的跟報告不一樣
  - 找不到：原文裡沒有這句話

## 1. 核對表

### 1.1 蒸餾（T1a、T1b）

| # | 主張 | 出處 | 報告頁碼 | 我找到的頁碼與原文 | 判定 | 備註 |
|---|---|---|---|---|---|---|
| 1a | Naive Fix 在 K=20/50/100 拿到 FullKD 收益的 73%/83%/97% | SLS（ACL 2025） | Table 2 p.4 | p.4 Table 2：「Naive Fix: Remaining Probability to Ground Truth … Top-k 20 2.83 2.76 73 … Top-k 50 2.80 2.76 83 … Top-k 100 2.77 2.75 97」 | 相符 | 設定：300M 學生、3B 老師、10B tokens。<br>CE 到 FullKD 的差距只有 0.07 nats（2.81→2.74），K=20 和 K=50 的 loss 都是 2.76；以兩位小數計，每格約 ±7 個百分點，73% 和 83% 的差距落在捨入範圍內。<br>沒有寫 seed 數 |
| 1b | ghost token 59%；label smoothing −73% | SLS | Table 2 p.4 | p.4 Table 2：「Smoothing 2.80 2.85 −73 … Ghost Token 2.80 2.77 59」；說明：「Smoothing … and Ghost Token use 50 tokens.」 | 相符 | 兩者都是 K=50 |
| 1c | RS-KD 12 個樣本 ≈ FullKD | SLS | Table 6 p.6 | p.6 Table 6：12.1 個 unique token 時 2.75，FullKD 2.75。p.18 App. C：「For example, N = 50, we find K = 12.」 | 部分相符 | 12 是 **unique token 數**，實際抽樣 50 次。以快取容量來比時要用 12 |
| 1d | 梯度夾角 top-50 48°、RS 4° | SLS | Table 3 p.5 | p.5 Table 3：「Top-K 50 48° 1.8 … Random Sampling 12 4° 1.0」 | 相符 | 量的是 **vanilla top-K**（tail 不處理），只量一個 batch，不是 Naive Fix。T1a 第 174 行拿它描述我們現在的做法，不能直接套用 |
| 1e | 8B→3B（Table 8）：vanilla top-50 2.40 > CE 2.37 > RS 2.35 ≈ Full 2.34 | SLS | Table 8 p.7 | p.7 Table 8：「CE 2.37 … Top-K 12 2.50 … Top-K 50 2.40 … Ours (12) 2.35 … FullKD 2.34」 | 相符 | 原文只寫「Top-K KD」，沒說是 vanilla 還是 Naive Fix。從 ECE 看比較像 vanilla，但這是推斷 |
| 1f | 「Naive Fix」等於我們的 target | SLS | §3.3 p.4 | p.4：「the probability of the target token is adjusted to ensure that the target probabilities sum up to 1.」p.22 的程式：`naive_fix_dist[:top_k] = data_dist[:top_k]; naive_fix_dist += data_dist * (1 - np.sum(naive_fix_dist))` | 部分相符 | **差異：**<br>- Naive Fix 的老師 top-K 機率不重新正規化，殘差 1−Σ 全給正確答案，而且是純 KD，沒有另外的 CE 項，等於我們的 **α=1**（也就是 CI）<br>- 我們的 α·p_T(top-K) + (1−α·m)·e_y = **α·NaiveFix + (1−α)·e_y**，也就是 Naive Fix 再加上權重 1−α 的 CE<br>- SLS 沒有測過「Naive Fix + CE」；CE 混入只在 RS-KD 上測過（Table 10，LR ratio 1.0 那一列：CE 0.3/0.2/0.1/0 分別是 101/111/95/98%，看不出明確方向） |
| 1g | 「現在做法的偏差有證明：SLS App. A.4 式 (5)」（T1a 第 172 行）；10-candidates 寫「SLS 證明現在的做法（Naive Fix）有偏差」 | SLS | A.4 p.16 | p.16 A.4 的標題是「Vanilla Top-K KD provides scaled teacher as target」，式 (5) 是 vanilla top-K 的梯度。Naive Fix 只在 p.4 有一句定性描述：「The methods above are either biased estimators of the teacher probability distribution, and/or lack adequate supervision in the tail.」 | 誇大 | 式 (5) 證明的是 vanilla top-K，不是 Naive Fix。Naive Fix 的 tail 偏差可以由式 (4) 自己推出來（T1a 算的「tail 只拿到資料機率的 0.28 倍」代數是對的），但那是我們的推導，不是論文證明的 |
| 1h | 「hard token」lr ×2 加 CE 0.1，比 FullKD 多拿 25% | SLS | Table 10 p.8 | p.8 Table 10：LR Ratio 2.0、CE 0.1 → 125 | 相符 | 是 RS-KD 加這兩個技巧，不是 top-K。表的標題寫 8B→300M，內文寫 3B 老師，論文自己前後不一致 |
| 2a | top-k=128 截斷 +0.11 nats | Distillation Scaling Laws（ICML 2025） | G.4 Fig.54 p.63 | p.63：「at k = 128, validation loss increases by 0.11 nats compared to full distillation (k = 32768)」 | 相符 | top-k 有**重新正規化**（式 55），λ=1 |
| 2b | k=128 加 λ=0.7 時差距 <0.01 nats | 同上 | p.63 | p.63：「Using λ = 0.7 with k = 128 maintains performance within 0.01 nats」；「statistically indistinguishable from those obtained using the complete teacher distribution」 | 相符 | 跟我們的設定最接近的證據：截斷後混入 30% 的 CE 就能救回大部分 |
| 2c | reverse KL 比 forward KL 差 0.28 nats | 同上 | G.5 p.63–64 | p.64：「The forward KL achieves a lower data cross-entropy compared to the reverse KL (Table 7), with an average improvement of 0.28 nats.」Table 7：2.42 vs 2.70 | 相符 | 只有一組設定（1.82B→546M） |
| 2d | τ=1 最好 | 同上 | G.2 p.61 | p.61：「We find optimal performance at τ = 1 across all model scales」 | 相符 | |
| 2e | λ>0.5 時曲線平坦 | 同上 | G.1 p.60–61 | p.61：「the student cross-entropy LS remains mostly flat for choices of λ > 0.5」 | 相符 | 弱老師時偏好較小的 λ |
| 2f | M=20 時 λ=1 ≈ λ* | 同上 | p.4 | p.4：「We verify the choice λ = 1 produces results statistically similar to the optimal λ∗」 | 相符 | 完整分布、M=20；Fig.51b 的 λ* 有些明顯小於 1 |
| 3a | KLD 2.107、CLM+KLD 2.117、RKLD 2.140 | Minitron（NeurIPS 2024） | A.5 p.17–18 | p.18 Table 15：「LCLM + Llogits(RKLD) 2.140 … LCLM + Llogits(KLD) 2.117 … Llogits(KLD) 2.107」 | 相符 | RKLD 那列其實是 CLM+RKLD，同條件下該跟 2.117 比（差 0.023）。<br>情境是剪枝後重新訓練（Nemotron-3 8B），不是從頭訓練 |
| 3b | top-K ≤100 掉分 | Minitron | A.5 p.18 | p.18：「a low value of top-K (<=100) results in a significant drop in accuracy.」 | 相符 | 沒有數字。它的 top-K 是老師和學生的 logits 都只留 top-K，跟我們「tail 給答案」不同 |
| 4a | 靜態 α 0.9 最好 | Peng et al.（ACL 2025） | Table 4 p.5 | p.5 Table 4：α 0.5/0.6/0.7/0.8/0.9/0.95/1.0 → 1.5/1.4/2.9/2.0/3.6/2.5/1.6（%）；「with the best performance at α = 0.9」 | 相符 | 數列不單調（0.7 > 0.8 < 0.9），雜訊約 ±1 個百分點 |
| 4b | WSD-α + WSD-LR +8.0%、Linear Dec +4.1% | Peng | Table 5 p.6 | p.6 Table 5：「Linear Dec … ↑4.1% … 1-α+WSD-LR … ↑5.0% … WSD-α+WSD-LR … ↑8.0%」 | 相符 | **WSD-α 的定義（p.6）：**「linearly increase α from 0 to 1.0 during the warm up stage … cosine decay … from 1.0 to 0. We set the warmup ratio at 10% and the decay ratio at 1%.」<br>也就是說 α 只在**最後 1%** 降到 0，lr 也只在最後 1% 衰減。<br>baseline（LLM-LM）是 cosine 降到 10%。WSD-LR 本身就有 +1.2%（0-α+WSD-LR） |
| 4c | 單一 seed、指標是 benchmark | Peng | — | p.3：「we conduct supervised fine-tuning (SFT; Ouyang et al., 2022) with additional 10B high-quality instruct-tuning data after pre-training.」全文沒有 seed 的字樣 | 相符 | 指標是 **10B tokens SFT 之後**的 8 個 benchmark 平均，不是 ppl。老師是 top-p 0.95 加 top-100，有重新正規化 |
| 5a | 純 KD 比最佳 α 差 | Strong Teacher Not Needed（NeurIPS 2026） | Table 5–6 p.18 | p.18：「In almost all configurations, there exists a better loss mixing than pure distillation.」 | 相符 | 8B 老師的 in-domain ppl 差距：老師 tokens 10B −11.5%、30B −1.8%、50B −0.9%、80B −0.6%、100B −0.4%、300B −0.1% |
| 5b | 8B 老師的最佳 α 依老師 tokens 是 0.4/0.6/0.8/1.0/1.0/0.8 | 同上 | p.8 | p.8：「In downstream accuracy, for example, the 8.0B teacher’s optimal α across token budgets (10B, 30B, 50B, 80B, 100B, 300B) is 0.4, 0.6, 0.8, 1.0, 1.0, 0.8」 | 部分相符 | 這是 **downstream 準確率**的最佳 α；T1b 沒寫指標。<br>以 in-domain ppl 來看（Table 6），每個老師 tokens 都是 α<1 比較好，只是差距縮到 0.1–0.9% |
| 5c | 300B tokens、α=0.2：in-domain +0.9% | 同上 | p.21–22 | p.21 Table 9：8.0B 老師的 in-domain ppl +0.9% | 相符 | 學生和老師都訓練 300B tokens，而且只測 α=0.2 |
| 6a | λ* = E⟨∇f_ξ(x*),∇f_ξ(θ)⟩ / (E‖∇f_ξ(θ)‖² + ‖∇f(θ)‖²/γ) | KD Performs Partial Variance Reduction（NeurIPS 2023） | p.7 | p.7 式 (14)，跟報告寫的一樣；p.20 Lemma 1 式 (33) 是一般化的版本（cγ） | 相符 | |
| 6b | 「步長越小 λ* 越小」 | 同上 | p.7 | 論文沒有這句話。論文對 λ* 的解讀（p.8）是：「the better the teacher’s model θ is, the bigger λ∗≤1 gets」 | 部分相符 | 從式 (14) 確實推得出來：只要老師不完美、分子 >0，γ 變小時分母變大。<br>但有三個限制：<br>- 這是固定步長下的穩態鄰域，條件是 self-distillation、凸或 PL，實驗是線性模型<br>- 論文沒討論排程<br>- 照字面套用，lr→0 時 λ*→0，也就是 D2Z 結束時 α 該降到 0，跟 K3 的下限 0.7 以及「不要把 α 降到 0」都不一致 |

### 1.2 超參數遷移與批次（T3a、T3b）

| # | 主張 | 出處 | 報告頁碼 | 我找到的頁碼與原文 | 判定 | 備註 |
|---|---|---|---|---|---|---|
| 7a | τ_opt = 1.084·TPP^−0.527 | Power Lines（NeurIPS 2025） | p.1、4–5 | p.1 Fig.1：「τ = 1.084(TPP)−0.527」；p.5：「A precise power law emerges (R2=0.975) … 10th and 90th percentiles of fitted mτ over all points are (-0.529, -0.507)」 | 相符 | p.24 有一處寫 mτ = −0.520，論文自己前後差一點 |
| 7b | B_opt ∝ D^0.383 | 同上 | — | p.1 Fig.1、p.7 Fig.5：「Bopt(D) = 0.0306D0.383」；p.8：10–90% 區間 (0.367, 0.391) | 相符 | 同一張圖的 B_crit = 0.0471·D^0.462 |
| 7c | 適用範圍：20–1280 TPP、AdamW、D2Z | 同上 | p.3、17 | p.2：「We study TPPs from 20 to 1280.」p.3：「a linear LR schedule, with a 10% warmup followed by decay-to-zero」p.17：「Our work focuses on AdamW」「We present results with a single (standard) learning rate schedule.」 | 相符 | 另外：μP，AdamW β2 0.95，只用一份資料集。我們的 1B→6B 是 0.8–4.8 TPP，全部在範圍外 |
| 8a | Wang & Aitchison 的「τ 固定」來自多 epoch 實驗 | ICML 2025 | p.4–5 | p.5：「We trained the model using the first 1/4, 1/2, and the whole dataset for approximately 4 epochs」；ImageNet 子集跑 100 epoch（p.4–5） | 相符 | |
| 8b | 作者自己在 p.8 改引 Power Lines | 同上 | p.8 | p.8：「Bergsma et al. (2025a) show that optimal τepoch should not stay constant in one epoch setting, but rather follows a power law τ optimal epoch ≈(TPP)−0.527 … and is always smaller than 1 when TPP > 1.」 | 相符 | p.8 也提到 Dey 2025、Narayan 2025 說 τ 能**跨寬度**遷移。那是固定 TPP、換模型大小，不是我們要的「固定模型、加 tokens」 |
| 9a | lr 指數 −0.32 | Bjorck（ICLR 2025） | p.7、14 | p.7 式 (4)：「C ∼1.55 × 10−3, α ∼0.23, β ∼0.32」；「valid in the regime of large models (≥ 760M parameters)」 | 相符 | 各尺寸（p.14）：760M 0.316、1.3B 0.317、2.7B 0.418。條件（p.13）：wd 固定 0.1，cosine 降到 0.1×，批次 0.5M |
| 9b | D2Z 下 lr 幾乎不用動 | Straight to Zero（ICLR 2025） | p.8 | p.8 Finding 3：「optimal LR shifts substantially lower for Constant, somewhat lower for 10×, and hardly at all for D2Z.」 | 相符 | 這些 sweep 的 wd 是**固定**的，所以 τ ∝ 1/D 會自動變小。T3a 第 18 行說「又已經用 τ 調過 wd 的情況下」，不是這篇的條件（Power Lines p.5 才有「只調 λ」的實驗） |
| 9c | Shen 的 −0.51 | Power Scheduler（arXiv 2408.13359，非六大會議） | p.5 | p.5 式 (5)：「γ = 4.6T −0.51」，γ = ηopt/β；p.3：「We use the WSD scheduler with 10% tokens in the decay phase」 | 部分相符 | 指數對。但 T3a 說「WSD 的固定 lr 段」不對，實際是完整的 WSD，最後 10% 有衰減。模型只有 12M/36M/121M |
| 10a | 臨界批次 ∝ D^0.47 | Zhang et al.（ICLR 2025） | — | p.24 Table 7 說明：「B∗= 22.91 × D0.47 where … data size D is in billions.」 | 相符 | 只有 302M 一種模型；臨界的定義是比線性多用 20% 資料。改成 10%/50% 時指數是 0.48/0.44 |
| 10b | Filatov 的 D^0.45 | arXiv 2510.03871（非六大會議） | p.2、8 | p.2：「B∗(D) ∝D0.45±0.07」 | 部分相符 | 數字對，但這是**最佳批次**，不是臨界批次。條件：Scion，沒有 wd、沒有 momentum，固定 lr |
| 10c | Step Law 的 D^0.571 | arXiv 2503.04715（非六大會議） | — | p.2–3：「B(D) = 0.58D0.571」 | 部分相符 | 同 10b，這也是最佳批次。T3b 的表 B 寫的是 B_opt，比較正確；任務清單把它歸成「臨界批次」 |
| 11a | Naganuma：Muon 160M 131k 3.3041 vs 漸增 3.3061、步數 −67% | ICML 2026 | — | p.8 Table 2（10 seeds）：「MUON 3.3041 ± 0.0021 3.3411 ± 0.0019 3.3181 ± 0.0072 3.3061 ± 0.0027 66.77」 | 相符 | −66.77% 是「達到 baseline 最低 loss 所需步數」的中位數。1B 時 AdamW 漸增追不上對照（p.8） |
| 11b | Merrill（OLMo）：好 0.005 nats、步數 −43% | NeurIPS 2025 | — | p.9：「outperforming the small-batch control in loss by a margin of 0.0166. After mid-training … a smaller margin of 0.0053 … using 43% fewer gradient steps.」 | 相符 | |

### 1.3 排程與優化器（T4、T5）

| # | 主張 | 出處 | 報告頁碼 | 我找到的頁碼與原文 | 判定 | 備註 |
|---|---|---|---|---|---|---|
| 12a | AdEMAMix 中途開啟時越早越好 | ICLR 2025 | Fig.5 | p.8 Fig.5 說明：「In both cases, the earlier AdEMAMix is used, the better the final loss.」p.9：「the earlier the switch, the larger the gain, with diminishing returns.」 | 相符 | 機制不同：AdEMAMix 是梯度的慢 EMA（β3 0.9999），EMA-Nesterov 是權重位移的 EMA。拿來支持 K7 只能算類比 |
| 12b | ramp 0–100k 步，最終值一樣 | 同上 | Fig.27 | p.38：「We observe how increasing Tβ3 = Tα only slows down the convergence.」 | 部分相符 | 文字只說「只是收斂變慢」，「最終一樣」是看圖得來的。條件：110M，在 300k 步（約 60%）才切換 |
| 13a | 用 Schaipp 把 1B 最佳衰減比例從 0.76 校正到 ≥0.86 | Schaipp et al.（ICML 2025） | §4.1、Fig.12、Fig.21 | p.5：「if the base learning-rate γ is fully tuned, then the optimal cooldown fraction is 1 (linear decay). For fixed γ, the optimal cooldown fraction can be smaller than one.」p.5：「for large enough γ, the parabola turns into a monotonically decreasing curve.」p.8：「γ(0.994) ≈e0.7γ⋆(0.2)」 | 部分相符 | **方向**有論文支持：lr 偏低時，衰減比例的拋物線頂點會偏短。<br>**0.76→≥0.86 這個數字**是 T4 自己的計算，論文沒有這種「修正擬合」的做法；結果還取決於用哪種 lr 位移模型（0.86 / 0.87 / ≥0.95）。<br>e^0.7 是 AdamW 124M、線性 cooldown 的結果，T4 套到 Muon 上 |
| 13b | 「horizon 變 2 倍時，lr 降為 0.525×，或衰減比例拉到約 0.6（p.7）」 | Schaipp | p.7 | p.6：「our analysis suggests to decrease the schedule by ρ = 0.525 for T2 = 2T1」；「When doubling the training length (T2 = 2T1), the adapted cooldown fraction is roughly c1 = 0.6.」 | 部分相符 | 頁碼是 p.6。0.525 是分段排程 (B2) 裡**延長那一段**的 lr 倍率（前段維持原 lr），而且來自凸理論模擬，不是「整個 lr ×0.525」。T4 沒有拿它做計算，影響小 |
| 14a | 「Secret Sauce」的完整標題和出處 | — | — | *In Search of Adam’s Secret Sauce*，Orvieto & Gower，**NeurIPS 2025**（arXiv 2505.21829，有在索引裡） | 相符 | T5 用的是 arXiv 版 |
| 14b | β1 0.9 配 β2 0.9875 很差，β1 0.975 配同一個 β2 很好 | 同上 | p.5 | p.5 Fig.3 旁的文字：「We notice high correlation between beta values (e.g., β2 = 0.9875 is a terrible option at β1 = 0.9, but a good one at β1 = 0.975). While results are noisy …」 | 相符 | 「很差」的量級：Fig.3 的色階上限約 0.34 ppl，對應 ppl 約 21.9，大約 1.5% |
| 14c | 實驗設定（模型大小、全模型都用 Adam） | 同上 | p.5–6 | p.2 Table 1：「160M parameters, 3.2B SlimPajama tokens, sequence length 2048, batch size 256 – Chinchilla optimal」；p.6 Fig.5 驗證了 410M、8.2B tokens | 相符 | **全模型都用 Adam**，wd 0.1，cosine。我們只有 embedding 和 norm 用 AdamW |
| 15a | Nesterov AdamW 一律勝過 AdamW | Fantastic（ICLR 2026） | p.9 | p.9：「all variance-reduced Adam variants (NAdamW, Mars, Cautious) surpass vanilla AdamW—except for a small lag at the smallest experiment.」 | 誇大 | 「一律」不對，原文有例外，雖然是小例外 |
| 15b | 1.2B 時 Nesterov AdamW ≈ Muon | 同上 | p.9 | p.9：「leading to similar performance between Muon and Nesterov AdamW.」Fig.5：「no longer significantly speed up over NAdamW」 | 相符 | |
| 15c | 低 TPP 時 Muon 最好；SOAP、Kron 要 ≥8× Chinchilla 才超過 | 同上 | p.8–9 | p.3：「while Muon is consistently the best optimizer in smaller Chinchilla ratio regimes, it is outperformed by Kron and Soap when the data-to-model ratio increases to 8x or larger」；p.8 也有同樣的話 | 相符 | 它的 1× Chinchilla 是以非 embedding 參數算 20 TPP |
| 15d | 加速到 1.2B 只剩約 1.1× | 同上 | p.1–3、9 | p.1：「decreases with model size to only 1.1× for 1.2B parameter models」 | 相符 | |
| 16a | Track 3 #24：輔助 Adam 衰減 0.4、矩陣 0.8，3225 → 3175 步，n=10 | modded-nanogpt（非六大會議） | 本地 README 56–103 行 | README #24：「Setup from #11, with split LR cooldown: aux Adam cooldown_frac=0.4 and matrix cooldown_frac=0.8」3175 步，3.2782（n=10）。<br>**#11 的 log：**`def set_hparams(step, cooldown_frac=0.7)`，兩組都是 0.7。#24 對 #11 的 diff 只有 train_steps 和這兩行 cooldown_frac | 部分相符 | 數字對，但 #24 **同時改了兩件事**：Adam 0.7→0.4（變短）、矩陣 0.7→0.8（變長）。<br>省下的 50 步（−1.6%）不能全歸給「Adam 衰減變短」。我們的矩陣組已經是 0.8。<br>沒有對 #11 的 p 值；#11 是 n=16 |
| 16b | Track 3 #46：Adam 組平到 51% 才降，Muon 組 18% 就開始降 | 同上 | — | 從 #46 log 的 PowerCool 常數算（power 1.2，t_end 2900）：embed 和其他 Adam 組 t_cd ≈ 1487（51%），Muon t_cd ≈ 514（18%） | 相符 | #46 本身是「#45 + RowUpdateFloor + CWD」，不是分開衰減的消融。另外 #37（MuonH）把輔助 Adam 的衰減調到 0.85、矩陣 1.0 |

### 1.4 資料、精度（T6、T2、T5）

| # | 主張 | 出處 | 報告頁碼 | 我找到的頁碼與原文 | 判定 | 備註 |
|---|---|---|---|---|---|---|
| 17 | SkyLadder：上下文窗口由短到長的收益、模型大小、是否搭配 doc masking | NeurIPS 2025 | — | p.1：「we pretrain 1B-parameter models (up to 32K context) and 3B-parameter models (8K context) on 100B tokens … consistent gains of up to 3.7% on common benchmarks」。p.6 Table 1：IntraDoc 47.4 → 加 SkyLadder 49.3（+1.9）。p.7 Table 6（32K）：IntraDoc 54.0 → 54.9（+0.9） | 相符 | 3.7% 是對 Random（不遮罩）的收益，疊在 IntraDoc 上只剩 +0.9 到 +1.9 個百分點（平均準確率）。<br>p.4：30B tokens 的池子訓練到 100B（約 3.3 epoch）。<br>主要指標是 benchmark，IntraDoc + SkyLadder 沒有 ppl。T6 的「長文 13.0→13.2」是長文 benchmark 的準確率，不是 ppl |
| 18 | Muon-VS 1.2B 比 NorMuon −0.45%，是否單 seed | arXiv 2601.14603（非六大會議） | 4、14 | p.14 Table 5：「NorMuon 2.7367 15.44 … Muon-VS 2.7323 15.37」；p.17：「we use same-nominal-seed reruns as a repeatability check rather than an independent-seed robustness analysis.」 | 相符 | 確定是單 seed；1.2B 連同 seed 的重跑都沒做（重跑只做了 210M、720M、GPT-2） |
| 19 | Beyond Outliers：Muon 1.5B W4A4 loss 9.7、AdamW 3.8 | ICLR 2026 | Table 6 | p.13 Table 6（row-wise W4A4 PTQ）：1.5B「AdamW … 3.769」「Muon … 9.725」 | 相符 | 50M/125M 時 Muon 反而比較好（4.305 vs 5.445、4.145 vs 4.233）。是 4-bit RTN，跟 bf16 差很多 |
| 20 | 事後量化的損失隨訓練 tokens 增加 | Scaling Laws for Precision（ICLR 2025） | Table 2 p.27 | p.1：「the degradation introduced by post-training quantization increases as models are trained on more data」；p.27 Table 2：C_T 0.0598、γD 0.5068、γN 0.3439、γ 0.5907 | 相符 | T2 引用的係數都對 |

**統計（共 53 列）：**
- 相符 40
- 部分相符 11：1c、1f、5b、6b、9c、10b、10c、12b、13a、13b、16a
- 誇大 2：1g、15a
- 不符 0
- 找不到 0

## 2. 改變結論的發現

### 2.1 該降級的候選

**K8（AdamW 組用較短的衰減）：降級**
- 唯一的直接 A/B 是 Track 3 #24 對 #11
  - #11 兩組的衰減比例都是 0.7
  - #24 同時把 Adam 改成 0.4、矩陣改成 0.8
- 50 步的收益有多少來自「矩陣衰減變長」，無法分開
- 我們的矩陣組已經是 0.8，能拿到的部分可能接近 0
- 之後的紀錄（#46：Adam 約 0.49、Muon 約 0.82；#37：Adam 0.85、矩陣 1.0）都維持「Adam 比矩陣短」，但都沒有消融
- 建議：信心從「低到中」降到「低」，預期改成 0 到 −0.15%；在 10-18 的 1B 批次裡排到後面

**K3（α 跟 lr 連動）：文獻支持變弱，只剩我們自己的數據**
- Peng 的 WSD-α：
  - α 在最後 **1%** 才降到 0，前 10% 從 0 暖身
  - lr 的衰減也只有 1%
  - 跟「α 隨長達 80% 的衰減一起下降」不是同一件事
  - 指標是 SFT 之後的 benchmark、單 seed，Table 4 的數列不單調（雜訊約 ±1 個百分點）
- KD 變異數縮減理論：
  - 式子正確，但條件是固定步長、self-distillation、凸問題
  - 照字面套用，lr→0 時 α 該降到 0，跟 K3 的下限 0.7 不一致
- Strong Teacher 的「α 先升後降」：
  - 是對**老師**的訓練 tokens，不是學生訓練中的時間
  - 而且是 downstream 指標
- 結論：K3 只靠 CD/CE/CH/CI 的擬合（最佳 α 隨時間變小）撐著。信心從「中低」降到「低」，預期 −0.1 到 −0.3% 偏樂觀
- 順帶一提：10-candidates D 段用「CD 的 α 0.5 明顯差」排除「最後把 α 降到 0」，理由不成立
  - CD 是全程固定 0.5，沒有測「只在最後一小段降到 0」
  - Peng 的最佳設定正好包含這件事，只是跟 α 暖身混在一起
  - 這個方向沒有證據說有害，但也沒有證據說有益

**K1（無偏 tail target）：前提的引用要修正，預期收益偏樂觀**
- SLS 的 73%/83% 是 **α=1 的 Naive Fix**（等於我們的 CI），不是我們的 α=0.8
- 把 CE 混進截斷 target 的直接證據是 Busbridge G.4：
  - 重新正規化的 top-128 加 30% CE，跟完整分布差 <0.01 nats
  - 我們 α=0.8 的 target 在精神上接近這個設定，所以剩下的 tail 偏差可能已經很小
  - 這跟 T6 的「真實偏差只有理想值的 1/20」一致
- 「SLS 證明現在的做法有偏差」要改寫：式 (5) 證明的是 vanilla top-K；Naive Fix 的偏差是我們自己用式 (4) 推的
- SLS 的 K=20 和 K=50 都是 2.76，73% 和 83% 的差距落在兩位小數的捨入範圍內，用來內插 K=32 的「78%」精度不高
- 建議：把「45% 機率超過 0.3%」下修到約 30%；先做 K2 的 D0（CPU 從快取算 tail 少給了多少），再決定要不要用 1B 的時間

**K9（AdamW β1 0.9 → 0.975）：維持低，傾向不測**
- Secret Sauce 的句子屬實，但條件是全模型用 Adam（160M/410M）
- Fantastic 的「一律勝過」有例外
- 新發現的反向證據：Track 3 調得最細的設定（Muon + 輔助 Adam，跟我們的結構最像）
  - #11、#36 的輔助 Adam 用 β1 **0.8**、β2 0.95
  - #46 用 β1 0.8、β2 0.99
  - 方向是**更低**的 β1，不是 0.975
- 我們的 (0.9, 0.9873) 剛好是 Secret Sauce 點名很差的組合（只差 0.0002），但那個「很差」是全模型 Adam 時約 1.5% ppl；只作用在 embedding 和 norm 時，效果應該小很多

### 2.2 不受影響或略為加強的

**K4（τ 0.39 → 0.2）：引用全部屬實，結論不變**
- Power Lines 的 −0.527、適用範圍（20–1280 TPP、AdamW、D2Z、μP）都對
- Wang & Aitchison 自己在 p.8 承認單趟訓練時 τ 不該固定
- 剩下的風險跟原本一樣：TPP<20 是外推，而且是 Muon。這一點引用核對解決不了，只能靠 E1（3B 的 τ 測試）或 CM

**K5（Muon lr 1.2e-3 → 1.05e-3）：不變**
- Bjorck −0.32（wd 固定、cosine 到 10%）、Straight to Zero（D2Z 時幾乎不變，wd 固定）、Shen −0.51 的數字都對
- Shen 的條件描述要修正：是 WSD 加 10% 衰減，不是固定段
- Straight to Zero 的「lr 不用動」是在 wd 固定、τ ∝ 1/D 自動變小的情況下測的。這反而說明：τ 固定時，lr 可能需要降得比 D2Z 文獻顯示的多

**K6（衰減比例 0.80 → 0.85）：方向有支持，略為加強**
- Schaipp 的定性結論屬實：lr 調好時越長越好，lr 夠大時也是單調變好
- 0.76→≥0.86 是 T4 自己的計算，結果取決於模型
- Track 3 #24 也把矩陣衰減從 0.7 拉到 0.8，而且有收益（雖然混了變因）；之後的紀錄矩陣衰減都在 0.82–1.0。這是「矩陣組衰減長一點」的旁證

**K7（EMA-Nesterov 提早開啟）：不變，維持低**
- 「越早越好」屬實，但原文還說收益遞減
- 「ramp 只變慢、最終一樣」只有看圖
- 機制不同，只能算類比

**K12（批次維持 131k）：不變**
- Naganuma（10 seeds）、Zhang、Merrill 的數字都對
- Filatov 和 Step Law 是最佳批次，不是臨界批次，但不影響「不改 262k」的結論

**K13（上下文窗口由短到長）：維持低，略降**
- 疊在 IntraDoc 上只剩 +0.9 到 +1.9 個百分點的 benchmark 收益
- 訓練是多 epoch，主要指標是 benchmark，在我們的條件（ppl、單趟）下沒有證據

**D 段（不建議清單）：結論不變**
- reverse KL 差 0.28 nats（雖然只有一組設定）、label smoothing −73%、ghost 59%、τ=1 最好都屬實

**T2 的精度結論：不變**
- Beyond Outliers 的 9.725 vs 3.769 是 4-bit RTN；Scaling Laws for Precision 的係數都對
- T2 已經說明不能外推到 bf16，這個判斷正確

## 3. 總結

- 報告裡的**數字幾乎都抄對了**：53 列裡沒有不符，也沒有找不到
- 問題集中在**條件和歸因**，有四處會影響候選：
  1. SLS 的 Naive Fix 等於我們 α=1 的版本，A.4 的證明是 vanilla top-K。K1 的前提要改寫；把 CE 混入截斷 target 的效果，Busbridge 已經顯示接近完整分布，所以 K1 的空間可能比估的小
  2. Peng 的 WSD-α 只在最後 1% 衰減，而且看 SFT 後的 benchmark、單 seed，不能拿來支持 K3 的長時間 α 衰減；K3 只剩我們自己的擬合
  3. Track 3 #24 同時改了 Adam 和矩陣兩組的衰減，K8 的證據被混在一起，對我們（矩陣已經是 0.8）可能接近 0
  4. Track 3 的輔助 Adam 用 β1 0.8，方向跟 K9 相反
- 跟 6B 設定有關的 K4、K5、K6，引用全部成立，風險仍然是「在文獻的 TPP 範圍外、而且是 Muon」，這要靠實驗回答，不是靠查文獻
- 對 10-18 的 1B 批次的含意：
  - K8、K9 往後排
  - K1 先做 D0 診斷再決定
  - K3 若要測，要了解它只有內部數據支持

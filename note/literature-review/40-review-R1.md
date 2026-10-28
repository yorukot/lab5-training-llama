# 40 審稿意見 R1（科學面）

- 審稿人：R1，新加入，之前沒參與這份研究
- 日期：2026-10-18
- 審的對象：`99-summary.md` 草稿；需要時對照 `00`、`10`、`20`、`21`、`30-*`、`T1a`、`T3a`
- 我自己的驗算腳本和查詢結果都在 scratchpad：
  - 玩具模擬：`/tmp/claude-1000/-home-yorukot-ml-lab5-training-llama/94460c4a-2e94-42fc-8d1d-1c1378fd5e26/scratchpad/r1bin/ema_wd_sim.py`、`ema_wd_sim2.py`
  - 43 組查詢的命中清單：`/tmp/claude-1000/-home-yorukot-ml-lab5-training-llama/94460c4a-2e94-42fc-8d1d-1c1378fd5e26/scratchpad/r1/q/R01–R43.txt`
- 新下載的論文放在 `~/ml/papers/pdf/review/`（SODA、SWD 兩篇，含 txt）

---

## 1. 摘要（我理解的主張）
- **K1：** 現在的 KD target 在答案落在老師 top-32 外時，只給答案 0.34–0.41 的權重，無偏應該是 1
  - 改約 4 行就能修正，1B 預期 −0.4% 到 −1%，列為第一優先
  - 前提由 D0（只用快取的 CPU 診斷）「證實」
- **6B 的 τ：** 從 0.39 降到 0.25
  - 理由一：單趟訓練的文獻都指向「tokens 越多，τ 越小」
  - 理由二：EMA-Nesterov 可能已經讓有效 τ 變短，所以不降到 0.2
  - 能跑 3B 的 E1 就依結果調整
- **6B 推估：** CM = 11.97 之後，現在的配方是 in-training 約 11.55、OJ 約 11.69；舊的 11.1–11.2 排除
- **不建議清單：** 換 loss 形式、其他 tail 分配法、權重平均、262k 批次、多種優化器等
- **實驗計畫：** 1B 的 2×2（CJ′、K1、K3、K1+K3），3B 的 E1，6B 的兩條路

## 2. 優點
- **數字大多可追溯，而且我驗算過的都對：**
  - D0 的分解是自洽的：0.9254 × 0.9425 + 0.0746 × 0.7321 = 0.9268 = E[m]
  - 答案在外時給答案的權重：1 − α × 0.7321，α 0.8 / 0.9 / 1.0 是 0.414 / 0.341 / 0.268
  - 理想學生的額外 CE，用 D0 的條件平均算：α 0.8 是 0.024、α 0.9 是 0.034、α 1.0 是 0.047 nats，跟 21 的「約 0.02 / 0.03」一致
  - 三點飽和擬合：β 0.359、底部 ppl 8.48、6B 11.571；CL、CM 兩點 log-linear 是 11.52。總結的 11.55 是兩者的中間
  - wd：1/(1.2e-3 × 0.25 × 45,796) = 0.0728；1/(6.8e-3 × 0.25 × 45,796) = 0.0128
- **D0 是好的一步：** 把 T1a 玩具裡的「m ≈ 0.9」換成實測的條件分布，發現答案在外時 m 只有 0.73，偏差比玩具溫和約一半，而且照實寫出來
- **引用查核做得紮實：** 54 條逐條比對原文，抓到 SLS 式 (5) 證的是 vanilla top-K、Peng 的 WSD-α 形式不同等關鍵歸因錯誤
- **決策規則先寫好再跑：** E1 的換算表、K1 的 0.25% 門檻、K3 的 0.15% 門檻都在實驗前定義，避免事後解讀
- **營運檢查清單（第 4 節）實用：** 程式版本、環境變數靜默退回、續跑沒測過，這些對一次性的 6B 比多數候選更重要

## 3. 主要問題

### M1（重大）τ：權重 RMS 的量測被誤讀，總結的 τ 推理因此自相矛盾

**問題：** 21 第 2 節和總結 §5 把「有 EMA 的權重大了 22–23%」解讀成「沒證實 wd 被放大」。但 RMS 根本不能檢驗這件事，而且正確的簡單模型本來就預測會變大。

**證據一：21 的「簡單模型」錯在哪裡**
- 21 的說法：隨機部分的平衡範數會變小，持續部分不變
- 實際上，EMA-Nesterov 會把所有慢於約 100–250 步的成分都放大 1/(1−β)，白雜訊的低頻部分也包括在內
  - 照 `ema_nesterov.py` 的公式推：每步位移 d = step + β·m，m 的穩態是 step 的 1/(1−β) 倍
  - 長時間累積的位移，由更新的零頻率成分決定，所以白雜訊同樣被放大
- 對隨機漫步來說：
  - 範數平方的成長率放大 A² 倍
  - wd 的收縮率只放大 A 倍
  - 所以平衡範數是 ×√A，**變大**
- 只有來回震盪（相鄰步負相關）的分量才會變小

**證據二：我的玩具模擬**
- 設定：照 `ema_nesterov.py` 的公式，1B 的排程（1% warmup、最後 80% 線性降到 0、β = 0.6 × lr 比例、15–74% 視窗），τ 0.39，更新 RMS 固定 0.2
- 「初始化保留」= ⟨W_T, W_0⟩/⟨W_0, W_0⟩，它直接量 wd 的總收縮

| 更新的型態 | RMS（有 EMA ÷ 沒有） | 初始化保留（沒有 → 有 EMA） | wd 的 e-fold |
|---|---|---|---|
| 白雜訊 | ×1.15 | 0.217 → 0.075 | 1.53 → 2.59 |
| 慢漂移加雜訊（相關時間 10–3000 步） | ×1.22–1.27 | 0.17–0.28 → 0.01–0.17 | 放大 1.4–2.5 倍 |
| 來回震盪 | ×0.70 | 0.217 → 0.075 | 1.53 → 2.59 |

- **解讀：**
  - RMS 變大或變小，取決於更新的頻譜，跟 wd 有沒有被放大無關
  - 實測的 +22–23% 正好落在「放大成立」的玩具範圍裡
  - 白雜訊那列的 e-fold 從 1.53 到 2.59（×1.70），跟理論派 `ema_sim.py` 的 ×1.70 完全一致
- **所以：**
  - 外推派「權重會小 5–15%」的預測本身就錯（`30-perspective-transfer.md` 2.1 節）
  - 21 把這個預測落空解讀成「沒證實放大」，同樣是錯的
  - wd 寫在 step 裡（`src/optimizer.py` 第 70 行：`step = -lr * (update + weight_decay * p)`），整個 step 進了 m。所以 wd 的放大是程式的機械性質，不是要靠實驗證實的假說
  - 真正不確定的是另一件事：有 EMA 時，最佳 τ 是不是該用 e-fold 來對齊

**證據三：總結內部的矛盾**
- §0、§1、`30-perspectives.md` K4：「EMA-Nesterov 可能已縮短有效 τ」是不降到 0.2 的**唯一**理由
- §5、21：這個放大「沒有證實」、「可能存在、大小未知」
- 如果真的懷疑放大，論證應該往 0.2 移，不是停在 0.25：
  - 不算放大時，T3a 的 HyperP 換算給 6B 約 0.18
  - 算放大時，理論派給 0.21–0.25
  - 0.25 是「放大成立」那一側的上緣，不是兩個模型的中間值
  - 總結說的「取四方的交集」，四方其實都假設放大成立
- **另外：** 第 3.4 節的不對稱風險（短的一側平、長的一側陡）支持往短的方向移，跟「停在 0.25」的方向相反，總結沒提

**影響：** 這是 6B-A（不等 E1 就送）選 τ 的主要依據。結論（0.22–0.25）在我看來仍然站得住，但理由要換成「放大是機械性質，大小 ×1.6–1.7 可以直接量」，而不是「可能存在」。

### M2（重大）K1：「前提已證實」和預期收益都說過頭

**一、D0 只量了 target 端的偏差**
- D0 回答的是：「理想學生」最多會因為 target 偏差吃虧多少，這是**上界**
- 學生實際實現了多少，D0 量不到，必須在學生端量：
  - 紅隊的 D1（幾分鐘 GPU）：比較學生放在老師 top-32 上的總機率、和答案落在 top-32 的頻率
  - T1a 的 C2/K2：量最佳溫度 T*
  - 這兩個都沒做
- 21 寫「K1 的前提通過驗證」，總結 §6 寫「通過，證實偏差」，都應改成：「target 端的偏差大小已量到（理想上界 0.02–0.03 nats）；學生端還沒驗證」

**二、預期範圍跟其他文件、也跟自己不一致**
- 各文件的說法：
  - 總結：1B −0.4% 到 −1%
  - 紅隊：−0.1% 到 −0.4%
  - T1a：中位 −0.5%，範圍 −0.2% 到 −1.5%
  - 查核：超過 0.3% 的機率從 45% 下修到約 30%
- 矛盾：如果 −0.4% 到 −1% 是合理區間，超過 0.3% 的機率應該 >85%，跟總結表格的「中低」信心、查核的 30% 對不上
- −0.4% 這個下限的來源混了兩種分母：
  - 「實現比例 15–20%」= SLS 實測差距 0.012–0.019 nats ÷ **玩具**的理想值（α=1 時 0.09–0.12）
  - 但 D0 尺度的理想值，在 α=1 時只有 0.047 nats，約是玩具的一半
  - 用同一份 SLS 差距配 D0 尺度的分母，實現比例會是 26–41%
  - 分子、分母不同源，所以 15–35% 乘上 D0 的理想值，得不出一個有根據的下限
- 這些都是**毛收益**：理論派 1.2 節算出 K1 的標籤雜訊是 V0 的 2–3 倍，這個代價的大小和正負都沒估，但不是 0
- **比較乾淨的上界：**
  - SLS 在 α=1 時，Naive Fix 和完整蒸餾只差 0.012–0.019 nats
  - V1 不可能贏過完整蒸餾
  - α 0.9 的理想偏差約是 α=1 的 0.72 倍（D0 尺度：0.034 ÷ 0.047）
  - 所以 1B 的毛收益上界約 0.9–1.4%；−1% 已經接近上界，不是區間中段

**三、反證據沒進總結**
- Busbridge G.4：重新正規化的 top-128，混入 30% 的 CE，就跟完整分布差不到 0.01 nats
  - 我估那個設定的理想偏差：假設 top-128 的質量 0.96，約 0.02 nats
  - 也就是說，實際實現的比例 <50%
  - 這跟 15–35% 相容，但也不排除更低
- 總結表格把「SLS（ACL 2025）」列成 K1 的證據，應該註明兩件事：
  - SLS 只量過 α=1 的 Naive Fix
  - V1 和「Naive Fix 加 CE」都沒有任何文獻直接測過

**四、措辭**
- §0「是唯一四個視角都同意要測的改動」不正確：K4 四方也都同意要在 3B 測（見 `30-perspectives.md` 第 1 節的表）

**影響：** 1B 的測試設計不用改。但 §0 的「最值得投入」和 6B 的「加上 K1：11.45–11.5」都應該降調。

### M3（次要）6B 推估的組成跟建議不一致
- **§1「只改 τ：約 11.53」的來源：** 取自外推派 5.1 節，那是「τ 0.24 加衰減 0.85」的整包；總結本身明確不採用 0.85
- **τ 收益的估計差很多，總結沒有調和：**
  - 外推派：整包只有 −0.17%
  - 理論派 3.4 節：如果最佳點左移，τ 0.39 在 6B 會被罰約 1%
  - 這個差距直接決定 E1 值不值 27 H200-h
- **K1 的 6B 收益沒換算 α：**
  - 「加 K1：11.45–11.5」用的是外推派「K1 配 α 0.9」的 −0.45%
  - 總結的 6B 是 α 0.8，外推派自己說要再乘約 0.7（`30-perspective-transfer.md` 2.3 節第 2 點）
- **CM 偏低的歸因說得太強：**
  - 總結寫「基準線正在飽和」
  - 外推派原文是「飽和加 KD 收益衰減，一支 CM 分不出兩者」，τ 最多解釋 0.1–0.3%

### M4（次要）採用 K1 時的 α，跟支撐文件衝突
- **各方的說法：**
  - 總結：α 0.8，只有 K1+K3 通過時才改成排程
  - 理論派 1.3、1.5 節：K1 拿掉偏差後，最佳 α 往上移，6B 用 0.85–0.9
  - 外推派 5.2 節：K1 通過就配 α 0.9
  - 紅隊：維持 0.8，沒測過不要動
- **問題：**
  - 1B 在 α 0.9 測 K1，6B 卻在 α 0.8 用 K1，這個組合沒測過
  - 總結沒寫為什麼選紅隊的版本
  - 也沒寫 1B 的收益換到 α 0.8 要打約 0.7 折

### M5（次要）其他不一致與措辭
- **bf16 的占比：**
  - §3 寫「bf16 只佔 OJ 和 in-training 差距的約一成」，這是 T2 的推估（信心低到中），不是量測
  - §5 又說三個來源都還沒量
  - 應標成「估計」
- **6B-A 的 τ：** 用 0.27，主建議是 0.25，沒說明為什麼不同（0.27 是實務派的保底值）
- **CJ′ 的程式漂移門檻 0.05%：** 紅隊估的雜訊 σ 是 0.05–0.15%，這個門檻會常常誤報
  - 建議改成約 0.15%，或改看整條曲線是否全程往同一邊偏
- **不建議清單的「衰減段權重平均、schedule-free：收益 ≤0.12%」：** 文獻有分歧
  - SODA（NeurIPS 2026，見 4.2）在 WSD 排程、wd 已調好的 Muon 上，仍然贏 0.015 nats（124M）
  - 這條要限定成「在我們 80% 線性衰減到 0 的設定下」

## 4. 漏掉的文獻

### 4.1 查詢紀錄（43 組，全部是我新寫的 regex）
- 範圍：整個索引，除非另外標年份
- 命中數來自 `search.py` 的 stderr
- 依六個指定方向分組：

**A. top-k 或 sparse 蒸餾的偏差修正**
- R01 `(truncat|sparse|top-?k|partial).{0,40}(teacher|logit|soft ?label).{0,80}(bias|unbias|correct|debias)`：0
- R02 `(importance (sampling|weight)|unbiased).{0,60}(distill|soft label|teacher distribution)`：0
- R03 `(offline|cached|stored|precomputed|pre-computed) (teacher )?(logits|soft labels|predictions|probabilit).{0,120}(distill|student)`：0
- R11 `(distill|teacher).{0,80}(long[- ]tail|low-probability|tail (token|probabilit|mass|class))`：6
- R12 `logit.{0,30}(compress|quantiz|sparsif|subsampl|storage)|(compress|quantiz|sparsif)\w* (teacher )?logits`：3
- R17 `(Rao-?Blackwell|control variate).{0,80}(distill|KL|language model|soft)`：3
- R25 `sparse logit|logit sampling|(random|stochastic) (top-?k|sampling).{0,60}(teacher|knowledge distillation)`：2
- R36 `sampled softmax|negative sampling.{0,60}(bias|softmax)`：3
- R38 `top-?[kp]\b.{0,60}(teacher|distill|soft label)`（2025–2026）：2
- R39 `importance sampl.{0,60}(logit|vocab|token|distill)`：4

**B. 預訓練蒸餾的 α 排程與 KD 的其他設定**
- R04 `(distill|teacher).{0,80}(schedul|anneal|decay|curriculum).{0,40}(weight|coefficient|alpha|ratio|mixing)`：2
- R05 `pre-?train.{0,60}(knowledge )?distill|distill.{0,60}pre-?train`（2025–2026）：111
- R13 `(distill).{0,100}(calibrat|over-?confiden|temperature scaling)`（2024–2026）：20
- R21 `distill.{0,100}(token budget|compute-optimal|data-efficien|sample[- ]efficien|fewer tokens)`：9
- R22 `(teacher|distill).{0,80}(mix|interpolat|combin).{0,40}(ground[- ]truth|hard label|one-hot|cross-entropy|next-token loss)`：4
- R29 `(distill).{0,80}(scaling law|capacity gap|teacher size)`：5
- R35 `(knowledge distillation|distillation).{0,100}(hard (tokens|examples)|difficult tokens|token-level weight|per-token weight)`：1
- R37 `(distill).{0,120}(perplexity)`（2025–2026）：1
- R42 `(knowledge distillation|distillation).{0,100}(learning rate|weight decay|batch size|critical batch)`（2024–2026）：1
- R43 `(soft (labels?|targets?)).{0,100}(gradient (noise|variance)|variance reduction|sample efficien)`：0

**C. weight decay 隨 tokens 的 scaling**
- R06 `weight decay.{0,80}(scal|token|dataset size|training (length|duration|horizon)|compute)`：13
- R10 `(muon|orthogonal(ized)? (update|optimizer)).{0,80}(weight decay|scaling law|hyperparameter transfer)`：3
- R16 `weight decay.{0,60}(schedul|warm-?up|anneal|decay to zero|cooldown|time-varying)`：6
- R23 `(weight norm|parameter norm).{0,100}(weight decay|equilibri).{0,100}(effective learning rate|angular|rotation)`：0
- R24 `effective learning rate.{0,100}weight decay|weight decay.{0,100}effective learning rate`：0
- R34 `(weight decay).{0,100}(language model|LLM).{0,100}(pre-?train)`（2025–2026）：2
- R40 `(model growth|weight decay).{0,40}(towards?|to) (the )?(init|initial)|anchor(ed)? (to|at) (the )?init`：2

**D. EMA 或 lookahead 與 weight decay 的交互**
- R07 `(timescale|time scale|EMA view|exponential moving average).{0,60}weight decay|weight decay.{0,60}(timescale|time scale)`：1
- R08 `(lookahead|look-ahead|nesterov|extrapolat).{0,80}weight decay|weight decay.{0,80}(lookahead|nesterov)`：1
- R18 `(muon|normuon|orthogonaliz).{0,100}(lookahead|nesterov|momentum|extrapolat|EMA|averag)`：25
- R19 `(schedule-?free|primal averag|iterate averag|tail averag).{0,100}(weight decay|language model|pre-?train)`：2
- R31 `lookahead optimi[sz]er|\blookahead\b.{0,60}(slow weights|fast weights)|slow weights`：1
- R32 `(decoupled weight decay|AdamW).{0,120}(momentum|moving average|EMA).{0,80}(interact|implicit|equivalen)`：0
- R33 `(outer|meta|slow) (nesterov|momentum).{0,100}(language model|pre-?train|LLM)`：0
- R41 `(weight|parameter) (averag|EMA).{0,80}(weight decay|implicit regulari)`：0

**E. 資料少的區間（≤5 TPP）**
- R09 `(data-?constrained|data-?limited|limited data|data scarc|few tokens|small token budget|token budget).{0,80}(language model|pre-?train)`：37
- R28 `(small|limited|fixed) (compute|token|data) (budget|regime).{0,120}(optimi[sz]er|hyperparameter|learning rate)`：3
- R30 `(over-?parameteri[sz]ed|undertrain|under-train|below chinchilla|tokens per parameter).{0,100}(language model|LLM|pre-?train)`：15

**F. 其他能在固定 tokens 下降 ppl 的方向**
- R14 `(temperature scaling|post-?hoc (calibration|temperature)).{0,100}(perplexity|language model|next-token)`：5
- R15 `(weight averag|checkpoint averag|model soup|LAWA|averag\w* (of )?checkpoints|EMA of (the )?weights).{0,100}(pre-?train|language model|LLM)`：5
- R20 `(token|loss)[- ](re-?weight|weighting|selection).{0,80}(pre-?train|language model)`：7
- R26 `(over-?confiden|under-?confiden|miscalibrat).{0,100}(pre-?train|base model|next-token).{0,60}(language model|LLM)?`：6
- R27 `(cooldown|cool-down|warmup-stable-decay|\bWSD\b).{0,120}(weight decay|averag|momentum)`：3

**統計：**
- 43 組查詢，命中合計 367 筆（R05 的 111 筆大多是擴散模型和視覺，逐條看過標題）
- 讀了 12 篇的摘要，2 篇讀全文的相關頁（SODA p.1、4–9、19、21；SLS p.4 的 Table 2）
- 命中的相關論文，大多已經在團隊的報告裡：
  - Calibrated Progressive Distillation、Completed-P、Scaling with Collapse、Rotational Equilibrium、Through the River
  - Strong Teacher Not Needed、Distilled Pretraining、Token Weighting、AdaGC、Distillation Traps、ATS、Denoise First

### 4.2 團隊沒引用、我認為值得記錄的論文

**1. Optimistic Dual Averaging Unifies Modern Optimizers（SODA，NeurIPS 2026，arXiv 2605.11172）**
- **內容：**
  - 把 Muon、Lion、AdEMAMix、NAdam 都看成「optimistic dual averaging」的特例
  - 提出一個包裝：x_{k+1} = z0/(k+2) + (1 − 1/(k+2))·x_k + 基礎更新
    - 等於 wd 以 1/k 的排程把權重拉回**初始化** z0，同時做均勻的權重平均（p.5 式 4）
    - z0 可以從 seed 重建，不用額外記憶體（p.5 Remark 3.1）
  - 實驗：124M、FineWeb、WSD 排程（28.5% 衰減）
    - SODA(Muon) 贏過 wd 掃描後最好的 Muon：1× Chinchilla 3.2815 → 3.2665，4× 3.1414 → 3.1262，約 0.015 nats（p.8 Fig.1–2）
  - p.8 也寫：Muon 的最佳 wd 隨訓練長度下降
- **會不會改變結論：** 不改 6B 的建議（來不及測，而且只有 124M、20–80 TPP、看起來是單 seed）。但有三點要修：
  - (a)「Muon 最佳 wd 隨長度下降」不能當 K4 的證據：lr 固定時，「τ 固定」（λ ∝ 1/S）和「τ 變小」都會產生這個現象，它分不出兩派
  - (b) 它把 wd 看成「權重平均」，這正是理解 EMA-Nesterov × wd 的框架；索引裡沒有直接研究這個交互的論文（見 4.3 的 D）
  - (c) 不建議清單裡的「權重平均 ≤0.12%」要限定範圍（M5）；可以列入截止後的 150M 候選

**2. Sampled Estimators For Softmax Must Be Biased（NeurIPS 2025）**
- **內容：** 只看部分類別的 softmax 估計必定有偏差
- **會不會改變結論：** 不會。那是學生端的 softmax 近似；K1 是 target 端，學生仍算完整 softmax，期望 target 可以無偏（理論派 1.1 節）。記錄下來，避免之後有人拿它反對 K1

**3. On the Overlooked Pitfalls of Weight Decay（SWD，NeurIPS 2023）**
- **內容：** 依梯度範數調整 wd 的排程；主要是 Adam 和視覺任務，沒有 LM ppl
- **會不會改變結論：** 不會

**4. 已引用、但沒用上的工具：Rotational Equilibrium（Kosson，ICML 2024）**
- T3a 已經引用，但只拿來換算 HyperP
- 它的「平衡範數 ∝ √(η/λ)」正好能用來推 M1：放大 A 時，擴散 ×A²、收縮 ×A，範數 ×√A
- 21 沒有用上，所以誤讀了 RMS

### 4.3 六個方向的結論
- **A. top-k 偏差修正：**
  - 索引裡直接相關的仍然只有 SLS
  - V1 這個形式沒有任何論文測過，K1 只能靠我們自己的 1B 測試
- **B. α 排程：**
  - 只有 CPD（團隊已找到，是指令微調）和 Peng（形式不同，查核已指出）
  - K3 只靠內部擬合撐著的判斷，維持不變
- **C. wd 隨 tokens：**
  - Power Lines、Wang & Aitchison、Completed-P、SODA
  - 新找到的 SODA 分不出兩派，所以總結「方向中高、幅度低」的信心是恰當的
- **D. EMA 或 lookahead × wd：**
  - 索引裡是**空白**（R07、R08、R31–R33、R41 幾乎沒有命中）
  - 這表示這個問題只能靠自己量，所以 M1 的初始化投影檢驗更重要
- **E. ≤5 TPP：**
  - 沒有任何優化器或超參數論文在這個區間（R09、R28、R30）
  - 所有 τ 的證據都是從 ≥20 TPP 外推，總結對幅度標「低」信心是對的
- **F. 其他方向：**
  - 沒找到團隊沒考慮過、又能在截止前做的
  - T6 對資料順序、課程、輔助目標的覆蓋很完整

**總結：** 我的查詢沒有找到會推翻 K1 或 K4 結論的論文。漏掉的主要是 SODA，影響在框架和不建議清單的範圍，不在 6B 設定。

## 5. 具體修改要求
1. **τ（M1）：**
   - **改寫理由：** §0、§1、§5 的 τ 理由改成「wd 寫在 step 裡，被放大是機械性質（×1.6–1.7 e-fold）；不確定的是最佳 τ 是否以 e-fold 對齊」
   - **改寫 21 第 2 節的解讀：** RMS 不能檢驗這件事
   - **在 6B-A 送出前，加一個零 GPU 的檢驗（初始化投影）：**
     - 用 `set_seed(42)` 加 `from_config` 重建初始權重 W_0（`src/training.py` 第 81–82 行），在 CPU 上算 c = ⟨W_T, W_0⟩/⟨W_0, W_0⟩
     - 對象：BU/CF、CH/CJ、BY/CL 的 Muon 矩陣；全零初始化的矩陣跳過
     - BU 和 CF 在 15% 之前完全相同，所以兩者的比值只反映視窗內的差別
     - **預測：**
       - 沒有 EMA：c ≈ e^−1.53 ≈ 0.22
       - 有 EMA：c ≈ e^−2.6 ≈ 0.07，比值約 0.34
       - 放大不存在：比值 ≈ 1
     - **判讀：**
       - 比值 ≤ 0.5：維持名目 0.22–0.25，偏 0.22
       - 比值接近 1：6B 改用 0.2 左右
     - **注意：** 如果更新本身跟 W_0 相關，c 會有偏差；但兩支同 seed 的 run 共用這個偏差，比值仍然可用
   - **補上不對稱風險：** 寫明 3.4 節的「短的一側平」，並說明為什麼仍不選 0.2
2. **K1（M2）：**
   - **改措辭：** 「前提通過驗證」改成「target 端的偏差大小已量到，學生端未驗證」
   - **改 1B 預期：** 寫成「毛收益 0 到 −1.4%，中位約 −0.5%，扣掉雜訊代價可能打平；超過 0.25% 門檻的機率約 35–45%」，跟表格的「中低」一致
   - **補上兩點：** 證據欄加入 Busbridge G.4 的反證據；註明 SLS 只測過 α=1
   - **可選：** 1B 跑的同時做紅隊的 D1，用來解讀 K1 的結果
   - **改 §0：** 拿掉「唯一四方都同意」
3. **推估（M3）：**
   - 重算「只改 τ」，不要包含衰減 0.85
   - τ 的收益寫成範圍（約 −0.1% 到 −1%），並說明 E1 的價值取決於這個範圍
   - K1 的 6B 收益乘上 α 0.8 的折扣（約 0.7）
   - CM 偏低的歸因改回外推派的原話
4. **α（M4）：** 寫明 K1 通過時 6B 用多少 α、為什麼，以及這個組合沒測過
5. **M5：**
   - bf16 的占比標成「估計」
   - 說明 6B-A 為什麼是 0.27
   - CJ′ 的漂移門檻改成約 0.15%
   - 權重平均那條加上範圍限定，並引用 SODA
6. **記錄修改：** 第 8 節記下哪些意見採納、哪些不採納，以及理由

## 6. 整體評分
- **評分：大修**
  - 要大修的是文字和推理，實驗計畫本身大致站得住
    - 1B 的 2×2、E1、名目 τ 0.22–0.25 都可以照走
  - 兩個核心論證有實質問題：
    - τ 的理由建立在一個被誤讀的量測上，而且前後矛盾
    - K1 的前提和收益說得比證據強
  - 這兩點都會影響使用者怎麼看待 6B-A 和 K1 的優先度
- **信心：**
  - M1：4/5。玩具模擬照程式的公式寫，三種更新型態的結果都一致，而且跟理論派的 ×1.70 對得上；但玩具沒有梯度回饋
  - M2：3/5。方向確定；確切的實現比例本來就辨識不了
  - 文獻部分：4/5。43 組查詢沒找到會推翻結論的論文

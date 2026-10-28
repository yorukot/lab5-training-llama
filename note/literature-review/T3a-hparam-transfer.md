# T3a 6B 遷移：固定模型、tokens 從 1B 增加到 6B 時，lr、τ、β2 怎麼變

第 4 號研究員，2026-10-17。PDF 和轉好的文字檔放在 `/home/yorukot/ml/papers/pdf/T3a/`，其中 `nonmajor/` 子資料夾放的是六大會議以外的論文。

## 0. 先講結論
- **τ 是最該改的。**
  - 單趟訓練（每個 token 只看一次）而且有實際量測的研究，**全部**都發現：最佳 τ 會隨 tokens 增加而變小。這類研究有 Power Lines、Bjorck、Shen、Semenov、Fantastic 的 Muon 表格、HyperP。
  - 主張「τ 固定」的只有兩篇：Wang & Aitchison 和 Completed-P。前者是多 epoch 的實驗，作者自己在單趟訓練的情況下改引用 Power Lines（p.8）；後者的 τ 固定是從理論推出來的，沒有實際量過 λ。
  - 建議 6B 的 τ 用 **0.2**（範圍 0.15–0.27），不要用 0.39。
  - 換算成 wd：Muon 0.104（原計畫 0.0467），embedding 0.016（原計畫 0.0082）。
- **lr 的指數**之所以互相矛盾，主要來自三個差異：
  1. 用什麼座標描述：λ 固定，還是 τ 固定
  2. 排程：衰減到 0，還是衰減到 10% 或固定 lr
  3. 批次大小有沒有跟著 tokens 變大
- **lr 的結論：**
  - 排程是線性衰減到 0、又已經用 τ 調過 wd 的情況下，peak lr 幾乎不用動（Straight to Zero、Power Lines）。
  - 我們 BY 的軌跡拆解出來，τ 固定時的指數約 **−0.2**（範圍 −0.06 到 −0.33），比 −0.125 稍陡。
  - 建議 6B 的 Muon lr 用 **1.05e-3**（原計畫 1.2e-3，範圍 0.9–1.2e-3），AdamW lr 維持 **6.8e-3**。
  - 這部分影響只有約 0.1%，信心低。
- **β2 與 momentum：** 都不改。
  - AdamW β2 0.9873、NorMuon β2 0.95、Muon momentum 0.95 全部沿用。
  - 「以 token 計的半衰期固定」只是處理批次大小改變的規則，對訓練長度沒有任何主張。
  - 文獻有「訓練越長 β2 越大」的說法（Semenov；Completed-P 的理論推導），但證據弱，而且我們的 BD、BZ 都顯示把平均拉長在 1B 是虧的。
- **便宜的驗證：** 一個 3B 的跑，配方同 CL，只把 τ 改成 0.22（wd ×1.77），跟 CL 做成對比較。
  - 大約 3.4 小時、27 H200-h，可以在 10-18 前完成。
  - 結果直接用下面第 6 節的對照表換成 6B 的 τ。
  - 可以同時加跑第二個：同樣 τ 0.22，lr ×0.8，用來決定 Muon lr。

---

## 1. 搜尋紀錄
### 1.1 索引查詢
索引是 `papers.jsonl`，共 68,363 篇。總共下了 45 組 regex，合計命中 757 次，去重後 555 個標題，全部看過標題。

| # | regex（不分大小寫） | 命中 |
|---|---|---|
| q01 | `optimal learning rate.{0,60}(token\|horizon\|dataset size\|data size\|training length\|training duration)` | 2 |
| q02 | `(token\|training) horizon` | 24 |
| q03 | `learning rate.{0,40}scal(e\|ing) law\|scaling law.{0,40}learning rate` | 11 |
| q04 | `hyperparameter.{0,40}(transfer\|scaling law\|scale)` | 41 |
| q05 | `weight decay.{0,60}(scal\|timescale\|time scale\|dataset\|token\|EMA)` | 11 |
| q06 | `AdamW.{0,80}(weight decay\|EMA\|timescale)` | 10 |
| q07 | `power lines\|tokens[- ]per[- ]parameter` | 6 |
| q08 | `batch size.{0,60}(scal\|optimal\|critical)` | 35 |
| q09 | `critical batch size` | 8 |
| q10 | `beta_?2\|β2\|β_2\|second[- ]moment.{0,40}(decay\|EMA\|beta)` | 8 |
| q11 | `\bMuon\b` | 81 |
| q12 | `(warmup\|warm-up).{0,60}(length\|duration\|fraction\|why\|need)` | 3 |
| q13 | `(cooldown\|cool-down\|decay phase\|annealing).{0,60}(scal\|length\|fraction\|law)` | 9 |
| q14 | `(learning rate\|lr) schedule.{0,60}(scal\|law\|predict\|horizon\|optimal)` | 8 |
| q15 | `compute[- ]optimal.{0,60}(hyperparameter\|learning rate\|batch)` | 1 |
| q16 | `maximal update\|\bmu-?P\b\|μP\|muP` | 86 |
| q17 | `(step law\|predictable scale\|opt-laws\|optimization hyper-?parameter law)` | 1 |
| q18 | `(rotational equilibrium\|effective learning rate\|angular update\|scale[- ]invariant).{0,80}(weight decay\|norm)` | 4 |
| q19 | `(SDE\|stochastic differential).{0,80}(scaling rule\|batch\|learning rate)` | 4 |
| q20 | `momentum.{0,60}(batch size\|scal\|horizon\|half-life\|tokens)` | 17 |
| q21 | `(straight to zero\|linear(ly)? decay.{0,30}zero\|decay to zero)` | 6 |
| q22 | `(over-?train\|overtrain\|data-constrained\|beyond chinchilla\|token budget).{0,80}(hyperparameter\|learning rate\|lr)` | 1 |
| q23 | `(spectral\|orthogonali[sz]ed).{0,40}(optimizer\|update\|momentum).{0,80}(scal\|transfer\|learning rate)` | 3 |
| q24 | `(small batch\|batch size 1\|gradient accumulation).{0,80}(Adam\|language model)` | 1 |
| q25 | `convex.{0,60}(learning rate\|schedule).{0,60}(large model\|LLM\|transformer)` | 0 |
| q26 | `(learning rate\|weight decay).{0,80}(sensitiv\|robust\|tuning).{0,80}(language model\|LLM\|pretrain)` | 4 |
| q27 | `learning rate.{0,80}(training (duration\|length\|budget)\|number of (training )?tokens\|longer training\|dataset size)` | 2 |
| q28 | `(optimal\|best) (peak )?learning rate` | 25 |
| q29 | `weight decay` | 94 |
| q30 | `(Adam\|AdamW).{0,60}(beta\|β\|momentum).{0,60}(batch\|scal\|tokens\|half)` | 1 |
| q31 | `(learning rate\|weight decay\|batch size).{0,80}(language model\|LLM).{0,80}(pre-?train)` | 6 |
| q32 | `(schedule-free\|anytime\|WSD\|warmup-stable-decay\|warmup stable decay)` | 122 |
| q33 | `(convex\|last[- ]iterate).{0,80}(schedul\|learning rate)` | 16 |
| q34 | `(Shampoo\|SOAP\|Scion\|Dion\|NorMuon\|AdaMuon\|PolarGrad\|spectral descent\|modular (norm\|duality))` | 56 |
| q35 | `(benchmark\|fantastic).{0,40}optimi[sz]er` | 9 |
| q36 | `(depth\|width).{0,40}(transfer\|μP\|muP).{0,80}(weight decay\|learning rate)` | 1 |
| q37 | `(model\|data).{0,10}size.{0,60}(learning rate\|hyperparameter).{0,40}(scal\|depend)` | 1 |
| q38 | `(loss curve\|training curve).{0,60}(predict\|scaling law)` | 5 |
| q39 | `(training\|run) (duration\|length).{0,80}(learning rate\|hyperparameter\|weight decay\|transfer)` | 1 |
| q40 | `(learning rate\|hyperparameter\|weight decay).{0,80}(training (duration\|length\|steps)\|number of (steps\|iterations))` | 10 |
| q41 | `(independent\|decoupled) weight decay` | 8 |
| q42 | `(Adam\|optimizer).{0,40}(epsilon\|ε).{0,40}(scal\|transfer)` | 0 |
| q43 | `(learning rate\|lr).{0,40}(decrease\|decay\|shrink\|lower).{0,60}(longer\|more (data\|tokens)\|larger (dataset\|data))` | 0 |
| q44 | `(data\|dataset)[- ]size.{0,60}(learning rate\|weight decay\|hyperparameter)` | 3 |
| q45 | `(effective learning rate\|equilibrium norm\|weight norm).{0,80}(weight decay\|AdamW\|language model)` | 0 |
| q46 | 引用追蹤用：`Optimal Scaling Needs Optimal Norm\|norm transfer\|Power scheduler\|MiniCPM\|Spectral Scaling Laws of Muon\|Farseer\|...` | 12（都不相關，Filatov 不在索引裡） |

### 1.2 篩選、下載與精讀
- **篩選：**
  - 看過 555 個標題，讀了大約 70 篇的摘要
  - 會議論文下載 38 份 PDF，其中 9 份走 arXiv
  - 另外直接從 arXiv 下載 7 份六大會議以外的論文：Step Law、DeepSeek LLM、MiniCPM、Moonlight、Practical Efficiency of Muon、Power Scheduler、Filatov
  - arXiv 合計 16 份，每份間隔 2–4 秒，沒有遇到 429
- **精讀：** 共 24 篇，用 grep 找公式、擬合出來的指數和表格
  - Bjorck、Power Lines、Wang & Aitchison、Completed-P、Step Law、DeepSeek LLM、ACL 2026 Zhou、Fantastic、Benchmarking (Semenov)、Marek、Porian、Schaipp、Shulgin、Moonlight、Practical Muon、Filatov、Wortsman、Kosson（ICLR 2026）、HyperP、Straight to Zero、Riabinin warm-up、Kalra warmup、Kosson warmup、Scaling with Collapse
- **快速翻過：** 共 8 篇
  - Power Scheduler、MiniCPM、Hägele、Critical Batch Size Scale、Rotational Equilibrium、How to Scale Your EMA、Pre-training under infinite compute、NorMuon
- **引用追蹤：**
  - Power Lines → Shen（Power Scheduler）、Wang & Aitchison、Bjorck
  - Shulgin 的表 1 → Filatov、von Rütte
  - Straight to Zero → Filatov、Bjorck
  - HyperP → Bjorck
  - Wang & Aitchison 的定稿 → Power Lines
  - Filatov 不在索引裡，用 WebSearch 找到 arXiv 2510.03871。網頁說它是 ICML 2026，但我們的索引沒有收錄，下面標成「索引缺」。
- **WebSearch：** 2 次。第二次在找「Muon 的 lr 隨訓練長度變化」的專門研究，沒有找到新東西。

---

## 2. 論文表
**縮寫：**
- TPP：tokens 數 / 參數數
- D2Z：線性衰減到 0
- 10×：衰減到 peak 的 10%
- γ = ηλ：每步的衰減率
- τ = 1/(η·λ·步數)，取 peak lr 時的值

### A. 最佳 lr 怎麼隨 tokens 變（固定模型）
| 論文 | 設定 | 擬合式 / 結論 | 指數（對 D） | 適用範圍與注意 | 頁碼 |
|---|---|---|---|---|---|
| Bjorck et al., *Scaling Optimal LR Across Token Horizons*（ICLR 2025） | GPT-3 架構 50M–7B（另有 Llama 50M）；25B–800B tokens；AdamW β2 0.95、**wd 0.1 固定**；cosine 到 **10%**；warmup max(1000, 1%)；批次 0.5M 固定 | LR\*(D)=B·D^−β；合併式 LR\*=1.55e-3·N^−0.23·D^−0.32；Llama-1 7B 的 lr 太大 2.6 倍 | ≥760M 是 −0.32；各尺寸：125M −0.65、350M −0.38、760M −0.32、1.3B −0.32、2.7B −0.42 | TPP 大約 10–1000，**比我們高很多**；λ 固定，所以 τ ∝ D^−0.68 是一起變的；批次加倍時斜率不變 | p.4–5、7（式 3–4）、9、14（表 5） |
| Ren et al., *HyperP*（NeurIPS 2026） | 208M；10.4B–166.4B tokens（TPP 50–800）；Muon 在 Frobenius 球面上，**沒有 wd**；批次 2M；線性衰減到 10%；momentum 0.95 | η\*=24.27·T^−0.320，留一交叉驗證誤差 1.5% | −0.32，跟 Bjorck 一樣，作者稱之為「magic exponent」 | 球面上的 lr 就等於角步長，可以拿來對照 wd 平衡下的 √(2ηλ) | p.5、7 |
| Mlodozeniec et al., *Completed Hyperparameter Transfer*（ICLR 2026） | 42M–743M；0.4B–32B tokens（TPP 約 1.6–300）；AdamW；cosine；批次固定 | η_opt ∝ κ^−1/2（κ = 步數變成幾倍）；SDE「等時長」規則：η、λ ∝ √(m_B/m_D)，(1−β1)、(1−β2) ∝ m_B/m_D | −0.5 | 涵蓋低 TPP；schedule 是 cosine；β 的縮放只是推導，沒有單獨驗證 | p.4（表 1）、6–7（2.3 節、圖 5）、18（圖 15） |
| Filatov et al., *Optimal Scaling Needs Optimal Norm*（arXiv 2510.03871；網頁說 ICML 2026，**索引缺**） | Scion，沒有 wd、沒有 momentum、**固定 lr**；v2 加了 Adam；最大 1.3B / 138B tokens | η\*(B,D) ∝ B^0.62·D^−0.56；B\* ∝ D^0.45 | 批次固定時 −0.56；沿著最佳批次的路徑是 −0.28 | 固定 lr 的排程讓指數最陡 | p.2、8 |
| Shen et al., *Power Scheduler*（arXiv 2408.13359，**非六大**） | µP、WSD 的固定 lr 段；3 種尺寸 | lr/批次 = 4.6·T^−0.51 | −0.51 | Power Lines 用 τ 律解釋了這個指數（m_ηD=−0.48） | p.5 |
| Zhou et al., *How to Set the LR for Large-Scale Pre-training?*（ACL 2026 Findings） | Qwen3-MoE；AdamW β2 0.95；warmup 1000 步後**固定 lr**；批次 4M 固定；擬合時 wd 0.1 | Lr(N,D)=38.46·N^−0.222·D^−0.351 | −0.35 | 衡量的是 WSD 的固定段 | p.4 |
| Bergsma et al., *Straight to Zero*（ICLR 2025） | 111M / 610M、µP、AdamW；TPP 2 / 20 / 80 / 200；比較 D2Z、10×、固定 lr | 最佳 peak lr：固定 lr 時明顯下降、10× 時略降、**D2Z 時幾乎不變**；如果步數固定、改用加大批次來增加 tokens，最佳 lr 反而**上升** | D2Z 約 0 | 唯一有 2 TPP 點的 lr 研究；但在 2 TPP 時 D2Z 反而輸 10× | p.7（圖 5）、8（Finding 3）、9、20（Finding 11） |
| Bergsma et al., *Power Lines*（NeurIPS 2025） | 見 B 組 | λ 固定時推得 η_opt ∝ D^−(mτ+1) = D^−0.48；**如果 λ 有調，η 可以維持 µP 的值**（111M、200 TPP：預設 2.810、只調 η 2.808、只調 λ 2.805） | λ 固定 −0.48；τ 有調時約 0 | 跟 Bjorck、Shen 的結果一致 | p.5、23–24 |
| DeepSeek LLM（arXiv 2401.02954，**非六大**） | 計算量 1e17–2e19（1e20 驗證）；AdamW (0.9, 0.95)、wd 0.1；multi-step 排程；warmup 2000 步 | η_opt=0.3118·C^−0.125；B_opt=0.292·C^0.327；落在最佳 0.25% 內都算「近最佳」，帶子很寬 | 對 C 是 −0.125 | **N、D、B 同時變**；拿來描述固定模型只增加 tokens 並不適用。把批次增長扣掉，大約變成 C^−0.29 | p.5、9 |
| Li et al., *Step Law / Predictable Scale I*（arXiv 2503.04715，**非六大**） | 60M–1B；2B–100B tokens；AdamW (0.9, 0.95)、wd 0.1；warmup 2000 步；cosine 到 1e-5 | η=1.79·N^−0.713·D^**+0.307**；B=0.58·D^0.571 | +0.307（批次跟著變大） | 固定批次時，用平方根規則換算後約 D^+0.02，也就是不變。Shulgin 的附錄 C.3 解釋了為什麼指數是正的 | p.3、4、6 |
| Wen et al., *Fantastic Pretraining Optimizers*（ICLR 2026） | 130M–1.2B；1–16× Chinchilla（20–320 TPP）；批次 128–256 × 4096；WSD 線性衰減 80–100%；每個尺寸逐項掃描 | **Muon 只有 lr 會隨規模改變**（表 5）。Muon lr：130M 從 0.016（1×）→ 0.008（2–8×）→ 0.004（16×）；1.2B 在 1–8× 都是 0.004。AdamW lr：130M 在 1–16× 都是 0.008，1.2B 在 1–8× 都是 0.002 | Muon 130M 約 −0.5；1.2B 和 AdamW 約 0（格點間距 2 倍） | 1.2B 的 2–8× 是用他們擬合的超參數律預測後只做驗證；TPP 高 | p.29、41–42、67–70、95–97 |
| Semenov et al., *Benchmarking Optimizers*（NeurIPS 2026） | 124M–720M；在 2.1B tokens 調參，再延伸到 16.8B / 33.6B；批次 256×512 = 131k（**跟我們一樣**） | 多數優化器的最佳 lr 在 8 倍長度內「平順轉移」（格點 1e-4…2e-3） | 約 0（格點粗） | 124M、高 TPP | p.6、9 |
| Schaipp et al.（ICML 2025） | 凸優化 last-iterate 上界，對照 124M / 210M 實驗 | WSD：γ\*=0.47·T^−0.488；cosine：0.92·T^−0.492 | −0.49 | 純理論，沒有 wd | p.4 |
| Shulgin et al., *Deriving HP Scaling Laws*（NeurIPS 2026） | LMO 類方法的收斂上界，涵蓋 normalized SGD、signSGD（近似 Adam）、Muon | momentum 和批次固定：η\* ∝ T^−1/2；momentum 也調：1−β ∝ b·T^−1/2、η\* ∝ b·T^−3/4；重調 momentum 只能邊際改善 | −0.5 到 −0.75 | 理論；也解釋了 Step Law 的正指數 | p.3–5、7（表 1） |

### B. weight decay / τ
| 論文 | 設定 | 結論 | 指數 | 適用範圍 | 頁碼 |
|---|---|---|---|---|---|
| Wang & Aitchison（ICML 2025） | ResNet-18、ViT 在 ImageNet 子集上跑 100 epoch；NanoGPT 124M 在 OpenWebText 子集上跑 4 epoch | τ_epoch 在不同資料量下大致不變，λ ∝ 1/資料量 | τ 是 0 次方，即不變 | **多 epoch**。作者自己寫：單趟 LLM 訓練要照 Power Lines，τ ∝ TPP^−0.527，「TPP>1 時恆小於 1」 | p.4–5、8 |
| Power Lines（NeurIPS 2025） | GPT2 類（ALiBi、SwiGLU）、SlimPajama；AdamW β2 0.95；µP；10% warmup 加 D2Z；111M–3.3B；**TPP 20–1280** | τ_opt = 1.084·TPP^−0.527（R² 0.975；bootstrap 10–90% 區間 −0.529 到 −0.507）；B ≤ B_crit 時 τ_opt 不隨 B 變（λ ∝ B）；λ 和 η 二選一時，調 λ 比較好 | **−0.527** | AdamW、單一排程；**TPP<20 是外推** | p.1（圖 1）、3（式 2–4、註 1：τ 取 peak lr）、4–5（Finding 1–3）、17（限制） |
| Bergsma et al., *Scaling with Collapse*（ICLR 2026） | Celerity 系列 300M–3.9B，234 TPP | 正式訓練就用 Power Lines 的 τ(TPP)，τ=0.05；只有 τ 設成該 TPP 的最佳值時，loss 曲線才會重合 | 沿用 −0.527 | 高 TPP | p.1–2 |
| Completed-P（同 A 組） | — | SDE 規則推出 η 和 λ 都乘 κ^−1/2，所以 **τ 不變** | 0 | 是推導，λ 沒有單獨驗證；cosine | p.4 |
| Fantastic（同 A 組）：**我重新分析的部分** | Muon 130M，λ 掃 0 / 0.1 / 0.2 | 用拋物線擬合，最佳 λ 在 1×、2×、4×、8× 都是 0.124–0.137，**不隨 tokens 變**；對應 τ_opt = 0.098、0.092、0.051、0.024 | 2× 到 8×：τ ∝ D^−0.97，也就是 γ 不變 | 格點粗；只有 130M；論文自己的結論是「Muon 的 wd 不需要隨規模調」 | p.67–70（表 83–86） |
| Semenov（同 A 組） | 124M，lr 固定 | 短訓練 wd 0.5 最好，長訓練 0.1 最好 | 方向：τ 隨 D 變小，但變得比 1/D 慢 | — | p.8 |
| Wortsman et al., *Small-scale proxies*（ICLR 2024） | 小模型代理實驗 | 固定 γ=ηλ（獨立式 wd）時，loss 對 lr 很不敏感 | — | 支持「τ / γ 才是主要變數」 | p.5–6 |
| Kosson et al., *Rotational Equilibrium*（ICML 2024）與 *WD may matter more than µP*（ICLR 2026） | AdamW / SGD；1B、20B tokens | 平衡時角步長約 √(2ηλ)；前期過後決定更新幅度的是 wd，不是 µP | — | 這是把 HyperP 的球面 lr 換算成 τ 的依據 | ICLR 2026 p.7 |
| Kim et al., *Pre-training under infinite compute*（ICLR 2026） | 資料有限、跑很多 epoch | 最佳 wd 是一般的 30 倍 | — | **重複資料**造成的過擬合，不適用於我們 | p.4 |

### C. β2 與 momentum
| 論文 | 設定 | 結論 | 頁碼 |
|---|---|---|---|
| Marek et al., *Small Batch Size Training*（NeurIPS 2025） | 30M 模型 600M tokens；GPT-2 124M 2.5B tokens | **換批次時**，二階動量以 token 計的半衰期 t2 固定；β1 = 0.9 在各種批次都可以；lr 隨批次增加得比平方根還慢。**對訓練長度沒有任何主張** | p.7 |
| Porian et al.（NeurIPS 2024） | compute-optimal 的參數掃描 | 小批次時 β2 0.95 不夠好，要 0.99–0.999 | p.6 |
| Semenov（同上） | 124M，131k 批次，16.8B tokens | AdamW β2 0.999 比 0.95 好，「訓練越長 β2 該越大」；但 SOAP 重調 β2 幾乎沒差，而且作者也承認業界訓練 trillions tokens 都用 0.95 | p.6、39 |
| Completed-P 表 1 | SDE 推導 | 1−β ∝ m_B/m_D；套到 6 倍 tokens：AdamW β2 0.9873 → 0.9979，Muon momentum 0.95 → 0.9917 | p.4 |
| Shulgin（同上） | 理論 | 最佳 1−β ∝ T^−1/2，但重調只有邊際收益 | p.4–5 |
| Fantastic（同上） | AdamW 130M | β2 在 1× 和 8× 都是 0.98 最好（格點 0.9 / 0.95 / 0.98，比 0.98 更大的沒測）；Muon momentum 多數配置是 0.98（批次 0.5–1M tokens） | p.41–42、67–70 |
| Busbridge et al., *How to Scale Your EMA*（NeurIPS 2023） | 只處理批次 | EMA 的 ρ 隨批次的換算規則 | — |

### D. warmup 與比例
| 論文 | 結論 | 頁碼 |
|---|---|---|
| Kalra & Barkeshli, *Why Warmup*（NeurIPS 2024） | warmup 的作用是讓 sharpness 先降下來，目標 lr 才不會不穩；長度由目標 lr 和前期動態決定，跟總長無關 | p.5 |
| Kosson et al., *Reducing the Need for Warmup*（NeurIPS 2024） | 需要多長的 warmup，取決於早期更新的大小 | — |
| Riabinin et al., *Where Does Warm-Up Come From?*（NeurIPS 2026） | 針對 Muon、Lion：warmup 長度可以在訓練一開始自適應決定，是早期的性質 | p.1 |
| 業界做法 | DeepSeek、Step Law 用固定 2000 步；Bjorck 用 max(1000, 1%)；Fantastic 的 AdamW warmup 會隨規模變（500–4000 步），Muon 不用 warmup | 見上 |
| Hägele et al.（NeurIPS 2024） | 衰減段 10–20% 通常就夠，越長的訓練可以越短（第 6 號研究員負責） | p.5 |

### E. Muon 特有
| 論文 | 結論 | 頁碼 |
|---|---|---|
| Moonlight（arXiv 2502.16982，**非六大**） | 更新 RMS 對齊到 0.2·√max(A,B) 之後，**直接沿用 AdamW 的 lr 和 wd**；scaling law 的實驗也直接用 AdamW 的 compute-optimal 超參數。沒有研究訓練長度 | p.4、6 |
| Practical Efficiency of Muon（arXiv 2505.02222，**非六大**） | µP 加 Muon，跨寬度搜尋 lr / wd；沒有研究訓練長度 | — |
| Fantastic | Muon 只有 lr 會隨規模改變；130M 時 Muon 的 lr 隨 tokens 下降，AdamW 不變 | p.29、67–70 |
| HyperP、Filatov、Shulgin | Muon 類方法的指數跟 AdamW 相同，分別是 −0.32、−0.56、−0.5 | 見 A 組 |

---

## 3. 為什麼結論互相矛盾
### 3.1 統一換算：1B → 6B（6 倍 tokens、批次固定）
下表把各規則統一換算成三個倍數：η、τ（固定前提的 τ）、每步的 γ=ηλ。

| 規則 | η 倍數 | τ 倍數 | γ 倍數 | 前提 |
|---|---|---|---|---|
| **目前**（DeepSeek lr，τ 固定） | 0.80 | 1.00 | 0.167 | — |
| Wang & Aitchison（τ 固定、η 不變） | 1.00 | 1.00 | 0.167 | 多 epoch |
| Completed-P（SDE 推導） | 0.41 | 1.00 | 0.167 | cosine，推導 |
| Power Lines（η 不變，調 λ） | 1.00 | **0.389** | 0.428 | D2Z，AdamW，20–1280 TPP |
| Shen（λ 固定） | 0.40 | 0.41 | 0.40 | WSD |
| Bjorck（λ 固定） | 0.56 | 0.30 | 0.56 | 10× cosine |
| HyperP（角步長 ∝ D^−0.32，換成 wd 平衡） | 不確定 | **0.525** | 0.32 | Muon 球面，衰減到 10% |
| Straight to Zero、Fantastic Muon（η、λ 都固定） | 1.00 | 0.167 | 1.00 | D2Z / WSD 衰減到 0 |
| Step Law（批次跟著變大） | 換回固定批次約 1.0 | — | — | 批次的路徑不同 |

### 3.2 三個矛盾來源
1. **座標不同。**
   - 大多數 lr 研究固定 λ。這時候降 lr，γ=ηλ 也會一起降，τ=1/(γS) 就跟著變小。
   - Bjorck 的 −0.32 和 Shen 的 −0.51，在 (η, τ) 座標裡其實等於「τ ∝ D^−0.5 到 −0.7」。
   - Power Lines 實際做了這個切換：固定 η、只調 λ，效果不輸只調 η（p.5）。Wortsman 也發現 γ 固定時 loss 對 η 不敏感。
   - 所以「lr 指數 −0.3 到 −0.5」**大半其實是 τ 該變小**的訊號，不是 τ 固定時的 η 該降這麼多。
   - 我們的規則是 τ 固定、只降 η，等於把兩件事搞混了。
2. **排程不同。**
   - lr 研究的指數依排程排列：
     - Filatov（固定 lr）−0.56
     - Shen（WSD 固定段）−0.51
     - Zhou（固定 lr 段）−0.35
     - Bjorck（cosine 到 10%）−0.32
     - HyperP（衰減到 10%）−0.32
     - Straight to Zero（D2Z）約 0
   - Straight to Zero 直接證明：衰減到 0 時，「降 lr 來降雜訊」這件事已經由排程完成了。
   - 我們的排程是 1% warmup 加最後 80% 線性降到 0，接近 D2Z，所以應該**靠近 0 的那一端**。
3. **批次的路徑不同。**
   - DeepSeek（批次 ∝ C^0.33）、Step Law（批次 ∝ D^0.57）、von Rütte 的批次都跟著變大。批次越大最佳 lr 越高，所以指數被拉向 0 甚至變成正的（Shulgin 附錄 C.3）。
   - DeepSeek 的 N 也一起變大。它的 −0.125 是 N、D、B 混在一起的結果，**拿來描述固定模型只增加 tokens 沒有根據**。

### 3.3 一個沒有人涵蓋的區間：TPP < 2
- 只有 Straight to Zero（111M、2 TPP）和 Completed-P（500M、1.6 TPP）摸到我們的區間（0.8–4.8 TPP）。
- Power Lines 的 τ 律是從 20 TPP 往下外推。
- 我們 1B 量到的最佳 τ（約 0.3–0.4）**比 Power Lines 外推的 1.1–1.2 低大約 3 倍**。因此只能借用它的**斜率**，不能借它的**絕對值**。

### 3.4 Muon 和 AdamW 的遷移規則是否不同
- **沒有證據顯示根本不同。**
  - HyperP 的 −0.32 跟 Bjorck 的 AdamW 相同
  - Filatov 的 Scion 和 Adam 共用同一個式子
  - Shulgin 的理論把兩者視為同一類方法
  - Moonlight 對齊 RMS 後，直接沿用 AdamW 的 lr 和 wd
- **有一點跡象顯示 Muon 的 lr 該降得快一點。**
  - Fantastic 130M：Muon 的 lr 在 16 倍 tokens 內降了 4 倍，AdamW 一直不變
  - 機制上：Adam 的 m/√v 在雜訊主導時會自己變小，Muon 正交化後的更新 RMS 卻固定，不管訊噪比高低
  - 所以 Muon 要靠 lr 或排程來壓雜訊。NorMuon 的每列二階正規化只恢復了一部分自適應性。
- **τ 方面：** Muon 偏好較短的 τ。
  - Fantastic 130M 在 20 TPP 時 τ_opt 約 0.1；Power Lines 的 AdamW 在同樣 TPP 是 0.22
  - 我們 1B 的 0.3–0.4，對照 Power Lines 外推的 1.1
  - 兩邊都差 2–3 倍
- **從兩個 Muon 定點粗估的斜率：**
  - 我們 1B：TPP 約 1，τ 約 0.28（已換成 80% 衰減的等效值）
  - Fantastic 130M：TPP 約 28，τ 約 0.095
  - 兩點連線的斜率約 −0.33，跟 HyperP 換算出的 −0.36 接近

---

## 4. 用我們自己的數據檢驗各規則
### 4.1 BY 的「收益只剩 1/3」代表什麼
- **先釐清：** BY 同時改了兩件事，warmup 5%→1% 和 lr ×1.25，所以不能直接解讀成「lr 的收益只剩 1/3」。
- **1B 時兩個收益在不同時間點的領先幅度**（相對 BN）：

| 時間點 | 20% | 50% | 75% | 最後 |
|---|---|---|---|---|
| BR（只改 warmup） | −2.9% | −0.91% | −0.55% | −0.52% |
| BU 對 BR（只改 lr ×1.25） | 約 −0.19% | −0.09% | −0.22% | −0.20% |
| BY（3B，兩個一起，相對 BP） | −2.7% | −0.56% | −0.34% | −0.27% |

- **warmup 的收益是前段大、之後縮小。** 1B 從 50% 到最後縮成 0.57 倍；lr 的收益在衰減段不會縮小。
- **假設一：3B 的 lr 部分跟 1B 一樣（−0.2%）。**
  - 那麼 warmup 部分要從 −0.36% 縮到 −0.07%，只剩 0.19 倍，比 1B 快很多，不太合理。
- **假設二：lr 部分接近 0。**
  - warmup 部分從 −0.56% 縮到 −0.27%，是 0.48 倍，跟 1B 的 0.57 倍一致。
  - **比較自洽的解讀是：3B 的 lr 部分大約 −0.1% 到 +0.1%。**
- **換算成指數：**
  - 用 1B 的 lr 拋物線（BR、BU、BV：曲率 0.0224 / (log2 lr)²，最佳 ×1.23），預測 3B 時 lr ×1.25 能帶來多少：

| τ 固定時的指數 β | 0 | 0.125（目前） | 0.2 | 0.25 | 0.32 | 0.5 |
|---|---|---|---|---|---|---|
| 預測的 lr 部分 | −0.49% | −0.20% | −0.03% | +0.09% | +0.25% | +0.66% |

  - 「lr 部分 −0.17% 到 +0.13%」對應 **β 約 0.1–0.3，中心約 0.2**。曲率減半時是 0.06–0.33。
- **結論：**
  - 最佳 lr 確實比 −0.125 降得**稍微快一點**，但沒有到 Bjorck 的 −0.32，更沒有到 Completed-P 的 −0.5。
  - β=0.5 會讓 lr 部分變成 +0.33 到 +0.66%，warmup 部分就得大於 1B 的 −0.52%，可以排除。
  - 「只剩 1/3」主要是 warmup 收益本來就前段大、後段小，加上所有百分比收益都會隨 ppl 下降而縮小（KD 也從 9% 縮到 7%）。
- **對 6B 的影響：**
  - β=0.2 時，6B 最佳 lr 是原計畫的 0.86 倍。照原計畫的懲罰約 0.05–0.1%。
  - β=0.32 時，懲罰是 0.3–0.6%。

### 4.2 τ：我們 1B 的曲線
- **資料：** 用 plain Muon、50% 衰減、524k 批次、lr 1.2e-3 的幾組跑（都是 1B）：
  - AI：τ 1.09
  - AK、AL：τ 0.55，比 AI −2.0%
  - AN：τ 0.27，再 −0.32%
- **擬合：** 在 log2 τ 上畫拋物線，最佳 τ 約 0.34，曲率約 0.0086 / (log2 τ)²。
- **兩個注意：**
  - 長 τ 的那一側很陡，短 τ 的那一側只有 AN 一個點，看起來比較平，所以拋物線會**高估**短側的懲罰。
  - 換成 80% 衰減後，lr 在整段的平均倍數從 0.725 降到 0.595。要維持相同的總衰減量，peak 時的 τ 要乘 0.82，也就是約 0.28。NorMuon 會部分取代 wd 的作用（AH/AJ/AM：wd 越大，NorMuon 的收益越小），這又可能把最佳值推高。
  - 綜合起來，**1B 時 0.39 就在平坦區內**。
- **用 1B 曲率外推 6B 時，各規則下的懲罰**（實際值大概只有一半到四分之一）：

| 真實規則 | 6B 的最佳 τ | 用 0.39 的懲罰 | 用 0.27 | 用 0.20 | 用 0.15 |
|---|---|---|---|---|---|
| τ 固定（W&A、CP） | 0.34 | 0.03% | 0.10% | 0.50% | 1.20% |
| HyperP 換算（6^−0.36） | 0.18 | 1.10% | 0.31% | 0.02% | 0.05% |
| Power Lines（6^−0.527） | 0.13 | 2.10% | 0.91% | 0.31% | 0.03% |
| Bjorck 換算（6^−0.68） | 0.10 | 3.30% | 1.75% | 0.85% | 0.29% |

- **用最小化最大遺憾來選：** 0.2 在四種規則下最大遺憾約 0.85%；0.39 是 3.3%；0.15 是 1.2%。
- 單趟訓練的實測研究一致指向「τ 變小」，因此**選 0.2**。

### 4.3 CM（4.5B）能檢驗什麼
- CM 用的就是目前的規則（lr ×0.83、τ 0.39），沒有對照組，**沒辦法區分不同規則**。
- 它只能當作健全性檢查：如果比 11.75 差 1% 以上，代表規則可能偏了，但看不出是 lr 還是 τ。
- 4.5B 時 Power Lines 預測的 τ 約 0.18，CM 的 τ 已經是它的 2.2 倍。

---

## 5. 6B 建議值與範圍
**計算前提：** 6B 的步數 = 6e9 / 131,016 ≈ 45,796 步。wd = 1/(lr·τ·步數)。1B 的參照是 BU（Muon 1.5e-3、AdamW 8.5e-3、τ 0.39）。

### 5.1 各規則推出的值
| 規則 | Muon lr | AdamW lr | τ | Muon wd | embedding wd |
|---|---|---|---|---|---|
| R0 目前（BP/CM 的推導） | 1.2e-3 | 6.8e-3 | 0.39 | 0.0467 | 0.0082 |
| R1 Power Lines（η 不變，τ 用斜率 −0.527） | 1.5e-3 | 8.5e-3 | 0.152 | 0.096 | 0.017 |
| R2 目前的 lr 加 Power Lines 的 τ | 1.2e-3 | 6.8e-3 | 0.152 | 0.120 | 0.021 |
| R3 Bjorck（λ 維持 1B 的值） | 0.85e-3 | 4.8e-3 | 0.115 | 0.224 | 0.040 |
| R4 Completed-P SDE | 0.61e-3 | 3.5e-3 | 0.39 | 0.091 | 0.016 |
| R5 Straight to Zero / Fantastic（η、λ 都不變） | 1.5e-3 | 8.5e-3 | 0.065 | 0.224 | 0.040 |
| R6 Shen（λ 固定，η ∝ T^−0.51） | 0.60e-3 | 3.4e-3 | 0.16 | 0.224 | 0.040 |
| **建議** | **1.05e-3** | **6.8e-3** | **0.20** | **0.104** | **0.016** |

R4 另外會要求 momentum 0.992、NorMuon β2 0.992、AdamW β2 0.998。

### 5.2 建議值的推導
- **Muon lr：** 1.5e-3 × 6^−0.2 = 1.05e-3
  - 指數 0.2 是 4.1 節拆解 BY 的中心值，夾在 Straight to Zero / Power Lines 的「幾乎不變」和 Bjorck / HyperP 的 −0.32 之間。
  - 範圍 0.9–1.2e-3，對應 β 在 0.125–0.3。
- **AdamW lr：** 維持 6.8e-3（範圍 5.9–7.4e-3）
  - 1B 時 AdamW 的擬合最佳值約 1.1e-2（AZ），一直高於實際用的值。
  - Fantastic 和 Straight to Zero 的 AdamW lr 隨 tokens 變化不大。
  - 所以 AdamW / Muon 的比例從 5.67 放寬到 6.5。這點的信心低，不想改比例的話，兩組都乘 0.73（Muon 1.1e-3、AdamW 6.2e-3）也可以。
- **τ：** 0.2，兩組都用（範圍 0.15–0.27）
  - 斜率在 HyperP 換算的 −0.36、Muon 兩定點粗估的 −0.33、Power Lines 的 −0.527 之間，偏保守。理由是我們的區間在外推範圍外，而且 1B 的短 τ 側看起來比較平。
  - 0.39 × 6^−0.36 = 0.205；0.39 × 6^−0.527 = 0.152。
  - embedding 那組也用同一個 τ。AR 顯示 embedding 的 wd 影響小，不值得分開處理。
- **跟原計畫的差距：**
  - Muon lr −12.5%
  - Muon wd ×2.2（0.0467 → 0.104）
  - embedding wd ×2.0（0.0082 → 0.016）
  - γ=ηλ 每步從 5.6e-5 變成 1.09e-4
- **預期影響：**
  - 如果「τ 變小」的研究是對的：比原計畫好約 0.3–1%
  - 如果「τ 固定」的研究是對的：差約 0.1–0.4%，因為短 τ 側比拋物線預測的平
  - lr 的部分在 ±0.1% 以內
- **信心：**
  - τ 該變小的方向：中高
  - τ 的幅度：低到中
  - Muon lr：低（差異小）
  - AdamW lr：低到中

### 5.3 β2 與 momentum
- **AdamW β2（只用在 embedding 和 norm）：** 維持 0.9873，半衰期 54 步、約 7.1M tokens
  - Marek 的規則只處理批次大小，6B 批次不變，所以不用動。這個做法是對的。
  - 「訓練越長 β2 越大」只有 Semenov（124M，全部參數都用 AdamW）支持，Fantastic 在 1× 到 8× 之間沒看到變化。
  - 我們的 AdamW 只管 embedding 和 norm，預期影響小於 0.1%，不值得冒險。
  - 真的想改的話，0.99 是低風險的上限。
- **NorMuon β2：** 維持 0.95。BZ 在 1B 把它拉長到 0.9873 是虧的（+0.12%），因為列尺度要追得上目前的更新。
- **Muon momentum：** 維持 0.95
  - Completed-P 的 SDE 推導會要求 0.992；Shulgin 的理論會要求約 0.98，但他也證明重調只有邊際收益。
  - BD（0.98）在 1B 輸 0.88%，而且主要輸在前段。
  - Moonlight、Kimi 等用 trillions tokens 訓練的模型都維持 0.95。

### 5.4 比例類參數與衰減比例的交互作用
- **warmup：** 文獻一致認為 warmup 長度由早期動態決定，以絕對步數算，不是總長的比例。
  - 1% 在 6B 是 458 步、60M tokens，安全，但比需要的長。
  - BY、CL 用 229 步沒有出現尖峰；BW 只用 19 步就出現尖峰。
  - 可以改成 229 步（0.5%），預期收益不到 0.05%，是可選的小改動。
- **衰減比例**（第 6 號研究員負責）的交互作用：
  1. **τ 是用 peak lr 定義的。** 衰減比例改了，lr 在整段的平均倍數 m(f) ≈ 0.995 − f/2 也會變。要維持相同的總衰減量，peak 時的 τ 要乘 m(f_new)/m(0.8)。例如 f=0.6 時 τ 要乘 1.17。
  2. **Straight to Zero 指出：** 衰減越長、越接近 D2Z，最佳 peak lr 隨 tokens 變化越小。如果 6B 改成較短的衰減，lr 的指數要更陡，可能到 −0.3。
  3. **BQ 顯示：** 衰減從 50% 拉長到 80% 時，最佳 peak lr 變高。衰減比例和 lr 要一起考慮。
- **EMA-Nesterov：** 視窗照比例沿用沒問題。γ 0.99 是以步數計，約 100 步，在 6B 只占 0.2%。wd 變強時，位移裡衰減的成分會變多，前瞻也會一起外推，預期影響很小。
- **KD：** soft target 會降低梯度雜訊。Power Lines 的註 2 推論：雜訊較小相當於有效 TPP 較低，所以最佳 τ 可能稍微偏長。下面的 E1 用的是 CL 配方，已經把這個效果算進去了。

---

## 6. 便宜的驗證實驗（可在 10-18 完成）
### E1（主實驗，只跑一次）：3B 測 τ
- **配置：** CL 的配方，只改 τ 0.39 → **0.22**（3B 時 Power Lines 的預測值 = 0.39 × 3^−0.527）
  - lr 不動：Muon 1.3e-3、AdamW 7.4e-3
  - wd：Muon 0.086 → **0.153**，embedding 0.0151 → **0.0268**
  - 其他全部照 CL：KD α 0.8、EMA-Nesterov 0.6、1% warmup、80% 衰減、seed 42、同樣的資料順序
  - 約 22,898 步，預計約 3.4 小時、約 27 H200-h
- **對照組：** 直接用 CL（12.635），同 seed、同資料，是成對比較。
- **為什麼選 3B、只跑一次，而不是兩個 2B：**
  - 對比幅度大約是 2B 的兩倍
  - 不用另外補一個對照組
  - 總 GPU-h 比兩個 2B（約 36 H200-h）少
- **預期結果（E1 − CL）**，數字是 1B 曲率推估的 0.25–1 倍：

| 真實規則 | 預期 E1 − CL | 換成 ppl |
|---|---|---|
| τ 固定（W&A、CP） | +0.08% 到 +0.31% | 12.65–12.67 |
| HyperP 換算（τ ∝ D^−0.36） | −0.13% 到 −0.50% | 12.57–12.62 |
| Power Lines（τ ∝ D^−0.527） | −0.22% 到 −0.88% | 12.52–12.61 |

- **怎麼把 E1 的結果換成 6B 的 τ：**
  - 做法：假設 3B 的 τ 曲線形狀跟 1B 一樣，用 Δ 解出 3B 的最佳 τ，再用 1B→3B 的斜率延伸到 6B。

| E1 − CL | +0.3% | +0.1% | 0 | −0.1% | −0.2% | −0.3% | −0.5% | −0.8% |
|---|---|---|---|---|---|---|---|---|
| 6B 的 τ（曲率 = 1B） | 0.34 | 0.29 | 0.27 | 0.25 | 0.23 | 0.21 | 0.18 | 0.14 |
| 6B 的 τ（曲率減半） | 0.39 | 0.31 | 0.27 | 0.23 | 0.19 | 0.17 | 0.12 | 0.08 |

- **決策規則：**
  - E1 比 CL 好 0.1% 以上：用表中兩列的平均，下限 0.12
  - 差異在 ±0.1% 以內：τ 用 0.27
  - E1 比 CL 差 0.1% 以上：維持 0.39
- **雜訊：** 1B 時一對 seed 的差異是 0.02%，所以 ±0.1% 的門檻還算可靠。

### E2（可以跟 E1 同時跑）：3B 測 lr
- **配置：** E1 的設定，兩個 lr 都乘 0.8：Muon 1.04e-3、AdamW 5.92e-3
  - τ 維持 0.22，所以 wd 變成：Muon 0.191、embedding 0.0335
- **比較方式：** E2 對 E1，是在建議的 τ 下測 lr。
- **預期結果（E2 − E1）：**
  - 最佳 lr 的指數 β = 0.125（目前規則）：+0.1 到 +0.2%
  - β = 0.2：大約打平（+0.03%）
  - β = 0.32：−0.12 到 −0.25%
  - β ≈ 0（Straight to Zero）：+0.24 到 +0.48%
- **決策規則：**
  - E2 比 E1 好 0.1% 以上：6B Muon lr 用 0.9e-3、AdamW 用 5.9e-3
  - 打平：Muon 1.05e-3、AdamW 6.8e-3
  - E2 比 E1 差 0.1% 以上：Muon 1.2e-3、AdamW 6.8e-3

### 如果只能跑 2B
- 需要兩個跑：2B、CL 配方，τ 0.39 和 τ 0.27 各一個，各約 2.3 小時。
- 預期差異只有 E1 的一半左右：Power Lines 下約 −0.1 到 −0.4%，τ 固定下約 +0.03 到 +0.1%，比較容易被雜訊蓋過。

### 如果都不跑
- 直接用第 5 節的建議：τ 0.2、Muon 1.05e-3、AdamW 6.8e-3。

---

## 7. 反證據
1. **Wang & Aitchison 的「τ 固定」**
   - 這是我們目前規則的根據。
   - 但它的證據是多 epoch 的實驗；作者自己在定稿版說單趟 LLM 訓練要用 Power Lines（p.8）。
2. **Completed-P 的 SDE 規則等於 τ 固定**
   - 它的 lr 遷移有實驗支持。
   - 但 λ 的部分是推導出來的，而且只用 cosine 排程。
   - 它同時也要求 lr ×0.41。如果它是對的，我們的 6B lr 會太高。不過 4.1 節的 BY 拆解已經排除了 β ≈ 0.5。
3. **Fantastic 說「Muon 的 wd 不需要隨規模調」**
   - 但 λ 固定、η 也幾乎固定，正好代表 τ ∝ 1/D。我重算的結果也是 λ 的最佳值不變。
   - 所以它不是「τ 固定」的證據，反而指向 τ 要降得比 Power Lines 更快。
   - 格點只有 0、0.1、0.2，而且只有 130M。
4. **Straight to Zero 的「D2Z 時 peak lr 幾乎不變」**
   - 跟我們 BY 推出的 β ≈ 0.2 有輕微衝突。
   - 可能的解釋：BY 的 τ 固定在 0.39，依 Power Lines 在 3B 已經偏長，降 lr 會部分彌補這一點；或者 Muon 本來就需要降得比 AdamW 快（3.4 節）。
5. **Step Law、von Rütte 的正指數**：批次跟著變大的路徑造成的，固定批次時不適用。
6. **Semenov 說「lr 在 8 倍長度內平順轉移」**：格點只有 2–3 倍間距，解析度不足。
7. **Pre-training under infinite compute（wd 要 30 倍）**：來自重複資料，我們每個 token 只看一次。
8. **我們自己的 1B 最佳 τ 比 Power Lines 外推值低 3 倍**
   - 這證明 Power Lines 的絕對值不適用於 Muon 和低 TPP。
   - 它的斜率在 TPP < 20 也沒有被驗證過。這是整份建議最大的不確定性，E1 就是為了這點設計的。
9. **Straight to Zero 在 2 TPP 時 D2Z 輸給 10×**
   - 代表在極低 TPP 時，「偏差」項，也就是需要高 lr 去走遠，比雜訊重要。
   - 這可能讓 1B 到 2B 之間的行為跟高 TPP 不同。這點交給第 6 號研究員處理。

---

## 8. 結論
- **τ：** 從「固定 0.39」改成「隨 tokens 變小」，6B 用 **0.2**（0.15–0.27）。
  - 單趟訓練的實測研究全部支持「變小」。
  - 1B 的數據顯示短 τ 側比較平，所以往下調的風險小。
  - 這是 T3a 裡最值得改、預期收益最大（0.3–1%）的一項。
- **lr：** Muon 從 1.2e-3 微調到 **1.05e-3**，AdamW 維持 **6.8e-3**。
  - DeepSeek 的 −0.125 是混合了 N、D、B 的擬合，對固定模型沒有根據。
  - 文獻中 −0.3 到 −0.5 的指數，大多是 λ 固定加上衰減不到 0 造成的，不能直接套用。
  - 我們的 BY 拆解出 β ≈ 0.2。
  - 預期影響在 0.1% 以內。
- **β2、NorMuon β2、momentum：** 都不改。
- **warmup：** 1% 可以；改成 229 步（0.5%）是可選的小改動。
- **10-18 前：** 跑 E1（3B、τ 0.22、對照 CL）；有餘裕的話同時跑 E2（3B、τ 0.22、lr ×0.8）。用第 6 節的表定下 6B 的 τ 和 lr。

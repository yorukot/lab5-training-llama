# 01 可優化面向與假設（步驟 1–2）

讀完 `src/` 與 66 個實驗紀錄後的判斷。每一軌列出：
- 機制
- 要成立的前提
- 預期效果
- 跟已試過的關係
- 要查的關鍵詞

收益大小以 1B 為準；「6B」欄是推估效果到 6B 剩多少。

## 總覽：剩下的空間在哪裡
- **已調得很細的部分（再調多半 <0.3%）：** 1B 的 lr、τ、warmup、衰減長度、momentum、norm lr
- **收益大、但在 6B 會變小的：** 蒸餾。1B −9.0%，3B 約 −7%，6B 推估約 −6%
  - 怎麼讓 6B 留下更多，是最值得投入的問題
- **完全沒碰過的四件事：**
  1. OJ 用純 bf16 推論造成的損失
  2. 固定模型、tokens 從 1B 增加到 6B 時，最佳 lr、τ、批次怎麼變；現在是沿用三條規則，只在 3B、4.5B 各有一點驗證
  3. 批次大小在訓練中途改變（臨界批次會隨 loss 下降而變大）
  4. 蒸餾目標的細節：α 隨時間變化、tail 怎麼分、token 加權

## T1a 蒸餾 A：損失形式、top-k 偏差、tail 處理
- **現況：**
  - target = α·p_T(top-32)，加上 (1 − α + α·tail_T) 全部給正確答案，對學生做 cross-entropy，也就是對這個混合 target 做 forward KL
  - tail_T = 1 − Σ top-32，平均約 9%
- **假設 1（tail 的分配）：** tail 全部給正確答案，等於告訴學生「老師 top-32 以外，除了正確答案都是 0」
  - 這會讓學生在 tail 太自信，而評分看的是整個分布
  - 替代做法一：tail 依學生自己的 tail 分布分配（stop-grad），只約束 tail 的總量
  - 替代做法二：tail 均勻分配
  - 偏差來源：Sparse Logit Sampling（ACL 2025）指出 top-k 快取的 KD 是有偏差的
  - 預期：0.1–0.5%，不確定
- **假設 2（損失形式）：** ppl 的評分就是 forward CE
  - 理論上 forward KL 跟評分一致；reverse KL、skew KL 偏向找眾數，對 ppl 可能更差
  - 要查文獻確認，在 ppl 指標下有沒有反例
- **不能做的：**
  - 換溫度（快取的 logsumexp 只對 T=1）
  - 換 top-K
  - 重算快取
- **關鍵詞：** knowledge distillation language model pretraining; top-k / sparse logits; logit caching; offline distillation; forward vs reverse KL; skew KL; TVD; f-divergence distillation; calibration of distilled student; tail probability; label smoothing via teacher; DistiLLM; MiniLLM; generalized KD (GKD); "Rethinking Kullback-Leibler divergence"

## T1b 蒸餾 B：α 排程、token 加權、收益隨 tokens 的變化
- **現況與證據：**
  - α 在整段固定，1B 最佳約 0.86，6B 計畫 0.8
  - CI（α 1.0）在前 10% 贏 CH（α 0.9），20% 之後輸，而且越差越多
  - 也就是說，最佳 α 隨訓練下降
- **假設 3（α 排程）：** α 從約 1.0 降到約 0.6，可能比固定 0.8 好
  - 可能的形式：線性下降、跟 lr 連動、依「學生 vs 老師 loss 差距」自適應
  - 預期 1B −0.2% 到 −0.5%；在 6B 可能更重要，因為後期占比更長
- **假設 4（token 加權）：** 依老師的熵、老師的信心、或學生與老師 loss 的差距，給每個位置不同的 α
  - 正確答案不在 top-32 的 7% 位置，可能該給正確答案更多權重
  - 參考：Rho-1 的 excess loss，以及 ATKD 一類的自適應教學
  - 注意：評分是所有 token，丟掉任何 token 都可能傷 ppl
- **假設 5（收益的 scaling）：** Distillation Scaling Laws（ICML 2025）可以預測固定老師下，學生 loss 隨 tokens 的變化
  - 用途：交叉檢驗 6B 推估（11.1–11.2）和 α 該怎麼隨 tokens 調整
- **關鍵詞：** distillation scaling law; teacher weight / mixing coefficient schedule; annealed distillation; curriculum distillation; adaptive distillation weight; token-level distillation weighting; teacher entropy / confidence; excess loss; Rho-1; MiniPLM; pre-training distillation; capacity gap; teacher-student gap; early-phase distillation; distillation then pure LM

## T2 精度與評測對齊（bf16 推論）
- **現況：** OJ 把 fp32 checkpoint 轉成 bf16，整個 forward 用 bf16
  - 我們訓練與 eval 時 residual 和 norm 保持 fp32
  - OJ 一直比 in-training 高 1.15–1.19%，但其中 bf16 和文件集各佔多少還沒量過
- **假設 6：** bf16 的 residual 加總每層都會捨入，16 層、8192 位置下累積，可能造成 0.3–1% 的 ppl 損失
  - 量化 scaling law（Scaling Laws for Precision，ICLR 2025）說：訓練 tokens 越多，事後量化的損失越大
- **可能的對策：**
  - (a) 最後一段訓練用純 bf16 forward（權重仍保留 fp32 主副本），讓模型適應 bf16 的雜訊，類似 QAT
  - (b) 上傳前自己挑 bf16 的捨入方向，例如做少量 bf16 約束的校正
  - (c) 如果量出來主要是文件集差異，就沒得改，只是校正 6B 的推估
- **第一步是量測（需要你同意，約 30 分 GPU）：** 用 OJ 的評估程式評 CL 在 10k 篇上的分數，比較 bf16 與 fp32
- **關鍵詞：** bfloat16 inference degradation; low-precision inference perplexity; precision scaling law; quantization-aware training; post-training quantization degradation with tokens; mixed precision residual stream; training-inference mismatch precision; stochastic rounding; bf16 accumulation error; fp32 master weights

## T3a 6B 遷移：lr、τ、β2 隨 tokens 變化
- **現況：**
  - lr × (tokens/1B)^−0.125，來自 DeepSeek，但那是跨模型大小的 compute fit，不是固定模型只增加 tokens
  - τ 固定 0.39；β2 半衰期固定
- **證據：**
  - BY（3B）時，1B 的「lr ×1.25」收益只剩 1/3，顯示 3B 的最佳 lr 可能比規則預測的更低
  - Power Lines（NeurIPS 2025）說最佳 τ 會隨 tokens/參數比下降，3B 推估約 0.225
- **假設 7：** 固定模型時，最佳 lr 隨 tokens 下降得比 −0.125 快
  - 參考：Scaling Optimal LR Across Token Horizons（ICLR 2025）的指數可能更陡
  - 如果是這樣，6B 應該用更低的 lr
- **假設 8：** 6B 的最佳 τ 比 0.39 小，例如 0.2–0.3
  - 但 1B 時 τ 在 0.27–0.55 之間曲線很平，影響應該小
- **驗證：** CM（4.5B）的結果可以直接檢驗：實際值比推估好還是差
- **Muon 特有的問題：** Muon 的 lr 遷移跟 AdamW 是否不同
  - 參考：Moonlight、Practical Efficiency of Muon、Muon 的超參數 scaling
- **關鍵詞：** optimal learning rate token horizon; learning rate scaling with dataset size; hyperparameter transfer across data scale; weight decay scaling; AdamW timescale; EMA view of weight decay; power lines; tokens per parameter; compute-optimal hyperparameters; DeepSeek LLM hyperparameter scaling; Step law; Muon hyperparameter scaling; beta2 batch size scaling

## T3b 6B 遷移：批次大小與吞吐
- **現況：**
  - 1B 時 131k 最好：262k +1.53%，lr 沒重調；65k +17.7%，lr 也沒重調
  - 6B 沿用 131k
- **假設 9：** 最佳批次隨資料量變大
  - 臨界批次 CBS ∝ D^~0.4–0.5（How Does Critical Batch Size Scale，ICLR 2025；Power Lines）
  - 6B 是 1B 的 6 倍，最佳批次可能變成約 2 倍，也就是 262k
  - 如果成立，262k 在 6B 至少打平，而且步數減半、GPU 利用率更好，時間更少
- **假設 10（批次漸增）：** 臨界批次在訓練中隨 loss 下降而變大
  - 前期用 131k、後期用 262k，可能比全程固定更好又更快
  - 參考：GPT-3 / Llama 的 batch ramp、Seesaw 一類的方法
  - 要處理的換算：lr、τ、β2 怎麼跟著批次一起改
- **吞吐：**
  - Newton-Schulz 分散到多卡約 −16% 時間
  - FP8 等加速方法不直接降 ppl，但給 64 H200-h 留出空間
- **關鍵詞：** critical batch size; optimal batch size scaling data; batch size warmup / ramp / schedule; gradient noise scale; batch size and learning rate scaling rule; square-root scaling; SDE scaling rules; Muon batch size; large batch Muon; Seesaw; FP8 training throughput

## T4 排程與權重平均
- **現況：** 1% warmup，最後 80% 線性降到 0；1B 擬合最佳衰減長度約 76%
  - EMA-Nesterov 一步開到 0.6，開啟時會跳升
- **假設 11：** 6B 的步數是 1B 的 6 倍，最佳衰減比例可能改變
  - 參考：Scaling Law with LR Annealing、Straight to Zero、Hägele 等
- **假設 12：** 在衰減段做權重平均（LAWA、EMA）
  - 學到 0 的線性衰減下，通常收益很小；要查有沒有反例
- **假設 13：** EMA-Nesterov 的 β 在幾 % 的訓練內漸進開啟
  - 目的：保留 CF 的最終收益，又消除開啟時的跳升
  - 3B 的跳升比 1B 更大，所以 6B 可能更重要
- **關鍵詞：** warmup-stable-decay; cooldown length; decay fraction; learning rate annealing scaling law; linear decay to zero; river valley loss landscape; weight averaging LAWA; EMA of weights pretraining; checkpoint averaging; schedule-free; lookahead optimizer; Nesterov outer momentum; DiLoCo outer Nesterov; momentum warmup

## T5 優化器
- **現況：**
  - NorMuon + Polar Express，Moonlight RMS 縮放，EMA-Nesterov
  - 試過但沒贏的：Sophia、long-side NorMuon、head damping 等
- **找什麼：** ICLR / ICML 2026、NeurIPS 2025–2026 中，在約 1B、資料少的區間，跟調好的 Muon 或 NorMuon 比仍然贏的方法
  - 例如：AdaMuon、Dion、PolarGrad、MARS、AdEMAMix、cautious update mask（只試過 cautious wd）、SOAP / Kron 跟 Muon 在低 tokens/參數區間的比較
  - embedding 用什麼優化器
- **門檻：** 只接受對照組有調參的證據
  - 優化器 benchmark 論文（Fantastic Pretraining Optimizers、Benchmarking Optimizers for LLM Pretraining）指出多數宣稱的加速在公平比較下會消失
- **關鍵詞：** Muon optimizer; orthogonalized momentum; spectral norm descent; Newton-Schulz; Shampoo; SOAP; Kron / PSGD; AdEMAMix; MARS variance reduction; cautious optimizer; optimizer benchmark pretraining; matrix-whitening optimizer; Dion; Scion / unconstrained LMO; NorMuon; AdaMuon; embedding optimizer

## T6 資料順序、課程與盲點
- **限制：** 只能在快取的 6B 列內重排，不能多選資料
- **假設 14：** 依長度或品質安排順序，可能讓前期學得更快
  - 例如先放短文件，或衰減段放高品質文件
  - 但 eval 是自然分布，衰減段偏向高品質可能傷 ppl，要查證據
- **盲點清單（還沒碰過）：**
  - 初始化 std 的設法，例如 depth-scaled、µP 式
  - grad clip 1.0 是否太緊：前期 grad norm 到 8.5
  - z-loss
  - output logit 的處理
  - 訓練時加輔助 loss，例如 multi-token prediction；評測時拿掉，所以架構不變
  - 文件邊界 token 的處理
- **關鍵詞：** curriculum learning pretraining; data ordering; sequence length curriculum; dataset decomposition; quality-based ordering; annealing data; initialization scaling; gradient clipping adaptive; z-loss; multi-token prediction small models; auxiliary loss pretraining; token dropping; data repetition order

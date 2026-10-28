# T1a 蒸餾 A：損失形式、top-k 快取的偏差、tail 機率的處理

研究員 1 號，2026-10-17。範圍：只能用現有快取（T=1 的 top-32 logits 加上完整 logsumexp）時，蒸餾 loss 怎麼改能讓 OJ ppl 更低。

## 0. 先講結論（一段）
- **現在的 target 有系統性偏差，方向是「低估 tail」：**
  - 現在的 target 是 α·p_T(top-32)，剩下的 1 − α·m 全部給正確答案（m = top-32 總機率）
  - 對資料取期望值後：top-32 多拿了 α(1−m)·p_data，tail 只剩 (1−α·m) 倍
  - 這就是 Sparse Logit Sampling（ACL 2025）裡的「Naive Fix」，它在 K=20/50 只拿到完整蒸餾（FullKD）收益的 73%/83%
- **最值得試的改動（C1）：** tail 改成「只有答案真的落在 tail 時才給它，而且給滿」，在期望值上無偏，程式約 4 行
- **最便宜的第一步（C2）：** 不用訓練，先量現有 checkpoint 的最佳溫度 T* 與 tail 校準
  - 量到 KD 學生的 T* > 1、CE 學生的 T* ≈ 1，就證實偏差真的存在；否則 C1 降級
  - T* 可以直接併進最後一層 RMSNorm 的權重，算一個免訓練的小改進
- **損失形式不要換：** 預訓練、以 ppl 為指標時，forward KL（也就是我們現在的 CE 到混合 target）一致勝出
  - reverse KL、skew KL、TVD、α-β、AKL 的證據幾乎都來自指令微調，指標是 ROUGE-L
  - 01-levers 提的兩個替代做法（tail 依學生分配、tail 均勻分配），文獻已經直接測過，都比現在的做法差

## 1. 搜尋紀錄

### 1.1 文獻庫查詢（共 50 組，含 1 組只算總數）
- 命中 >300 的 Q00（2654）、Q01（1684）沒有直接看，而是細分成 Q01a/Q01b/Q01c 和 Q32–Q39
- 篩選欄空白表示沒有第二個條件；有的話，標題加摘要要同時符合兩個 regex
- 範圍：整個索引（6 大會議 2023–2026），沒有限制會議或年份

| 編號 | regex | 篩選（也要符合） | 命中 |
|---|---|---|---|
| Q00_all_distill | `distill` | — | 2654 |
| Q01_kd_lm | `knowledge distillation\|distill(ation\|ing\|ed)?` | `language model\|\bLLMs?\b\|\bGPT\|transformer\|pre-?train\|autoregressive\|next-token\|token-level` | 1684 |
| Q02_pretrain_kd | `pre-?train(ing)? (with \|via )?(knowledge )?distillation\|distillation (for\|during\|in) pre-?train\|distill.{0,60}pre-?train(ing)? (of\|stage\|phase\|corpus)` | — | 13 |
| Q03_topk_sparse | `top-?k (logit\|probabilit\|token\|distill\|knowledge)\|sparse (logit\|knowledge distillation\|soft ?label\|teacher)\|logit (cach\|sampling\|compression\|storage)\|cached (teacher\|logit)` | — | 9 |
| Q04_fwd_rev_kl | `reverse KL\|reverse Kullback\|forward KL\|mode-?seeking\|mode-?covering\|inclusive KL\|exclusive KL` | `distill` | 27 |
| Q05_skew_tvd | `skew(ed)? (KL\|Kullback\|reverse\|forward)\|DistiLLM\|total variation.{0,60}distill\|distill.{0,80}total variation\|\bTVD\b` | — | 6 |
| Q06_fdiv | `f-divergence\|alpha-?beta divergence\|\bAB-?div\|Jensen-?Shannon\|Renyi\|Rényi\|alpha-divergence\|α-divergence` | `distill` | 11 |
| Q07_onpolicy | `MiniLLM\|\bGKD\b\|on-?policy distill\|generalized knowledge distillation\|student-generated\|self-generated (output\|sequence).{0,40}distill\|imitation.{0,40}distill` | — | 74 |
| Q08_rethinkkl | `rethinking (the )?(Kullback\|KL)\|adaptive (KL\|Kullback\|divergence)\|\bAKL\b\|divergence (choice\|selection).{0,40}distill\|which divergence` | — | 6 |
| Q09_calib | `calibrat\|overconfiden\|under-?confiden\|expected calibration error` | `distill` | 82 |
| Q10_ls | `label smoothing\|soft labels?\|soft targets?\|dark knowledge` | `language model\|\bLLMs?\b\|\bGPT\|transformer\|pre-?train\|autoregressive\|next-token\|token-level` | 49 |
| Q11_temp | `temperature` | `knowledge distillation\|distill(ation\|ing)` | 20 |
| Q12_tail | `(long\|heavy)[- ]tail(ed)? (of the )?(vocab\|token\|distribution\|probabilit)\|tail (probabilit\|mass\|token)\|probability mass (outside\|beyond\|of the tail)\|residual (probability\|mass)\|truncat(ed\|ion) (teacher\|distribution\|softmax\|vocab)` | — | 134 |
| Q13_offline | `offline (knowledge )?distillation\|pre-?comput(ed\|e) (teacher\|logit\|soft)\|teacher (logit\|output)s? (are\|were) (stored\|cached\|precomputed)\|storage.{0,60}(teacher\|logit)` | — | 9 |
| Q14_kdscaling | `distillation scaling law\|scaling law.{0,60}distill\|distill.{0,60}scaling law\|compute-optimal distill` | — | 4 |
| Q15_impsamp | `importance sampl.{0,60}distill\|distill.{0,80}importance sampl\|unbiased.{0,60}distill\|distill.{0,60}unbiased\|random sampling knowledge distillation\|sampled (logit\|softmax).{0,40}distill` | — | 1 |
| Q16_dkd | `decoupled knowledge distillation\|non-?target (class\|logit\|token\|knowledge)\|target class.{0,40}non-?target\|\bDKD\b\|\bNTKD\b` | — | 12 |
| Q17_gap | `capacity gap\|teacher-student gap\|curse of capacity\|larger teacher.{0,60}(worse\|hurt\|not always)\|stronger teacher` | `distill` | 24 |
| Q18_seq_tok | `sequence-level knowledge distillation\|token-level (knowledge )?distillation\|word-level distill` | — | 14 |
| Q19_minitron | `Minitron\|prun(e\|ing).{0,40}distill.{0,60}(LLM\|language model)\|compact language models? via\|Sheared LLaMA` | — | 8 |
| Q20_ppl | `perplexity` | `knowledge distillation\|distill(ation\|ing)` | 22 |
| Q21_entropy | `teacher('s)? (entropy\|confidence\|uncertainty)\|entropy-?(based\|aware\|weighted).{0,40}distill\|distill.{0,60}entropy` | — | 14 |
| Q22_logitstd | `logit standardi[sz]ation\|normali[sz]ed (KD\|knowledge distillation\|logit)\|logit (normali[sz]ation\|matching)\|\bMSE\b.{0,40}logit\|logit.{0,40}\bMSE\b` | `distill` | 3 |
| Q23_bias | `bias(ed)?.{0,40}(teacher\|distill\|gradient estimat)` | `knowledge distillation\|distill(ation\|ing)` | 23 |
| Q24_whykd | `why (does )?(knowledge )?distillation (work\|help)\|understanding (knowledge )?distillation\|does knowledge distillation really\|statistical perspective.{0,40}distill\|bias-variance.{0,40}distill\|distillation.{0,40}bias-variance` | — | 3 |
| Q25_miniplm | `MiniPLM\|difference sampling\|teacher-guided (data\|pre-?train)\|knowledge distillation for pre-?training\|distillation.{0,40}from scratch` | `language model\|\bLLMs?\b\|\bGPT\|transformer\|pre-?train\|autoregressive\|next-token\|token-level` | 2 |
| Q26_mixdiv | `(interpolat\|mixture\|mix\|combin)(ed\|ing\|e)? (of )?(forward and reverse\|KL divergences\|divergences)\|(forward\|reverse) and (forward\|reverse) KL\|symmetric KL\|bidirectional KL\|Jeffreys` | — | 18 |
| Q27_wass | `(Wasserstein\|optimal transport).{0,80}distill\|distill.{0,80}(Wasserstein\|optimal transport)` | `language model\|\bLLMs?\b\|\bGPT\|transformer\|pre-?train\|autoregressive\|next-token\|token-level` | 5 |
| Q28_smalllm_kd | `(small\|compact\|tiny\|efficient) language models?.{0,120}distill\|distill.{0,120}(small\|compact\|tiny) language model` | — | 41 |
| Q29_gemma_tech | `Gemma\|Llama 3\.2\|Qwen ?2\.5\|SmolLM\|MiniCPM\|Phi-[1234]` | `distill` | 66 |
| Q30_vocab_proj | `(full\|entire) vocabulary.{0,80}(distill\|teacher)\|distill.{0,80}(full\|entire) vocabulary\|top-?k (approximation\|truncation)\|truncated (KL\|distribution)` | — | 4 |
| Q31_ce_kl | `cross-?entropy.{0,60}(KL\|Kullback).{0,60}(distill\|teacher)\|(distill\|teacher).{0,60}cross-?entropy.{0,40}(KL\|Kullback)\|hard (label\|target).{0,40}soft (label\|target)\|ground[- ]truth (label\|token).{0,60}(teacher\|distill)` | — | 7 |
| Q01a_kd_lm_div | `(knowledge distillation\|distill(ation\|ing)).{0,300}(KL\|Kullback\|divergence\|objective\|loss function)\|(KL\|Kullback\|divergence).{0,300}(knowledge distillation\|distill(ation\|ing))` | `language model\|\bLLMs?\b` | 122 |
| Q01b_kd_lm_pretrain | `(knowledge distillation\|distill(ation\|ing))` | `(pre-?train(ing\|ed)? (a \|the )?(student\|small\|language\|LLM\|from scratch)\|pretraining stage\|pre-?training corpus\|during pre-?training\|tokens)` | 253 |
| Q01c_kd_teacher_prob | `(teacher'?s? (probabilit\|distribution\|logits\|soft\|output distribution\|predictions))` | `language model\|\bLLMs?\b\|token` | 25 |
| Q32_tokenlevel_kd_lm | `token-level\|per-token\|vocabulary` | `(knowledge distillation\|distill(ation\|ing)).{0,200}(language model\|\bLLMs?\b)\|(language model\|\bLLMs?\b).{0,200}(knowledge distillation\|distill(ation\|ing))` | 51 |
| Q33_dense_vs_sparse | `(dense\|full).{0,30}(teacher )?(logit\|distribution\|soft label)s?.{0,80}(sparse\|top-?k\|truncat)\|(sparse\|top-?k\|truncat).{0,80}(dense\|full) (teacher )?(logit\|distribution)` | — | 1 |
| Q34_specdec_div | `DistillSpec\|speculative decoding.{0,80}distill\|distill.{0,80}speculative decoding` | — | 6 |
| Q35_proper_scoring | `proper scoring\|strictly proper\|log(arithmic)? score.{0,40}calibrat\|calibration.{0,60}(perplexity\|log-?likelihood\|cross-?entropy)` | `language model\|\bLLMs?\b\|next-token\|token` | 9 |
| Q36_temp_scaling_lm | `temperature scaling\|post-?hoc calibration\|recalibrat` | `language model\|\bLLMs?\b\|token` | 74 |
| Q37_variance_kd | `variance.{0,60}(distill\|soft label\|teacher)\|(distill\|soft label\|teacher).{0,60}variance (reduction\|of the gradient)\|noisy labels?.{0,60}distill` | — | 18 |
| Q38_kd_mixing_alpha | `(mix(ing)?\|interpolat\|weight(ing)?\|balanc).{0,40}(ground[- ]truth\|hard[- ]label\|cross-?entropy\|NLL).{0,60}(distill\|teacher\|KL)\|distillation (weight\|coefficient)` | — | 4 |
| Q39_rkd_pretrain_kl | `(KL\|Kullback\|forward\|reverse).{0,200}pre-?train` | `knowledge distillation\|distill(ation\|ing)` | 12 |
| Q40_renorm | `renormali[sz]\|unnormali[sz]ed\|re-?normali[sz]` | `teacher\|distill` | 5 |
| Q41_bucket | `(catch-?all\|ghost\|residual\|remaining\|rest-of-vocabulary\|other)[- ](token\|class\|bucket\|bin\|mass)` | `distill\|teacher\|top-?k` | 10 |
| Q42_censored | `censored\|truncated (output\|logit\|probabilit\|distribution)\|partial (logit\|probabilit\|distribution) (access\|information)\|top-?k (API\|access\|log-?prob)` | — | 55 |
| Q43_kd_ppl_eval | `(lower\|better\|improv\w*\|reduc\w*) (validation \|held-?out \|test )?perplexity` | `knowledge distillation\|distill(ation\|ing)` | 3 |
| Q44_kd_variance_theory | `(knowledge distillation\|distill\w*).{0,200}(variance\|bias\|unbiased\|Bayes\|calibrat)` | `theor\|prov\|analy` | 79 |
| Q45_lm_kd_loss_compare | `(compar\w*\|empirical study\|systematic\|benchmark\w*\|revisit\w*).{0,100}(knowledge distillation\|distillation) (objective\|loss\|method\|divergence)s?` | — | 22 |
| Q46_minitron_nemotron | `Nemotron\|Minitron\|Llama-?3\.[12]\|Gemma-?[23]\|Phi-?4\|Qwen-?3` | `distill\|teacher` | 97 |

### 1.2 統計
- **標題：** 合併去重後看過 1,127 個
  - Q02–Q31：618 個
  - Q01a–c、Q32–Q39 新增：357 個
  - Q40–Q46 新增：152 個
- **摘要：** 約 40 篇，用 `--abstracts` 或 jsonl 讀
- **全文：** 下載並轉文字 41 篇，放在 `/home/yorukot/ml/papers/pdf/T1a/`；精讀約 18 篇，其餘用 grep 找段落
- **arXiv 與網路：** 共 8 次 arXiv 存取（5 篇 PDF 經 download.py、1 篇 curl、2 次 WebFetch），每次間隔 ≥2 秒
  - WebSearch 4 次
  - WebFetch：Gemma 3 技術報告、arXiv 2608.03796、arXiv 2609.34386 的摘要
- **引用追蹤：**
  - Sparse Logit Sampling 的參考文獻：FIRST、Sultan 2023（KD≈LS）、Zhou 2021 bias-variance（ICLR 2021，不在庫的年份）、Raman 2023（workshop，不在庫）、Rethinking KL（COLING 2025，用 arXiv 補）
  - Distillation Scaling Laws 的參考文獻：Fan 2024 calibration、AKL
  - Minitron、Peng 2024、Strong Teacher 等的參考文獻：Gemma 2/3、Llama 3.2、AFM
- **反證據搜尋：** Q04–Q08、Q26、Q40–Q41 專門找「替代 loss 或 tail 處理沒用、更差」的結果

## 2. 論文表

相符度的標準：
- **高：** 從頭預訓練、top-k 快取或完整分布，而且看 LM loss 或 ppl
- **中：** 預訓練但看下游分數，或模型是剪枝後再訓練
- **低：** 指令微調或分類任務

| 出處 | 標題 | 設定 | 主張 | 效果大小 | 相符度 | 位置 |
|---|---|---|---|---|---|---|
| ACL 2025 | Sparse Logit Sampling: Accelerating KD in LLMs | 300M 學生、3B 老師、約 10B tokens（§6.1，約 33 tokens/參數）、預訓練、forward KL；另有 8B→3B、100B tokens | top-K 快取的 target 有偏差，學生在 top-K 過度自信、tail 信心不足；用 importance sampling 的隨機取樣（RS-KD）無偏 | CE 2.81、FullKD 2.74（nats）。tail 給答案（=我們 α=1）：K=20 是 2.76（73%）、K=50 是 2.76（83%）、K=100 是 2.75（97%）。ghost token（K=50）2.77（59%）。label smoothing（K=50）2.85（−73%，比 CE 還差）。RS-KD 12 個樣本 ≈ FullKD；梯度夾角 top-50 48°、RS 4°。8B→3B：vanilla top-50 2.40 > CE 2.37 > RS 2.35 ≈ Full 2.34 | 高 | Table 1 p.2；§2.2 p.2–3；Table 2 p.4；Table 3 p.5；Table 6 p.6；Table 8 p.7；Table 10 p.8；App. A.4–A.5 p.16–17 |
| ICML 2025 | Distillation Scaling Laws | 學生 198M–2.72B、老師 546M–7.75B，都是 20 tokens/參數、預訓練、完整分布 | top-k/top-p 截斷（重新正規化）有害，混入 CE 可救回；forward KL 勝 reverse KL；τ=1 最好；λ>0.5 時曲線平 | k=128：+0.11 nats；p=0.9：+0.13 nats；k=128 加 λ=0.7：與完整分布差 <0.01 nats。forward KL 比 reverse KL 低 0.28 nats。用完整分布蒸餾的學生對資料校準良好 | 高 | App. G.1 p.60–61；G.2 p.61；G.4 Fig.54 p.63；G.5 Table 7 p.63–64；E.8 p.53–56 |
| NeurIPS 2024 | Compact LMs via Pruning and KD（Minitron） | 15B 剪成 8B/4B 再訓練（約 94B tokens），也就是預訓練式的重新訓練 | logit 的 forward KL 是最好的 loss；top-K ≤100 明顯掉分；τ=1 最好 | LM loss：只用 KLD 2.107、CLM+KLD 2.117、RKLD 2.140、cosine 2.134、MSE 2.144；WikiText ppl 8.72 vs RKLD 9.01。depth-pruned：RKLD 2.665 vs KLD 2.155 | 中 | App. A.5 p.17–18，Table 15–16 |
| ACL 2025 | Pre-training Distillation for LLMs: A Design Space Exploration | GLM-4-9B → 1.9B、100B tokens；top-p 0.95 + top-100 重新正規化；看下游 | NLL ≈ KLD > MSE；靜態 α 0.9 最好；k、p 影響小；τ ≤2 差不多 | 下游平均 +1.6%（相對）；各個 k 在 +1–4% 之間；MSE 明顯下降 | 中（不是看 ppl） | Fig.3、Table 2 p.4；Table 4–5 p.5–6 |
| NeurIPS 2026 | Strong Teacher Not Needed? On Distillation in LLM Pretraining | 1.7B 學生、50B tokens；老師 0.7–8B、10–300B tokens；完整 logits、forward KL、τ=1 | 純蒸餾（α=1）幾乎總是比最佳 α 差，老師越強差距越小；同一 domain 的 ppl 收益比 OOD 小 | 8B 老師、80B tokens、純 KD：in-domain ppl +3.7%；弱老師純 KD 會差 60% 以上 | 中高 | Finding 1 p.6；Table 5–6 p.18 |
| ICML 2026 | Don't Ignore the Tail（TAD） | 0.5–1.2B 學生，用老師權重初始化，約 2B tokens；完整 vocab | 把 top-K 與 tail 拆開，放大 tail 的 KL，held-out KL 更低、下游更好 | 下游比 MiniPLM 好 50–120%（相對變化）；需要完整 tail logits | 低–中（我們沒有 tail logits） | §2 p.2–3；Fig.1；Table 1 |
| ICLR 2025 | TAID | TinyLlama 1.1B 以 Phi-3-medium 為老師繼續預訓練，20B tokens | 預訓練時各種 divergence 差異很小，forward KL 接近最好 | Open LLM 平均：KL 39.91、TVD 39.41、AKL 39.11、GJS 39.81、SKL 39.84、SRKL 39.60、TAID 40.10 | 中低（看下游） | Table 2 p.7 |
| ACL 2023 | MixCE | GPT-2 微調 | 混入 reverse CE：生成品質好，但 ppl 變差 | ppl 一律較差（論文自述） | 中（指出 ppl 方向） | §4 p.8 |
| ICML 2024 | Language Generation with Strictly Proper Scoring Rules | 翻譯與 LLaMA 微調 | 用 Brier / spherical 取代 log score，BLEU 上升，但 log score（即 ppl）下降 | 定性：log score 顯著下降 | 中低 | §4.4 p.7 |
| COLING 2025（非六大） | Rethinking KL Divergence in KD for LLMs | 指令微調 | FKL 與 RKL 的最佳點相同；訓練初期 FKL 偏 head、RKL 偏 tail；提出 AKL | ROUGE-L 小幅提升 | 低 | 摘要 p.1 |
| ICML 2024 / ICML 2025 / ICLR 2026 / EMNLP 2025 / ICLR 2024 ×2 / ACL 2023 / ICLR 2026 | DistiLLM（skew KL）、ABKD（α-β）、AMiD、ToDi、MiniLLM（RKL）、GKD、f-div seq KD、Concrete Score Matching | 都是指令微調（Dolly 等），指標 ROUGE-L | 偏 RKL 或 skew 對生成任務較好 | ROUGE-L 提升 1–3 點；沒有 ppl | 低 | 各篇實驗節 |
| NeurIPS 2025 | Why KD Works in Generative Models | GMM 模擬 + SmolLM2 多階段蒸餾 | 老師越尖銳，學生 precision 越高、recall（涵蓋率）越低 | 定性曲線 | 中（top-k 截斷 ≈ 讓老師變尖；ppl 是 recall 類指標） | §1 p.2 |
| ICLR 2026 | Distilled Pretraining | 預訓練蒸餾 + bigram 模擬 | 蒸餾加速學習高熵位置，但傷害 induction / ICL | 定性 | 中（屬於 T1b 的 token 加權） | §4 p.7 |
| NeurIPS 2023 | KD Performs Partial Variance Reduction | 線性模型理論 | KD 等於部分的 variance reduction，權重 λ 要調 | 理論 | 中低 | p.2 |
| ICLR 2026 | SGD-Based KD with Bayesian Teachers | 理論 + 分類 | 準確的 Bayes 機率帶來 variance reduction；有雜訊的老師效果遞減 | +4% 準確率（分類） | 低 | 摘要 |
| NeurIPS 2025 | Better Estimation of the KL Divergence Between LMs | RLHF 的 KL 估計 | Rao-Blackwellized 估計：能精確算的部分精確算，其餘才抽樣，無偏且變異較小 | 變異大幅下降 | 低–中（支持 C1 的結構） | §3 |
| EMNLP 2023 | KD ≈ Label Smoothing: Fact or Fallacy? | 文字分類 | KD 和 LS 把信心推向相反方向；學生會繼承老師的信心 | 定性 | 中低 | 摘要 |
| NeurIPS 2026 | Identified-Set Geometry ... Top-K Censored API | 理論 + Qwen3 | 只有 top-K 時，tail 有不可辨識的範圍，大小由 TV 直徑 U_K 決定 | top-K 蒸餾恢復 12% 能力，完整 logits 56% | 中低 | 摘要、§3 |
| 技術報告（非六大） | Gemma 3 | 預訓練蒸餾 | 每個 token 依老師機率抽 256 個 logits，沒抽到的設 0 後重新正規化 | 沒有消融數字 | 中（業界做法） | §2.2 |
| 技術報告（非六大） | Gemma 2、Llama 3.2、AFM（經 Peng 2024 p.4 轉述） | 預訓練蒸餾 | Gemma 2 用完整分布；Llama 3.2 用 8B/70B 的 logits 當 token target；AFM 用 top-1 | 沒有數字 | 低–中 | — |
| arXiv 2608.03796（非六大） | Efficient KD for LLMs: Offline Top-K Logits and a Fused Chunked KL Loss | 剪枝後恢復訓練，K=100 | 不正規化，tail 不處理 | 沒有 ppl | 低 | — |
| ICLR 2025 | MiniPLM | 離線預訓練 KD，靠資料重新加權 | 用老師與參考模型的差異挑資料 | 下游 +；test loss 下降 | 低（我們不能換資料） | §3 |
| ICML 2024 / ICML 2025 / EMNLP 2024 / ACL 2023 / ACL 2025 | Revisit KD through Calibration、Calibrated LMs & Label Smoothing、FIRST、A Close Look into PLM Calibration、Law of Capacity Gap | 分類或 SFT 校準；容量差距 | 跟老師校準、SFT 校準、老師大小有關 | — | 低 | 摘要 |

## 3. 核心推導：現在的 target 偏在哪裡

定義：
- S：老師的 top-32
- m = Σ_S p_T
- y：資料中的正確答案
- e_y：只在 y 為 1 的向量

| 版本 | 每個位置的 target q | 對 y ~ p_data 的期望值 E[q] | Σq |
|---|---|---|---|
| 現在（V0，= SLS 的 Naive Fix） | α·p_T\|S + (1 − α·m)·e_y | head：α·p_T + (1−αm)·p_data；tail：(1−αm)·p_data | 1 |
| C1（V1，tail 由答案代表） | α·p_T\|S + (1 − α + α·1[y∉S])·e_y | head：α·p_T + (1−α)·p_data；tail：p_data | 期望值為 αm + 1 − α + α·P(y∉S) ≈ 1 − 0.02α |
| ghost / 學生 tail（V2） | α·p_T\|S + α(1−m)·sg(p_S 的 tail 形狀) + (1−α)·e_y | tail 總量對，但形狀只從 (1−α)·e_y 學 | 1 |

- **V0 的偏差：**
  - tail 只拿到 (1−αm) 倍的資料機率
  - 在老師不確定（m 小）的位置，這個倍數大約 0.3–0.5，被拿掉的質量都移到 head
  - α 越大，偏差越大
- **V1 的性質：**
  - 若老師的 top-32 機率對資料是校準的，即 P(y∈S|x) ≈ m，V1 的期望值就是「head 用老師、tail 用資料」，沒有偏差
  - 實際數字：P(y∉S) = 7%，E[1−m] = 9%，所以 tail 會跟資料頻率（7%）對齊，這正是 ppl 要的
  - 梯度要改成 (Σq)·softmax − q，因為 Σq 不再固定為 1
- **玩具模擬（腳本在 `/tmp/claude-1000/-home-yorukot-ml-lab5-training-llama/94460c4a-2e94-42fc-8d1d-1c1378fd5e26/scratchpad/bin/sim.py`）：**
  - 設定：Zipf 型老師、V=32768、K=32、top-32 平均質量 0.88–0.90
  - 理想學生（完全擬合期望 target、p_data = p_T）下，V0 的額外 CE：
    - α 0.8：+0.039–0.050 nats（+4–5% ppl）
    - α 0.9：+0.056–0.075 nats（+6–8%）
    - α 1.0：+0.09–0.12 nats（+10–13%）
  - V1 是 0
- **實際會實現多少：**
  - SLS 在 α=1、K=20–50、300M 的實測只差 FullKD 約 0.02 nats（約 2%），大概是理想值的 15–20%
  - 學生越接近 target，實現的比例越高
  - SLS 在 8B→3B、100B tokens 時，vanilla top-K 與 FullKD 的差距變大（Table 8）
  - 這也可能是「KD 收益隨 tokens 變小」（1B −9%，3B 約 −7%）以及「α 1.0 前期贏、後期輸」（CI vs CH）的部分原因。這點是推測，還沒驗證
- **溫度修正的模擬（同一資料夾的 `sim2.py`）：** 在同一個玩具裡，一個全域溫度 T ≈ 1.1 可以收回 V0 理想偏差的約 57%（0.048 → 0.021 nats）

## 4. 候選改動

### C1：tail 由答案代表（label-sampled tail，無偏版）
- **機制：**
  - 老師 top-32 照舊給 α·p_T
  - 答案在 top-32 外時，答案拿 1；答案在 top-32 內時，答案只拿資料份額 1−α
  - 換句話說，tail 不再「部分」塞給 head 的答案，而是用資料樣本當 tail 的 Monte Carlo 估計
  - 結構上是 Rao-Blackwell 的做法：能精確算的 head 精確算，tail 才抽樣
- **證據：**
  - 現在做法的偏差有證明：SLS App. A.4 式 (5)、§2.2 p.2–3
  - 實測：Naive Fix 在 K=32 附近只拿到 FullKD 收益的約 78%（SLS Table 2 p.4，K=20 是 73%、K=50 是 83%）
  - 無偏的取樣法 RS-KD 用 12 個樣本就追平 FullKD（Table 6 p.6），梯度夾角 4°，top-50 則是 48°（Table 3 p.5）
  - 加重 tail 答案的權重方向對 LM loss 有利：SLS Table 10 p.8，「hard token」lr ×2 加 CE 0.1，比 FullKD 多拿 25% 的差距
  - Busbridge G.4 p.63：截斷有害，但混入真實答案可以救回
  - 注意：沒有任何論文直接測過 V1 這個形式，這是我依上面的證據推導出來的
- **1B 預期：**
  - ppl −0.5%（範圍 −0.2% 到 −1.5%）
  - 信心中低：大約 45% 機率超過 0.3% 的雜訊門檻
  - 推估方式：剩下的空間約是 KD 收益 9% 的 10–25%，V1 能收回其中一部分
- **6B 預期：** −0.5% 到 −1%，可能比 1B 大
  - 理由是學生越接近 target，偏差實現得越多
  - 也可能減緩 KD 收益隨 tokens 的縮小
- **交互作用：**
  - NorMuon：無關，只改 target；每個位置的權重 Σq 落在 0.8–1.8，NorMuon 的列正規化不受影響
  - EMA-Nesterov：正交，作用在權重上
  - α：偏差跟 α 成正比，所以 C1 之後最佳 α 可能往上移（見 C3）；6B 計畫的 α 0.8 不要再往下降
- **有沒有試過：** 沒有。00-context 第 6–7 節沒有任何 tail 處理的變化
- **實作草圖（`src/distill.py` 的 `MixedCrossEntropy.forward`，約 4 行，再改 docstring）：**
  ```python
  ids, teacher = kd_ids[s : s + CHUNK], alpha * kd_probs[s : s + CHUNK]
  tail = (ids != label).all(1, keepdim=True)        # the label is outside the teacher's top 32
  data = 1 - alpha + alpha * tail                   # the teacher's tail, sampled by the data
  total = teacher.sum(1, keepdim=True) + data       # the target's mass, ~1 on average
  ...
  grad = log_p.exp_().mul_(total)  # total * softmax - target
  ```
  - 計算 loss 的兩行不用改
  - 印出的訓練 loss 尺度會稍微變，但 eval ppl 不受影響
- **成本與風險：**
  - 成本：每個位置多一次 32 個 id 的比較，時間可以忽略
  - 風險一：7% 的位置變異增加，權重最高到 1+αm
  - 風險二：失去「tail 質量推向 head 答案」的銳化效果，如果那個效果其實有幫助，C1 可能打平
  - 風險三：老師在 tail 對資料的校準差（9% vs 7%），所以 V1 跟的是資料頻率，不是老師的 tail；對 ppl 這是對的方向
- **1B 測試設計：**
  - **主測：** CJ 配方（α 0.9 + EMA-Nesterov，1B）加 C1，對照 CJ（15.324）
    - ≤ 15.278（−0.3%）算贏
    - > 15.309（贏不到 0.1%）算失敗
  - **看曲線：** C1 的收益應該集中在後段（20% → 100%），跟 CI 輸給 CH 的時段相同；如果只有前段贏，機制就不是 tail 偏差
  - **加測（可選）：** C1 配 α 1.0，見 C3

### C2：先量測，再做免訓練的溫度修正（併進最後一層 RMSNorm）
- **機制：**
  - NLL 是嚴格 proper 的評分，可以拆成「校準」和「銳利度」兩部分
  - 學生如果 head 過度自信、tail 太少，把 logits 除以 T > 1 就能直接降低對資料的 CE
  - Llama 的 logits = RMSNorm(h)·g·W_Eᵀ，把 `model.model.norm.weight` 除以 T 就等於 logits/T；只改權重，架構不變
- **證據：**
  - top-K 學生校準變差：SLS Table 1 p.2，ECE 在 K=25 是 3.2、CE 是 1.2；Table 2 p.4，Naive Fix 在 K=20 是 2.3、K=50 是 1.3、FullKD 0.7
  - 3B 規模也一樣：Table 8 p.7，top-12 的 ECE 4.7，RS 0.2
  - 用完整分布蒸餾的學生是校準的：Busbridge E.8 p.53–56
  - 玩具模擬：T ≈ 1.1 收回約一半的理想偏差
  - 缺口：沒有論文直接報告「預訓練 LM 做 token 級溫度修正後 ppl 下降多少」。ATS（EMNLP 2024）等溫度修正研究都在 RLHF/SFT 後的模型上
- **1B 預期：**
  - CJ 上 −0.1%（範圍 0 到 −0.4%），CE 學生（BU/CF）上約 0
  - 信心：「量得到 T* > 1」中；「超過 0.3%」低
- **6B 預期：**
  - 沒採用 C1：跟 1B 相當或略大
  - 採用 C1：應該接近 0，可以當成 C1 有效的旁證
- **交互作用：** 跟優化器無關；跟 C1 部分重疊，C1 從源頭消除偏差
- **有沒有試過：** 沒有
- **實作草圖：**
  - 不改 `src/distill.py`，另寫一個約 25 行的評估腳本
  - 步驟：讀 checkpoint；在 2–5M 個訓練列的 token 上算 T ∈ {0.95, 1.00, 1.03, 1.06, 1.10, 1.15} 的 CE；取 T*；把 norm.weight 除以 T*；存檔
  - 同時輸出診斷：學生 top-32 的平均質量，以及答案落在學生 top-32 的頻率。過度自信時前者會大於後者
  - 還可以加一個純 CPU 的診斷（D0）：從快取直接算 E[(1−αm)·1[y∉S]]，跟 P(y∉S) 比，就知道現在 target 的 tail 少給了多少
- **成本與風險：**
  - 成本：每個 checkpoint 約 10–20 GPU 分鐘，不用訓練
  - 風險一：不能在 in-training eval 那 10k 篇上調 T。那是 OJ 50k 篇的一部分，等於在測試集上調參
  - 風險二：用來調 T 的 token 可能要算進 6B 預算，需要你確認規則；用 ≤5M tokens，約佔預算 0.08%
  - 風險三：bf16 的捨入影響可以忽略
- **1B 測試設計（這也是 C1 前提最便宜的檢驗）：**
  - 對 BU、CF（無 KD）和 CD/CE/CH/CI/CJ（有 KD）的最終 checkpoint 量 T* 和 ΔCE；CL（3B）也量，看 tokens 的趨勢
  - 預測：KD 的 T* 落在 1.02–1.10，而且隨 α 單調（CI > CH > CE > CD）；無 KD 的 T* ≈ 1.00
  - 判定失敗：KD checkpoint 的 T* ≈ 1.00 且 |ΔCE| < 0.05%，代表偏差沒有實現，C1 降級

### C3：採用 C1 後，重新檢查 α（0.9 → 1.0）
- **機制：** V0 的偏差隨 α 變大，CI（α 1.0）後期輸可能有一部分是偏差造成；偏差拿掉後，最佳 α 可能往上移
- **證據：**
  - 完整分布時 λ > 0.5 的曲線很平，強老師的最佳值接近 1：Busbridge G.1 p.60–61
  - 但純蒸餾通常仍略差於最佳 α：Strong Teacher Table 6 p.18
  - Peng 2024 靜態 α 0.9 最好（p.5）
  - 綜合起來，預期最佳值在 0.9–1.0
- **1B 預期：** 在 C1 + α 0.9 之上再 0 到 −0.3%；信心低
- **6B 預期：** 主要用途是決定 6B 的 α，避免沿用「α 隨 tokens 下降」的舊推論把 α 降太多
- **交互作用：** 與 T1b 的 α 排程直接相關，兩者要一起決定
- **有沒有試過：** CD/CE/CH/CI 掃過 α，但都是在 V0 之下
- **實作：** 只改設定 KD_ALPHA
- **成本：** 一次 1B run（約 70 分鐘、8 GPU）
- **1B 測試：** C1 + α 1.0 對 C1 + α 0.9。贏 ≥0.3% 就改用 α 1.0；輸的話維持 0.9，6B 用 0.85–0.9

## 5. 反證據與不建議的做法
- **tail 均勻分配（等同 label smoothing）：不要做**
  - SLS Table 2 p.4，K=50：2.85，比 CE（2.81）還差，佔差距 −73%
  - 機制：真實 token 分布是 Zipf，均勻分配會把機率給垃圾 token
  - KD≈LS（EMNLP 2023）：LS 和 KD 把信心推向相反方向
- **tail 依學生自己的分布分配（stop-grad，ghost token）：不要做**
  - SLS Table 2 p.4，K=50：ghost 只有 59%，Naive Fix（我們現在）83%，ghost 輸給現在的做法
  - 原因：tail 總量對了，但 tail 形狀完全沒有監督（App. A.5 p.16–17）
  - 我們有 α 0.8 的資料份額，會比 SLS 的 α=1 好一點，但 tail 答案的權重仍只有 1−α，比 V0 和 V1 都少
- **top-32 重新正規化（Gemma 3 的做法）：不要做**
  - 它把 tail 質量按比例分給 head，偏差方向跟 V0 相同而且更糟，因為 tail 的答案拿不到額外質量
  - Busbridge G.4：k=128 重新正規化 +0.11 nats
  - Minitron A.5：top-K ≤100 明顯掉分
- **reverse KL、skew KL、TVD、α-β、AKL、GJS：不要換**
  - 預訓練、以 LM loss 為指標的證據：
    - Busbridge G.5：RKL 比 FKL 差 0.28 nats
    - Minitron Table 15–16：RKLD 2.140 vs KLD 2.107；depth-pruned 時 2.665 vs 2.155
    - TAID 繼續預訓練（Table 2 p.7）：KL 已接近最好，TVD、AKL、SRKL 都較差
  - 改用其他 loss 會傷 ppl：MixCE（p.8）、Strictly Proper Scoring Rules（p.7）
  - 支持 RKL/skew/AB 的論文（DistiLLM、ABKD、AMiD、ToDi、MiniLLM、GKD）全是指令微調、看 ROUGE-L，追求的是「找眾數」的生成品質，跟 ppl 方向相反
  - 只有 top-32 快取時，RKL 也算不準：需要老師在學生支撐上的機率
- **MSE 或 cosine 對 logits：不要做**：Minitron Table 15、Peng Table 5 都明顯較差
- **溫度 ≠ 1：做不到，也沒必要**：Busbridge G.2、Minitron p.17 都是 τ=1 最好；Peng τ ≤2 差不多
- **TAD（放大 tail 的 KL）：做不到**：需要完整 tail logits，我們沒有
- **反證據：C1 的空間可能很小**
  - Busbridge G.4：k=128 加 λ=0.7 的真實答案份額，就已經和完整分布差 <0.01 nats
  - 我們有 α 0.8–0.9（等於 10–20% 的 CE），再加上 tail 給答案，可能已經很接近 FullKD
  - SLS 的 Naive Fix 在 K=50 已有 83%
  - 所以 C1 有相當機率小於 0.3%，C2 的量測就是用來先判斷這件事
- **提醒（跟 T1b 有關）：** 加重 hard token（SLS Table 10）能降 LM loss，但模型會變得信心不足；對 OJ 的 ppl 無害，但不要過頭

## 6. 校準與 ppl 的關係（總結）
- **校準直接計入 ppl：** ppl = exp(NLL)，NLL 是嚴格 proper 的評分，校準誤差會直接加到 NLL 上
  - CE 或完整分布訓練出來的 LM，在 token 層級本來就校準良好（Busbridge E.8）
  - top-K 蒸餾會破壞校準：head 過度自信（SLS Fig.3a、Table 1–2）
  - 無偏的蒸餾（RS-KD）能恢復校準（SLS Table 6、Table 8）
- **ECE 不夠用：** ECE 只看 top-1，跟 ppl 的關係間接；要診斷 tail 偏差，應該量「top-32 質量 vs 答案落在 top-32 的頻率」和最佳溫度 T*（見 C2）
- **建議順序：**
  1. C2 量測（免訓練，約 1 GPU 小時涵蓋所有 1B checkpoint）
  2. 若量到偏差，跑 C1 的 1B 主測（對 CJ）
  3. C1 贏了再跑 C3（α 1.0）
  - 6B 截止前時間有限，C1 的 1B 主測可以跟 C2 平行跑

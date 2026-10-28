# T6：資料順序、課程與盲點（第 8 號研究員）

日期：2026-10-17。範圍：00-context 第 3、6、7 節、01-levers 的 T6，加上 `src/prepare.py`、`src/training.py`、`src/distill.py`、`src/distillation/generate_logits.py`、`note/prepare-AB.md`，以及 OJ 評分程式 `perplexity.py`。

**先講結論：**
- **資料順序本身幾乎沒有空間。** 在線性降到 0、單次走過、評分看自然分布的設定下，文獻裡的課程收益多半只出現在固定 lr 或下游任務上，到了衰減段就消失。
- **值得試的只有兩件事：**
  - 不改資料、只改注意力範圍的「上下文窗口由短到長」
  - 一個零訓練成本的檢查：KD 學生的輸出溫度
- **其餘盲點大多有直接反證**，包括 MTP、metadata、Coupled Adam、input 梯度放大、模型長大。

---

## 1. 搜尋紀錄

**工具：** `papers.jsonl`（68,363 篇，六大會議 2023–2026），自寫批次 regex 腳本，每組都存下命中清單。

### 1.1 查詢與命中數（共 54 組；A 部分 19 組，B 部分 27 組，補充 8 組）

| 編號 | 主題 | 命中 | 編號 | 主題 | 命中 |
|---|---|---|---|---|---|
| A01 | curriculum × pretrain/LLM | 89 | B01 | initialization × transformer/LLM | 29 |
| A02 | data order / training order | 32 | B02 | depth-scaled / small init / init std | 36 |
| A03 | sequence length curriculum / warmup / dataset decomposition | 3 | B03 | μP / maximal update / HP transfer | 29 |
| A04 | short-to-long / context length schedule | 12 | B04 | embedding init / scale / multiplier | 18 |
| A05 | intra-document / document mask / packing | 12 | B05 | gradient clipping / adaptive clipping | 77 |
| A06 | anneal / cooldown 段的資料品質、mid-training | 31 | B06 | loss spike / training instability | 158 |
| A07 | easy-to-hard / 難度課程 | 66 | B07 | z-loss / output logit / log-partition | 49 |
| A08 | perplexity/loss-based selection、learnability、RHO-LOSS | 184 | B08 | multi-token prediction / future token | 40 |
| A09 | Rho-1 / token selection | 16 | B09 | auxiliary loss × pretrain | 5 |
| A10 | MiniPLM / difference sampling / KD 資料選擇 | 5 | B10 | tied embedding / weight tying | 9 |
| A11 | forgetting × pretrain、recency | 158 | B11 | attention sink / BOS / document boundary / EOS | 84 |
| A12 | data-efficient pretraining | 35 | B12 | model growth / stacking / progressive | 82 |
| A13 | DoReMi / data mixing law / 動態混合 | 11 | B13 | calibration × LM/distill、temperature scaling | 292 |
| A14 | quality-ordered curriculum | 1 | B14 | label smoothing / soft target | 6 |
| A15 | data-constrained / repeated data | 28 | B15 | Adam epsilon / Adam-atan2 | 5 |
| A16 | in-context pretraining / document grouping | 7 | B16 | weight decay × embedding/norm | 3 |
| A17 | online batch selection / importance sampling | 27 | B17 | small-scale proxies / qk-norm | 4 |
| A18 | data pruning × in-distribution perplexity | 2 | B18 | layer/token dropping | 16 |
| A19 | shuffle / seed / data-order variance | 11 | B19 | token/loss reweighting | 1 |
| C01 | metadata conditioning | 4 | B20 | lr warmup analysis | 12 |
| C02 | distillation × token/sample weighting | 9 | B21 | tokens per parameter / data-limited | 42 |
| C03 | teacher/reference × curriculum | 5 | B22 | output layer / lm head training | 11 |
| C04 | ordering by perplexity/loss | 2 | B23 | rare token / anisotropy | 22 |
| C05 | lr decay × data curriculum | 1 | B24 | next-latent prediction / JEPA-LLM | 5 |
| C06 | Coupled Adam / embedding × optimizer | 3 | B25 | checkpoint averaging | 6 |
| C07 | sample reweighting × pretrain | 4 | B26 | pretraining distillation | 9 |
| C08 | noisy data × pretrain | 2 | B27 | position frequency / effective context | 20 |

完整 regex 與命中清單在暫存目錄 `/tmp/claude-1000/-home-yorukot-ml-lab5-training-llama/94460c4a-2e94-42fc-8d1d-1c1378fd5e26/scratchpad/t6/`，檔名 `queries.tsv`、`queries2.tsv`、`hits/*.jsonl`。

### 1.2 篩選與精讀
- **規模：**
  - 命中合計 1,830 筆，不重複 1,738 篇，標題全部看過
  - 讀了約 80 篇的摘要
  - 下載 56 篇 PDF 並全部轉成文字，放在 `/home/yorukot/ml/papers/pdf/T6/`
- **精讀：** 30 篇，指讀了方法和結果表，見第 2 節
  - 另從 T1a 的資料夾讀了 2 篇（Sparse Logit Sampling、Distillation Scaling Laws）的 top-K 段落
- **arXiv：** 4 篇 PDF 經 arXiv 下載，間隔 2 秒，另有 2 次 abs/html 頁面讀取；沒有遇到 429
- **引用追蹤：**
  - SkyLadder → Dataset Decomposition、Sequence Composition（intra-doc）
  - LR-decay 論文 → TREC 和課程相關文獻
  - Sparse Logit Sampling → Distillation Scaling Laws（top-k + λ）
  - WeSaR → Spike No More（scaled embedding）
  - MTP → MTP curriculum、TOP、MuToR

### 1.3 非六大會議來源（都已標明）
- 「Beyond Random Sampling」（Zhang et al.，EACL 2026，arXiv 2506.11300）：讀了全文摘要
- ZClip（arXiv 2504.02507）：只看摘要
- torchtitan v0.1.0 的 Llama 3 初始化程式：用 gh 讀原始碼
- OLMo 2（arXiv 2501.00656）：只有搜尋摘要提到「全部 0.02 初始化」，沒有核對原文
- Llama 3.2 1B 是由 8B 剪枝再蒸餾而來（Meta 公告）：這條是記憶，沒有重新查證

---

## 2. 論文表（精讀 30 篇 + 2 篇 T1a 段落）

| # | 論文 | 會議 | 設定 | 關鍵結果 | 對我們 |
|---|---|---|---|---|---|
| 1 | How Learning Rate Decay Wastes Your Best Data in Curriculum-Based LLM Pretraining | ICLR26 | 1.5B / 30B DCLM | 依品質遞增的課程在固定 lr 下明顯贏隨機；換成 WSD 或 cosine，優勢幾乎消失。PreSelect 分數排序在 WSD 下，最終 val loss 還比隨機高，而且 val 本身是高品質子集。解法是溫和衰減（降到約 1/3）加權重平均，下游 +1.64% | 我們是線性降到 0，衰減段占 80%：品質課程不建議 |
| 2 | Predicting Training Re-evaluation Curves (TREC) | ICLR26 | 111M–3.9B，20–1280 TPP | 最終模型對各時段訓練資料的 loss（TREC）形狀由 AdamW 時間尺度 τ 決定。高品質資料放在 TREC 最低點最好；線性降到 0 時，最低點略早於結尾。TPP 越低，TREC 的落差越大（越像記憶）。「高品質」的定義是對目標領域 val 有幫助 | 我們的 eval 和訓練同分布，沒有「目標領域資料」可以挪位置 |
| 3 | SkyLadder: Context Window Scheduling | NeurIPS25 | 120M–3B，100B tokens（第 3 節為 30B 池、約 3.3 epoch） | 固定 tokens 下，短窗口訓練的 ppl 較低（評在 512）。窗口由 32 線性長到 8K（約 64% 時到滿）：Qwen2.5-0.5B 的 PPL@8K 12.5→12.1，Gemma3-like 120M 13.0→12.4。疊在 IntraDoc 上：標準任務 47.4→49.3，長文 13.0→13.2。反向（長到短）無效（52.6 vs 52.5）。IntraDoc 本身較好，部分原因是它的平均窗口短 | 我們已有 doc masking，可以再加窗口排程；最有根據的 A 部分候選 |
| 4 | Dataset Decomposition | NeurIPS24 | OpenLM-1B，8K | doc masking 讓一般任務 51.5→52.4。依長度的循環課程（Grow-P2，8 輪）：一般任務 53.8→54.4，MDQA 23.0→28.0。只縮短、不再拉長（Shrink）明顯較差 | 長度課程對一般任務只有小幅幫助，主要幫長文；沒有 ppl 數字 |
| 5 | Analysing the Impact of Sequence Composition | ACL24 | 1.3B | intra-document 遮罩的 ppl 和下游都比跨文件 causal 好 | 我們已採用，屬背景 |
| 6 | Preference Curriculum (PDPC) | ACL25 | 1.3B/100B、3B/1T，cosine | 用弱、強模型的 ppl 差排序，弱模型覺得難的放後面；只報下游準確率 | 沒有自然分布 ppl 的證據 |
| 7 | MiniPLM | ICLR25 | 200M–1.2B | 用老師與參考模型的差異，從較大的資料池做 difference sampling。test loss 是在 DCLM 子集上量的；也指出 vanilla KD 對較大的學生有效 | 需要比 6B 更大的池子，我們沒有，不適用 |
| 8 | Not All Tokens (Rho-1) | NeurIPS24 | 1B/7B，持續預訓練 | 只在 60–70% 的 token 上算 loss，對齊目標分布；沒被選中的 token loss 會上升 | 評分算全部 token，丟 token 有風險，不適用 |
| 9 | Dynamic Loss-Based Sample Reweighting | ICLR25 | 120M–7B，SlimPajama | LinUpper 把低 loss 樣本降權：GPT2-mini 各領域平均 log-ppl 3.32→3.30，但 GitHub 1.84→1.89 變差 | 指標是領域平均，不是 token 加權；對我們（code 多）有風險 |
| 10 | Exploring Forgetting in LLM Pre-Training | ACL25 | — | 重放可減少遺忘 | 重放算預算，不適用 |
| 11 | Metadata Conditioning then Cooldown (MeCo) | ICML25 | 1.6B / 160B | 下游 +1.0，等於省 33% 資料；但 val PPL 13.2→13.3 變差。同表的資料選擇也是 13.3 | 反證：下游變好，ppl 不變或變差 |
| 12 | Better & Faster LLMs via Multi-token Prediction | ICML24 | 300M–13B | 小模型做 MTP 比基準差，大模型才贏；7B 在 200B 自然語言上，n=2 打平，n=4 變差 | 反證：1B 不該做 MTP |
| 13 | Pre-Training Curriculum for MTP | ACL25 | 1.3B / 3B | 1.3B 在 MiniPile 的 BPB：NTP 1.08，2-head MTP 1.12（差 3.7%）；課程只能部分補回 | 反證 |
| 14 | Predicting the Order of Upcoming Tokens (TOP) | ICML26 | 340M / 1.8B / 7B，最多 104B | 輔助頭做未來 token 的排序。LAMBADA ppl：1.8B 11.38→11.19（−1.7%），340M 30.34→28.76；同條件下 MTP 是 13.69（變差）。只多一個 unembedding，推論時拿掉 | 少數在約 2B 規模有 ppl 正訊號的輔助目標；但只有 LAMBADA |
| 15 | Weight Tying Biases Token Embeddings Towards the Output Space | ACL26 | OLMo-1B，20B | tied 矩陣主要被輸出端塑形；把 input 端梯度放大 5 倍，WikiText-2 ppl 35.71→36.64 變差 | 反證：不要放大 input 端 |
| 16 | Better Embeddings with Coupled Adam | ACL25 | 125M–2.6B | 1.3B/26B（20 TPP）：loss 2.433→2.448 變差；105B：2.278→2.277 持平 | 反證：我們只有 5 TPP |
| 17 | Small-scale proxies for large-scale Transformer training instabilities | ICLR24 | 9M–4.8B | z-loss 只解決「沒有 wd」時的輸出 logit 發散，wd 也能解決。eps 1e-8→1e-15 只在 grad RMS 接近 eps（4.8B）時有幫助。µParam 不改善 loss | z-loss、eps 對我們幾乎沒有空間 |
| 18 | Scaling Exponents Across Parameterizations and Optimizers | ICML24 | 到 27B | eps underflow；提出 Adam-atan2 去掉 eps | 同上 |
| 19 | WeSaR: Initialization via Reparameterization | EMNLP24 | 130M–13B，AdamW | 1.3B 的 WikiText ppl 16.55→14.51，但基準是有 spike 的 small init | 增益來自 Adam 加上 spike，Muon 下不一定有 |
| 20 | Completed Hyperparameter Transfer | ICLR26 | 50M→7.2B，AdamW | 逐模組調 lr、init、eps、β、wd，在 7.2B 仍有 1.32× 加速 | 主要靠逐模組 lr，屬 T5/T3 範圍 |
| 21 | Stacking Your Transformers (G_stack) | NeurIPS24 | 410M–7B | 7B：用 194B tokens 達到從零訓練 300B 的 loss；小模型另外吃 10B tokens | 比較基準 lr 未調；token 預算下不划算（見反證） |
| 22 | No Train No Gain | NeurIPS23 | BERT / T5 | 和調好、完整衰減的基準比，layer stacking、dropping、selective backprop、RHO-loss、Lion、Sophia 的優勢全部消失 | 反證：模型長大和批次選擇 |
| 23 | Sparse Logit Sampling（T1a 資料夾） | ACL25 | 300M 學生、3B 老師 | top-K KD 的學生在 top-K 過度自信、其餘不足；ECE：CE 1.2、FullKD 0.7、K=25 3.2、K=12 4.7；K<25 比純 CE 還差 | 支持「我們的 KD 學生可能有溫度偏差」 |
| 24 | Distillation Scaling Laws（T1a 資料夾） | ICML25 | 198M–7.75B | 只用 top-128 比完整分布差 0.11 nats；加 λ=0.7 混正確答案後，差距在 0.01 nats 內 | 有混 CE 時偏差小，所以溫度收益預期小 |
| 25 | AdaGC | ICML26 | 到 7B | 逐 tensor 的自適應 clipping，收益來自壓 spike | 我們沒有 spike |
| 26 | PolyPythias | ICLR25 | 14M–410M，50 次 | 種子（init + 資料順序）不同，訓練動態高度一致 | 順序雜訊小 |
| 27 | Convergence and Divergence under Different Random Seeds | EMNLP25 | Pythia | 跨種子的 KL 有四個階段；常見 token 收斂得快 | 背景 |
| 28 | YuLan-Mini | ACL25 | 2.4B / 1.08T | 穩定性技巧加資料課程，只報下游 | 無 ppl 證據 |
| 29 | NITP: Next Implicit Token Prediction | ICML26 | 0.5B–9B | 表示空間的輔助 loss，只報下游 | 無 ppl 證據 |
| 30 | Beyond MTP: Future Summaries | ICLR26 | 3B / 8B | 只報下游 | 無 ppl 證據 |
| 31 | Beyond Random Sampling（**非六大會議**，EACL26） | arXiv 2506.11300 | 0.5B/10B、1B/20B，**固定 lr 2e-4** | 課程當 warmup，下游最多 +3.5%（取最佳 checkpoint）；以 ppl 排序的效果不穩定 | 固定 lr、看下游，不能套用到我們 |
| 32 | torchtitan Llama 3 init（**非六大會議**，程式碼 v0.1.0） | — | — | wq/wk/wv/w1 用 0.02；wo、w2、w3 用 0.02/√(2(l+1))；embedding std 1.0；untied 的輸出層用 dim^-0.5 | 為 untied 設計，tied 的 1B 不能直接用 |

只讀摘要、沒有精讀的（列出以示查過）：
- **資料：**
  - Muon Outperforms Adam in Tail-End Associative Memory（ICLR26）
  - Pre-training under infinite compute（ICLR26）
  - Fresh in memory（ICLR26）
  - Data Temporality（ICML26）
  - Demystifying Data Organization（ACL26）
  - Self-Influence reweighting（EMNLP23）
  - Perplexed by Perplexity（ICLR25）
  - URLs Help, Topics Guide（NeurIPS25）
  - Scaling Data-Constrained LMs（NeurIPS23）
  - Why Does the Effective Context Length Fall Short（ICLR25）
  - Token Weighting for Long-Range LM（NAACL25）
- **模型長大：**
  - Curriculum-Guided Layer Scaling（ICML26）
  - zero/one-layer expansion（ICML26）
  - progressive subnetworks（ICLR25）
  - Inductive Bias of Stacking（NeurIPS24）
  - Taming Curvature（ICLR26）
- **溫度與蒸餾：**
  - Post-Hoc Reversal（NeurIPS24）
  - Temperature Scaling in Discrete Sequence Models（ICML26）
  - Progressive distillation implicit curriculum（ICLR25）
  - Calibrated Progressive Distillation（ACL26）
  - Distillation Traps and Guards（ACL26）
  - Training-Trajectory-Aware Token Selection（ICML26）
- **初始化、穩定性與其他：**
  - Analyzing & Reducing the Need for LR Warmup（NeurIPS24）
  - Do Deep Networks Forget Initialization（NeurIPS26）
  - GradientStabilizer（ICML26）
  - SPAM（ICLR25）
  - Adaptive Preconditioners Trigger Loss Spikes（ICML26）
  - MuToR（NeurIPS25）

---

## 3. A 部分（資料順序與課程）候選總表

| 排名 | 候選 | 1B 預期 | 機率 | 6B 預期 | 成本 | 建議 |
|---|---|---|---|---|---|---|
| A1 | 維持隨機順序（基準） | 0 | — | 0 | 0 | 預設 |
| A2 | 上下文窗口由短到長（只改遮罩，資料順序不變） | −0.1 ~ −0.5% | ~30% | −0.05 ~ −0.3%（~25%） | 改約 15 行、1 次 1B | **建議測** |
| A3 | 老師 loss「由易到難」的局部 warmup 課程 | 0 ~ −0.15% | ~15% | ~0 | 掃一次快取，加 1 次 1B | 低優先 |
| A4 | 依當下 loss 調整每列權重（LinUpper） | ±0.3% | ~15% | 不明 | 改約 20 行、1 次 1B | 低優先，有負面風險 |
| A5 | 依文件長度重排列（短在前） | ~0 | ~10% | ~0 | 低 | 不建議（會改變各段的領域組成） |
| ✗ | 品質遞增或衰減段放高品質、資料選擇 | 預期變差 | — | — | — | 不建議 |

**「資料順序重要嗎」的答案：** 如果順序和資料性質無關（隨機重排），影響大約是種子雜訊。我們自己的數字：BM 與 BE 重跑差 0.02%；PolyPythias 也一致。

系統性的課程只有在三種情況下明顯有效：
- 固定 lr（LR-decay 論文、Beyond Random Sampling）
- 目標分布和訓練分布不同（TREC、MiniPLM、MeCo 的 HQ val）
- 評分看下游任務（PDPC、YuLan-Mini、NITP）

我們三個條件都不符合：
- 衰減占 80%，而且降到 0
- eval 就是訓練分布本身
- 評分看 ppl

所以預期效果是 0 到 ±0.1%。唯一例外是 A2，因為它不改變資料分布，只改模型在前期看得到多少上下文。

---

## 4. B 部分（盲點）候選總表：依「預期收益 × 機率」排序

| 排名 | 候選 | 若成立的收益 | 機率 | 收益×機率 | 成本 | 備註 |
|---|---|---|---|---|---|---|
| B1 | **事後溫度：把最終 RMSNorm 權重乘上純量 c** | 0.1–0.3%（只對 KD） | ~40% | ~0.06% | **不用訓練**，約 0.5 GPU-h 的 eval | 同時是 T1a「tail 偏差」的免費量測 |
| B2 | TOP 輔助頭（未來 token 排序，評測時拿掉） | 0.2–0.5% | ~15% | ~0.06% | 1 次 1B，加約 0.5 天實作；時間 +約 20% | 6B 時間可能超過 64 H200-h |
| B3 | input 端 embedding 尺度（訓練時乘倍數，存檔時折回，架構不變） | 0.1–0.3% | ~15% | ~0.03% | 先免費量測，再 1–2 次 1B | 有反證（input 梯度 ×5 變差） |
| B4 | 依深度縮小輸出投影的初始化（torchtitan 式） | ±0.15% | ~15% | ~0.02% | 1 次 1B | 零初始化 BB 已輸 0.47% |
| B5 | embedding 與 norm 那組的 AdamW eps 1e-8 → 1e-12 | ~0.05% | ~10% | ~0.005% | 改 1 行 | 可以搭便車，不值得單獨一次 |
| B6 | z-loss（1e-4） | 0 ~ +0.05%，可能為負 | ~10% | ~0 | 改約 10 行 | 只有在 T2 量到 logits 捨入是問題時才考慮 |
| B7 | 調整 grad clip | ~0 | <10% | ~0 | 改 1 行 | 不建議改（理由見 5.B7） |
| ✗ | MTP、FSP、NITP、MeCo、Coupled Adam、input 梯度放大、模型長大、layer drop、WeSaR、Rho-1 | 有反證或無 ppl 證據 | — | — | — | 不建議 |

B1 和 B2 的期望值差不多，B1 排第一是因為成本幾乎是 0，又能量出 KD 的偏差有多大，可以幫 T1a 決定 tail 修正值不值得做。

---

## 5. 每個候選的細節

### A2. 上下文窗口由短到長（SkyLadder 式，疊在 doc masking 上）
- **機制：**
  - 訓練前段把每篇文件切成長度 w 的區塊，每個 token 只看同區塊內的前文；w 隨訓練線性長到 8192
  - 資料、順序、標籤、KD target 都不變，只改 `cu_seqlens`
  - SkyLadder 認為前期短上下文讓注意力學得較集中，而且 doc masking 的好處有一部分就來自「平均窗口變短」
  - 它和「依長度重排」不同：各段的資料（領域組成）仍是 i.i.d.；SkyLadder 也證明「長到短」無效，所以關鍵是排程本身，不是資料混合
- **證據：**
  - SkyLadder（NeurIPS25）：
    - 疊在 Random packing 上，PPL@8K 降 3–5%
    - 疊在 IntraDoc 上，標準任務 +1.9、長文 +0.2；沒有給 IntraDoc 加 SkyLadder 的 ppl
  - Dataset Decomposition（NeurIPS24）：長度課程讓一般任務 +0.6、長文 +5
  - 弱點：
    - 兩者都是 100 TPP 量級（甚至多 epoch），我們是 5 TPP
    - 兩者都有 decay schedule，但 SkyLadder 是 cosine、全長 100B
- **預期與信心：**
  - 1B：−0.1% 到 −0.5%，信心約 30%
  - 6B：步數多 6 倍，窗口仍以比例排程時，前期短窗口占的 token 也更多；但 5 TPP 下長上下文能力本來就不足，是否要更早給長窗口不確定。預估收益變小，約 −0.05% 到 −0.3%，信心約 25%
- **交互作用：**
  - **KD：** 老師的 logits 是用完整前文算的。被切斷的 token，老師分布條件比學生多，等於被要求預測它看不到的資訊；α 0.8 時這部分像 target 雜訊，前期 KD 收益可能被吃掉一些
  - **EMA-Nesterov：** 從 15% 開始，和窗口長滿的時間重疊，應該無關
  - **BOS：** 被切開的區塊沒有 BOS 當 attention sink，模型會把區塊第一個 token 當 sink；窗口長滿後就回到和 OJ 一致
  - **速度：** 前期 attention 較便宜，SkyLadder 報 −13%；不影響預算
- **有沒有已試過：** 沒有。BW、BR 是 lr warmup，不是窗口
- **實作草圖：** 改 `LabTrainer.training_step`，在 `super()` 之前做：
  ```python
  w = window(self.state.global_step)  # 例如 min(8192, 64 + (8192-64) * step / (0.3 * max_steps))
  if w < 8192:
      pos = inputs["position_ids"].flatten()
      starts = (pos % w == 0).nonzero().flatten()          # 每篇的 position 從 0 起算，所以 pos % w == 0 就是區塊起點
      cu = torch.cat([starts, starts.new_tensor([pos.numel()])]).int()
      inputs |= {"cu_seq_lens_q": cu, "cu_seq_lens_k": cu, "max_length_q": w, "max_length_k": w}
  ```
  - position_ids、labels、kd_* 都不動，RoPE 是相對位置，區塊內不需要重設
  - padding 段也會被切，沒有影響
- **成本與風險：**
  - 1 次 1B，約 70 分鐘
  - 風險在長文件：
    - Dolma 3 的長文件 token 占比高（PDF、code、arXiv），如果窗口太晚長滿，長位置的訓練不足會在 OJ 直接扣分
    - 建議先用 `doc_lens` 離線算出 >2k、>4k 位置的 token 占比，不用 GPU
- **1B 測試設計與失敗條件：**
  - 設定：CJ 的配方（KD α 0.9 + EMA-Nesterov），只加窗口排程，w 從 64 線性長到 8192，在 30% 時長滿
  - 失敗條件一：最終 eval 沒有比 CJ 的 15.324 好 0.2% 以上
  - 失敗條件二：70% 到結尾的差距一路縮小，外推到 6B 會歸零
  - 若成功，第二輪可試「50% 時長滿」，把排程掃出來

### A3. 老師 loss 的局部 warmup 課程
- **機制：** 由易到難只排在前段，例如前 15% 的列在自己內部依老師 NLL 由低到高排序；後 85% 維持隨機
  - 因為只在該段內部排序，整體領域組成不變，衰減段完全不受影響
- **老師 NLL 怎麼算：**
  - 快取有 top-32 的 logits 和 logsumexp
  - 正確答案在 top-32 內（93%）時：log p = logit − lse
  - 不在時：用 log(tail) 或 top-32 最小 logit − lse 當上界
  - 每列取平均
- **證據：**
  - 支持：PDPC（ACL25，只看下游）、Beyond Random Sampling（**非六大會議**，固定 lr，只看下游，取最佳 checkpoint）
  - 反對：LR-decay 論文（衰減下課程優勢消失）、DD（課程對一般任務 +0.6）、Beyond Random Sampling 自己也說以 ppl 排序不穩定
- **預期：** 1B 0 到 −0.15%，信心 15%；6B 約 0，因為前段占比的影響在 6B 更容易被後段抹平（TREC：早期資料的影響被遺忘）
- **交互作用：** 前期 KD α 高；老師容易的列，KD 和 CE 的 target 接近，前期等於多一些容易的資料
- **有沒有已試過：** 沒有
- **實作草圖：**
  - 先做一次 CPU 掃描，算 732,756 列的老師 NLL：讀 1.1 TB，約 1–2 小時 I/O；也可以每列只抽 512 個位置，讀取量降到約 1/16
  - 再改 Trainer 的 sampler：傳入一個固定的索引順序，取代 RandomSampler
  - 快取以列號查找，重排只影響讀取模式（每列約 1.5 MB 的隨機讀取），要先測 DataLoader 吞吐
- **成本與風險：** 1 次 1B 加掃描；收益很可能測不出來
- **1B 測試與失敗條件：** 沒有 ≤ −0.2% 就放棄；這條只在 A2 成功、還有預算時才排

### A4. 依當下 loss 調整每列權重（Dynamic Loss-Based Reweighting）
- **機制：** 每步依各列目前的 loss 給權重，高 loss 的列加權；權重正規化到平均 1，上限 2
- **證據：**
  - ICLR25：GPT2 等級，各領域平均 log-ppl −0.02
  - 但低 loss 的 GitHub 從 1.84 變差到 1.89
  - 我們的評分以 token 加權，長的 code 和 PDF 占很多 token，低 loss 的列被降權可能直接扣分
- **預期：** ±0.3%，正向機率約 15%
- **交互作用：** 和 T1b 的 token 加權重疊。如果要做，應該由 T1b 統一處理「列層級」和「token 層級」的權重，避免兩邊重複
- **實作：** 在 `distill.use_mixed_loss` 的 forward 裡，用每列的 loss 算權重，乘上每 token 的 loss；約 20 行
- **1B 測試與失敗條件：** 沒有 ≤ −0.2% 就放棄
- **結論：** 不建議優先做

### B1. 事後溫度：把最終 norm 權重乘上 c（只對 KD 學生）
- **機制（最意外的發現）：**
  - 我們的 KD target 是：老師 top-32 給 α·p_T，tail 質量與 1−α 全部給正確答案
  - 假設老師就是真實分布 p*，對資料取期望後，學生的最佳解 q 為：
    - top-32 內：p*·(1 + α·tail)
    - top-32 外：p*·(1 − α + α·tail)
  - 以 α 0.8、tail 0.09 代入：老師 top-32 以外的 token 只拿到真實機率的約 27%，也就是結構性的「頭部過度自信、尾部不足」
  - Sparse Logit Sampling（ACL25）在 300M 學生上量到同方向的偏差：top-K 學生 ECE 3–5，CE 學生約 1
  - 以溫度 T > 1 攤平分布，可以部分補回尾部
  - 在架構不變的前提下，T 正好等於把 `model.norm.weight` 乘上 c = 1/T：最終 norm 只用在 lm_head 之前，和 tied embedding 無關
- **玩具模擬：**
  - 設定：12.8 萬詞表，Zipf 分布，top-32 以外占 6%
  - α 0.8：理想學生的 KL 是 0.034 nats，最佳 T≈1.08 時降到 0.017，約補回一半
  - α 1.0：0.095 → 0.052（T≈1.15）
  - 但實際觀察到的 α 效應比理想小約 20 倍：CI（α 1.0）只比 CH（α 0.9）差 0.26%，理想推估差約 6%，表示真實學生離這個最佳解很遠
  - 因此保守估計可補回 0.05–0.3%，最佳 c 約 0.96–0.99
  - Distillation Scaling Laws 指出，top-128 混 λ=0.7 的 CE 後偏差在 0.01 nats 內，也支持「收益偏小」
- **1B / 6B 預期：**
  - 若 KD 偏差在大 tokens 時仍存在，6B 也有效
  - 6B 的 α 0.8 比 1B 的 0.9 低，偏差應該更小
  - 預估 6B 收益 0.05–0.2%，信心約 35–40%
- **交互作用：**
  - **T1a：** 如果 T1a 改了 tail 分配（例如依學生自己的 tail 分配），偏差在訓練時就修好了，B1 會歸零；B1 同時可以量出這個偏差有多大，作為 T1a 的前導數據
  - **T1b：** α 往下降的排程也會縮小偏差
  - **T2（bf16）：** c 在 bf16 下仍可表示；norm 權重本來就會被捨入成 bf16，乘 c 不增加誤差；但最佳 c 應該在 OJ 的 bf16 eval 下找
- **有沒有已試過：** 沒有。in-training eval 都是 c = 1
- **實作草圖：** 不用訓練
  ```python
  model.model.norm.weight.mul_(c)   # 存檔前做一次
  ```
  - c 在 {0.94, 0.96, 0.98, 1.00, 1.02} 中找
  - 擬合集：建議用不在訓練集、也不在 judge 50k 的新文件，例如 seed-42 排列中 `order[7_000_000 : 7_005_000]`，去掉近似重複，約 5k 篇
  - 用 judge 的前 10k 篇當確認集
- **規則：**
  - 這是對權重的事後處理，只擬合一個純量，不在測試集上訓練，也不吃訓練預算
  - 但仍建議先請使用者確認是否合乎規則
- **成本與風險：**
  - 每個 checkpoint 約 5 個 c × 10k 篇的 eval，約 0.5 GPU-h；需要使用者同意送 job
  - 風險很低：c = 1 永遠是備選
- **1B / 3B 測試與失敗條件：**
  - 評 BU（無 KD）、CH、CJ（1B KD）、CL（3B KD），之後再評 CM
  - 預期：BU 的最佳 c ≈ 1.00（健全性檢查）；KD 的最佳 c < 1，且收益 ≥ 0.1%
  - 失敗條件：KD 模型的最佳 c 也 ≈ 1，或收益 < 0.05%。若失敗，代表偏差已被 CE 那部分吸收，T1a 的 tail 修正優先度也該降低

### B2. TOP 輔助頭（Token Order Prediction）
- **機制：**
  - 額外一個 128k × 2048 的 unembedding 頭，以排序 loss 預測未來 N 個 token 中「誰先出現」
  - 這個頭在推論時拿掉，所以最終權重仍是原本的 Llama
  - 它比 MTP 寬鬆，不必精確預測遠處的 token，所以小模型也能從中受益
- **證據：**
  - TOP（ICML26）：1.8B 的 LAMBADA ppl −1.7%，340M −5%；同條件下 MTP 和 DeepSeek 式 MTP 都變差
  - 但只有 LAMBADA 的 ppl，沒有語料 ppl
  - 而且基準沒有 KD
- **預期：**
  - 1B：−0.2% 到 −0.5%，或變差；正向機率約 15%
  - 為什麼不高：KD 已經提供豐富的軟標籤；小模型的容量要分給輔助目標
- **交互作用：**
  - 輔助頭要不要和 embedding 綁在一起：論文是另外一層
  - 與 Liger、自寫 `MixedCrossEntropy` 的融合計算：要再寫一個分塊的排序 loss
- **有沒有已試過：** 沒有
- **實作草圖：**
  - 在 `use_mixed_loss` 的 forward 裡，hidden 另外接 `top_head`（Linear 2048→128256，用 embedding 複製初始化）
  - target 由 input_ids 在 N=16 的窗口內建出「距離分數」，用 listwise softmax loss，和主 loss 相加
  - `save_model` 前刪掉 `top_head`
- **成本與風險：**
  - 計算約 +20%：lm head 在 1B 模型中約占四分之一的 FLOPs
  - 記憶體約 +4 GB/GPU
  - **6B 的時間預算是硬限制：** 目前估 55 H200-h，+20% 後約 66，超過 64
  - 除非 Newton-Schulz 分散到多卡（−16%）先完成，否則 6B 用不上
- **1B 測試與失敗條件：**
  - 只在 B1、A2 之後，而且 6B 時間有餘裕時才做
  - 設定：CJ 的配方加 TOP，權重 1
  - 失敗條件：沒有 ≤ −0.3%（因為成本較高，門檻也較高）

### B3. input 端 embedding 尺度（先量測，再決定要不要測）
- **機制：**
  - tied embedding 的 input 和 output 兩個角色共用同一個矩陣，也共用同一個 AdamW lr
  - residual stream 中「embedding 本身」相對於「各 block 輸出」的比例，目前沒有任何直接調過的參數
  - 一開始 embedding std 0.02，而 block 輸出 RMS 約 0.8，token 身分的訊號被 block 的雜訊淹沒，要靠 AdamW 很快拉起來
  - 零初始化 BB 讓 embedding 一開始占主導，前期快 4.4%，最後輸 0.47%，表示這個比例確實會影響結果
- **可以不改架構：** 訓練時把 input embedding 乘 m。因為每個 block 讀的都是 RMSNorm 後的值，存檔時把 `o_proj`、`down_proj` 除以 m，函數完全相同（只差 RMSNorm eps 1e-5 的微小影響）
- **證據：**
  - Spike No More 與 WeSaR 都用 scaled embedding，但都是 Adam 加 spike 的情境
  - Weight Tying Biases（ACL26）：input 端梯度 ×5 讓 ppl +2.6%。m > 1 會連帶放大 input 端梯度，方向偏不利
- **預期：** ±0.2%，正向機率 15%
- **第一步：免費量測。** 從 CJ 或 CL 的 checkpoint 算各層 residual 中 embedding 與各 block 輸出的 RMS 比例、embedding 的平均向量長度
  - 如果 embedding 只占很小比例，才值得試 m ∈ {0.5, 2}
- **有沒有已試過：** AR（embedding wd 0）打平、BB（零初始化）輸了，都是相關但不同的實驗
- **1B 測試與失敗條件：** 在 m = 2 和 0.5 中先試量測結果指向的那一個；沒有 ≤ −0.2% 就放棄

### B4. 依深度縮小輸出投影的初始化
- **機制：** `o_proj`、`down_proj` 用 std 0.02/√(2(l+1))（torchtitan 的 Llama 3 預設，**非六大會議**），其他維持 0.02
  - 這介於現行配方和零初始化（BB）之間
- **證據：**
  - OLMo 2（**非六大會議**，未核對原文）改回全部 0.02 是為了穩定
  - BB 的結果是「前期快、最後輸」
  - Do Deep Networks Forget Initialization（NeurIPS26，只看摘要）：Adam 類方法大多會抹掉初始化的影響
  - 我們的 τ 0.39 讓初始權重到結尾只剩約 exp(−0.6/0.39) ≈ 21%
- **預期：** ±0.15%；最可能的結果和 BB 同一形狀但幅度較小。信心 15%
- **Llama 3.2 的原始初始化：** 不存在可複製的版本。3.2 1B 是由 8B 剪枝再蒸餾而來（Meta 公告，**非六大會議**，記憶未查證），而且用老師權重初始化違反「隨機初始化」的規則。torchtitan 的 embedding std 1.0 是為 untied 設計，對 tied 會讓初始 logits std 約 45，不能用
- **實作：** 和 `ZERO_INIT` 相同的位置，用 `normal_(std=0.02/(2*(l+1))**0.5)`，約 5 行
- **1B 測試與失敗條件：** 沒有 ≤ −0.2% 就放棄；如果曲線形狀和 BB 一樣（前期快、尾段輸），直接結案

### B5. AdamW eps（embedding 與 norm 那組）
- **機制：** grad RMS 接近 eps（1e-8）時，eps 會壓小更新
  - 1B 後期全域 grad norm 約 0.12：即使 embedding 只占 10%，每個元素的 RMS 也約 2e-6，遠大於 1e-8
  - 只有極罕見 token 的列會碰到 eps；我粗算這些 token 殘留的機率質量 < 1e-4，對 ppl 沒影響
- **證據：** Small-scale proxies 在 4.8B 時才看到效果（grad RMS ~1e-8 的情況）
- **預期：** ~0
- **建議：** 不單獨做；如果其他實驗要動 `src/adamw.py`，可以順便改成 1e-12

### B6. z-loss
- **證據：** Small-scale proxies 指出 z-loss 只修「無 wd 時的 logit 發散」，我們有 wd，也沒有發散
- **和 T2 的交互作用：**
  - OJ 的 lm_head 輸出是 bf16。logit 在 16–32 之間時，捨入間隔 0.125，標準差約 0.036
  - 二階效應估計 CE 增加約 ½σ²(1 − Σp²) ≈ 4×10⁻⁴ nats，也就是 ppl 約 +0.04%
  - z-loss 讓 logsumexp 接近 0，高機率 token 的 logit 會變小、捨入誤差跟著變小；但 z-loss 本身也有成本
- **建議：** 只有在 T2 的量測顯示 logits 捨入是主要來源時才考慮

### B7. grad clipping
- **為什麼幾乎沒有影響：**
  - BU 的 grad norm 只在前 30 步達到 8.5，之後很快降到 1 以下，後期約 0.1–0.15；clip 只在最前面幾十到一百多步生效
  - Muon 的正交化對梯度尺度不敏感，clip 只改變前期 momentum 中各步的權重
  - AdamW 除了 eps 也不受尺度影響
  - 所以改 clip 的期望影響約為 0
- **反而有用的一面：** BW（warmup 0.25%）的 spike 顯示，clip 是前期的保險；6B 的 warmup 有 460 步，保留 1.0 沒有成本
- **ZClip、AdaGC 的收益：** 都來自壓制大規模 spike，不適用我們的情況
- **建議：** 不動

### 不建議（都有反證或沒有 ppl 證據）
- **MTP：**
  - Gloeckle（ICML24）：小模型做 MTP 變差
  - ACL25：1.3B 的 BPB 1.08→1.12
  - MuToR（NeurIPS25）主要是微調場景
- **FSP、NITP：** 只有下游任務的證據，沒有 ppl
- **MeCo、metadata：** PPL 13.2→13.3。而且 metadata 前綴是非 padding token，算預算
- **Coupled Adam：** 1.3B/26B 時 loss 2.433→2.448，我們只有 5 TPP，更不利
- **input 端梯度放大：** ppl +2.6%
- **模型長大（stacking、zero/one-layer、CGLS、RaPTr）：**
  - 這些方法省的是 FLOPs，我們受限的是 tokens；小模型吃的 tokens 也算預算
  - No Train No Gain 在調好、完整衰減的基準下，連 FLOPs 優勢都消失
  - 1B 的 embedding 有 263M，四分之一深度的小模型也有約 0.5B，不算小
- **layer、token dropping：** 同上
- **WeSaR：** 收益來自 Adam 加 spike
- **Rho-1、selective LM：** 丟 token
- **MiniPLM：** 需要更大的資料池
- **重放（replay）：** 重複算預算
- **In-context pretraining（相關文件放在一起）：** 需要跨文件 attention，和 OJ 的每篇獨立評分不符
- **品質遞增課程、衰減段放高品質資料、資料選擇：**
  - LR-decay 論文：衰減下沒有優勢，甚至更差
  - MeCo 表中的資料選擇：PPL 13.2→13.3
  - eval 是自然分布

### 已確認一致、沒有槓桿的地方：文件邊界與 BOS
- 訓練與 OJ 完全一致：
  - 每篇 BOS + text、不加 EOS
  - 截斷 8192，doc masking，position 每篇從 0 開始
  - BOS 不當 target，最後一個 token 不預測下一篇的 BOS
  - loss 以 token 加權（`num_items_in_batch`），等於語料 ppl
- BOS 占預算 0.1%（約 6.1M 篇），但模型需要它當 sink，不能省

---

## 6. 反證據（整理）
1. **課程在衰減段會消失：**
   - LR-decay 論文（ICLR26）在 1.5B/30B：WSD 或 cosine 下，品質課程的優勢幾乎沒了；PreSelect 排序甚至比隨機差
   - Beyond Random Sampling（**非六大會議**）只在固定 lr、取最佳 checkpoint 時有收益
2. **「下游變好、ppl 不變或變差」的方法：** MeCo（13.2→13.3）、資料選擇（13.3）、MIDAS stacking（摘要：ppl 相近或略差）、Rho-1（沒被選中的 token loss 上升）、PDPC/YuLan/NITP/FSP（只報下游）。我們的目標是 ppl，這類證據不能拿來用
3. **小模型的輔助目標：** MTP 在 1.3B 變差 3.7% BPB；input 端梯度 ×5 讓 ppl +2.6%；Coupled Adam 在 20 TPP 變差
4. **模型長大：** No Train No Gain 顯示，和調好的基準比，FLOPs 優勢消失；在 token 預算下更不利
5. **A2 的反證：**
   - SkyLadder 在 IntraDoc 基準上的長文增益很小（13.0→13.2），而且沒有給 IntraDoc 的 ppl
   - 他們的 ppl 改善主要在 Random packing（跨文件 attention）上，我們已經有 doc masking，可能已經吃掉大部分好處
   - 而且他們是 100 TPP、多 epoch
4. **B1 的反證：** Distillation Scaling Laws 的 top-128 加 λ=0.7 幾乎沒有偏差；我們的 CI vs CH 也顯示真實偏差只有理想值的約 1/20
5. **順序雜訊本身很小：** 我們的 BM 重跑差 0.02%；PolyPythias 一致；TREC 在同質資料上，最終 val 不受順序影響（受影響的是「目標領域資料放哪裡」）

---

## 7. 結論
- **A 部分：** 資料順序與課程在我們的設定下幾乎沒有空間
  - 原因：線性降到 0、衰減 80%、eval 和訓練同分布、評分看 ppl
  - 唯一值得一試的是 **A2 上下文窗口由短到長**：它不動資料分布、不動快取，只改 `cu_seqlens`
  - 1B 預期 −0.1% 到 −0.5%，信心約 30%
  - MiniPLM、Rho-1 都不適用：一個需要更大的池子，一個會丟 token
  - 老師 loss 最適合的用途是 T1b 的 token 加權，不是排序
- **B 部分：** 最便宜也最有資訊量的是 **B1 事後溫度**（norm 權重乘 c）
  - 不用訓練，約 0.5 GPU-h，能量出我們 KD 的「tail 不足」偏差
  - 有效時可直接套到 6B 最終權重上，收益 0.05–0.3%
  - 也能告訴 T1a 的 tail 修正值不值得做
- **其他：** TOP 有少量正向證據，但 6B 時間預算吃緊；input embedding 尺度與初始化建議先量測，再決定要不要花 1B 的 run；clip、eps、z-loss 不建議花 run
- **建議順序：**
  1. B1：評 BU、CH、CJ、CL，需要同意送 eval job
  2. A2：1B，疊在 CJ 的配方上
  3. 視結果與時間，再考慮 B3 的量測或 B2
- **和其他軌的分工：**
  - token 與列的加權歸 T1b
  - tail 分配歸 T1a；B1 是 T1a 的前導量測
  - bf16 logits 捨入歸 T2；z-loss 只在 T2 量到問題時考慮
  - 權重平均或溫和衰減歸 T4

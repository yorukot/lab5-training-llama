# TCD → loss-variance covariance 修正紀錄

研究目標：在使用者選定的固定 GPT-2 衍生架構 benchmark 上，以相同 GPU
成本降低至少 0.5 PPL，或以明顯較低成本達到相近品質。模型實際為
162,354,816 參數；舊 150M Llama campaign 已退役，不重啟。

狀態：**已依使用者指示停止這輪研究，不得自動重啟或重試。**
已取消 array 557707 中正在執行的 6–8 與尚未啟動的 9–11；0–5 和短測
557706 在停止前已完成。已確認這兩個 job ID 沒有任何工作留在 active queue。
既有結果原樣保留，未繼續調參、訓練或分析。恢復實驗需使用者另行指示。
這是探索性研究，沒有已成功或全球首次的宣稱。

停止證據：`outputs/gpt2-cov-20261016-v1/STOPPED_BY_USER.json`、
`slurm-stop-accounting.txt`、`stop-snapshot.json`。遠端研究根目錄也已寫入
`STOPPED_BY_USER.json`，其中 `resume_authorized=false`。
以下內容保留為停止前的歷史紀錄，不代表現在仍有實驗在進行。

## 1. 原版的實測與推論範圍

相同 seed 42、1,000 steps、524,288,000 tokens、70% cooldown、FineWeb
development validation；auxiliary AdamW 與架構不變。下表是最終 PPL，越低越好。

| 方法 | LR .00625 | LR .025 | LR .1 |
|---|---:|---:|---:|
| 調好的 Muon 配方，只改 matrix LR / 縮短預算 | 35.52050 | 33.65865 | 初始化網路失敗 |
| TCD1，rho=3 | 84.89795 | 364.02407 | 334.21684 |
| Fold-RMS，移除 TCD 幾何項 | 48.51728 | 292.33488 | 298.50301 |

同 LR=.00625，TCD 比控制組差，表示這個設定下幾何項有害。但兩個方法的
最佳 LR 都落在下邊界，不能宣稱已充分調參，更不能把它們與 Muon 的最佳
LR 強制綁定。Muon .1 那一組在建立 NCCL 通訊時出現 `No route to host`，
尚無 step 0 驗證；這不是 optimizer divergence，會另外重試並保留失敗成本。

完整 3,250-step Muon 基線（seed 42）另有 PPL 26.50544、8 H200 allocation
566 秒。不能把這個結果直接當作上述 1,000-step 新方法的同預算對照。
原始結果與曲線保存在 `outputs/gpt2-tcd-20261016-v1/results-snapshot.json`。
遠端凍結研究：`/work/u9603854/lab5-training-llama/experiments/gpt2-tcd-20261016-v1`。

Slurm accounting：八組成功 screen 共 2,044 allocation seconds × 8 GPUs，
即 4.54222 GPU-hours；網路失敗 667 秒，即 1.48222 GPU-hours；前輪短測
127 秒，即 .28222 GPU-hours。這個 TCD campaign 合計 **6.30667 GPU-hours**，
不含更早的 baseline 與其他研究。不能把失敗配置從搜尋成本中刪除。
來源：`outputs/gpt2-tcd-20261016-v1/slurm-accounting.txt`。

## 2. 要修正的具體機制

原版風險是

\[
R_E(D)=\tfrac14\{\|E^TD+D^TE\|_F^2+\|ED^T+DE^T\|_F^2\}.
\]

它是損失一階擾動平方的啟發式代理，而不是實際 covariance，也不構成
普遍的方差上界。當 E=zI，
真正的一階擾動為 z tr(D)。對稱且 trace=0 的 D 不受這種雜訊影響，但
R_E(D)=2z²||D||² 仍會懲罰它。求導沒有算錯，問題是代理目標過寬。

另一個獨立問題是步長尺度。Frobenius norm 相同的 rank-one 更新與 polar
更新，其 operator norm 比值最高可達 sqrt(min(m,n))。在 768×768 矩陣上，
這個上限約 27.7；這是數學上的可能差距，**不是已量測到模型中的差距**。
因此「相同 RMS + 相同 LR」不足以保證步長可比。本輪同時延伸較小 LR。

## 3. 從實際損失擾動到 covariance 估計

令每張 GPU 的 local mean gradient 為 x_i，i=1,...,8，每個含 65,536 target
tokens。實作先收集原本的 SUM gradients，再除以 65,536；沒有增加反向傳播。
將前四張與後四張分成 A、B，各 k=4。以 A 為例：

\[
a=\tfrac1k\sum_i x_i,\quad Z_{i,:}=\operatorname{vec}(x_i-a)^T,
\quad \widehat\Sigma_A=\frac{Z^TZ}{k(k-1)}=\frac{Z^TZ}{12}.
\]

若給定過去後 x_i 獨立同分布、均值為 mu、covariance 為 Sigma，則
E[Z^TZ]=(k-1)Sigma，因此 E[hatSigma_A]=Sigma/k，正好是組平均的 covariance。
這個期望等式已用二點分布的所有四樣本組合完整枚舉驗證。

給定方向 d，一階損失擾動為 xi^T d，其 variance=d^T Sigma d。定義

\[
v_A=\operatorname{tr}\widehat\Sigma_A,\quad
B_A=Z/\|Z\|_F,\quad r_A=\frac{v_A}{v_A+\|a\|_F^2},\quad
\lambda_A=\rho r_A.
\]

非零雜訊時 tr(B_A^T B_A)=1，且 B_A^T B_A=hatSigma_A/v_A。
rho 是待驗證的風險強度，首輪固定 3；r_A 是有偏但有界的啟發式 gate，
不能把它當成無偏 SNR 估計。v_A=0 時定義 P_A=I。

## 4. 更新的閉式解與成本

對另一組的 numerator u，解

\[
d_A=\arg\min_d\left\{\tfrac12\|d-u\|^2+
\tfrac{\lambda_A}{2}\|B_A d\|^2\right\}
=(I+\lambda_A B_A^TB_A)^{-1}u.
\]

用 Woodbury 恆等式：

\[
d_A=u-\lambda_A B_A^T
(I_4+\lambda_A B_AB_A^T)^{-1}B_Au.
\]

只需 4×4 線性系統，而不是 parameter-count 維度的 inverse。矩陣 factor
運算仍有 O(k²mn) 成本；跨 GPU 梯度收集也仍存在，全部計入時間。沒有 SVD、
Muon polar 或 trace compensation。核心 inverse 的條件數不超過 1+rho；
使用 FP32 的 `solve_ex`，沒有把浮點近似當成精確保證。

因 B_A^TB_A 是 PSD、trace=1，

\[
\frac{1}{1+\rho}I\preceq P_A\preceq I.
\]

因此它不放大向量範數，且 B_Au=0 時 **P_Au=u**，精確保留已觀測雜訊
子空間以外的方向。這直接修正 E=zI、trace-free 訊號的已知反例。
不做 trace compensation，避免把未觀測雜訊的方向無差別放大。

## 5. 保留對照組的 momentum 與尺度

為了單獨檢查幾何項，維持前輪 beta=.95 的 EMA-Nesterov 與 opposite-fold
RMS scaling。設歷史 momentum 為 M：

\[
N_A=\beta^2M+(1-\beta^2)a,\quad
N_B=\beta^2M+(1-\beta^2)b,\quad
s_A=\max(\operatorname{RMS}(N_A),10^{-8}),
\]

並同樣定義 s_B。fan-in 為 n 時：

\[
D=\frac{P_AN_B/s_A+P_BN_A/s_B}{2\sqrt n},\quad
M^+=\beta M+(1-\beta)(a+b)/2,\quad
W^+=(1-\eta\,wd)W-\eta D.
\]

rho=0 恢復前輪 Fold-RMS 控制組，除了浮點加總次序的差異。八張 GPU
的資料、參數同步與 auxiliary SUM reduction 都用獨立 reference 檢查。

假設兩組條件獨立，A_A=P_A/(sqrt(n)s_A)，nu=beta²M+(1-beta²)mu，則
E[D|past]=.5(E[A_A]+E[A_B])nu。Cross-fitting 消除這個 numerator/geometry
耦合，但 E[P] 不等於 (I+lambda E[B^TB])^-1，並未消除所有估計偏差。
若 s_min<=s_A,s_B<=s_max，可取 a0=1/(sqrt(n)s_max(1+rho))、
b0=1/(sqrt(n)s_min)。令 delta=M-mu，則

\[
\langle\mu,E[D]\rangle\ge a_0\|\mu\|^2
-\beta^2b_0\|\mu\|\|\delta\|.
\]

可見 stale momentum 仍能破壞下降方向。即使 P 是 contraction，也能把
梯度轉向高曲率的未知方向：u=(1,0)、雜訊沿 (1,1)、rho=3 時 P u=(.625,-.375)。
在 F(w)=.5(w1+1)²+5000w2²、w=0、eta=.1，loss 仍從 .5 上升到約 7.47。
此例已作為必須保留的失敗測試，而不是只保留通過的新反例。

Packed FineWeb 連續資料不保證上述 iid 假設；有限樣本噪聲 span 最多三維，
容易漏掉大量雜訊方向；gate、尺度、decay 與 LR 尚未充分調參。這些限制都
不能由「修好一個反例」消除。

### 5.1 一步期望下降的充分條件

這裡先分析單一參數 block，固定其他參數，並暫時令 weight decay=0。
假定 F 對這個 block 是 L-smooth，兩組條件獨立且 local gradient covariance
為 Sigma，尺度滿足上一段的確定上下界。所有期望均條件於歷史。定義

\[
c=a_0\|\mu\|^2-\beta^2b_0\|\mu\|\|M-\mu\|,\quad
V=\|\nu\|^2+(1-\beta^2)^2\operatorname{tr}(\Sigma)/4.
\]

由 squared norm 的凸性與 ||A_A||,||A_B||<=b0，
E||D||² <= .5 b0² E(||N_A||²+||N_B||²)=b0²V。
將此式及前述 inner-product 下界代入 smoothness inequality，得到

\[
\mathbb E[F(W-\eta D)]-F(W)
\le -\eta c+\tfrac{L\eta^2}{2}b_0^2 V.
\]

因此 c>0、V>0 且 0<eta<2c/(L b0²V) 是嚴格期望下降的充分條件。
它直接暴露三種失效途徑：momentum 過時使 c<=0；噪聲／尺度造成 V 或 b0
過大；LR 超出由曲率 L 決定的安全範圍。這是保守充分條件，違反時不代表
必然上升。也沒有證明它比 Muon 的可用步長大。

實際有 decoupled decay 時，更新向量是 D+wd W，精確上界應改成

\[
-\eta(\langle\mu,\mathbb E D\rangle+wd\langle\mu,W\rangle)
+\tfrac{L\eta^2}{2}\{\mathbb E\|D\|^2
+2wd\langle W,\mathbb E D\rangle+wd^2\|W\|^2\}.
\]

所以沒有把無 decay 的定理冒充成整個訓練器的保證。多 block 同時更新時
必須用完整 Hessian 的 smoothness bound，不能忽略 block 之間的交互作用；
auxiliary AdamW 也需另行分析。實際 L、尺度上界和獨立性尚未驗證，這份
推導解釋可能成功與失敗的條件，不能取代 LLM 實驗。

## 6. 原創性與參考資料

Covariance preconditioning、Woodbury inverse 與 cross-fitting 都是既有工具。
[Nayak et al.](https://arxiv.org/html/2605.20756v1) 已研究 optimizer 中的
cross-fitting 與 inverse-estimator bias；
[Mandt et al.](https://www.cs.columbia.edu/~blei/papers/MandtHoffmanBlei2017.pdf)
也討論以 gradient-noise covariance 選擇 preconditioner。
這個修正版的用途是驗證原 TCD 的風險代理是否造成退步，**不把已知方法
重新命名當成原創 optimizer**。它仍是通往新方法的診斷階段。

## 7. 下一輪先登記的實驗

所有方法累積使用相同 matrix LR 集合：
{.000390625,.0015625,.00625,.025,.1}，matrix decay=.05、seed 42、1,000 steps。
方法：Muon、原 TCD1、Fold-RMS、covariance 修正版。重用已完成且條件相符的
舊結果；補六個較小 LR 組合、新版五個 LR，以及網路失敗的 Muon .1。
共十二個新 full-screen jobs，加一個不參與品質比較的短測。

這不是全域最佳參數搜尋：若最佳 LR 仍在邊界，仍需延伸範圍；rho、decay
與 auxiliary LR 也要在正式勝出宣稱前公平調整。1000-step cooldown 是等比
縮短的，可能與 3250-step 排名不同。若新版連短程強基線都差很多，不會
直接投入大量 full-budget confirmation。

`note/SOTA.md` 也顯示 LR 與 decay 的時間尺度有交互作用。本輪固定 wd
並不固定累積收縮量：純 decay 的剩餘權重比例是 product_t(1-eta_t wd)，
近似 exp(-wd sum_t eta_t)。因此這一輪測的是各 LR 下完整配方，不能把
所有改善都單獨歸因於 gradient step。正式確認前還需做固定 eta×wd 的
對照及各方法等預算的 decay 搜尋；不能直接搬用另一架構 SOTA 的 PPL。

原研究保留的 validation targets [80,000,000,90,485,760) 在本輪仍不使用。
它們應在鎖定超參數後才用於確認，不將公開 dev 上反覆挑選的結果當成獨立測試。

程式隔離在 `src/exp/covariance_revision/`，從前輪凍結版本複製，保留
`base_source.json`。不覆寫同一工作區其他研究正在修改的 optimizer 入口。
遠端新研究預定為 `gpt2-cov-20261016-v1`，沿用已驗證 FineWeb shards。

## 8. 執行結果

截至 2026-10-16 06:33 UTC：

- 本地 25 個測試通過（covariance 7、整合 6、既有 TCD 12），Ruff / ty 通過。
- nano4 上 13 個 CPU 測試通過；**八卡 collector 與實際模型短測尚未通過**。
- 已凍結、部署程式與 12 組 preregistered configs；archive SHA256：
  `c64881025a199c9ec1caffcc7cbb85a32c79cf00cbc6cdd21746732f0c683abc`。
- `sbatch` 回覆 `Socket timed out on send/recv operation`，沒有取得 job ID。
  後續 `squeue` 也 timeout；`sacct` 最近一次成功查詢未找到本次短測。
  尚需確認 active queue，避免把沒有收到 acknowledgment 誤判成沒有提交。
- 12 組品質篩選尚未提交，也沒有新版 LLM PPL；排程失敗不是 optimizer 失敗。
- 額外的有限分布推導檢查枚舉 256 個八樣本組合：conditional mean identity
  最大絕對誤差 4.44e-16；更新二階矩 1.73791 <= 上界 2.15613。在所構造的
  L=2 quadratic 上，測試 eta=.0892204，期望 loss change=-.183199，低於理論
  上界 -.0171634。這是定理數值稽核，**不是 LLM 效能證據**。

狀態與完整數值：`outputs/gpt2-cov-20261016-v1/deployment.json`、
`theory-check.json`、`protocol.json`。推導檢查可重現：

```bash
PYTHONPATH=src .venv/bin/python -m exp.covariance_revision.theory_probe \
  --output outputs/gpt2-cov-20261016-v1/theory-check.json
```

只有已完成、通過資料與來源核對的 run 才計入結果；基礎設施失敗單列，
仍計入已發生的資源成本。相同 token 的 PPL 與相同 GPU 時間的品質分開呈現。

### 2026-10-16 07:12 UTC 更新

使用者提示 Slurm 應可提交後重新檢查：`squeue` 和 `sacct` 均成功回應，
確認前次沒有留下短測工作。短測 job **557706** 已在 `25a-hgpn034` 執行；
十二組 screen 的 array job **557707** 已接受，使用
`--array=0-11%3 --dependency=afterok:557706 --kill-on-invalid-dep=yes`，
最多同時三組，八卡 correctness 與模型短測全部成功後才開始。
這是啟動狀態更新，尚無新版完整訓練結果，不能把已提交當成品質驗證成功。

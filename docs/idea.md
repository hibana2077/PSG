# Path-Stability Generalization (PSG) Framework

---

## 研究核心問題

1. **可觀測的「訓練路徑複雜度」是否能比最終權重（endpoint）更穩定地解釋/預測泛化？**
   既有研究顯示：SGD 的泛化可由「演算法穩定性」解釋（與訓練步數/步長相關）([Proceedings of Machine Learning Research][1])，也有資訊論界指出泛化上界可依賴「沿路徑的梯度局部統計」，且經驗上「trajectory length」與 test loss/complexity 有顯著相關([NSF 公共存取庫][2])。PSG 的核心就是把這些線索變成**可量測、可控制、可改進泛化**的框架。

2. **能否建立（至少在 toy/可控假設下）「generalization gap 與 PL/GE 單調相關」的可證命題？**
   讓你的 toy 指標（PL/GE）不是只有 correlation，而是能接到 stability / IT bound 的形式。

3. **路徑正則（path regularization）能否在「不惡化訓練誤差」或「同等訓練誤差」下，系統性縮小 generalization gap？**
   把觀察提升到 intervention：你主動把 PL/GE 壓低，看 gap 是否可重現地縮小。

---

## 研究目標

1. **定義 PSG 指標族**：以你給的兩個為主

   * 位移路徑長：(\mathrm{PL}=\sum_{t=1}^T \lVert w_t-w_{t-1}\rVert_2)
   * 梯度能量：(\mathrm{GE}=\sum_{t=1}^T \eta_t^2 \lVert \nabla \ell_t\rVert_2^2)

2. **建立理論骨架（分層可證）**

   * Level-A（最小可證，凸/平滑）：用演算法穩定性把 generalization gap 上界寫成「步長與路徑步幅」的函數（接到 Hardt-Recht-Singer 路線）([Proceedings of Machine Learning Research][1])。
   * Level-B（離散 SGD/非凸的路徑統計）：把「沿路徑的梯度統計」上界（資訊論）轉成可估的 proxy，對齊你定義的 GE 。
   * Level-C（路徑長 → 泛化）：對齊「短路徑 ⇒ 好泛化」的既有長度型結果（多以 gradient flow 或特定條件表述）([OpenReview][3])。

3. **提出可操作的 path regularizer**（你已經有方向）並系統驗證。

4. **用 toy → 小型真實資料集 → 更真實設定**的階梯式實驗，證明：

   * 指標可預測（prediction）
   * 正則可改善（intervention）
   * 效果可解釋（mechanism）

---

## 預期貢獻

1. **概念貢獻**：把「泛化依賴訓練動態」落在一組可計算的 path-metrics，而不是只談 flatness/endpoint norm。
2. **理論貢獻**：給出一條清楚的推導鏈：
   [
   \text{(stability / IT bound)} \Rightarrow f(\text{path statistics}) \Rightarrow \text{PL/GE proxy}
   ]
   對齊 SGD 穩定性([Proceedings of Machine Learning Research][1])與路徑梯度統計型上界，並補上「路徑長連到泛化」的既有長度型理論線索([OpenReview][3])。
3. **方法貢獻**：提供一個簡單可插拔的 path regularization，與既有 trick（weight decay / clipping / schedule / early stopping）統一在「路徑穩定化」視角下。
4. **實證貢獻**：從 toy 到 CIFAR-10 子集等設定，給出可重現的 PL/GE ↔ gap 以及 intervention 的 ablation（對照 Loukas 等人的 correlation 視角）([NSF 公共存取庫][2])。

---

## 創新點

1. **把「路徑」當成主要解釋變量**，不是輔助觀察：不只量測，還做正則化干預。
2. **雙指標（PL 與 GE）對齊兩條理論路線**：

   * PL 更像「幾何路徑複雜度」與長度型泛化結果對齊([OpenReview][3])
   * GE 更像「沿路徑梯度統計」與資訊論 SGD 泛化上界對齊
3. **把常見訓練技巧重新詮釋為 path-stabilizers**：例如 clipping/小步長/較大 batch 的效果，是否可被 PL/GE 共同解釋（這是可檢驗的）。

---

## 理論洞見（你可以寫在 Introduction 的“Why”）

1. **SGD 泛化與穩定性**：經典結果指出 SGD 的 generalization 與演算法穩定性相關，並受步長、步數等因素影響([Proceedings of Machine Learning Research][1])。PSG 把「這些因素的總效應」壓縮成一個可觀測的路徑量（PL/GE）。
2. **沿路徑的梯度局部統計很關鍵**：資訊論觀點直接把泛化上界寫成沿 SGD 迭代路徑上的梯度變異/平滑度等局部統計，這與 GE 的精神高度一致。
3. **短路徑 ⇒ 好泛化的理論線索已存在，但需要落地到離散 SGD + 可操作正則**：已有工作在特定條件/連續時間（gradient flow）下把「optimization path length」連到泛化([OpenReview][3])；PSG 的價值在於把它變成 PyTorch 裡可直接驗證與可改進的 pipeline。

---

## 方法論

### 1) 指標計算（training log 即可）

* PL：每一步 optimizer step 後計算 (\sum |w_t-w_{t-1}|_2)
* GE：用當步的 gradient norm 與 lr：(\sum \eta_t^2|\nabla \ell_t|_2^2)

> 備註：對 SGD，(\Delta w_t = -\eta_t g_t)，所以 **PL 與 GE 其實分別對應 (\sum \eta_t|g_t|) 與 (\sum \eta_t^2|g_t|^2)**；前者偏「路徑長」，後者偏「能量/累積擾動」。

### 2) Path regularization（最小可用版本）

* 參數位移 penalty：(\lambda|w_t-w_{t-1}|_2^2)
* 或 gradient penalty：(\lambda|g_t|_2^2)（等價於推向較小 GE）

**PyTorch 插入點（概念碼）**：

```python
# after loss.backward()
delta2 = 0.0
for p, p_prev in zip(model.parameters(), prev_params):
    delta2 += (p - p_prev).pow(2).sum()
loss = loss + lam * delta2

optimizer.step()

# update snapshot
prev_params = [p.detach().clone() for p in model.parameters()]
```

---

## 數學理論推演與證明（建議寫成「可完成的 2~3 個命題」）

### 命題 A（stability → PL 上界；凸/平滑作為第一步）

**設定**：(\ell(w;z)) 為 (L)-Lipschitz、(\beta)-smooth；資料集 (S,S') 僅差一筆；用（S）GD/SGD 訓練得到路徑 ({w_t})、({w'_t})。
**目標**：證明 generalization gap 可被「兩路徑距離」控制，而兩路徑距離又可用「每步位移」累積界住，進而得到與 PL/GE 相關的上界。

你可以沿 Hardt-Recht-Singer 的 uniform stability 證明架構走（追蹤 leave-one-out 的 iterates 差距，再用 Lipschitz 把 loss 差距界住）([Proceedings of Machine Learning Research][1])，但把最後一步改寫成：

* 先得到（典型形式）(|w_t-w'_t|) 的遞推界
* 再把每步更新 (|w_t-w_{t-1}|=\eta_t|g_t|) 帶入，推出「當 PL 小時，兩條路徑不易分岔」
* 最後得到 (\text{gen gap} \le c\cdot \mathrm{PL}/n) 或 (\le c'\cdot \sqrt{\mathrm{GE}}/n) 類型的 bound（常數視假設而定）

> 這一段你的目標不一定要追求最 tight，而是把 **PSG 的“可證鏈條”建立起來**。

### 命題 B（資訊論泛化 bound → GE proxy；非凸也能講）

Neu 等人的結果明確指出：SGD 的泛化上界可依賴「沿迭代路徑的 stochastic gradient 局部統計（例如 variance、局部 smoothness、對輸出擾動的敏感度）」。
你的路線：把其中的「沿路徑梯度統計」簡化成可估 proxy（例如用 minibatch 梯度 norm/方差估計），最後落到 GE：
[
\mathrm{GE}=\sum_t \eta_t^2|g_t|^2 \approx \sum_t \eta_t^2 \cdot \text{(local gradient scale)}
]
然後提出一個可檢驗假設：**在其他因素控制下，GE 越小，generalization gap 越小**（對照你 toy 的操作）。

### 命題 C（短路徑 → 好泛化；對齊既有「path length generalization」文獻）

Liu 等系列工作直接在特定條件下把「optimization trajectory length」連到泛化（包含 “short optimization paths lead to good generalization” 的論述）([OpenReview][3])。
你的創新是：把它改造成 **離散 SGD、可插拔正則、可在 toy 與小型真實資料集驗證**。

此外，若你想補強「為什麼 PL 本身不會無限大」：可引用 GD/GF 的 path length bounds 文獻，說明在某些收斂條件（如 PL/PKL）下路徑長可被控制，用來合理化「路徑長是一個 meaningful 的 complexity measure」。

---

## 預計使用 dataset

### Toy（最小可驗證）

* `two-moons`（scikit-learn 生成）
* `teacher-student` 合成分類（可控 label noise / margin）

---

## 與現有研究之區別（你可以直接放 Related Work）

1. **vs. SGD stability（Hardt et al.）**：他們證明 SGD 的泛化可由演算法穩定性界住，但主要以步長/步數/平滑性條件表達，未把「實際走過的路徑複雜度」當成主要可觀測量來做干預([Proceedings of Machine Learning Research][1])。
2. **vs. 資訊論 SGD bounds（Neu et al.）**：他們的上界依賴沿路徑的梯度局部統計，你的 PSG 可以把它工程化成 GE 這種「訓練 log 即可計算」的 proxy，並提出 path regularizer 做實證干預。
3. **vs. path length generalization（Liu 系列）**：既有結果多在 gradient flow / 特定條件下連結 path length 與 generalization，你補上「離散 SGD + 實作正則 + 更直接的 PL/GE 指標」並做最小可驗證實驗鏈([OpenReview][3])。
4. **vs. 經驗相關性工作（Loukas 等）**：已有觀察指出 trajectory length 與 test loss/complexity 相關，你把它推進成「可控變量（regularize path）」來驗證因果方向，並用 gap/robustness 指標做更完整的評估([NSF 公共存取庫][2])。
5. **補充：寬網路/NTK 設定下的 trajectory length**：寬網路理論中也出現「SGD trajectory length 有界」與泛化分析的脈絡，你可用來說明 PSG 與現代理論接得上([yuancaohku.github.io][4])。

---

## Experiment 設計

### A. Correlation（驗證 PSG 指標是否“像”泛化）

* 固定資料/模型，掃描：

  * optimizer：SGD / SGD+momentum / Adam
  * lr：常數 vs cosine vs step decay
  * weight decay
  * batch size
  * gradient clipping threshold
* 每次訓練記錄：

  * train loss、test loss、generalization gap
  * PL、GE（以及可選：(\max_t |g_t|)、平均 (|g_t|)）
* 分析：

  * PL vs test error / gap（Pearson + Spearman）
  * GE vs test error / gap
  * **控制變因**：同訓練誤差分桶比較（避免只是 “train 得不夠好”）

### B. Intervention（路徑正則是否真的縮小 gap）

* 對同一組設定加入 path regularizer（(\lambda) sweep）
* 觀察是否同時成立：

  1. (\mathrm{PL}\downarrow) 或 (\mathrm{GE}\downarrow)
  2. test error 或 gap 改善
  3. 不只是 early stopping 的替代（可固定訓練步數、或匹配 train loss）

### C. Stress test（把“過擬合”做得更明顯）

* label noise（隨機翻轉比例）
* 小資料 regime（減少 n）
* data augmentation 強弱
  看 PSG 指標是否能穩定反映「何時開始走向記憶化」——這點能呼應「沿路徑局部統計」與「訓練動態揭露複雜度」的文獻脈絡([Proceedings of Machine Learning Research][5])。

---
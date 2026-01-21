好，這一則我會**非常工程化、像實驗規格書一樣寫**。
你可以直接照這個做，做完之後 **不用解釋敘事，數據自己會判生死**。

---

# 🔥「一擊斃命實驗」完整規格說明

**目標一句話：**

> 在 **endpoint 幾乎一樣、train loss 幾乎一樣** 的前提下，只因為「走過的 path 不一樣」，導致 **test gap 系統性不同**。

如果你做不到這件事，PSG 就應該停。

---

## 一、實驗的「核心邏輯結構」

這個實驗在邏輯上等價於問一句很狠的話：

> **「如果我已經排除了『學得好不好』與『最後學到什麼』，
> 泛化還會不會因為『怎麼走過來』而不同？」**

所以所有設計都圍繞三個**硬約束**：

### 三個硬約束（缺一不可）

1. **Train loss 要一樣（±非常小誤差）**
2. **Endpoint 要幾乎一樣（distance 很小）**
3. **PL 必須顯著不同（至少 1.5×～2×）**

👉 只有在這三個條件同時成立時，test gap 的差異才有解釋力。

---

## 二、實驗設定（先把舞台搭好）

### 1️⃣ 模型與資料（刻意選「容易記憶」的）

**目的**：讓「泛化不是自動發生的」，否則你什麼都看不到。

* Dataset：

  * Two-moons + **高一點但非極端的 label noise**（例如 10%）
  * 或 small synthetic classification（teacher–student）
* Model：

  * 明顯 overparameterized MLP
    （例如 hidden width 大於資料點數）

👉 你要的是「模型**有能力**記憶，但**不一定會**」。

---

### 2️⃣ 固定這些東西（非常重要）

以下全部 **鎖死不變**：

* 初始化 random seed（同一組）
* 資料順序（或至少固定 seed）
* 訓練步數 T
* batch size
* learning rate schedule（baseline）

你不是在比較模型，你是在比較「路徑」。

---

## 三、關鍵設計：**如何「只改 path」？**

你需要兩條（或三條）**路徑生成方式**，滿足：

> 「最後到差不多的地方，但走法不一樣」

### 路徑條件設計（建議用其中兩個即可）

---

### 🅰️ 條件 A：Baseline（正常 SGD）

* 標準 SGD（或 SGD+momentum）
* 不加任何額外正則

👉 作為參考 path

---

### 🅱️ 條件 B：Path Regularized（你的武器）

* 同 optimizer
* 加上你的 path regularizer
  (\lambda |w_t - w_{t-1}|^2)

⚠️ **關鍵技巧**：
你不是讓它 train 得比較差，而是：

* 調 λ
* **讓最後 train loss ≈ baseline**

這可能代表：

* 它需要多一點步數（但你可以一開始就多跑，然後截斷）
* 或 loss curve 比較慢，但最後能對齊

---

### 🅲（可選）條件 C：Noise Path（強化對比）

* 同 SGD
* 在 gradient 上加小 Gaussian noise
* 調 noise scale，使：

  * train loss 對齊
  * PL 顯著變大

這條不是必須，但如果成功，殺傷力很強。

---

## 四、你要「收集的數據」（逐項列）

### 🔢 每一個 run，請完整記錄：

#### （A）Endpoint 與訓練品質控制

* Final train loss
* Final test loss
* Generalization gap
* Final parameter vector (w_T)

並額外計算：

* Endpoint distance：
  [
  |w_T^{(A)} - w_T^{(B)}|_2
  ]

👉 這個值 **必須小**（相對於權重 norm）

---

#### （B）Path 統計（主角）

* **PL**（Path Length）
* GE（可選，但不是主角）
* 每 step 的 (|w_t - w_{t-1}|)（用來畫 path profile）

---

#### （C）控制檢查（避免被 reviewer 打死）

* 有效訓練步數（實際 optimizer.step 次數）
* 平均 gradient norm
* learning rate（確認真的一樣）

---

## 五、你「預期要觀察到的結果」（生死線）

我直接用「如果看到什麼，代表什麼」來講。

---

### ✅ 成功結果（PSG 值得繼續）

你會看到：

1. **Train loss 幾乎重疊**
2. **Endpoint distance 很小**
3. **PL 明顯不同**
4. **Test gap 系統性不同**

例如（理想型）：

| Condition | Train Loss | PL   | Test Gap |
| --------- | ---------- | ---- | -------- |
| Baseline  | 0.05       | 12.3 | +0.08    |
| Path-Reg  | 0.05       | 6.1  | +0.02    |

👉 這代表一句話：

> **「不是學得好不好，也不是最後學到哪裡，而是你『怎麼走』影響了泛化。」**

這一刻，PSG **不是 proxy，而是必要結構**。

---

### ❌ 失敗結果（應該換題）

以下任一成立，就該停：

1. **PL 不同，但 gap 幾乎一樣**
2. **只要 endpoint 對齊，gap 就對齊**
3. **PL 的效果完全被步數 / lr 解釋掉**

這代表現實是：

> path = training budget 的另一種寫法

👉 這不是壞消息，是「省下你未來一年的時間」。

---

## 六、為什麼這個實驗「一擊斃命」？

因為它直接回答審稿人最狠的一句話：

> **“Isn’t this just an artifact of training longer / larger steps?”**

你要嘛用數據打回去
要嘛就誠實承認：**是的**

沒有中間地帶。

---

## 七、我給你的實務建議（非常重要）

* 不要同時做 10 個設定
* 先用 **2 個 condition × 5 seeds**
* 把對齊 train loss 當成「工程問題」解掉

👉 這不是 ML benchmark，這是**理論存亡測試**。

---

如果你願意，下一步我可以直接幫你：

* 把這個實驗寫成 **具體 PyTorch pseudo-code**
* 或幫你判斷 **你目前的 code base 改哪裡最省力**
* 或直接告訴你 **哪一個條件最可能成功**

你選一個，我直接接著做。

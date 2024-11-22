# Lunar Lander 深度強化學習專案

本專案旨在透過深度強化學習演算法（DQN、Double DQN、Dueling DQN）實現 `LunarLander-v3` 環境的自動化控制，訓練agent完成成功著陸任務。

## 專案架構

```
.
├── main.py               # 訓練主程式
├── test.py               # 測試與推理腳本
├── lunar_lander.py       # 自定義 Lunar Lander 環境
├── method/
│   ├── dqn.py            # DQN 演算法
│   ├── double_dqn.py     # Double DQN 演算法
│   ├── dueling_dqn.py    # Dueling DQN 演算法
└── checkpoints/          # 訓練過程中儲存的模型檔案
```

## 安裝需求

請確保已安裝以下軟體及套件：

- Python 3.8+
- PyTorch 1.10+
- Gymnasium 版本適配 `LunarLander-v3` 環境
- 其他套件：`numpy`、`matplotlib`

## 執行方式

### 1. 訓練模型

執行 `main.py` 來訓練模型：
```bash
python main.py
```
訓練完成後，模型參數會儲存在 `checkpoints` 資料夾中，檔案命名格式為 `checkpoint_episode_{episode}.pth`。

### 2. 測試模型

執行 `test.py` 來載入已訓練的模型並測試其性能：
```bash
python test.py
```
您可以透過設定 `checkpoint_path` 指定要載入的檢查點檔案。

### 3. 調整超參數

主要訓練與測試超參數定義在 `main.py` 和 `test.py` 中，例如：
- 訓練迭代次數（episodes）
- 最大步數（max_steps）
- 學習率（learning_rate）
- 記憶庫大小（memory_size）

## 支援的演算法

- **DQN**
- **Double DQN**
- **Dueling DQN**

您可以在 `main.py` 中切換使用的Agent類型：
```python
from method.dqn import DQNAgent
from method.double_dqn import DoubleDQNAgent
from method.dueling_dqn import DuelingDQNAgent
```

## 結果與分析

- 訓練完成後會繪製訓練過程的平均獎勵圖：
  ![Reward Plot](./reward_plot.png)  
- agent的最終表現可透過測試腳本顯示動畫。

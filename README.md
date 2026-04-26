# Quality_Management_Battery

本專案致力於研究鋰電池健康狀態 (State of Health, SOH) 的預測方法，主要基於 NASA 鋰電池數據集。

## 主要內容

### 1. CNN-LSTM 前 30 分鐘數據預測 SOH (依 battery_id 拆分)
位於目錄：`1. CNN_LSTM前30分鐘數據預測SOH_依battery_id拆分/`

此子項目實現了：
*   **混合模型**：結合 1D CNN 與 LSTM 的優點，提取時序數據特徵。
*   **前 30 分鐘預測**：僅需放電初期的數據即可預測剩餘容量。
*   **嚴謹驗證**：數據按電池 ID 拆分，驗證模型對未知電池的泛化能力。

詳細說明請參閱該目錄下的 [README.md](./1.%20CNN_LSTM前30分鐘數據預測SOH_依battery_id拆分/README.md)。

---

## 快速開始

### 資料集準備
下載資料集：[NASA Battery Dataset (Kaggle)](https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset)

解壓後確保 `metadata.csv` 與 `data` 資料夾與執行腳本位於同一層級。

### 環境需求
*   Python 3.8+
*   PyTorch
*   NumPy, Pandas, Matplotlib, Scikit-learn, Scipy

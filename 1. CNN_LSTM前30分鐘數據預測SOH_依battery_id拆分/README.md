# Quality_Management_Battery - 鋰電池健康狀態 (SOH) 預測

本專案利用 NASA 鋰電池數據集，開發了一套基於 **CNN-LSTM 混合模型** 的健康狀態 (State of Health, SOH) 預測系統。該模型僅需使用放電循環前 **30 分鐘** 的感測數據（電壓、電流、溫度等），即可精準預測電池的剩餘容量與 SOH。

## 更新亮點
*   **模型架構升級**：從純 1D CNN 升級為 **CNN-LSTM**。利用 CNN 提取空間特徵，再由 LSTM 捕捉時間序列的長程依賴性。
*   **更嚴謹的數據拆分**：改採 **依電池 ID 拆分 (Split by Battery ID)**。確保訓練集與測試集完全屬於不同的電池實體，有效防止數據洩漏（Data Leakage），提升模型在未知電池上的泛化能力。
*   **自動化診斷報告**：訓練與測試流程會自動生成多維度的視覺化圖表，包含殘差分析、誤差分布與 SOH 健康度評估。

---

## 快速開始

### 1. 資料集準備
1.  下載 NASA 鋰電池資料集：[Kaggle 連結](https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset)
2.  解壓後確保 `metadata.csv` 與 `data` 資料夾（內含原始 CSV 檔案）與本專案腳本置於同一目錄下。

### 2. 環境安裝
確保已安裝 Python 3.8+ 與以下套件：
```bash
pip install torch numpy pandas matplotlib scikit-learn scipy
```

### 3. 執行流程
依序執行以下三個腳本：
1.  **預處理**：`python preprocess.py` (生成 .npy 數據與標準化參數)
2.  **訓練**：`python CNN_LSTM_train.py` (訓練模型並存檔)
3.  **測試**：`python CNN_LSTM_test.py` (評估模型效能)

---

## 腳本詳細說明

### 1. `preprocess.py` - 數據預處理與電池拆分
*   **關鍵邏輯**：
    *   **特徵工程**：提取放電前 1800 秒（30 分鐘）的數據，並以 5 秒間隔進行線性插值，統一輸入維度。
    *   **清洗機制**：排除異常容量值（< 1.2Ah 或 > 2.0Ah）及已知的損壞電池（B0049-B0052）。
    *   **依 ID 拆分**：將所有電池按 **7:1:2** 的比例分配至訓練、驗證與測試集。這模擬了實際應用中「新電池」的預測情境。
    *   **標準化**：執行 Z-score 標準化，並儲存參數於 `norm_params.npy`。

### 2. `CNN_LSTM_train.py` - CNN-LSTM 混合架構訓練
*   **模型架構**：
    *   **CNN 區塊**：三層 1D 卷積層，通道數從 32 增至 128，捕捉放電曲線的局部波動特徵。
    *   **LSTM 區塊**：將 CNN 提取的特徵序列輸入 LSTM 層，強化對時間維度趨勢的理解。
    *   **預測頭**：全連接層結合 Dropout (0.2) 輸出最終預測容量。
*   **訓練機制**：
    *   使用 Adam 優化器與 MSE 損失函數。
    *   **早停機制 (Early Stopping)**：Patience 設定為 200，確保在最優點停止訓練。
    *   **視覺化報告**：產出 `training_results_CNN_LSTM.png`，包含 Loss 曲線、預測散點圖與誤差分布。

### 3. `CNN_LSTM_test.py` - 模型驗證與 SOH 分析
*   **性能評估**：在完全獨立的測試電池組上進行推理。
*   **SOH 換算**：將預測容量轉化為 SOH 百分比（基於標稱容量 2.0Ah）。
*   **輸出圖表**：
    *   `testing_analysis_CNN_LSTM.png`：容量預測的殘差分析與高斯擬合。
    *   `soh_analysis_CNN_LSTM.png`：SOH 健康度預測的精準度分析。

---

## 專案結構
```text
.
├── preprocess.py                 # 數據預處理與電池 ID 拆分
├── CNN_LSTM_train.py             # CNN-LSTM 模型訓練與視覺化
├── CNN_LSTM_test.py              # 模型推理與 SOH 效能分析
├── best_model_CNN_LSTM.pth      # 訓練後保存的最優模型權重
├── norm_params.npy               # 特徵標準化參數
├── *.npy                         # 預處理後的訓練/驗證/測試集
└── *.png                         # 自動生成的診斷報告與圖表
```

# Quality_Management_Battery
## 資料集準備、程式準備和套件安裝

下載資料集：https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset?resource=download

解壓縮後會看到一個資料夾cleaned_dataset，裡面有data資料夾、extra_infos資料夾、metadata.csv。

從此github儲存庫中的資料夾"1.CNN前30分鐘數據預測SOH"中，把preprocess.py、train.py、test.py下載下來並放到cleaned_dataset資料夾裡。

安裝python，並裝pytorch套件

pytorch套件安裝網址：https://pytorch.org/get-started/locally/

## 執行程式
先執行preprocess.py，再執行train.py，最後執行test.py。

以下是程式說明：
### 1. `preprocess.py` - 數據預處理與工程化
這是專案的基礎模組，負責將原始觀測數據轉化為適合神經網路訓練的標準化特徵。
*   **數據篩選與清洗**：從原始數據集中提取 `discharge` 事件，並嚴格限制容量範圍在 **1.2Ah 至 2.0Ah** 之間，自動剔除異常值與空數據。
*   **時序特徵對齊**：
    *   提取放電開始前 **30 分鐘**（1800 秒）的關鍵特徵。
    *   利用**線性插值**技術處理不穩定的採樣率，將每個樣本統一重取樣為 **61 個時間點**（每 30 秒一個取樣點）。
*   **資料集拆分 (7:1:2)**：將 1,800+ 樣本精確劃分為訓練集（Train）、驗證集（Val）與測試集（Test）。
*   **Z-score 標準化**：計算訓練集特徵的均值與標準差，並對所有數據進行縮放，解決不同物理量（V, A, °C）量綱不一的問題。

### 2. `train.py` - 1D CNN 模型構建與訓練
本模組定義了深度學習模型架構，並實現了自動化訓練流程。
*   **1D CNN 模型架構**：
    *   採用一維卷積神經網路（1D CNN），專門設計用於提取時間序列數據中的局部形狀特徵（如電壓跌落曲線的斜率）。
    *   整合了 **Batch Normalization** 與 **Dropout** 層，增強模型的收斂速度與泛化能力。
*   **智慧訓練機制**：
    *   使用 **MSE (Mean Squared Error)** 作為損失函數，**Adam** 作為優化器。
    *   **早停機制 (Early Stopping)**：實時監控驗證集損失，若連續 15 個週期無改善則自動終止訓練，確保模型具備最佳的泛化效能。
    *   自動保存性能最優的模型權重檔案 (`best_model.pth`)。

### 3. `test.py` - 模型評估與視覺化分析
用於檢驗模型在未知數據上的表現，並提供實務應用工具。
*   **多維度性能指標**：計算並輸出 MAE、RMSE、$R^2$ Score 等回歸分析指標。
*   **專業視覺化報告**：生成 `testing_analysis.png` 與 `soh_analysis.png`，包含：
    *   **預測 vs 真實對比圖**：觀察預測點與 $y=x$ 對角線的貼合程度。
    *   **殘差分析圖**：檢查模型在不同電量階段的預測穩定性。
    *   **誤差高斯擬合**：利用常態分佈統計，分析模型誤差的均值（Bias）與標準差（Variance）。
*   **SOH 換算分析**：自動將容量預測結果換算為 **SOH (State of Health)**，提供 0-100 的健康度數值評估。
*   **實務預測接口**：內建單一檔案預測功能，可直接輸入原始 CSV 格式的感測器數據並即時輸出預測結果。

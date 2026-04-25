import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

# 從 CNN_LSTM_train.py 匯入
from CNN_LSTM_train import BatteryCNNLSTM, BatteryDataset

def load_norm_params():
    if not os.path.exists('norm_params.npy'):
        raise FileNotFoundError("找不到 norm_params.npy")
    params = np.load('norm_params.npy', allow_pickle=True).item()
    return params['mean'], params['std']

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")

    # 1. 載入模型
    model = BatteryCNNLSTM().to(device)
    if not os.path.exists('best_model_CNN_LSTM.pth'):
        print("錯誤：找不到 best_model_CNN_LSTM.pth")
        return
    model.load_state_dict(torch.load('best_model_CNN_LSTM.pth', map_location=device))
    model.eval()

    # 2. 載入測試數據
    test_ds = BatteryDataset('X_test.npy', 'y_test.npy')
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.numpy())

    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()
    residuals = all_preds - all_targets

    # 3. 圖表 1: 容量分析 (4 欄位)
    plt.figure(figsize=(15, 10))
    
    # 圖 1: 預測 vs 真實
    plt.subplot(2, 2, 1)
    plt.scatter(all_targets, all_preds, alpha=0.5, color='blue')
    plt.plot([min(all_targets), max(all_targets)], [min(all_targets), max(all_targets)], 'r--')
    plt.title('Actual vs Predicted Capacity (CNN-LSTM)')
    plt.xlabel('Actual (Ah)')
    plt.ylabel('Predicted (Ah)')

    # 圖 2: 殘差圖
    plt.subplot(2, 2, 2)
    plt.scatter(all_targets, residuals, alpha=0.5, color='purple')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title('Residual Plot (CNN-LSTM)')
    plt.xlabel('Actual Capacity (Ah)')
    plt.ylabel('Error (Ah)')

    # 圖 3: 誤差高斯擬合
    plt.subplot(2, 2, 3)
    mu, std_err = norm.fit(residuals)
    plt.hist(residuals, bins=30, density=True, color='skyblue', edgecolor='black', alpha=0.6)
    xmin, xmax = plt.xlim()
    x_range = np.linspace(xmin, xmax, 100)
    plt.plot(x_range, norm.pdf(x_range, mu, std_err), 'r', linewidth=2, label=f'mu={mu:.4f}\nsigma={std_err:.4f}')
    plt.title('Error Distribution (Gaussian Fit)')
    plt.legend()

    # 圖 4: 誤差絕對值直方圖
    plt.subplot(2, 2, 4)
    abs_err = np.abs(residuals)
    plt.hist(abs_err, bins=30, color='green', edgecolor='black', alpha=0.6)
    plt.axvline(x=np.mean(abs_err), color='red', linestyle='--', label=f'MAE={np.mean(abs_err):.4f}')
    plt.title('Error Magnitude Distribution')
    plt.legend()

    plt.tight_layout()
    plt.savefig('testing_analysis_CNN_LSTM.png')

    # 4. 圖表 2: SOH 分析 (4 欄位)
    all_targets_soh = (all_targets / 2.0) * 100
    all_preds_soh = (all_preds / 2.0) * 100
    residuals_soh = all_preds_soh - all_targets_soh

    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.scatter(all_targets_soh, all_preds_soh, alpha=0.5, color='blue')
    plt.plot([min(all_targets_soh), max(all_targets_soh)], [min(all_targets_soh), max(all_targets_soh)], 'r--')
    plt.title('Actual vs Predicted SOH (%)')
    
    plt.subplot(2, 2, 2)
    plt.scatter(all_targets_soh, residuals_soh, alpha=0.5, color='purple')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title('Residual Plot (SOH %)')

    plt.subplot(2, 2, 3)
    mu_s, std_s = norm.fit(residuals_soh)
    plt.hist(residuals_soh, bins=30, density=True, color='skyblue', alpha=0.6)
    x_s = np.linspace(plt.xlim()[0], plt.xlim()[1], 100)
    plt.plot(x_s, norm.pdf(x_s, mu_s, std_s), 'r', label=f'mu={mu_s:.3f}')
    plt.title('SOH Error Dist.')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.hist(np.abs(residuals_soh), bins=30, color='green', alpha=0.6)
    plt.axvline(x=np.mean(np.abs(residuals_soh)), color='red', label=f'MAE={np.mean(np.abs(residuals_soh)):.3f}')
    plt.title('SOH Error Magnitude')
    plt.legend()

    plt.tight_layout()
    plt.savefig('soh_analysis_CNN_LSTM.png')
    
    print("\n測試分析完成！")
    print("容量分析圖表: testing_analysis_CNN_LSTM.png")
    print("SOH 分析圖表: soh_analysis_CNN_LSTM.png")

if __name__ == "__main__":
    main()

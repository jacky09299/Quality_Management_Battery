import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

# 從 train.py 匯入模型與數據集定義
from train import BatteryCNN, BatteryDataset

def load_norm_params():
    if not os.path.exists('norm_params.npy'):
        raise FileNotFoundError("找不到 norm_params.npy，請先執行 preprocess.py")
    params = np.load('norm_params.npy', allow_pickle=True).item()
    return params['mean'], params['std']

def predict_single_file(model, csv_path, device, mean, std, window_sec=1800, target_points=61):
    """
    輸入一個原始的 CSV 檔案路徑，直接輸出預測容量
    """
    df = pd.read_csv(csv_path)
    if df['Time'].max() < window_sec:
        return None, "數據時長不足 1800 秒"
    
    # 1. 插值處理
    new_time = np.arange(0, window_sec + 30, 30)
    features = ['Voltage_measured', 'Current_measured', 'Temperature_measured', 'Voltage_load']
    sample_data = []
    
    for feat in features:
        f_interp = interp1d(df['Time'], df[feat], kind='linear', 
                            bounds_error=False, fill_value=(df[feat].iloc[0], df[feat].iloc[-1]))
        resampled_feat = f_interp(new_time)
        sample_data.append(resampled_feat)
    
    # 2. 轉換為 Tensor 並標準化
    x = np.array(sample_data).reshape(1, 4, target_points) # (1, 4, 61)
    x_norm = (x - mean) / std
    x_tensor = torch.from_numpy(x_norm).float().to(device)
    
    # 3. 推論
    model.eval()
    with torch.no_grad():
        pred = model(x_tensor).cpu().item()
    
    return pred, None

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")

    # 1. 載入模型
    model = BatteryCNN().to(device)
    if not os.path.exists('best_model.pth'):
        print("錯誤：找不到 best_model.pth，請先執行 train.py 進行訓練。")
        return
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()

    # 2. 載入測試數據
    test_ds = BatteryDataset('X_test.npy', 'y_test.npy')
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    mean, std = load_norm_params()

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

    # 3. 計算指標
    mae = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    r2 = r2_score(all_targets, all_preds)
    mape = np.mean(np.abs((all_targets - all_preds) / all_targets)) * 100

    print("\n" + "="*30)
    print("      測試集評估結果")
    print("="*30)
    print(f"MAE  (平均絕對誤差): {mae:.4f} Ah")
    print(f"RMSE (均方根誤差)  : {rmse:.4f} Ah")
    print(f"R²   (決定係數)    : {r2:.4f}")
    print(f"MAPE (平均百分比誤差): {mape:.2f} %")
    print("="*30)

    # 4. 繪製進階圖表
    plt.figure(figsize=(15, 10))
    from scipy.stats import norm

    # 圖 1: 預測 vs 真實 (Regression Plot)
    plt.subplot(2, 2, 1)
    plt.scatter(all_targets, all_preds, alpha=0.5, color='blue')
    plt.plot([min(all_targets), max(all_targets)], [min(all_targets), max(all_targets)], 'r--')
    plt.title('Actual vs Predicted Capacity')
    plt.xlabel('Actual Capacity (Ah)')
    plt.ylabel('Predicted Capacity (Ah)')

    # 圖 2: 殘差圖 (Residual Plot)
    plt.subplot(2, 2, 2)
    residuals = all_preds - all_targets
    plt.scatter(all_targets, residuals, alpha=0.5, color='purple')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title('Residual Plot (Error vs Actual)')
    plt.xlabel('Actual Capacity (Ah)')
    plt.ylabel('Error (Ah)')

    # 圖 3: 絕對誤差與高斯分佈擬合 (Error Distribution)
    plt.subplot(2, 2, 3)
    mu, std_err = norm.fit(residuals)
    n, bins, patches = plt.hist(residuals, bins=30, density=True, color='skyblue', edgecolor='black', alpha=0.6, label='Error Dist.')
    xmin, xmax = plt.xlim()
    x_range = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x_range, mu, std_err)
    plt.plot(x_range, p, 'r', linewidth=2, label=f'Gaussian Fit\n$\mu={mu:.4f}, \sigma={std_err:.4f}$')
    plt.axvline(x=0, color='black', linestyle=':', alpha=0.5)
    plt.title('Error Distribution with Gaussian Fit')
    plt.xlabel('Absolute Error (Ah)')
    plt.ylabel('Probability Density')
    plt.legend()

    # 圖 4: 誤差絕對值直方圖 (Absolute Error Magnitude)
    plt.subplot(2, 2, 4)
    abs_errors = np.abs(residuals)
    mae_val = np.mean(abs_errors)
    plt.hist(abs_errors, bins=30, color='green', edgecolor='black', alpha=0.6)
    plt.axvline(x=mae_val, color='red', linestyle='--', linewidth=2, label=f'MAE = {mae_val:.4f}')
    plt.title('Distribution of Error Magnitudes')
    plt.xlabel('|Predicted - Actual| (Ah)')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    plt.savefig('testing_analysis.png')
    print("\n測試分析圖表已儲存為 'testing_analysis.png'")

    # 5. 繪製 SOH 圖表 (SOH = Capacity / 2 * 100)
    all_targets_soh = (all_targets / 2.0) * 100
    all_preds_soh = (all_preds / 2.0) * 100
    residuals_soh = all_preds_soh - all_targets_soh

    plt.figure(figsize=(15, 10))

    # 圖 1: 預測 vs 真實 (SOH)
    plt.subplot(2, 2, 1)
    plt.scatter(all_targets_soh, all_preds_soh, alpha=0.5, color='blue')
    plt.plot([min(all_targets_soh), max(all_targets_soh)], [min(all_targets_soh), max(all_targets_soh)], 'r--')
    plt.title('Actual vs Predicted SOH')
    plt.xlabel('Actual SOH')
    plt.ylabel('Predicted SOH')

    # 圖 2: 殘差圖 (SOH)
    plt.subplot(2, 2, 2)
    plt.scatter(all_targets_soh, residuals_soh, alpha=0.5, color='purple')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title('Residual Plot (SOH Error vs Actual)')
    plt.xlabel('Actual SOH')
    plt.ylabel('Error')

    # 圖 3: 誤差高斯擬合 (SOH)
    plt.subplot(2, 2, 3)
    mu_soh, std_soh = norm.fit(residuals_soh)
    plt.hist(residuals_soh, bins=30, density=True, color='skyblue', edgecolor='black', alpha=0.6, label='SOH Error Dist.')
    xmin_soh, xmax_soh = plt.xlim()
    x_range_soh = np.linspace(xmin_soh, xmax_soh, 100)
    p_soh = norm.pdf(x_range_soh, mu_soh, std_soh)
    plt.plot(x_range_soh, p_soh, 'r', linewidth=2, label=f'Gaussian Fit\n$\mu={mu_soh:.3f}, \sigma={std_soh:.3f}$')
    plt.axvline(x=0, color='black', linestyle=':', alpha=0.5)
    plt.title('SOH Error Distribution with Gaussian Fit')
    plt.xlabel('SOH Error')
    plt.ylabel('Probability Density')
    plt.legend()

    # 圖 4: SOH 誤差絕對值直方圖
    plt.subplot(2, 2, 4)
    abs_errors_soh = np.abs(residuals_soh)
    mae_soh = np.mean(abs_errors_soh)
    plt.hist(abs_errors_soh, bins=30, color='green', edgecolor='black', alpha=0.6)
    plt.axvline(x=mae_soh, color='red', linestyle='--', linewidth=2, label=f'MAE = {mae_soh:.3f}')
    plt.title('Distribution of SOH Error Magnitudes')
    plt.xlabel('|Predicted - Actual| SOH')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    plt.savefig('soh_analysis.png')
    print("SOH 分析圖表已儲存為 'soh_analysis.png'")

    # 6. 示範單一檔案預測 (選取一個測試集中的檔案作範例)
    print("\n[單一檔案預測示範]")
    # 這裡我們隨機抓取 metadata 中一個 discharge 的檔案
    metadata = pd.read_csv('metadata.csv')
    sample_row = metadata[metadata['type'] == 'discharge'].iloc[0]
    file_path = os.path.join('data', sample_row['filename'])
    
    pred_cap, err = predict_single_file(model, file_path, device, mean, std)
    if err:
        print(f"預測失敗: {err}")
    else:
        actual_cap = sample_row['Capacity']
        print(f"檔案: {sample_row['filename']}")
        print(f"預測容量: {pred_cap:.4f} Ah")
        print(f"實際容量: {actual_cap} Ah")
        print(f"絕對誤差: {abs(pred_cap - float(actual_cap)):.4f} Ah")

if __name__ == "__main__":
    main()

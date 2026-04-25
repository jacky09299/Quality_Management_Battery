import pandas as pd
import numpy as np
import os
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split

def preprocess_battery_data(metadata_path, data_dir, target_points=61, window_sec=1800):
    # (前面處理邏輯不變 ...)
    metadata = pd.read_csv(metadata_path)
    discharge_meta = metadata[metadata['type'] == 'discharge'].copy()
    
    # 處理 Capacity 欄位中的異常值 (如 '[]')，將其轉為 NaN 並剔除
    discharge_meta['Capacity'] = pd.to_numeric(discharge_meta['Capacity'], errors='coerce')
    discharge_meta = discharge_meta.dropna(subset=['Capacity'])
    
    # 額外篩選：剔除容量小於 1.2 Ah 或大於 2.0 Ah 的異常樣本
    discharge_meta = discharge_meta[(discharge_meta['Capacity'] >= 1.2) & (discharge_meta['Capacity'] <= 2.0)]
    
    processed_features = []
    capacities = []
    
    print(f"開始讀取並插值 {len(discharge_meta)} 個放電事件...")
    
    for idx, row in discharge_meta.iterrows():
        file_path = os.path.join(data_dir, row['filename'])
        if not os.path.exists(file_path):
            continue
        df = pd.read_csv(file_path)
        if df['Time'].max() < window_sec:
            continue
            
        new_time = np.arange(0, window_sec + 30, 30)
        features = ['Voltage_measured', 'Current_measured', 'Temperature_measured']
        sample_data = []
        
        for feat in features:
            f_interp = interp1d(df['Time'], df[feat], kind='linear', 
                                bounds_error=False, fill_value=(df[feat].iloc[0], df[feat].iloc[-1]))
            resampled_feat = f_interp(new_time)
            sample_data.append(resampled_feat)
            
        processed_features.append(sample_data)
        capacities.append(row['Capacity'])
        
    X = np.array(processed_features)
    y = np.array(capacities)
    
    # --- 新增：拆分與標準化 (7:1:2) ---
    print(f"原始樣本數: {len(X)}")
    
    # 1. 先拆出 20% 作為測試集
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    # 2. 再從剩下的 80% 中拆出 1/8 作為驗證集 (即總數的 10%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.125, random_state=42, shuffle=True
    )
    
    # 3. 標準化 (Z-score) - 僅使用訓練集的參數
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True)
    std[std == 0] = 1.0
    
    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std
    X_test_norm = (X_test - mean) / std
    
    return X_train_norm, X_val_norm, X_test_norm, y_train, y_val, y_test, mean, std

if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, mean, std = preprocess_battery_data('metadata.csv', 'data')
    
    print("\n--- 處理與拆分完成 (7:1:2) ---")
    print(f"訓練集形狀: {X_train.shape}")
    print(f"驗證集形狀: {X_val.shape}")
    print(f"測試集形狀: {X_test.shape}")
    
    # 儲存處理後的數據
    np.save('X_train.npy', X_train)
    np.save('X_val.npy', X_val)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_val.npy', y_val)
    np.save('y_test.npy', y_test)
    np.save('norm_params.npy', {'mean': mean, 'std': std})
    
    print("\n數據已儲存為 X_train.npy, X_val.npy, X_test.npy 等檔案")

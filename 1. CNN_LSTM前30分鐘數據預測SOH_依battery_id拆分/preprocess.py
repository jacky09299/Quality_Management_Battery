import pandas as pd
import numpy as np
import os
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split

def preprocess_battery_data(metadata_path, data_dir, delta_time=5, window_sec=1800):
    metadata = pd.read_csv(metadata_path)
    discharge_meta = metadata[metadata['type'] == 'discharge'].copy()
    
    # 1. 處理 Capacity 異常值
    discharge_meta['Capacity'] = pd.to_numeric(discharge_meta['Capacity'], errors='coerce')
    discharge_meta = discharge_meta.dropna(subset=['Capacity'])
    discharge_meta = discharge_meta[(discharge_meta['Capacity'] >= 1.2) & (discharge_meta['Capacity'] <= 2.0)]
    
    # 排除損壞的電池 ID
    excluded_batteries = ['B0049', 'B0050', 'B0051', 'B0052']
    discharge_meta = discharge_meta[~discharge_meta['battery_id'].isin(excluded_batteries)]
    
    processed_features = []
    final_meta_rows = []
    
    print(f"開始處理數據 (window_sec={window_sec})...")
    
    for idx, row in discharge_meta.iterrows():
        file_path = os.path.join(data_dir, row['filename'])
        if not os.path.exists(file_path):
            continue
            
        df = pd.read_csv(file_path)
        if df['Time'].max() < window_sec:
            continue
            
        # 插值
        new_time = np.arange(0, window_sec + delta_time, delta_time)
        features = ['Voltage_measured', 'Current_measured', 'Temperature_measured', 'Voltage_load']
        sample_data = []
        
        for feat in features:
            f_interp = interp1d(df['Time'], df[feat], kind='linear', 
                                bounds_error=False, fill_value=(df[feat].iloc[0], df[feat].iloc[-1]))
            resampled_feat = f_interp(new_time)
            sample_data.append(resampled_feat)
            
        processed_features.append(sample_data)
        final_meta_rows.append(row)
        
    # 重新構建同步的 meta 和 X, y
    discharge_meta = pd.DataFrame(final_meta_rows).reset_index(drop=True)
    X = np.array(processed_features, dtype=np.float32)
    y = discharge_meta['Capacity'].values.astype(np.float32)
    
    print(f"有效樣本數: {len(X)}")
    
    # --- 以 battery_id 為基準進行拆分 (7:1:2) ---
    unique_batteries = discharge_meta['battery_id'].unique()
    print(f"總電池數: {len(unique_batteries)}")
    
    batt_temp, batt_val = train_test_split(
        unique_batteries, test_size=0.2, random_state=929, shuffle=True
    )
    batt_train, batt_test = train_test_split(
        batt_temp, test_size=0.125, random_state=929, shuffle=True
    )

    print(f"訓練電池: {batt_train}, 驗證電池: {batt_val}, 測試電池: {batt_test}")
    
    print(f"訓練電池: {len(batt_train)}, 驗證電池: {len(batt_val)}, 測試電池: {len(batt_test)}")
    
    train_idx = discharge_meta[discharge_meta['battery_id'].isin(batt_train)].index
    val_idx = discharge_meta[discharge_meta['battery_id'].isin(batt_val)].index
    test_idx = discharge_meta[discharge_meta['battery_id'].isin(batt_test)].index
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    # 標準化
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
    
    np.save('X_train.npy', X_train)
    np.save('X_val.npy', X_val)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_val.npy', y_val)
    np.save('y_test.npy', y_test)
    np.save('norm_params.npy', {'mean': mean, 'std': std})
    
    print("\n數據已儲存為 X_train.npy, X_val.npy, X_test.npy 等檔案")

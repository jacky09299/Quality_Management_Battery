"""
preprocess.py - Data loading, feature extraction, dataset building, and saving to disk.
"""
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import norm
from sklearn.model_selection import train_test_split

SEED = 44
DATA_DIR = 'processed_data'
PLOTS_DIR = 'plots'
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def extract_features_full(df_raw, window_sec=600, delta_time=5):
    new_time = np.arange(0, window_sec + delta_time, delta_time)
    cols = ['Voltage_measured', 'Current_measured', 'Temperature_measured', 'Voltage_load']
    interped = {}
    for feat in cols:
        f = interp1d(df_raw['Time'], df_raw[feat], kind='linear', 
                    bounds_error=False, fill_value=(df_raw[feat].iloc[0], df_raw[feat].iloc[-1]))
        interped[feat] = f(new_time)
    v, i_raw, t, vl = [interped[c] for c in cols]
    ta = new_time
    mi = np.mean(np.abs(i_raw))
    if mi < 0.1: mi = 1.0
    mid = len(v)//2
    q4 = len(v)*3//4
    vn = (v[0]-v)/mi
    vs = np.polyfit(ta, vn, 1)[0]
    vse = np.polyfit(ta[:mid], vn[:mid], 1)[0]
    vsl = np.polyfit(ta[mid:], vn[mid:], 1)[0]
    vsq1 = np.polyfit(ta[:mid//2], vn[:mid//2], 1)[0]
    vsq2 = np.polyfit(ta[mid//2:mid], vn[mid//2:mid], 1)[0]
    vsq3 = np.polyfit(ta[mid:q4], vn[mid:q4], 1)[0]
    vsq4 = np.polyfit(ta[q4:], vn[q4:], 1)[0]
    va = np.trapz(vn, ta)
    vc = np.polyfit(ta, vn, 2)[0]
    dv = np.gradient(v, ta)
    tn = (t-t[0])/mi
    ts = np.polyfit(ta, tn, 1)[0]
    ta2 = np.trapz(tn, ta)
    tr = (t[-1]-t[0])/mi
    vld = (vl[0]-vl[-1])/mi
    q = np.cumsum(np.abs(i_raw))*5/3600
    
    feats_full = np.array([vs, vse, vsl, vsq1, vsq2, vsq3, vsq4, va, vc, np.std(v), 
                     np.mean(dv), dv[3], np.mean(dv[:mid]), np.mean(dv[mid:]),
                     ts, ta2, tr, vld, mi, q[-1]], dtype=np.float32)
                     
    v_drop_norm = vn[-1]
    v_drop_2min = vn[24]
    v_drop_5min = vn[60]
    v_drop_8min = vn[96]
    feats_short = np.array([v_drop_norm, v_drop_2min, v_drop_5min, v_drop_8min, va, 
                     tr, vld, mi], dtype=np.float32)
                     
    return feats_full, feats_short

def load_data():
    metadata = pd.read_csv('../metadata.csv')
    dm = metadata[metadata['type'] == 'discharge'].copy()
    dm['Capacity'] = pd.to_numeric(dm['Capacity'], errors='coerce')
    dm = dm.dropna(subset=['Capacity'])
    dm = dm[(dm['Capacity'] >= 1.2) & (dm['Capacity'] <= 2.0)]
    excl = ['B0049', 'B0050', 'B0051', 'B0052']
    dm = dm[~dm['battery_id'].isin(excl)].reset_index(drop=True)
    
    bat_feats_full, bat_feats_short, bat_sohs, bat_rows = {}, {}, {}, {}
    for idx, row in dm.iterrows():
        fp = os.path.join('..', 'data', row['filename'])
        if not os.path.exists(fp): continue
        df = pd.read_csv(fp)
        if df['Time'].max() < 600: continue
        bat = row['battery_id']
        if bat not in bat_feats_full: 
            bat_feats_full[bat], bat_feats_short[bat], bat_sohs[bat], bat_rows[bat] = [], [], [], []
        ff, fs = extract_features_full(df)
        bat_feats_full[bat].append(ff)
        bat_feats_short[bat].append(fs)
        bat_sohs[bat].append(row['Capacity']/2.0*100.0)
        bat_rows[bat].append(row)
    return bat_feats_full, bat_feats_short, bat_sohs, bat_rows

def build_datasets(bat_feats_full, bat_feats_short, bat_sohs, bat_rows):
    X_full, X_short, y_list, bat_list = [], [], [], []
    for bat in bat_feats_full:
        ff_list = bat_feats_full[bat]
        fs_list = bat_feats_short[bat]
        sohs_list = bat_sohs[bat]
        n = len(ff_list)
        ff_first = ff_list[0]
        fs_first = fs_list[0]
        amb = float(bat_rows[bat][0]['ambient_temperature'])
        for k in range(n):
            cur_f = ff_list[k]
            rel_f = (cur_f - ff_first) / (np.abs(ff_first) + 1e-6)
            rate_f = (cur_f - ff_list[k-1]) if k > 0 else np.zeros(20)
            rate_f_norm = rate_f / (np.abs(ff_first) + 1e-6)
            rate2_f = (cur_f - ff_list[k-2]) if k >= 2 else (cur_f - ff_first)
            rate2_f_norm = rate2_f / (np.abs(ff_first) + 1e-6)
            xf = np.concatenate([cur_f, ff_first, rel_f, rate_f_norm, rate2_f_norm, [k, k/max(n-1,1), amb]])
            X_full.append(xf.astype(np.float32))
            
            cur_s = fs_list[k]
            rel_s = (cur_s - fs_first) / (np.abs(fs_first) + 1e-6)
            rate_s = (cur_s - fs_list[k-1]) if k > 0 else np.zeros(8)
            xs = np.concatenate([cur_s, fs_first, rel_s, rate_s, [k, k/max(n-1,1), amb]])
            X_short.append(xs.astype(np.float32))
            
            y_list.append(sohs_list[k])
            bat_list.append(bat)
    return np.array(X_full), np.array(X_short), np.array(y_list, dtype=np.float32), np.array(bat_list)

def generate_analysis_plots(all_targets_soh, all_preds_soh, prefix, title_prefix, plots_dir='plots'):
    all_targets_cap = all_targets_soh * 2.0 / 100.0
    all_preds_cap = all_preds_soh * 2.0 / 100.0
    residuals_cap = all_preds_cap - all_targets_cap
    residuals_soh = all_preds_soh - all_targets_soh
    
    # 1. Capacity Plot
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.scatter(all_targets_cap, all_preds_cap, alpha=0.5, color='blue')
    plt.plot([min(all_targets_cap), max(all_targets_cap)], [min(all_targets_cap), max(all_targets_cap)], 'r--')
    plt.title(f'Actual vs Predicted Capacity ({title_prefix})')
    plt.xlabel('Actual (Ah)')
    plt.ylabel('Predicted (Ah)')

    plt.subplot(2, 2, 2)
    plt.scatter(all_targets_cap, residuals_cap, alpha=0.5, color='purple')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title(f'Residual Plot ({title_prefix})')
    plt.xlabel('Actual Capacity (Ah)')
    plt.ylabel('Error (Ah)')

    plt.subplot(2, 2, 3)
    mu, std_err = norm.fit(residuals_cap)
    plt.hist(residuals_cap, bins=30, density=True, color='skyblue', edgecolor='black', alpha=0.6)
    xmin, xmax = plt.xlim()
    x_range = np.linspace(xmin, xmax, 100)
    plt.plot(x_range, norm.pdf(x_range, mu, std_err), 'r', linewidth=2, label=f'mu={mu:.4f}\\nsigma={std_err:.4f}')
    plt.title('Error Distribution (Gaussian Fit)')
    plt.legend()

    plt.subplot(2, 2, 4)
    abs_err = np.abs(residuals_cap)
    plt.hist(abs_err, bins=30, color='green', edgecolor='black', alpha=0.6)
    plt.axvline(x=np.mean(abs_err), color='red', linestyle='--', label=f'MAE={np.mean(abs_err):.4f}')
    plt.title('Error Magnitude Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{prefix}_capacity_analysis.png'))
    plt.close()

    # 2. SOH Plot
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.scatter(all_targets_soh, all_preds_soh, alpha=0.5, color='blue')
    plt.plot([min(all_targets_soh), max(all_targets_soh)], [min(all_targets_soh), max(all_targets_soh)], 'r--')
    plt.title(f'Actual vs Predicted SOH (%) ({title_prefix})')
    
    plt.subplot(2, 2, 2)
    plt.scatter(all_targets_soh, residuals_soh, alpha=0.5, color='purple')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title(f'Residual Plot (SOH %) ({title_prefix})')

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
    plt.savefig(os.path.join(plots_dir, f'{prefix}_soh_analysis.png'))
    plt.close()

def main():
    print("Loading data and extracting features...")
    bat_feats_full, bat_feats_short, bat_sohs, bat_rows = load_data()
    
    print("Building datasets...")
    Xf, Xs, y, bat_arr = build_datasets(bat_feats_full, bat_feats_short, bat_sohs, bat_rows)
    
    print("Determining splits...")
    unique_batteries = np.array(list(bat_feats_full.keys()))
    batt_temp, batt_test = train_test_split(unique_batteries, test_size=0.2, random_state=SEED, shuffle=True)
    batt_train, batt_val = train_test_split(batt_temp, test_size=0.125, random_state=SEED, shuffle=True)
    
    # Save the splits to dict
    splits = {
        'train': batt_train,
        'val': batt_val,
        'test': batt_test
    }
    
    print("Saving processed data to disk...")
    np.save(os.path.join(DATA_DIR, 'Xf.npy'), Xf)
    np.save(os.path.join(DATA_DIR, 'Xs.npy'), Xs)
    np.save(os.path.join(DATA_DIR, 'y.npy'), y)
    np.save(os.path.join(DATA_DIR, 'bat_arr.npy'), bat_arr)
    with open(os.path.join(DATA_DIR, 'splits.pkl'), 'wb') as f:
        pickle.dump(splits, f)
        
    print(f"Preprocessing complete. Data saved to {DATA_DIR}/")

if __name__ == "__main__":
    main()

"""
test.py - Loads processed data and models, predicts on test set, and generates testing plots.
"""
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

DATA_DIR = 'processed_data'
MODELS_DIR = 'models'
PLOTS_DIR = 'plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

# Import plotting function from preprocess
from preprocess import generate_analysis_plots

def main():
    print("Loading test configuration and models...")
    with open(os.path.join(DATA_DIR, 'splits.pkl'), 'rb') as f:
        splits = pickle.load(f)
    batt_test = splits['test']
    
    print("Loading processed data...")
    Xf = np.load(os.path.join(DATA_DIR, 'Xf.npy'))
    Xs = np.load(os.path.join(DATA_DIR, 'Xs.npy'))
    y = np.load(os.path.join(DATA_DIR, 'y.npy'))
    bat_arr = np.load(os.path.join(DATA_DIR, 'bat_arr.npy'))
    
    test_mask = np.array([b in set(batt_test) for b in bat_arr])
    Xf_test = Xf[test_mask]
    Xs_test = Xs[test_mask]
    y_test = y[test_mask]
    
    with open(os.path.join(MODELS_DIR, 'scaler_full.pkl'), 'rb') as f: sc_f = pickle.load(f)
    with open(os.path.join(MODELS_DIR, 'scaler_short.pkl'), 'rb') as f: sc_s = pickle.load(f)
    with open(os.path.join(MODELS_DIR, 'svr_model.pkl'), 'rb') as f: svr = pickle.load(f)
    with open(os.path.join(MODELS_DIR, 'lgbm_model.pkl'), 'rb') as f: lgbm = pickle.load(f)
    with open(os.path.join(MODELS_DIR, 'xgb_model.pkl'), 'rb') as f: xgbr = pickle.load(f)
    with open(os.path.join(MODELS_DIR, 'weights.pkl'), 'rb') as f: best_w = pickle.load(f)
    
    Xf_test_s = sc_f.transform(Xf_test)
    Xs_test_s = sc_s.transform(Xs_test)
    
    print("Predicting on test set...")
    pred_svr_test = svr.predict(Xs_test_s)
    pred_lgbm_test = lgbm.predict(Xf_test_s)
    pred_xgb_test = xgbr.predict(Xf_test_s)
    
    ens_test = best_w[0] * pred_svr_test + best_w[1] * pred_lgbm_test + best_w[2] * pred_xgb_test
    
    t_mae = mean_absolute_error(y_test, ens_test)
    print(f"Test Set MAE: {t_mae:.4f}%")
    
    # Generate the standard analysis plots for testing
    print("Generating testing plots...")
    generate_analysis_plots(y_test, ens_test, "testing", "Test Set", plots_dir=PLOTS_DIR)
    
    # Also generate the time-series degradation plot
    print("Generating Degradation Curves Plot...")
    plt.figure(figsize=(15, 10))
    for i, bat in enumerate(sorted(batt_test)):
        plt.subplot(2, 3, i+1)
        bl = bat_arr[test_mask] == bat
        if bl.sum() > 0:
            cycles = np.arange(bl.sum())
            plt.plot(cycles, y_test[bl], 'b-', label='Actual SOH', linewidth=2)
            plt.plot(cycles, ens_test[bl], 'r--', label='Predicted SOH', linewidth=2)
            plt.title(f"{bat} SOH Degradation")
            plt.xlabel("Cycle")
            plt.ylabel("SOH (%)")
            plt.legend()
            plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'test_degradation_curves.png'), dpi=300)
    plt.close()
    
    print(f"Testing Complete. Generated Test plots in {PLOTS_DIR}/")

if __name__ == "__main__":
    main()

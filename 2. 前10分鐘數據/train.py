"""
train.py - Loads processed data, trains the Ensemble model, finds best weights, saves models, and generates train plots.
"""
import numpy as np
import os
import pickle
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import lightgbm as lgb
import xgboost as xgb

# Import plotting function from preprocess
from preprocess import generate_analysis_plots

DATA_DIR = 'processed_data'
MODELS_DIR = 'models'
PLOTS_DIR = 'plots'
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def main():
    print("Loading processed data...")
    Xf = np.load(os.path.join(DATA_DIR, 'Xf.npy'))
    Xs = np.load(os.path.join(DATA_DIR, 'Xs.npy'))
    y = np.load(os.path.join(DATA_DIR, 'y.npy'))
    bat_arr = np.load(os.path.join(DATA_DIR, 'bat_arr.npy'))
    
    with open(os.path.join(DATA_DIR, 'splits.pkl'), 'rb') as f:
        splits = pickle.load(f)
    
    batt_train = splits['train']
    batt_val = splits['val']
    
    train_mask = np.array([b in set(batt_train) for b in bat_arr])
    val_mask   = np.array([b in set(batt_val)   for b in bat_arr])
    
    Xf_train, Xf_val = Xf[train_mask], Xf[val_mask]
    Xs_train, Xs_val = Xs[train_mask], Xs[val_mask]
    y_train, y_val = y[train_mask], y[val_mask]
    
    print("Scaling features...")
    sc_f = StandardScaler()
    Xf_train_s = sc_f.fit_transform(Xf_train)
    Xf_val_s   = sc_f.transform(Xf_val)
    
    sc_s = StandardScaler()
    Xs_train_s = sc_s.fit_transform(Xs_train)
    Xs_val_s   = sc_s.transform(Xs_val)
    
    print("Training SVR...")
    svr = SVR(kernel='rbf', C=10, gamma='scale', epsilon=0.1)
    svr.fit(Xs_train_s, y_train)
    
    print("Training LGBM...")
    lgbm = lgb.LGBMRegressor(n_estimators=2000, num_leaves=63, learning_rate=0.02,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=5, random_state=42, n_jobs=-1, verbose=-1)
    lgbm.fit(Xf_train_s, y_train)
    
    print("Training XGB...")
    xgbr = xgb.XGBRegressor(n_estimators=1000, max_depth=5, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3, random_state=42, n_jobs=-1, verbosity=0)
    xgbr.fit(Xf_train_s, y_train)
    
    print("Finding optimal ensemble weights...")
    pred_svr_val = svr.predict(Xs_val_s)
    pred_lgbm_val = lgbm.predict(Xf_val_s)
    pred_xgb_val = xgbr.predict(Xf_val_s)
    
    best_ens_val = float('inf')
    best_w = None
    
    for w_svr in np.linspace(0, 1, 11):
        for w_lgbm in np.linspace(0, 1-w_svr, 11):
            w_xgb = 1.0 - w_svr - w_lgbm
            if w_xgb < -1e-6: continue
            ens_val = w_svr * pred_svr_val + w_lgbm * pred_lgbm_val + w_xgb * pred_xgb_val
            v_mae = mean_absolute_error(y_val, ens_val)
            if v_mae < best_ens_val:
                best_ens_val = v_mae
                best_w = (w_svr, w_lgbm, w_xgb)
                
    print(f"Best validation weights: SVR={best_w[0]:.2f}, LGBM={best_w[1]:.2f}, XGB={best_w[2]:.2f}")
    
    print("Saving models and weights...")
    with open(os.path.join(MODELS_DIR, 'svr_model.pkl'), 'wb') as f: pickle.dump(svr, f)
    with open(os.path.join(MODELS_DIR, 'lgbm_model.pkl'), 'wb') as f: pickle.dump(lgbm, f)
    with open(os.path.join(MODELS_DIR, 'xgb_model.pkl'), 'wb') as f: pickle.dump(xgbr, f)
    with open(os.path.join(MODELS_DIR, 'scaler_full.pkl'), 'wb') as f: pickle.dump(sc_f, f)
    with open(os.path.join(MODELS_DIR, 'scaler_short.pkl'), 'wb') as f: pickle.dump(sc_s, f)
    with open(os.path.join(MODELS_DIR, 'weights.pkl'), 'wb') as f: pickle.dump(best_w, f)
    
    # Generate Training Plot (combining Train and Val)
    pred_svr_train = svr.predict(Xs_train_s)
    pred_lgbm_train = lgbm.predict(Xf_train_s)
    pred_xgb_train = xgbr.predict(Xf_train_s)
    ens_train = best_w[0] * pred_svr_train + best_w[1] * pred_lgbm_train + best_w[2] * pred_xgb_train
    
    print("Generating train plots...")
    generate_analysis_plots(y_train, ens_train, "train", "Train Set", plots_dir=PLOTS_DIR)
    
    print(f"Training Complete. Generated Train plots in {PLOTS_DIR}/")

if __name__ == "__main__":
    main()

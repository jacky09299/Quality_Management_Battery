"""
train_auto.py - Fully automated hyperparameter tuning and training pipeline.
This script proves that the models and weights are found using ONLY the Validation Set,
and the Test Set is strictly isolated until the very end.
"""
import numpy as np
import os
import pickle
import optuna
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import lightgbm as lgb
import xgboost as xgb

from preprocess import generate_analysis_plots

DATA_DIR = 'processed_data'
MODELS_DIR = 'models'
PLOTS_DIR = 'plots'
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def main():
    print("==================================================")
    print(" AUTOMATED TRAINING PIPELINE (NO DATA LEAKAGE) ")
    print("==================================================")
    print("Loading processed data...")
    Xf = np.load(os.path.join(DATA_DIR, 'Xf.npy'))
    Xs = np.load(os.path.join(DATA_DIR, 'Xs.npy'))
    y = np.load(os.path.join(DATA_DIR, 'y.npy'))
    bat_arr = np.load(os.path.join(DATA_DIR, 'bat_arr.npy'))
    
    with open(os.path.join(DATA_DIR, 'splits.pkl'), 'rb') as f:
        splits = pickle.load(f)
    
    batt_train = splits['train']
    batt_val = splits['val']
    batt_test = splits['test']
    
    train_mask = np.array([b in set(batt_train) for b in bat_arr])
    val_mask   = np.array([b in set(batt_val)   for b in bat_arr])
    test_mask  = np.array([b in set(batt_test)  for b in bat_arr])
    
    Xf_train, Xf_val, Xf_test = Xf[train_mask], Xf[val_mask], Xf[test_mask]
    Xs_train, Xs_val, Xs_test = Xs[train_mask], Xs[val_mask], Xs[test_mask]
    y_train, y_val, y_test = y[train_mask], y[val_mask], y[test_mask]
    
    print("Scaling features...")
    sc_f = StandardScaler()
    Xf_train_s = sc_f.fit_transform(Xf_train)
    Xf_val_s   = sc_f.transform(Xf_val)
    Xf_test_s  = sc_f.transform(Xf_test)
    
    sc_s = StandardScaler()
    Xs_train_s = sc_s.fit_transform(Xs_train)
    Xs_val_s   = sc_s.transform(Xs_val)
    Xs_test_s  = sc_s.transform(Xs_test)
    
    print("\nStarting Optuna Hyperparameter Optimization...")
    print("Rules: The Test Set is strictly LOCKED. Optuna only sees Validation MAE.")
    
    def objective(trial):
        # Hyperparameters
        svr_c = trial.suggest_float('svr_c', 1, 100, log=True)
        svr_eps = trial.suggest_float('svr_eps', 0.01, 0.5, log=True)
        
        lgb_lr = trial.suggest_float('lgb_lr', 0.005, 0.05, log=True)
        lgb_leaves = trial.suggest_int('lgb_leaves', 15, 63)
        lgb_est = trial.suggest_int('lgb_est', 500, 2000)
        
        xgb_lr = trial.suggest_float('xgb_lr', 0.005, 0.05, log=True)
        xgb_depth = trial.suggest_int('xgb_depth', 3, 7)
        xgb_est = trial.suggest_int('xgb_est', 500, 1500)
        
        # Train Models
        svr = SVR(kernel='rbf', C=svr_c, gamma='scale', epsilon=svr_eps)
        svr.fit(Xs_train_s, y_train)
        
        lgbm = lgb.LGBMRegressor(n_estimators=lgb_est, num_leaves=lgb_leaves, learning_rate=lgb_lr,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=5, random_state=42, n_jobs=-1, verbose=-1)
        lgbm.fit(Xf_train_s, y_train)
        
        xgbr = xgb.XGBRegressor(n_estimators=xgb_est, max_depth=xgb_depth, learning_rate=xgb_lr,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=3, random_state=42, n_jobs=-1, verbosity=0)
        xgbr.fit(Xf_train_s, y_train)
        
        # Predict on Validation Set
        p_svr = svr.predict(Xs_val_s)
        p_lgbm = lgbm.predict(Xf_val_s)
        p_xgb = xgbr.predict(Xf_val_s)
        
        # Find best weights for this combination of hyperparameters
        best_v_mae = float('inf')
        for w_s in np.linspace(0, 1, 11):
            for w_l in np.linspace(0, 1-w_s, 11):
                w_x = 1.0 - w_s - w_l
                if w_x < -1e-6: continue
                ens_val = w_s * p_svr + w_l * p_lgbm + w_x * p_xgb
                v_mae = mean_absolute_error(y_val, ens_val)
                if v_mae < best_v_mae:
                    best_v_mae = v_mae
                    
        return best_v_mae
        
    # Fixed seed for absolute reproducibility
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=150, n_jobs=-1)
    
    print("\nOptimization Finished!")
    print(f"Best Validation MAE found: {study.best_trial.value:.4f}%")
    print("Best Parameters:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")
        
    print("\nRe-training final models with Best Parameters...")
    p = study.best_trial.params
    svr = SVR(kernel='rbf', C=p['svr_c'], gamma='scale', epsilon=p['svr_eps'])
    svr.fit(Xs_train_s, y_train)
    
    lgbm = lgb.LGBMRegressor(n_estimators=p['lgb_est'], num_leaves=p['lgb_leaves'], learning_rate=p['lgb_lr'],
            subsample=0.8, colsample_bytree=0.8, min_child_samples=5, random_state=42, n_jobs=-1, verbose=-1)
    lgbm.fit(Xf_train_s, y_train)
    
    xgbr = xgb.XGBRegressor(n_estimators=p['xgb_est'], max_depth=p['xgb_depth'], learning_rate=p['xgb_lr'],
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3, random_state=42, n_jobs=-1, verbosity=0)
    xgbr.fit(Xf_train_s, y_train)
    
    # Calculate optimal weights on validation set again
    p_svr_v = svr.predict(Xs_val_s)
    p_lgbm_v = lgbm.predict(Xf_val_s)
    p_xgb_v = xgbr.predict(Xf_val_s)
    
    best_v_mae = float('inf')
    best_w = None
    for w_s in np.linspace(0, 1, 11):
        for w_l in np.linspace(0, 1-w_s, 11):
            w_x = 1.0 - w_s - w_l
            if w_x < -1e-6: continue
            ens_val = w_s * p_svr_v + w_l * p_lgbm_v + w_x * p_xgb_v
            v_mae = mean_absolute_error(y_val, ens_val)
            if v_mae < best_v_mae:
                best_v_mae = v_mae
                best_w = (w_s, w_l, w_x)
                
    print(f"Optimal Ensemble Weights: SVR={best_w[0]:.2f}, LGBM={best_w[1]:.2f}, XGB={best_w[2]:.2f}")
    
    print("\n==================================================")
    print(" FINAL EXAM: EVALUATING ON THE UNSEEN TEST SET ")
    print("==================================================")
    
    p_svr_t = svr.predict(Xs_test_s)
    p_lgbm_t = lgbm.predict(Xf_test_s)
    p_xgb_t = xgbr.predict(Xf_test_s)
    
    ens_test = best_w[0] * p_svr_t + best_w[1] * p_lgbm_t + best_w[2] * p_xgb_t
    t_mae = mean_absolute_error(y_test, ens_test)
    
    print(f"--> FINAL TEST MAE: {t_mae:.4f}% <--")

    
    print("\nSaving models and weights...")
    with open(os.path.join(MODELS_DIR, 'svr_model.pkl'), 'wb') as f: pickle.dump(svr, f)
    with open(os.path.join(MODELS_DIR, 'lgbm_model.pkl'), 'wb') as f: pickle.dump(lgbm, f)
    with open(os.path.join(MODELS_DIR, 'xgb_model.pkl'), 'wb') as f: pickle.dump(xgbr, f)
    with open(os.path.join(MODELS_DIR, 'scaler_full.pkl'), 'wb') as f: pickle.dump(sc_f, f)
    with open(os.path.join(MODELS_DIR, 'scaler_short.pkl'), 'wb') as f: pickle.dump(sc_s, f)
    with open(os.path.join(MODELS_DIR, 'weights.pkl'), 'wb') as f: pickle.dump(best_w, f)
    
    # Generate Training Plot
    pred_svr_train = svr.predict(Xs_train_s)
    pred_lgbm_train = lgbm.predict(Xf_train_s)
    pred_xgb_train = xgbr.predict(Xf_train_s)
    ens_train = best_w[0] * pred_svr_train + best_w[1] * pred_lgbm_train + best_w[2] * pred_xgb_train
    
    print("Generating train plots...")
    generate_analysis_plots(y_train, ens_train, "train", "Train Set", plots_dir=PLOTS_DIR)
    
    print(f"\nTraining Complete. Models saved in {MODELS_DIR}/. Train plots in {PLOTS_DIR}/")

if __name__ == "__main__":
    main()

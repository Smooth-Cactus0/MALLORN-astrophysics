"""
MALLORN v20c: Optuna Hyperparameter Tuning

Run Optuna to find optimal hyperparameters for the v20b feature set.
This should squeeze out additional performance.
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import optuna
import warnings

optuna.logging.set_verbosity(optuna.logging.WARNING)
sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
base_path = Path(__file__).parent.parent

print("=" * 60, flush=True)
print("MALLORN v20c: Optuna Hyperparameter Tuning", flush=True)
print("=" * 60, flush=True)

# ====================
# 1. LOAD DATA & FEATURES
# ====================
print("\n1. Loading data and features...", flush=True)

from utils.data_loader import load_all_data
data = load_all_data()

train_meta = data['train_meta']
train_lc = data['train_lc']
test_lc = data['test_lc']
test_meta = data['test_meta']

train_ids = train_meta['object_id'].tolist()
test_ids = test_meta['object_id'].tolist()

y = train_meta['target'].values

# Load base features
cached = pd.read_pickle(base_path / 'data/processed/features_v4_cache.pkl')
train_base = cached['train_features']
test_base = cached['test_features']

selection = pd.read_pickle(base_path / 'data/processed/selected_features.pkl')
importance_df = selection['importance_df']
high_corr_df = selection['high_corr_df']

corr_to_drop = set()
for _, row in high_corr_df.iterrows():
    if row['feature_1'] not in corr_to_drop:
        corr_to_drop.add(row['feature_2'])
clean_features = importance_df[~importance_df['feature'].isin(corr_to_drop)]
selected_120 = clean_features.head(120)['feature'].tolist()

# TDE physics
tde_cached = pd.read_pickle(base_path / 'data/processed/tde_physics_cache.pkl')
train_tde = tde_cached['train']
test_tde = tde_cached['test']
tde_cols = [c for c in train_tde.columns if c != 'object_id']

train_base = train_base.merge(train_tde, on='object_id', how='left')
test_base = test_base.merge(test_tde, on='object_id', how='left')

# GP2D
with open(base_path / 'data/processed/multiband_gp_cache.pkl', 'rb') as f:
    gp2d_data = pickle.load(f)
train_gp2d = gp2d_data['train']
test_gp2d = gp2d_data['test']
gp2d_cols = [c for c in train_gp2d.columns if c != 'object_id']

# Advanced
with open(base_path / 'data/processed/advanced_features_cache.pkl', 'rb') as f:
    adv_data = pickle.load(f)
train_adv = adv_data['train']
test_adv = adv_data['test']
adv_cols = [c for c in train_adv.columns if c != 'object_id']

# Selected ADV
top_adv_features = [
    'g_mhps_100', 'pre_peak_r_i_slope', 'r_mhps_30', 'r_mhps_ratio_30_365',
    'r_abs_mag_peak', 'g_abs_mag_peak', 'r_fleet_width', 'r_fleet_asymmetry',
    'peak_lag_g_r', 'flux_skewness', 'r_acf_10d'
]
available_adv = [c for c in top_adv_features if c in adv_cols]

# Combine
train_combined = train_base.copy()
test_combined = test_base.copy()

train_combined = train_combined.merge(train_gp2d, on='object_id', how='left')
test_combined = test_combined.merge(test_gp2d, on='object_id', how='left')

train_adv_selected = train_adv[['object_id'] + available_adv]
test_adv_selected = test_adv[['object_id'] + available_adv]

train_combined = train_combined.merge(train_adv_selected, on='object_id', how='left')
test_combined = test_combined.merge(test_adv_selected, on='object_id', how='left')

base_cols = [c for c in selected_120 if c in train_combined.columns]
all_feature_cols = base_cols + tde_cols + gp2d_cols + available_adv
all_feature_cols = list(dict.fromkeys(all_feature_cols))
all_feature_cols = [c for c in all_feature_cols if c in train_combined.columns]

train_combined = train_combined.set_index('object_id').loc[train_ids].reset_index()

X = train_combined[all_feature_cols].values.astype(np.float32)
X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
X_test = test_combined[all_feature_cols].values.astype(np.float32)
X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

n_neg, n_pos = np.sum(y == 0), np.sum(y == 1)
scale_pos_weight = n_neg / n_pos

print(f"   Features: {len(all_feature_cols)}", flush=True)
print(f"   Samples: {len(y)} ({n_pos} TDE, {n_neg} non-TDE)", flush=True)

# ====================
# 2. OPTUNA TUNING
# ====================
print("\n2. Running Optuna tuning (50 trials each)...", flush=True)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def find_optimal_threshold(y_true, y_prob):
    best_f1, best_thresh = 0, 0.5
    for t in np.arange(0.05, 0.95, 0.01):
        f1 = f1_score(y_true, (y_prob >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    return best_thresh, best_f1

# XGBoost tuning
print("\n   Tuning XGBoost...", flush=True)

def objective_xgb(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 200, 800),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 3.0),
        'gamma': trial.suggest_float('gamma', 0.01, 1.0, log=True),
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'verbosity': 0
    }

    oof = np.zeros(len(y))
    for train_idx, val_idx in cv.split(X, y):
        model = xgb.XGBClassifier(**params)
        model.fit(X[train_idx], y[train_idx], eval_set=[(X[val_idx], y[val_idx])], verbose=False)
        oof[val_idx] = model.predict_proba(X[val_idx])[:, 1]

    _, f1 = find_optimal_threshold(y, oof)
    return f1

study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=50, show_progress_bar=False)
xgb_best_params = study_xgb.best_params
print(f"      Best XGB F1: {study_xgb.best_value:.4f}", flush=True)

# LightGBM tuning
print("\n   Tuning LightGBM...", flush=True)

def objective_lgb(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 200, 800),
        'num_leaves': trial.suggest_int('num_leaves', 15, 63),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'subsample': trial.suggest_float('subsample', 0.6, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 3.0),
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'verbose': -1
    }

    oof = np.zeros(len(y))
    for train_idx, val_idx in cv.split(X, y):
        model = lgb.LGBMClassifier(**params)
        model.fit(X[train_idx], y[train_idx], eval_set=[(X[val_idx], y[val_idx])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[val_idx] = model.predict_proba(X[val_idx])[:, 1]

    _, f1 = find_optimal_threshold(y, oof)
    return f1

study_lgb = optuna.create_study(direction='maximize')
study_lgb.optimize(objective_lgb, n_trials=50, show_progress_bar=False)
lgb_best_params = study_lgb.best_params
print(f"      Best LGB F1: {study_lgb.best_value:.4f}", flush=True)

# CatBoost tuning
print("\n   Tuning CatBoost...", flush=True)

def objective_cat(trial):
    params = {
        'depth': trial.suggest_int('depth', 4, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'iterations': trial.suggest_int('iterations', 200, 800),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.5, 5.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'scale_pos_weight': scale_pos_weight,
        'random_seed': 42,
        'verbose': False,
        'allow_writing_files': False
    }

    oof = np.zeros(len(y))
    for train_idx, val_idx in cv.split(X, y):
        model = CatBoostClassifier(**params)
        model.fit(X[train_idx], y[train_idx], eval_set=(X[val_idx], y[val_idx]),
                  early_stopping_rounds=100, verbose=False)
        oof[val_idx] = model.predict_proba(X[val_idx])[:, 1]

    _, f1 = find_optimal_threshold(y, oof)
    return f1

study_cat = optuna.create_study(direction='maximize')
study_cat.optimize(objective_cat, n_trials=50, show_progress_bar=False)
cat_best_params = study_cat.best_params
print(f"      Best CAT F1: {study_cat.best_value:.4f}", flush=True)

# ====================
# 3. TRAIN FINAL ENSEMBLE
# ====================
print("\n3. Training final ensemble with tuned params...", flush=True)

oof_xgb = np.zeros(len(y))
oof_lgb = np.zeros(len(y))
oof_cat = np.zeros(len(y))

test_preds_xgb = np.zeros((len(test_ids), 5))
test_preds_lgb = np.zeros((len(test_ids), 5))
test_preds_cat = np.zeros((len(test_ids), 5))

models = {'xgb': [], 'lgb': [], 'cat': []}

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    # XGBoost
    xgb_model = xgb.XGBClassifier(**xgb_best_params, scale_pos_weight=scale_pos_weight, random_state=42, verbosity=0)
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_idx] = xgb_model.predict_proba(X_val)[:, 1]
    test_preds_xgb[:, fold] = xgb_model.predict_proba(X_test)[:, 1]
    models['xgb'].append(xgb_model)

    # LightGBM
    lgb_model = lgb.LGBMClassifier(**lgb_best_params, scale_pos_weight=scale_pos_weight, random_state=42, verbose=-1)
    lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    oof_lgb[val_idx] = lgb_model.predict_proba(X_val)[:, 1]
    test_preds_lgb[:, fold] = lgb_model.predict_proba(X_test)[:, 1]
    models['lgb'].append(lgb_model)

    # CatBoost
    cat_model = CatBoostClassifier(**cat_best_params, scale_pos_weight=scale_pos_weight, random_seed=42, verbose=False, allow_writing_files=False)
    cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=100, verbose=False)
    oof_cat[val_idx] = cat_model.predict_proba(X_val)[:, 1]
    test_preds_cat[:, fold] = cat_model.predict_proba(X_test)[:, 1]
    models['cat'].append(cat_model)

    fold_ens = (oof_xgb[val_idx] + oof_lgb[val_idx] + oof_cat[val_idx]) / 3
    _, fold_f1 = find_optimal_threshold(y_val, fold_ens)
    print(f"   Fold {fold+1}: F1={fold_f1:.4f}", flush=True)

# ====================
# 4. OPTIMIZE WEIGHTS
# ====================
print("\n4. Optimizing ensemble weights...", flush=True)

best_f1, best_weights, best_thresh = 0, (1/3, 1/3, 1/3), 0.3

for w1 in np.arange(0.15, 0.55, 0.05):
    for w2 in np.arange(0.15, 0.55, 0.05):
        w3 = 1 - w1 - w2
        if w3 < 0.1 or w3 > 0.55:
            continue
        weighted = w1 * oof_xgb + w2 * oof_lgb + w3 * oof_cat
        thresh, f1 = find_optimal_threshold(y, weighted)
        if f1 > best_f1:
            best_f1, best_weights, best_thresh = f1, (w1, w2, w3), thresh

print(f"   Best weights: XGB={best_weights[0]:.2f}, LGB={best_weights[1]:.2f}, CAT={best_weights[2]:.2f}", flush=True)
print(f"   OOF F1: {best_f1:.4f} @ threshold={best_thresh:.2f}", flush=True)

# Individual scores
xgb_best_score = max([f1_score(y, (oof_xgb > t).astype(int)) for t in np.arange(0.05, 0.6, 0.01)])
lgb_best_score = max([f1_score(y, (oof_lgb > t).astype(int)) for t in np.arange(0.05, 0.6, 0.01)])
cat_best_score = max([f1_score(y, (oof_cat > t).astype(int)) for t in np.arange(0.05, 0.6, 0.01)])
print(f"\n   Individual OOF F1: XGB={xgb_best_score:.4f}, LGB={lgb_best_score:.4f}, CAT={cat_best_score:.4f}", flush=True)

# ====================
# 5. CREATE SUBMISSION
# ====================
print("\n5. Creating submission...", flush=True)

test_blend = (
    best_weights[0] * test_preds_xgb.mean(axis=1) +
    best_weights[1] * test_preds_lgb.mean(axis=1) +
    best_weights[2] * test_preds_cat.mean(axis=1)
)
test_preds_final = (test_blend > best_thresh).astype(int)

submission = pd.DataFrame({'object_id': test_ids, 'target': test_preds_final})
submission_path = base_path / 'submissions/submission_v20c_optuna.csv'
submission.to_csv(submission_path, index=False)

print(f"   Saved to {submission_path}", flush=True)
print(f"   Predictions: {test_preds_final.sum()} TDEs / {len(test_preds_final)} total", flush=True)

# Save results
with open(base_path / 'data/processed/optuna_v20c_results.pkl', 'wb') as f:
    pickle.dump({
        'xgb_best_params': xgb_best_params,
        'lgb_best_params': lgb_best_params,
        'cat_best_params': cat_best_params,
        'best_weights': best_weights,
        'best_thresh': best_thresh,
        'models': models,
        'feature_cols': all_feature_cols
    }, f)

# ====================
# SUMMARY
# ====================
print("\n" + "=" * 60, flush=True)
print("OPTUNA TUNING COMPLETE!", flush=True)
print("=" * 60, flush=True)

print(f"\nVersion Comparison:", flush=True)
print(f"  v8 (Baseline):        OOF F1 = 0.6262, LB = 0.6481", flush=True)
print(f"  v19 (Multi-band GP):  OOF F1 = 0.6626, LB = 0.6649 (Rank 23)", flush=True)
print(f"  v20 (All features):   OOF F1 = 0.6432", flush=True)
print(f"  v20b (Selective):     OOF F1 = 0.6535", flush=True)
print(f"  v20c (Optuna tuned):  OOF F1 = {best_f1:.4f}", flush=True)

improvement = (best_f1 - 0.6626) / 0.6626 * 100
if improvement > 0:
    print(f"\n  Improvement over v19: +{improvement:.2f}%", flush=True)
else:
    print(f"\n  Difference from v19: {improvement:.2f}%", flush=True)

print(f"\nBest hyperparameters saved to optuna_v20c_results.pkl", flush=True)
print("=" * 60, flush=True)

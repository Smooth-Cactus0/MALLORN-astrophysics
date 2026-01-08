"""
MALLORN v21: XGBoost Only (Benchmark)

XGBoost achieved 0.6708 OOF in v20c - testing if single model
generalizes better than ensemble on leaderboard.
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import xgboost as xgb
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
base_path = Path(__file__).parent.parent

print("=" * 60, flush=True)
print("MALLORN v21: XGBoost Only Benchmark", flush=True)
print("=" * 60, flush=True)

# ====================
# 1. LOAD DATA & FEATURES (same as v20c)
# ====================
print("\n1. Loading data and features...", flush=True)

from utils.data_loader import load_all_data
data = load_all_data()

train_meta = data['train_meta']
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

# Advanced (selected only)
with open(base_path / 'data/processed/advanced_features_cache.pkl', 'rb') as f:
    adv_data = pickle.load(f)
train_adv = adv_data['train']
test_adv = adv_data['test']
adv_cols = [c for c in train_adv.columns if c != 'object_id']

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
# 2. LOAD OPTUNA XGB PARAMS
# ====================
print("\n2. Loading Optuna XGBoost params...", flush=True)

with open(base_path / 'data/processed/optuna_v20c_results.pkl', 'rb') as f:
    optuna_data = pickle.load(f)

xgb_params = optuna_data['xgb_best_params']
print(f"   Params: {xgb_params}", flush=True)

# ====================
# 3. TRAIN XGBoost ONLY
# ====================
print("\n3. Training XGBoost (5-fold CV)...", flush=True)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(y))
test_preds = np.zeros((len(test_ids), 5))
models = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    model = xgb.XGBClassifier(
        **xgb_params,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbosity=0
    )

    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds[:, fold] = model.predict_proba(X_test)[:, 1]
    models.append(model)

    # Fold F1
    best_f1 = 0
    for t in np.arange(0.1, 0.9, 0.01):
        f1 = f1_score(y_val, (oof_preds[val_idx] > t).astype(int))
        if f1 > best_f1:
            best_f1 = f1

    print(f"   Fold {fold+1}: F1={best_f1:.4f}", flush=True)

# ====================
# 4. FIND OPTIMAL THRESHOLD
# ====================
print("\n4. Finding optimal threshold...", flush=True)

best_f1 = 0
best_thresh = 0.5

for t in np.arange(0.05, 0.95, 0.01):
    f1 = f1_score(y, (oof_preds > t).astype(int))
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"   OOF F1: {best_f1:.4f} @ threshold={best_thresh:.2f}", flush=True)

# Confusion matrix
final_preds = (oof_preds > best_thresh).astype(int)
tp = np.sum((final_preds == 1) & (y == 1))
fp = np.sum((final_preds == 1) & (y == 0))
fn = np.sum((final_preds == 0) & (y == 1))
tn = np.sum((final_preds == 0) & (y == 0))

print(f"   Confusion: TP={tp}, FP={fp}, FN={fn}, TN={tn}", flush=True)
print(f"   Precision: {tp/(tp+fp):.4f}", flush=True)
print(f"   Recall: {tp/(tp+fn):.4f}", flush=True)

# ====================
# 5. CREATE SUBMISSION
# ====================
print("\n5. Creating submission...", flush=True)

test_avg = test_preds.mean(axis=1)
test_final = (test_avg > best_thresh).astype(int)

submission = pd.DataFrame({
    'object_id': test_ids,
    'target': test_final
})

submission_path = base_path / 'submissions/submission_v21_xgb_only.csv'
submission.to_csv(submission_path, index=False)

print(f"   Saved to {submission_path}", flush=True)
print(f"   Predictions: {test_final.sum()} TDEs / {len(test_final)} total ({100*test_final.mean():.1f}%)", flush=True)

# Save model
with open(base_path / 'data/processed/models_v21_xgb.pkl', 'wb') as f:
    pickle.dump({
        'models': models,
        'best_thresh': best_thresh,
        'feature_cols': all_feature_cols,
        'xgb_params': xgb_params
    }, f)

# ====================
# SUMMARY
# ====================
print("\n" + "=" * 60, flush=True)
print("XGBoost Only Training Complete!", flush=True)
print("=" * 60, flush=True)

print(f"\nv21 XGBoost Only: OOF F1 = {best_f1:.4f}", flush=True)
print(f"\nCompare to v20c ensemble: OOF F1 = 0.6687, LB = 0.6518", flush=True)
print(f"\nIf XGB alone beats ensemble on LB, single model > ensemble for this problem.", flush=True)
print("=" * 60, flush=True)

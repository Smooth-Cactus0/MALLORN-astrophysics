"""
MALLORN v74: Top 100 Features + Selective Cesium Features

Key insight from v73: Adding all 80 Cesium features only improved OOF by +0.0002.
Most Cesium features add noise, not signal.

Strategy:
- Keep top 100 v34a features
- Add ONLY the top-performing Cesium features (by v73 importance):
  1. Anderson-Darling (normality test) - physics: TDE flux distributions differ
  2. Percent amplitude - captures variability scale
  3. Flux percentile ratios - shape characteristics

Goal: ~110-115 total features (minimal bloat, maximum signal)
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

print("=" * 80, flush=True)
print("MALLORN v74: Top 100 + Selective Cesium Features", flush=True)
print("=" * 80, flush=True)

# ====================
# 1. LOAD DATA
# ====================
print("\n1. Loading data...", flush=True)

from utils.data_loader import load_all_data
data = load_all_data()

train_meta = data['train_meta']
test_meta = data['test_meta']
train_ids = train_meta['object_id'].tolist()
test_ids = test_meta['object_id'].tolist()
y = train_meta['target'].values

print(f"   Training: {len(train_ids)} objects ({np.sum(y)} TDE)", flush=True)

# ====================
# 2. LOAD TOP 100 FEATURES
# ====================
print("\n2. Loading top 100 v34a features...", flush=True)

with open(base_path / 'data/processed/v34a_artifacts.pkl', 'rb') as f:
    v34a_artifacts = pickle.load(f)

v34a_importance = v34a_artifacts['feature_importance']
top_100_features = v34a_importance.sort_values('importance', ascending=False)['feature'].head(100).tolist()

# Load all feature data
cached = pd.read_pickle(base_path / 'data/processed/features_v4_cache.pkl')
train_base = cached['train_features']
test_base = cached['test_features']

tde_cached = pd.read_pickle(base_path / 'data/processed/tde_physics_cache.pkl')
train_tde = tde_cached['train']
test_tde = tde_cached['test']

with open(base_path / 'data/processed/multiband_gp_cache.pkl', 'rb') as f:
    gp2d_data = pickle.load(f)
train_gp2d = gp2d_data['train']
test_gp2d = gp2d_data['test']

with open(base_path / 'data/processed/bazin_features_cache.pkl', 'rb') as f:
    bazin_cache = pickle.load(f)
train_bazin = bazin_cache['train']
test_bazin = bazin_cache['test']

# Merge all
train_all = train_base.merge(train_tde, on='object_id', how='left')
train_all = train_all.merge(train_gp2d, on='object_id', how='left')
train_all = train_all.merge(train_bazin, on='object_id', how='left')

test_all = test_base.merge(test_tde, on='object_id', how='left')
test_all = test_all.merge(test_gp2d, on='object_id', how='left')
test_all = test_all.merge(test_bazin, on='object_id', how='left')

available_top100 = [f for f in top_100_features if f in train_all.columns]
print(f"   Top 100 features available: {len(available_top100)}", flush=True)

# ====================
# 3. SELECT TOP CESIUM FEATURES
# ====================
print("\n3. Selecting top Cesium features...", flush=True)

# Load cached Cesium features
cesium_cache_path = base_path / 'data/processed/cesium_features_cache.pkl'

with open(cesium_cache_path, 'rb') as f:
    cesium_cache = pickle.load(f)
train_cesium = cesium_cache['train']
test_cesium = cesium_cache['test']

# Based on v73 importance analysis, select only the most valuable Cesium features
# These are features that ranked in top 50 and showed physics-meaningful signal
top_cesium_patterns = [
    # Anderson-Darling: tests normality of flux distribution (TDEs differ!)
    'anderson_darling',
    # Percent amplitude: (max-min)/mean, captures variability scale
    'percent_amplitude',
    # Flux percentile ratios: shape characteristics
    'flux_percentile_ratio_mid50',
    'flux_percentile_ratio_mid65',
    'flux_percentile_ratio_mid80',
]

# Select only r, g, i, z bands (most informative)
important_bands = ['r', 'g', 'i', 'z']

selected_cesium = []
for col in train_cesium.columns:
    if col == 'object_id':
        continue
    # Check if matches our patterns and bands
    for pattern in top_cesium_patterns:
        for band in important_bands:
            if f'{band}_cesium_{pattern}' == col:
                selected_cesium.append(col)

print(f"   Selected Cesium features: {len(selected_cesium)}", flush=True)
for feat in selected_cesium:
    cov = train_cesium[feat].notna().sum() / len(train_cesium)
    print(f"      {feat}: {100*cov:.1f}% coverage", flush=True)

# ====================
# 4. COMBINE FEATURES
# ====================
print("\n4. Combining top 100 + selective Cesium features...", flush=True)

# Start with top 100
train_combined = train_all[['object_id'] + available_top100].copy()
test_combined = test_all[['object_id'] + available_top100].copy()

# Add only selected Cesium features
train_cesium_selected = train_cesium[['object_id'] + selected_cesium]
test_cesium_selected = test_cesium[['object_id'] + selected_cesium]

train_combined = train_combined.merge(train_cesium_selected, on='object_id', how='left')
test_combined = test_combined.merge(test_cesium_selected, on='object_id', how='left')

feature_names = [c for c in train_combined.columns if c != 'object_id']
print(f"   Total features: {len(feature_names)} (100 base + {len(selected_cesium)} Cesium)", flush=True)

X_train = train_combined.drop(columns=['object_id']).values
X_test = test_combined.drop(columns=['object_id']).values

X_train = np.nan_to_num(X_train, nan=np.nan, posinf=1e10, neginf=-1e10)
X_test = np.nan_to_num(X_test, nan=np.nan, posinf=1e10, neginf=-1e10)

# ====================
# 5. TRAIN MODEL
# ====================
print("\n5. Training XGBoost...", flush=True)

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 5,
    'learning_rate': 0.025,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'reg_alpha': 0.2,
    'reg_lambda': 1.5,
    'scale_pos_weight': len(y[y==0]) / len(y[y==1]),
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1
}

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

oof_preds = np.zeros(len(y))
test_preds = np.zeros((len(X_test), n_folds))
feature_importance = np.zeros(len(feature_names))
fold_f1s = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y), 1):
    print(f"\n   Fold {fold}/{n_folds}:", flush=True)

    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test, feature_names=feature_names)

    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=500,
        evals=[(dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=False
    )

    oof_preds[val_idx] = model.predict(dval)
    test_preds[:, fold-1] = model.predict(dtest)

    # Feature importance
    importance = model.get_score(importance_type='gain')
    for feat, gain in importance.items():
        if feat in feature_names:
            idx = feature_names.index(feat)
            feature_importance[idx] += gain

    # Fold F1
    best_f1 = 0
    for t in np.linspace(0.03, 0.3, 50):
        f1 = f1_score(y_val, (oof_preds[val_idx] > t).astype(int))
        if f1 > best_f1:
            best_f1 = f1
    fold_f1s.append(best_f1)
    print(f"      Fold F1: {best_f1:.4f}", flush=True)

# ====================
# 6. RESULTS
# ====================
print("\n" + "=" * 80, flush=True)
print("RESULTS", flush=True)
print("=" * 80, flush=True)

best_f1 = 0
best_thresh = 0.1
for t in np.linspace(0.03, 0.3, 200):
    f1 = f1_score(y, (oof_preds > t).astype(int))
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"\n   OOF F1: {best_f1:.4f} @ threshold={best_thresh:.3f}", flush=True)
print(f"   Fold F1s: {[f'{f:.4f}' for f in fold_f1s]}", flush=True)
print(f"   Fold Std: {np.std(fold_f1s):.4f}", flush=True)

final_preds = (oof_preds > best_thresh).astype(int)
tp = np.sum((final_preds == 1) & (y == 1))
fp = np.sum((final_preds == 1) & (y == 0))
fn = np.sum((final_preds == 0) & (y == 1))
print(f"\n   TP={tp}, FP={fp}, FN={fn}", flush=True)
print(f"   Precision: {tp/(tp+fp):.4f}, Recall: {tp/(tp+fn):.4f}", flush=True)

# Feature importance analysis
feature_importance = feature_importance / n_folds
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("\n   Top 20 Features:", flush=True)
print(importance_df.head(20).to_string(index=False), flush=True)

# Cesium features analysis
print("\n   Selected Cesium Features Importance:", flush=True)
for col in selected_cesium:
    if col in importance_df['feature'].values:
        row = importance_df[importance_df['feature'] == col].iloc[0]
        rank = list(importance_df['feature']).index(col) + 1
        print(f"      {rank:3d}. {col:45s} {row['importance']:8.1f}", flush=True)

# ====================
# 7. SUBMISSION
# ====================
print("\n" + "=" * 80, flush=True)
print("SUBMISSION", flush=True)
print("=" * 80, flush=True)

test_avg = test_preds.mean(axis=1)
test_binary = (test_avg > best_thresh).astype(int)

submission = pd.DataFrame({
    'object_id': test_ids,
    'target': test_binary
})

submission_path = base_path / 'submissions/submission_v74_selective_cesium.csv'
submission.to_csv(submission_path, index=False)

print(f"   Saved: {submission_path.name}", flush=True)
print(f"   Predicted TDEs: {test_binary.sum()}", flush=True)

# Save artifacts
artifacts = {
    'oof_preds': oof_preds,
    'test_preds': test_avg,
    'best_threshold': best_thresh,
    'oof_f1': best_f1,
    'feature_importance': importance_df,
    'feature_names': feature_names,
    'selected_cesium': selected_cesium,
    'fold_f1s': fold_f1s
}

with open(base_path / 'data/processed/v74_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

# ====================
# 8. COMPARISON
# ====================
print("\n" + "=" * 80, flush=True)
print("COMPARISON", flush=True)
print("=" * 80, flush=True)

print(f"""
   Model                       Features   OOF F1   LB F1
   -----                       --------   ------   -----
   v34a (baseline)             224        0.6667   0.6907
   v72 top100                  100        0.6723   0.6500
   v73 top100+ALL Cesium       180        0.6725   ???
   v74 top100+SELECT Cesium    {len(feature_names)}        {best_f1:.4f}   ???

   Selective Cesium features: {len(selected_cesium)}
   - Anderson-Darling (normality): captures TDE flux distribution
   - Percent amplitude: variability scale
   - Flux percentile ratios: shape characteristics
""", flush=True)

print("=" * 80, flush=True)
print("v74 Complete", flush=True)
print("=" * 80, flush=True)

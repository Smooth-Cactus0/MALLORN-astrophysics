"""
MALLORN v73: Top 100 Features + Cesium Features

Building on v72's finding that 100 features is optimal:
- Start with top 100 v34a features (LB 0.65)
- Add Cesium domain-specific features:
  - Stetson J, K (correlated variability)
  - Beyond N-std (outlier detection)
  - Flux percentile ratios (shape)
  - Maximum slope, linear trend
  - Anderson-Darling normality test

Goal: Add astronomy-specific features that capture physics
we're missing, without adding noise.
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
print("MALLORN v73: Top 100 + Cesium Features", flush=True)
print("=" * 80, flush=True)

# ====================
# 1. LOAD DATA
# ====================
print("\n1. Loading data...", flush=True)

from utils.data_loader import load_all_data
data = load_all_data()

train_meta = data['train_meta']
test_meta = data['test_meta']
train_lc = data['train_lc']
test_lc = data['test_lc']
train_ids = train_meta['object_id'].tolist()
test_ids = test_meta['object_id'].tolist()
y = train_meta['target'].values

print(f"   Training: {len(train_ids)} objects ({np.sum(y)} TDE)", flush=True)

# ====================
# 2. LOAD TOP 100 FEATURES
# ====================
print("\n2. Loading top 100 v34a features...", flush=True)

# Load v34a feature importance
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

# Select top 100
available_top100 = [f for f in top_100_features if f in train_all.columns]
print(f"   Top 100 features available: {len(available_top100)}", flush=True)

# ====================
# 3. EXTRACT CESIUM FEATURES
# ====================
print("\n3. Extracting Cesium features...", flush=True)

from features.cesium_features import extract_cesium_features

# Check cache
cesium_cache_path = base_path / 'data/processed/cesium_features_cache.pkl'

if cesium_cache_path.exists():
    print("   Loading cached Cesium features...", flush=True)
    with open(cesium_cache_path, 'rb') as f:
        cesium_cache = pickle.load(f)
    train_cesium = cesium_cache['train']
    test_cesium = cesium_cache['test']
else:
    print("   Extracting Cesium features (training)...", flush=True)
    train_cesium = extract_cesium_features(train_lc, train_ids)

    print("   Extracting Cesium features (test)...", flush=True)
    test_cesium = extract_cesium_features(test_lc, test_ids)

    # Cache
    with open(cesium_cache_path, 'wb') as f:
        pickle.dump({'train': train_cesium, 'test': test_cesium}, f)

cesium_cols = [c for c in train_cesium.columns if c != 'object_id']
print(f"   Cesium features: {len(cesium_cols)}", flush=True)

# Check coverage
print("\n   Cesium feature coverage (r-band):", flush=True)
for col in ['r_cesium_stetson_j', 'r_cesium_beyond_1std', 'r_cesium_maximum_slope']:
    if col in train_cesium.columns:
        cov = train_cesium[col].notna().sum() / len(train_cesium)
        print(f"      {col}: {100*cov:.1f}%", flush=True)

# ====================
# 4. COMBINE FEATURES
# ====================
print("\n4. Combining top 100 + Cesium features...", flush=True)

# Start with top 100
train_combined = train_all[['object_id'] + available_top100].copy()
test_combined = test_all[['object_id'] + available_top100].copy()

# Add Cesium features
train_combined = train_combined.merge(train_cesium, on='object_id', how='left')
test_combined = test_combined.merge(test_cesium, on='object_id', how='left')

feature_names = [c for c in train_combined.columns if c != 'object_id']
print(f"   Total features: {len(feature_names)} (100 base + {len(cesium_cols)} Cesium)", flush=True)

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
print("\n   Cesium Features in Top 50:", flush=True)
cesium_in_top50 = importance_df.head(50)
cesium_feats = cesium_in_top50[cesium_in_top50['feature'].str.contains('cesium')]
if len(cesium_feats) > 0:
    for _, row in cesium_feats.iterrows():
        rank = list(importance_df['feature']).index(row['feature']) + 1
        print(f"      {rank:3d}. {row['feature']:40s} {row['importance']:8.1f}", flush=True)
else:
    print("      None in top 50", flush=True)

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

submission_path = base_path / 'submissions/submission_v73_top100_cesium.csv'
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
    'fold_f1s': fold_f1s
}

with open(base_path / 'data/processed/v73_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

# ====================
# 8. COMPARISON
# ====================
print("\n" + "=" * 80, flush=True)
print("COMPARISON", flush=True)
print("=" * 80, flush=True)

print(f"""
   Model                    Features   OOF F1   LB F1
   -----                    --------   ------   -----
   v34a (baseline)          224        0.6667   0.6907
   v72 top100               100        0.6723   0.6500
   v73 top100+Cesium        {len(feature_names)}        {best_f1:.4f}   ???

   Cesium features added: {len(cesium_cols)}
   Cesium in top 50: {len(cesium_feats)}
""", flush=True)

print("=" * 80, flush=True)
print("v73 Complete", flush=True)
print("=" * 80, flush=True)

"""
MALLORN v111: LightGBM DART Boosting Variant

DART (Dropouts meet Multiple Additive Regression Trees) randomly drops trees
during boosting, which can help prevent overfitting by reducing the contribution
of individual trees.

Key differences from v110 (GBDT):
- boosting_type: 'dart' instead of 'gbdt'
- drop_rate: 0.15 (fraction of trees to drop)
- max_drop: 50 (max trees to drop per iteration)
- skip_drop: 0.5 (probability of skipping dropout)
- uniform_drop: False (weight by contribution)
- No early stopping (DART doesn't support it well)

Goal: DART's dropout may improve generalization beyond GBDT regularization.
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
base_path = Path(__file__).parent.parent

print("=" * 80, flush=True)
print("MALLORN v111: LightGBM DART Boosting Variant", flush=True)
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
# 2. LOAD v34a FEATURES
# ====================
print("\n2. Loading v34a features...", flush=True)

with open(base_path / 'data/processed/v34a_artifacts.pkl', 'rb') as f:
    v34a_artifacts = pickle.load(f)

feature_names = v34a_artifacts['feature_names']
print(f"   v34a features: {len(feature_names)}", flush=True)

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

# Select v34a features
available_features = [f for f in feature_names if f in train_all.columns]
print(f"   Available features: {len(available_features)}", flush=True)

X_train = train_all[available_features].values
X_test = test_all[available_features].values

# Handle infinities
X_train = np.nan_to_num(X_train, nan=np.nan, posinf=1e10, neginf=-1e10)
X_test = np.nan_to_num(X_test, nan=np.nan, posinf=1e10, neginf=-1e10)

# Calculate class weight
n_pos = np.sum(y)
n_neg = len(y) - n_pos
scale_pos_weight = n_neg / n_pos
print(f"   Class imbalance ratio: {scale_pos_weight:.2f}", flush=True)

# ====================
# 3. DART PARAMETERS
# ====================
print("\n3. DART boosting parameters:", flush=True)

lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'dart',    # KEY: DART instead of GBDT
    'verbose': -1,
    'n_jobs': -1,
    'random_state': 42,

    # DART-specific parameters
    'drop_rate': 0.15,          # Fraction of trees to drop
    'max_drop': 50,             # Max trees to drop per iteration
    'skip_drop': 0.5,           # Probability of skipping dropout
    'uniform_drop': False,      # Weight by contribution

    # Same regularization as v110
    'num_leaves': 15,
    'max_depth': 4,
    'learning_rate': 0.02,
    'n_estimators': 600,

    'feature_fraction': 0.4,
    'bagging_fraction': 0.5,
    'bagging_freq': 5,

    'reg_alpha': 5.0,
    'reg_lambda': 10.0,

    'min_child_samples': 30,

    'scale_pos_weight': scale_pos_weight,
}

print(f"   boosting_type: {lgb_params['boosting_type']}", flush=True)
print(f"   drop_rate: {lgb_params['drop_rate']} (fraction of trees to drop)", flush=True)
print(f"   max_drop: {lgb_params['max_drop']} (max trees to drop per iter)", flush=True)
print(f"   skip_drop: {lgb_params['skip_drop']} (probability of skipping dropout)", flush=True)
print(f"   uniform_drop: {lgb_params['uniform_drop']} (weight by contribution)", flush=True)
print(f"   n_estimators: {lgb_params['n_estimators']} (fixed, no early stopping)", flush=True)
print(f"   num_leaves: {lgb_params['num_leaves']}", flush=True)
print(f"   max_depth: {lgb_params['max_depth']}", flush=True)
print(f"   learning_rate: {lgb_params['learning_rate']}", flush=True)

# ====================
# 4. TRAIN WITH 5-FOLD CV
# ====================
print("\n4. Training LightGBM DART with 5-fold CV...", flush=True)
print("   Note: DART uses fixed iterations (no early stopping)", flush=True)

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

oof_preds = np.zeros(len(y))
test_preds = np.zeros((len(X_test), n_folds))
feature_importance = np.zeros(len(available_features))
fold_f1s = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y), 1):
    print(f"\n   Fold {fold}/{n_folds}:", flush=True)

    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=available_features)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=available_features, reference=train_data)

    # DART doesn't support early stopping well, use fixed iterations
    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=lgb_params['n_estimators'],
        valid_sets=[val_data],
        callbacks=[lgb.log_evaluation(period=0)]
    )

    # Predict without num_iteration parameter for DART
    oof_preds[val_idx] = model.predict(X_val)
    test_preds[:, fold-1] = model.predict(X_test)

    # Feature importance
    importance = model.feature_importance(importance_type='gain')
    feature_importance += importance

    # Fold F1 at best threshold
    best_fold_f1 = 0
    best_fold_thresh = 0.1
    for t in np.linspace(0.03, 0.5, 50):
        f1 = f1_score(y_val, (oof_preds[val_idx] > t).astype(int))
        if f1 > best_fold_f1:
            best_fold_f1 = f1
            best_fold_thresh = t
    fold_f1s.append(best_fold_f1)
    print(f"      Fold F1: {best_fold_f1:.4f} @ threshold={best_fold_thresh:.3f}", flush=True)
    print(f"      Trees: {model.num_trees()}", flush=True)

# ====================
# 5. FIND OPTIMAL THRESHOLD
# ====================
print("\n5. Finding optimal threshold...", flush=True)

best_f1 = 0
best_thresh = 0.1
for t in np.linspace(0.03, 0.5, 200):
    f1 = f1_score(y, (oof_preds > t).astype(int))
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"   Best threshold: {best_thresh:.4f}", flush=True)

# ====================
# 6. RESULTS
# ====================
print("\n" + "=" * 80, flush=True)
print("RESULTS", flush=True)
print("=" * 80, flush=True)

print(f"\n   OOF F1: {best_f1:.4f} @ threshold={best_thresh:.3f}", flush=True)
print(f"   Fold F1s: {[f'{f:.4f}' for f in fold_f1s]}", flush=True)
print(f"   Fold Std: {np.std(fold_f1s):.4f}", flush=True)

final_preds = (oof_preds > best_thresh).astype(int)
tp = np.sum((final_preds == 1) & (y == 1))
fp = np.sum((final_preds == 1) & (y == 0))
fn = np.sum((final_preds == 0) & (y == 1))
print(f"\n   TP={tp}, FP={fp}, FN={fn}", flush=True)
if tp + fp > 0:
    print(f"   Precision: {tp/(tp+fp):.4f}, Recall: {tp/(tp+fn):.4f}", flush=True)

# Feature importance analysis
feature_importance = feature_importance / n_folds
importance_df = pd.DataFrame({
    'feature': available_features,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("\n   Top 20 Features:", flush=True)
print(importance_df.head(20).to_string(index=False), flush=True)

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

submission_path = base_path / 'submissions/submission_v111_lgbm_dart.csv'
submission.to_csv(submission_path, index=False)

print(f"   Saved: {submission_path.name}", flush=True)
print(f"   Predicted TDEs: {test_binary.sum()}", flush=True)

# ====================
# 8. SAVE ARTIFACTS
# ====================
artifacts = {
    'oof_preds': oof_preds,
    'test_preds': test_avg,
    'best_threshold': best_thresh,
    'oof_f1': best_f1,
    'feature_importance': importance_df,
    'feature_names': available_features,
    'fold_f1s': fold_f1s,
    'params': lgb_params
}

with open(base_path / 'data/processed/v111_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print(f"   Artifacts saved to: data/processed/v111_artifacts.pkl", flush=True)

# ====================
# 9. COMPARISON
# ====================
print("\n" + "=" * 80, flush=True)
print("COMPARISON", flush=True)
print("=" * 80, flush=True)

print(f"""
   Model                           Features   OOF F1    LB F1
   -----                           --------   ------    -----
   v34a XGBoost (Optuna)           224        0.6667    0.6907  <- Best LB
   v77 LightGBM (Optuna GBDT)      223        0.6886    0.6714  <- Overfit
   v110 LightGBM (Heavy Reg GBDT)  224        0.6609    ???
   v111 LightGBM (DART)            {len(available_features)}        {best_f1:.4f}    ???

   DART boosting effect:
   - vs v110 GBDT: {best_f1 - 0.6609:+.4f} OOF
   - vs v34a XGB:  {best_f1 - 0.6667:+.4f} OOF

   DART adds dropout regularization on top of heavy regularization.
   If OOF is similar or lower than v110, DART may generalize better.
   Key: Does DART's dropout help LightGBM match XGBoost's generalization?
""", flush=True)

print("=" * 80, flush=True)
print("v111 Complete", flush=True)
print("=" * 80, flush=True)

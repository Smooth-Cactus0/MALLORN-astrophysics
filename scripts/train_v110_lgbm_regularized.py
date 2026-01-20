"""
MALLORN v110: LightGBM with Heavy Regularization

Key insight: v77 LightGBM achieved higher OOF (0.6886) but LOWER LB (0.6714)
than v34a XGBoost (OOF 0.6667, LB 0.6907). This means LightGBM overfit.

This version uses heavy regularization to improve generalization:
- num_leaves: 15 (v77 had up to 63)
- max_depth: 4 (v77 had up to 8)
- feature_fraction: 0.4 (v77 had 0.6-0.95)
- bagging_fraction: 0.5 (v77 had 0.6-0.95)
- reg_alpha: 5.0 (v77 had 0.01-1.0)
- reg_lambda: 10.0 (v77 had 0.1-3.0)

Goal: Lower OOF F1 (0.64-0.68) that generalizes better to LB.
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
print("MALLORN v110: LightGBM with Heavy Regularization", flush=True)
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
# 3. HEAVY REGULARIZATION PARAMETERS
# ====================
print("\n3. Heavy regularization parameters (vs v77 Optuna ranges):", flush=True)

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'verbose': -1,
    'n_jobs': -1,
    'random_state': 42,

    # Tree structure - MUCH smaller than v77
    'num_leaves': 15,         # v77: 15-63
    'max_depth': 4,           # v77: 4-8

    # Learning - slower
    'learning_rate': 0.02,    # v77: 0.01-0.1
    'n_estimators': 600,      # v77: 200-800

    # Regularization - MUCH stronger
    'feature_fraction': 0.4,  # v77: 0.6-0.95
    'bagging_fraction': 0.5,  # v77: 0.6-0.95
    'bagging_freq': 5,        # v77: 1-7
    'reg_alpha': 5.0,         # v77: 0.01-1.0 (5-500x stronger!)
    'reg_lambda': 10.0,       # v77: 0.1-3.0 (3-100x stronger!)

    # Leaf constraints - stricter
    'min_child_samples': 30,  # v77: 10-50

    # Class imbalance
    'scale_pos_weight': scale_pos_weight,
}

print(f"   num_leaves: {params['num_leaves']} (v77: 15-63)", flush=True)
print(f"   max_depth: {params['max_depth']} (v77: 4-8)", flush=True)
print(f"   feature_fraction: {params['feature_fraction']} (v77: 0.6-0.95)", flush=True)
print(f"   bagging_fraction: {params['bagging_fraction']} (v77: 0.6-0.95)", flush=True)
print(f"   reg_alpha: {params['reg_alpha']} (v77: 0.01-1.0)", flush=True)
print(f"   reg_lambda: {params['reg_lambda']} (v77: 0.1-3.0)", flush=True)
print(f"   min_child_samples: {params['min_child_samples']} (v77: 10-50)", flush=True)

# ====================
# 4. TRAIN WITH 5-FOLD CV
# ====================
print("\n4. Training LightGBM with 5-fold CV...", flush=True)

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

    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=0)
        ]
    )

    oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    test_preds[:, fold-1] = model.predict(X_test, num_iteration=model.best_iteration)

    # Feature importance
    importance = model.feature_importance(importance_type='gain')
    feature_importance += importance

    # Fold F1 at best threshold
    best_fold_f1 = 0
    for t in np.linspace(0.03, 0.5, 50):
        f1 = f1_score(y_val, (oof_preds[val_idx] > t).astype(int))
        if f1 > best_fold_f1:
            best_fold_f1 = f1
    fold_f1s.append(best_fold_f1)
    print(f"      Fold F1: {best_fold_f1:.4f}, best_iteration: {model.best_iteration}", flush=True)

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

submission_path = base_path / 'submissions/submission_v110_lgbm_reg.csv'
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
    'params': params
}

with open(base_path / 'data/processed/v110_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print(f"   Artifacts saved to: data/processed/v110_artifacts.pkl", flush=True)

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
   v77 LightGBM (Optuna)           223        0.6886    0.6714  <- Overfit
   v110 LightGBM (Heavy Reg)       {len(available_features)}        {best_f1:.4f}    ???

   Regularization impact:
   - vs v34a: {best_f1 - 0.6667:+.4f} OOF (target: similar or lower for generalization)
   - vs v77:  {best_f1 - 0.6886:+.4f} OOF (expected: lower due to heavy reg)

   Key question: Will lower OOF lead to better LB?
   If LB > 0.6714 (v77), heavy regularization helps LightGBM generalize.
   If LB > 0.6907 (v34a), this beats our best model!
""", flush=True)

print("=" * 80, flush=True)
print("v110 Complete", flush=True)
print("=" * 80, flush=True)

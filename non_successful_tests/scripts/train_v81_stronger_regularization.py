"""
MALLORN v81: Stronger Regularization on v34a

Key insight: v34a (OOF 0.6667) generalized to LB 0.6907 (+0.024 gain).
            Other models with higher OOF performed WORSE on LB.

Strategy: Deliberately underfit to improve generalization.
- Increase reg_lambda and reg_alpha
- Reduce max_depth
- Increase min_child_weight
- Target OOF of ~0.64-0.65 hoping for 0.70+ LB

Tests multiple regularization strengths:
- v81a: Mild increase (1.5x current)
- v81b: Moderate increase (2x current)
- v81c: Strong increase (3x current)
- v81d: Very strong (4x current)
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

print("=" * 80)
print("MALLORN v81: Stronger Regularization")
print("=" * 80)

# ====================
# 1. LOAD DATA
# ====================
print("\n1. Loading data...")

from utils.data_loader import load_all_data
data = load_all_data()

train_meta = data['train_meta']
test_meta = data['test_meta']
train_ids = train_meta['object_id'].tolist()
test_ids = test_meta['object_id'].tolist()
y = train_meta['target'].values

print(f"   Training: {len(train_ids)} objects ({np.sum(y)} TDE)")

# ====================
# 2. LOAD v34a FEATURES
# ====================
print("\n2. Loading v34a features...")

with open(base_path / 'data/processed/v34a_artifacts.pkl', 'rb') as f:
    v34a = pickle.load(f)

feature_names = v34a['feature_names']
print(f"   v34a features: {len(feature_names)}")

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
print(f"   Available features: {len(available_features)}")

X_train = train_all[available_features].values
X_test = test_all[available_features].values

# Handle infinities
X_train = np.nan_to_num(X_train, nan=np.nan, posinf=1e10, neginf=-1e10)
X_test = np.nan_to_num(X_test, nan=np.nan, posinf=1e10, neginf=-1e10)

# ====================
# 3. DEFINE REGULARIZATION VARIANTS
# ====================
print("\n3. Defining regularization variants...")

# v34a original parameters (baseline)
base_params = {
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

# Regularization variants
variants = {
    'v81a_mild': {
        'reg_alpha': 0.3,       # 1.5x
        'reg_lambda': 2.25,     # 1.5x
        'max_depth': 5,
        'min_child_weight': 4,
    },
    'v81b_moderate': {
        'reg_alpha': 0.4,       # 2x
        'reg_lambda': 3.0,      # 2x
        'max_depth': 4,
        'min_child_weight': 5,
    },
    'v81c_strong': {
        'reg_alpha': 0.6,       # 3x
        'reg_lambda': 4.5,      # 3x
        'max_depth': 4,
        'min_child_weight': 7,
    },
    'v81d_very_strong': {
        'reg_alpha': 0.8,       # 4x
        'reg_lambda': 6.0,      # 4x
        'max_depth': 3,
        'min_child_weight': 10,
    },
    'v81e_extreme': {
        'reg_alpha': 1.0,       # 5x
        'reg_lambda': 7.5,      # 5x
        'max_depth': 3,
        'min_child_weight': 15,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
    }
}

for name, overrides in variants.items():
    print(f"   {name}: reg_alpha={overrides['reg_alpha']}, reg_lambda={overrides['reg_lambda']}, max_depth={overrides['max_depth']}")

# ====================
# 4. TRAIN ALL VARIANTS
# ====================
print("\n" + "=" * 80)
print("TRAINING")
print("=" * 80)

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

results = {}

for variant_name, overrides in variants.items():
    print(f"\n   {variant_name}...")

    # Merge base params with overrides
    params = base_params.copy()
    params.update(overrides)

    oof_preds = np.zeros(len(y))
    test_preds = np.zeros((len(X_test), n_folds))
    fold_f1s = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y), 1):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=available_features)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=available_features)
        dtest = xgb.DMatrix(X_test, feature_names=available_features)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            evals=[(dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=False
        )

        oof_preds[val_idx] = model.predict(dval)
        test_preds[:, fold-1] = model.predict(dtest)

        # Fold F1
        best_f1 = 0
        for t in np.linspace(0.05, 0.5, 50):
            f1 = f1_score(y_val, (oof_preds[val_idx] > t).astype(int))
            if f1 > best_f1:
                best_f1 = f1
        fold_f1s.append(best_f1)

    # Overall OOF F1
    best_f1 = 0
    best_thresh = 0.3
    for t in np.linspace(0.05, 0.5, 200):
        f1 = f1_score(y, (oof_preds > t).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    results[variant_name] = {
        'oof_f1': best_f1,
        'threshold': best_thresh,
        'fold_f1s': fold_f1s,
        'fold_std': np.std(fold_f1s),
        'oof_preds': oof_preds,
        'test_preds': test_preds.mean(axis=1),
        'params': params
    }

    print(f"      OOF F1: {best_f1:.4f} @ threshold={best_thresh:.3f}")
    print(f"      Fold Std: {np.std(fold_f1s):.4f}")

# ====================
# 5. RESULTS SUMMARY
# ====================
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

print(f"\n{'Variant':<25} {'OOF F1':<10} {'Std':<10} {'Threshold':<10} {'Expected LB':<12}")
print("-" * 70)

# Reference: v34a
print(f"{'v34a (baseline)':<25} {'0.6667':<10} {'-':<10} {'0.386':<10} {'0.6907 (actual)':<12}")
print("-" * 70)

# Based on pattern: lower OOF = higher LB
# v34a: OOF 0.6667 -> LB 0.6907 (+0.024)
# Estimate: LB = OOF + 0.024 + bonus_for_lower_oof

sorted_results = sorted(results.items(), key=lambda x: x[1]['oof_f1'])

for name, res in sorted_results:
    # Estimate LB based on pattern
    oof_diff = res['oof_f1'] - 0.6667
    if oof_diff < 0:
        # Lower OOF might generalize better
        estimated_lb = f"~{0.6907 - oof_diff:.4f}?"
    else:
        estimated_lb = f"~{0.6907 - oof_diff*2:.4f}?"

    print(f"{name:<25} {res['oof_f1']:<10.4f} {res['fold_std']:<10.4f} {res['threshold']:<10.3f} {estimated_lb:<12}")

# ====================
# 6. SUBMISSIONS
# ====================
print("\n" + "=" * 80)
print("SUBMISSIONS")
print("=" * 80)

for name, res in sorted_results:
    test_binary = (res['test_preds'] > res['threshold']).astype(int)

    submission = pd.DataFrame({
        'object_id': test_ids,
        'target': test_binary
    })

    filename = f"submission_{name}.csv"
    submission.to_csv(base_path / f'submissions/{filename}', index=False)
    print(f"   {filename}: OOF={res['oof_f1']:.4f}, TDEs={test_binary.sum()}")

# Save artifacts
with open(base_path / 'data/processed/v81_artifacts.pkl', 'wb') as f:
    pickle.dump(results, f)

# ====================
# 7. RECOMMENDATION
# ====================
print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)

# Find variant with lowest OOF but still reasonable
lowest_oof = sorted_results[0]
print(f"\n   Lowest OOF variant: {lowest_oof[0]} (OOF={lowest_oof[1]['oof_f1']:.4f})")
print(f"   If pattern holds, this may generalize best to LB")

# Find variant with best stability
most_stable = min(results.items(), key=lambda x: x[1]['fold_std'])
print(f"\n   Most stable variant: {most_stable[0]} (std={most_stable[1]['fold_std']:.4f})")

print("\n   Submit in order of lowest OOF first (most underfit = best generalization)")

print("\n" + "=" * 80)
print("v81 Complete")
print("=" * 80)

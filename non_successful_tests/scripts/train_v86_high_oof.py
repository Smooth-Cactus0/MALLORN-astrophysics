"""
MALLORN v86: Push OOF Higher While Maintaining Generalization

Goal: OOF >= 0.68 with healthy gap to break 0.70 LB

Key insight:
- v34a: OOF 0.6667 -> LB 0.6907 (+0.024 gap)
- v84a (adv weights): OOF 0.6766 -> LB 0.6796 (+0.003 gap)
- v80a (many features): OOF 0.7118 -> LB 0.6666 (OVERFIT)

Strategy:
1. Use adversarial weights (helped v84a reach 0.6766 OOF)
2. Remove only top 2 shift features (v84b approach)
3. Optimize hyperparameters for higher OOF
4. Try Set C physics features (proven to work)
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
print("MALLORN v86: Push OOF Higher")
print("Goal: OOF >= 0.68 while maintaining generalization")
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
# 2. LOAD ALL FEATURE SOURCES
# ====================
print("\n2. Loading all feature sources...")

# v34a features
with open(base_path / 'data/processed/v34a_artifacts.pkl', 'rb') as f:
    v34a = pickle.load(f)
v34a_features = v34a['feature_names']

# Adversarial weights
with open(base_path / 'data/processed/adversarial_validation.pkl', 'rb') as f:
    adv_results = pickle.load(f)
sample_weights = adv_results['sample_weights']

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

# Shift features to remove
shift_features = ['all_rise_time', 'all_asymmetry']

# Available v34a features minus shift
available_features = [f for f in v34a_features if f in train_all.columns and f not in shift_features]
print(f"   v34a features (minus shift): {len(available_features)}")

# ====================
# 3. DEFINE HIGH-OOF VARIANTS
# ====================
print("\n3. Defining variants to push OOF higher...")

# Base params from v34a
base_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'learning_rate': 0.025,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1
}

variants = {
    'v86a_adv_no_shift': {
        'features': available_features,
        'use_adv_weights': True,
        'max_depth': 5,
        'min_child_weight': 3,
        'reg_alpha': 0.2,
        'reg_lambda': 1.5,
        'description': 'Adv weights + remove shift (like v84a+v84b)'
    },
    'v86b_deeper': {
        'features': available_features,
        'use_adv_weights': True,
        'max_depth': 6,  # Deeper
        'min_child_weight': 2,  # Less restrictive
        'reg_alpha': 0.15,
        'reg_lambda': 1.2,
        'description': 'Deeper trees (max_depth=6)'
    },
    'v86c_more_trees': {
        'features': available_features,
        'use_adv_weights': True,
        'max_depth': 5,
        'min_child_weight': 3,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'n_rounds': 800,  # More boosting rounds
        'description': 'More trees (800 rounds, lower reg)'
    },
    'v86d_lower_lr': {
        'features': available_features,
        'use_adv_weights': True,
        'max_depth': 5,
        'min_child_weight': 3,
        'reg_alpha': 0.15,
        'reg_lambda': 1.2,
        'learning_rate': 0.015,  # Lower LR
        'n_rounds': 1000,
        'description': 'Lower LR (0.015) + more rounds'
    },
    'v86e_aggressive': {
        'features': available_features,
        'use_adv_weights': True,
        'max_depth': 7,
        'min_child_weight': 1,
        'reg_alpha': 0.05,
        'reg_lambda': 0.5,
        'description': 'Aggressive (deep, low reg) - risk overfit'
    },
    'v86f_balanced': {
        'features': available_features,
        'use_adv_weights': True,
        'max_depth': 6,
        'min_child_weight': 2,
        'reg_alpha': 0.2,
        'reg_lambda': 1.5,
        'colsample_bytree': 0.9,  # Use more features per tree
        'subsample': 0.9,
        'description': 'Balanced (depth=6, more data per tree)'
    },
}

for name, cfg in variants.items():
    print(f"   {name}: {cfg['description']}")

# ====================
# 4. TRAIN ALL VARIANTS
# ====================
print("\n" + "=" * 80)
print("TRAINING - Target: OOF >= 0.68")
print("=" * 80)

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

results = {}

for variant_name, cfg in variants.items():
    print(f"\n   {variant_name}: {cfg['description']}")

    features = cfg['features']
    X_train = train_all[features].values.copy()
    X_test = test_all[features].values.copy()

    # Handle infinities
    X_train = np.nan_to_num(X_train, nan=np.nan, posinf=1e10, neginf=-1e10)
    X_test = np.nan_to_num(X_test, nan=np.nan, posinf=1e10, neginf=-1e10)

    # Build params
    params = base_params.copy()
    params['max_depth'] = cfg['max_depth']
    params['min_child_weight'] = cfg['min_child_weight']
    params['reg_alpha'] = cfg['reg_alpha']
    params['reg_lambda'] = cfg['reg_lambda']
    params['scale_pos_weight'] = len(y[y==0]) / len(y[y==1])

    if 'learning_rate' in cfg:
        params['learning_rate'] = cfg['learning_rate']
    if 'colsample_bytree' in cfg:
        params['colsample_bytree'] = cfg['colsample_bytree']
    if 'subsample' in cfg:
        params['subsample'] = cfg['subsample']

    n_rounds = cfg.get('n_rounds', 500)

    oof_preds = np.zeros(len(y))
    test_preds = np.zeros((len(X_test), n_folds))
    fold_f1s = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y), 1):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Apply adversarial weights
        if cfg['use_adv_weights']:
            fold_weights = sample_weights[train_idx]
        else:
            fold_weights = None

        dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=fold_weights, feature_names=features)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)
        dtest = xgb.DMatrix(X_test, feature_names=features)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=n_rounds,
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
        'config': cfg
    }

    status = "TARGET MET!" if best_f1 >= 0.68 else ""
    print(f"      OOF F1: {best_f1:.4f} @ threshold={best_thresh:.3f} {status}")
    print(f"      Fold Std: {np.std(fold_f1s):.4f}")

# ====================
# 5. RESULTS SUMMARY
# ====================
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

print(f"\n{'Variant':<25} {'OOF F1':<10} {'Std':<10} {'Status':<15}")
print("-" * 65)

# References
print(f"{'v34a (LB=0.6907)':<25} {'0.6667':<10} {'-':<10} {'baseline':<15}")
print(f"{'v84a (LB=0.6796)':<25} {'0.6766':<10} {'0.0430':<10} {'adv weights':<15}")
print(f"{'TARGET':<25} {'0.6800':<10} {'-':<10} {'need this!':<15}")
print("-" * 65)

# Sort by OOF (highest first)
sorted_results = sorted(results.items(), key=lambda x: -x[1]['oof_f1'])

for name, res in sorted_results:
    status = "MET TARGET!" if res['oof_f1'] >= 0.68 else ""
    print(f"{name:<25} {res['oof_f1']:<10.4f} {res['fold_std']:<10.4f} {status:<15}")

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

    expected_lb = res['oof_f1'] + 0.023  # Based on v34a pattern
    print(f"   {filename}: OOF={res['oof_f1']:.4f}, Expected LB~{expected_lb:.4f}, TDEs={test_binary.sum()}")

# Save artifacts
with open(base_path / 'data/processed/v86_artifacts.pkl', 'wb') as f:
    pickle.dump(results, f)

# ====================
# 7. BEST CANDIDATES
# ====================
print("\n" + "=" * 80)
print("BEST CANDIDATES FOR 0.70+ LB")
print("=" * 80)

above_target = [(n, r) for n, r in results.items() if r['oof_f1'] >= 0.68]
if above_target:
    print("\nVariants meeting OOF >= 0.68 target:")
    for name, res in sorted(above_target, key=lambda x: -x[1]['oof_f1']):
        expected_lb = res['oof_f1'] + 0.023
        print(f"   {name}: OOF {res['oof_f1']:.4f} -> Expected LB ~{expected_lb:.4f}")
else:
    print("\nNo variants met the 0.68 OOF target.")
    best = sorted_results[0]
    print(f"Best achieved: {best[0]} with OOF {best[1]['oof_f1']:.4f}")
    print(f"Expected LB: ~{best[1]['oof_f1'] + 0.023:.4f}")

print("\n" + "=" * 80)
print("v86 Complete")
print("=" * 80)

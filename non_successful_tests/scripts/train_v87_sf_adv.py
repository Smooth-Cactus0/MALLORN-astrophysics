"""
MALLORN v87: Structure Function + Adversarial Weights

Key insight:
- v80a with SF features got OOF 0.7118 but LB 0.6666 (overfit)
- v84a with adversarial weights got OOF 0.6766 -> LB 0.6796 (good gap)

Strategy: Combine SF features with adversarial weights to get high OOF that generalizes.
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
print("MALLORN v87: Structure Function + Adversarial Weights")
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
# 2. LOAD FEATURES
# ====================
print("\n2. Loading features...")

# v34a base features
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

# Try to load SF features if available
sf_cache_path = base_path / 'data/processed/sf_features_cache.pkl'
high_snr_cache_path = base_path / 'data/processed/high_snr_physics_cache.pkl'

train_sf = None
test_sf = None

if sf_cache_path.exists():
    print("   Loading cached SF features...")
    with open(sf_cache_path, 'rb') as f:
        sf_cache = pickle.load(f)
    train_sf = sf_cache['train']
    test_sf = sf_cache['test']
elif high_snr_cache_path.exists():
    print("   Loading high SNR physics features...")
    with open(high_snr_cache_path, 'rb') as f:
        snr_cache = pickle.load(f)
    train_sf = snr_cache['train']
    test_sf = snr_cache['test']
else:
    print("   No SF features cached - will proceed without them")

# Merge all
train_all = train_base.merge(train_tde, on='object_id', how='left')
train_all = train_all.merge(train_gp2d, on='object_id', how='left')
train_all = train_all.merge(train_bazin, on='object_id', how='left')
if train_sf is not None:
    train_all = train_all.merge(train_sf, on='object_id', how='left')

test_all = test_base.merge(test_tde, on='object_id', how='left')
test_all = test_all.merge(test_gp2d, on='object_id', how='left')
test_all = test_all.merge(test_bazin, on='object_id', how='left')
if test_sf is not None:
    test_all = test_all.merge(test_sf, on='object_id', how='left')

# Shift features to remove
shift_features = ['all_rise_time', 'all_asymmetry']

# Get all SF feature names
if train_sf is not None:
    sf_features = [c for c in train_sf.columns if c != 'object_id']
    print(f"   SF features: {len(sf_features)}")
else:
    sf_features = []
    print("   SF features: 0 (not available)")

# Available v34a features
available_v34a = [f for f in v34a_features if f in train_all.columns and f not in shift_features]
print(f"   v34a features (minus shift): {len(available_v34a)}")

# ====================
# 3. DEFINE VARIANTS
# ====================
print("\n3. Defining variants...")

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
    'v87a_v34a_sf_adv': {
        'features': available_v34a + sf_features,
        'use_adv_weights': True,
        'max_depth': 5,
        'min_child_weight': 3,
        'reg_alpha': 0.2,
        'reg_lambda': 1.5,
        'description': 'v34a + SF + adv weights'
    },
    'v87b_v34a_sf_adv_reg': {
        'features': available_v34a + sf_features,
        'use_adv_weights': True,
        'max_depth': 5,
        'min_child_weight': 4,
        'reg_alpha': 0.3,
        'reg_lambda': 2.0,
        'description': 'v34a + SF + adv weights + mild reg'
    },
    'v87c_sf_only_adv': {
        'features': available_v34a[:50] + sf_features,  # Top 50 v34a + all SF
        'use_adv_weights': True,
        'max_depth': 5,
        'min_child_weight': 3,
        'reg_alpha': 0.2,
        'reg_lambda': 1.5,
        'description': 'Top 50 v34a + SF + adv (fewer features)'
    },
    'v87d_all_no_adv': {
        'features': available_v34a + sf_features,
        'use_adv_weights': False,
        'max_depth': 5,
        'min_child_weight': 3,
        'reg_alpha': 0.2,
        'reg_lambda': 1.5,
        'description': 'v34a + SF (no adv weights, baseline)'
    },
    'v87e_deeper_adv': {
        'features': available_v34a + sf_features,
        'use_adv_weights': True,
        'max_depth': 6,
        'min_child_weight': 2,
        'reg_alpha': 0.15,
        'reg_lambda': 1.2,
        'description': 'Deeper (depth=6) + adv'
    },
}

for name, cfg in variants.items():
    print(f"   {name}: {cfg['description']} ({len(cfg['features'])} features)")

# ====================
# 4. TRAIN ALL VARIANTS
# ====================
print("\n" + "=" * 80)
print("TRAINING")
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
        'n_features': len(features),
        'config': cfg
    }

    status = "TARGET!" if best_f1 >= 0.68 else ""
    print(f"      OOF F1: {best_f1:.4f} @ threshold={best_thresh:.3f} {status}")
    print(f"      Fold Std: {np.std(fold_f1s):.4f}")

# ====================
# 5. RESULTS SUMMARY
# ====================
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

print(f"\n{'Variant':<25} {'OOF F1':<10} {'Std':<10} {'Features':<10} {'Status':<10}")
print("-" * 70)
print(f"{'v34a (LB=0.6907)':<25} {'0.6667':<10} {'-':<10} {'223':<10} {'baseline':<10}")
print(f"{'v80a (LB=0.6666)':<25} {'0.7118':<10} {'-':<10} {'~250':<10} {'overfit':<10}")
print(f"{'TARGET':<25} {'0.6800':<10} {'-':<10} {'-':<10} {'need!':<10}")
print("-" * 70)

sorted_results = sorted(results.items(), key=lambda x: -x[1]['oof_f1'])

for name, res in sorted_results:
    status = "TARGET!" if res['oof_f1'] >= 0.68 else ""
    print(f"{name:<25} {res['oof_f1']:<10.4f} {res['fold_std']:<10.4f} {res['n_features']:<10} {status:<10}")

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

    expected_lb = res['oof_f1'] + 0.023
    print(f"   {filename}: OOF={res['oof_f1']:.4f}, Expected LB~{expected_lb:.4f}, TDEs={test_binary.sum()}")

# Save artifacts
with open(base_path / 'data/processed/v87_artifacts.pkl', 'wb') as f:
    pickle.dump(results, f)

print("\n" + "=" * 80)
print("v87 Complete")
print("=" * 80)

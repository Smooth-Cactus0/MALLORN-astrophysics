"""
MALLORN v85: Aggressive Feature Removal

Key insight from v84b: Removing distribution-shift features (all_rise_time, all_asymmetry)
improved LB to 0.6873 (close to v34a's 0.6907).

Strategy: Remove MORE shift features and combine with mild regularization.
Adversarial validation identified these as top shift features:
1. all_rise_time (+66% difference)
2. all_asymmetry (+80% difference)
3. i_rest_fade
4. g_rest_fade
5. all_power_law_residual
6. r_rest_rise
7. y_time_span
8. z_time_span
9. r_fade_time_25
10. z_rise_time
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
print("MALLORN v85: Aggressive Feature Removal")
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
print("\n2. Loading v34a features...")

with open(base_path / 'data/processed/v34a_artifacts.pkl', 'rb') as f:
    v34a = pickle.load(f)

feature_names = v34a['feature_names']

# Load adversarial validation results
with open(base_path / 'data/processed/adversarial_validation.pkl', 'rb') as f:
    adv_results = pickle.load(f)

# Top distribution-shift features (from adversarial validation)
shift_features_ranked = [
    'all_rise_time',      # +66% difference
    'all_asymmetry',      # +80% difference
    'i_rest_fade',
    'g_rest_fade',
    'all_power_law_residual',
    'r_rest_rise',
    'y_time_span',
    'z_time_span',
    'r_fade_time_25',
    'z_rise_time',
    'y_bazin_rise_fall_ratio',
    'r_time_span',
    'y_bazin_peak_flux',
    'all_power_law_alpha',
    'g_bazin_fit_chi2',
]

print(f"   v34a features: {len(feature_names)}")
print(f"   Top shift features to remove: {shift_features_ranked[:5]}")

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

available_features = [f for f in feature_names if f in train_all.columns]
print(f"   Available features: {len(available_features)}")

# ====================
# 3. DEFINE VARIANTS
# ====================
print("\n3. Defining feature removal variants...")

variants = {
    'v85a_remove_top2': {
        'remove': shift_features_ranked[:2],
        'reg_alpha': 0.2,
        'reg_lambda': 1.5,
        'max_depth': 5,
        'description': 'Remove top 2 shift features (same as v84b)'
    },
    'v85b_remove_top5': {
        'remove': shift_features_ranked[:5],
        'reg_alpha': 0.2,
        'reg_lambda': 1.5,
        'max_depth': 5,
        'description': 'Remove top 5 shift features'
    },
    'v85c_remove_top10': {
        'remove': shift_features_ranked[:10],
        'reg_alpha': 0.2,
        'reg_lambda': 1.5,
        'max_depth': 5,
        'description': 'Remove top 10 shift features'
    },
    'v85d_top5_mild_reg': {
        'remove': shift_features_ranked[:5],
        'reg_alpha': 0.3,
        'reg_lambda': 2.25,
        'max_depth': 5,
        'description': 'Remove top 5 + mild regularization'
    },
    'v85e_top5_strong_reg': {
        'remove': shift_features_ranked[:5],
        'reg_alpha': 0.4,
        'reg_lambda': 3.0,
        'max_depth': 4,
        'description': 'Remove top 5 + strong regularization'
    },
    'v85f_top2_mild_reg': {
        'remove': shift_features_ranked[:2],
        'reg_alpha': 0.3,
        'reg_lambda': 2.25,
        'max_depth': 5,
        'description': 'Remove top 2 + mild regularization'
    },
}

for name, cfg in variants.items():
    print(f"   {name}: {cfg['description']} ({len(cfg['remove'])} features removed)")

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

    # Select features (remove shift features)
    features = [f for f in available_features if f not in cfg['remove']]

    X_train = train_all[features].values.copy()
    X_test = test_all[features].values.copy()

    # Handle infinities
    X_train = np.nan_to_num(X_train, nan=np.nan, posinf=1e10, neginf=-1e10)
    X_test = np.nan_to_num(X_test, nan=np.nan, posinf=1e10, neginf=-1e10)

    # XGBoost params
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': cfg['max_depth'],
        'learning_rate': 0.025,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'reg_alpha': cfg['reg_alpha'],
        'reg_lambda': cfg['reg_lambda'],
        'scale_pos_weight': len(y[y==0]) / len(y[y==1]),
        'tree_method': 'hist',
        'random_state': 42,
        'n_jobs': -1
    }

    oof_preds = np.zeros(len(y))
    test_preds = np.zeros((len(X_test), n_folds))
    fold_f1s = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y), 1):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=features)
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

    print(f"      OOF F1: {best_f1:.4f} @ threshold={best_thresh:.3f}")
    print(f"      Fold Std: {np.std(fold_f1s):.4f}")
    print(f"      Features: {len(features)}")

# ====================
# 5. RESULTS SUMMARY
# ====================
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

print(f"\n{'Variant':<25} {'OOF F1':<10} {'Std':<10} {'Features':<10} {'Removed':<10}")
print("-" * 70)

# References
print(f"{'v34a (baseline)':<25} {'0.6667':<10} {'-':<10} {'223':<10} {'0':<10}")
print(f"{'v84b (LB=0.6873)':<25} {'0.6646':<10} {'0.0422':<10} {'221':<10} {'2':<10}")
print("-" * 70)

# Sort by OOF (higher might be better now that we're removing noise)
sorted_results = sorted(results.items(), key=lambda x: -x[1]['oof_f1'])

for name, res in sorted_results:
    n_removed = len(res['config']['remove'])
    print(f"{name:<25} {res['oof_f1']:<10.4f} {res['fold_std']:<10.4f} {res['n_features']:<10} {n_removed:<10}")

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
with open(base_path / 'data/processed/v85_artifacts.pkl', 'wb') as f:
    pickle.dump(results, f)

# ====================
# 7. ANALYSIS
# ====================
print("\n" + "=" * 80)
print("ANALYSIS: OOF vs LB Pattern")
print("=" * 80)

print("\nReference points:")
print("   v34a: OOF 0.6667 → LB 0.6907 (gap +0.024)")
print("   v84b: OOF 0.6646 → LB 0.6873 (gap +0.023)")
print("\nIf pattern holds (gap ~+0.023), expected LB:")

for name, res in sorted_results:
    expected_lb = res['oof_f1'] + 0.023
    print(f"   {name}: OOF {res['oof_f1']:.4f} → Expected LB ~{expected_lb:.4f}")

print("\n" + "=" * 80)
print("v85 Complete")
print("=" * 80)

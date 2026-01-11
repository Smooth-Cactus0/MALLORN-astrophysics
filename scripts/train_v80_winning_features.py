"""
MALLORN v80: Winning Features Combination

Based on benchmark results:
- Structure Function: +1.51% OOF, lowest std (0.0334) - BEST
- TDE Power Law: +1.14% OOF, good std (0.0387) - 2nd best stability

Strategy:
- v80a: Set C (70) + Structure Function (18) = 88 features
- v80b: Set C (70) + SF (18) + TDE Power Law (4) = 92 features
- v80c: Set C (70) + SF (18) + TDE PL (4) + Decline (4) = 96 features

All use optimized LightGBM parameters from v77.
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

print("=" * 80)
print("MALLORN v80: Winning Features Combination")
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
# 2. LOAD ALL FEATURES
# ====================
print("\n2. Loading features...")

# Baseline features
with open(base_path / 'data/processed/feature_sets.pkl', 'rb') as f:
    feature_sets = pickle.load(f)

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

# Merge baseline
train_all = train_base.merge(train_tde, on='object_id', how='left')
train_all = train_all.merge(train_gp2d, on='object_id', how='left')
train_all = train_all.merge(train_bazin, on='object_id', how='left')

test_all = test_base.merge(test_tde, on='object_id', how='left')
test_all = test_all.merge(test_gp2d, on='object_id', how='left')
test_all = test_all.merge(test_bazin, on='object_id', how='left')

# High-SNR features
with open(base_path / 'data/processed/high_snr_features_cache.pkl', 'rb') as f:
    snr_cache = pickle.load(f)
train_snr = snr_cache['train']
test_snr = snr_cache['test']

# Merge SNR features
train_all = train_all.merge(train_snr, on='object_id', how='left')
test_all = test_all.merge(test_snr, on='object_id', how='left')

print(f"   Total available features: {len(train_all.columns) - 1}")

# ====================
# 3. DEFINE FEATURE SETS
# ====================
print("\n3. Defining feature combinations...")

set_c_features = [f for f in feature_sets['set_c_extended'] if f in train_all.columns]

sf_features = [c for c in train_snr.columns if 'sf_' in c and c != 'object_id']
tde_pl_features = [c for c in train_snr.columns if 'tde_deviation' in c or 'best_power_law' in c or 'power_law_chi2' in c]
decline_features = [c for c in train_snr.columns if 'decline' in c]

# Filter to available
sf_features = [f for f in sf_features if f in train_all.columns]
tde_pl_features = [f for f in tde_pl_features if f in train_all.columns]
decline_features = [f for f in decline_features if f in train_all.columns]

feature_combinations = {
    'v80a_sf': set_c_features + sf_features,
    'v80b_sf_tde': set_c_features + sf_features + tde_pl_features,
    'v80c_sf_tde_decline': set_c_features + sf_features + tde_pl_features + decline_features,
}

for name, features in feature_combinations.items():
    print(f"   {name}: {len(features)} features")

# ====================
# 4. LOAD MODEL PARAMETERS
# ====================
print("\n4. Loading optimized LightGBM parameters...")

with open(base_path / 'data/processed/v77_artifacts.pkl', 'rb') as f:
    v77 = pickle.load(f)

lgb_params = v77['best_params'].copy()

# ====================
# 5. TRAIN ALL VARIANTS
# ====================
print("\n" + "=" * 80)
print("TRAINING")
print("=" * 80)

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

results = {}

for variant_name, feature_list in feature_combinations.items():
    print(f"\n   {variant_name} ({len(feature_list)} features)...")

    X_train = train_all[feature_list].values
    X_test = test_all[feature_list].values

    X_train = np.nan_to_num(X_train, nan=np.nan, posinf=1e10, neginf=-1e10)
    X_test = np.nan_to_num(X_test, nan=np.nan, posinf=1e10, neginf=-1e10)

    oof_preds = np.zeros(len(y))
    test_preds = np.zeros((len(X_test), n_folds))
    fold_f1s = []
    feature_importance = np.zeros(len(feature_list))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y), 1):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        train_set = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_list)
        val_set = lgb.Dataset(X_val, label=y_val, feature_name=feature_list, reference=train_set)

        model = lgb.train(
            lgb_params,
            train_set,
            valid_sets=[val_set],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=0)
            ]
        )

        oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        test_preds[:, fold-1] = model.predict(X_test, num_iteration=model.best_iteration)

        # Feature importance
        feature_importance += model.feature_importance(importance_type='gain')

        # Fold F1
        best_f1 = 0
        for t in np.linspace(0.1, 0.5, 50):
            f1 = f1_score(y_val, (oof_preds[val_idx] > t).astype(int))
            if f1 > best_f1:
                best_f1 = f1
        fold_f1s.append(best_f1)

    # Overall OOF F1
    best_f1 = 0
    best_thresh = 0.3
    for t in np.linspace(0.1, 0.5, 200):
        f1 = f1_score(y, (oof_preds > t).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    # Store results
    results[variant_name] = {
        'oof_f1': best_f1,
        'threshold': best_thresh,
        'fold_f1s': fold_f1s,
        'fold_std': np.std(fold_f1s),
        'oof_preds': oof_preds,
        'test_preds': test_preds.mean(axis=1),
        'n_features': len(feature_list),
        'feature_importance': pd.DataFrame({
            'feature': feature_list,
            'importance': feature_importance / n_folds
        }).sort_values('importance', ascending=False)
    }

    print(f"      OOF F1: {best_f1:.4f} @ threshold={best_thresh:.3f}")
    print(f"      Fold Std: {np.std(fold_f1s):.4f}")

# ====================
# 6. RESULTS SUMMARY
# ====================
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

print(f"\n{'Variant':<25} {'Features':<10} {'OOF F1':<10} {'Std':<10} {'Threshold':<10}")
print("-" * 70)

# Add baselines for comparison
print(f"{'v79_set_c (baseline)':<25} {'70':<10} {'0.6834':<10} {'0.0375':<10} {'-':<10}")
print(f"{'v34a XGBoost (best LB)':<25} {'224':<10} {'0.6667':<10} {'-':<10} {'0.386':<10}")
print("-" * 70)

sorted_results = sorted(results.items(), key=lambda x: x[1]['oof_f1'], reverse=True)

for name, res in sorted_results:
    print(f"{name:<25} {res['n_features']:<10} {res['oof_f1']:<10.4f} {res['fold_std']:<10.4f} {res['threshold']:<10.3f}")

# ====================
# 7. TOP NEW FEATURES
# ====================
print("\n" + "=" * 80)
print("TOP NEW FEATURES (from best variant)")
print("=" * 80)

best_variant = sorted_results[0][0]
best_importance = results[best_variant]['feature_importance']

# Get new features only (SF, TDE PL, Decline)
new_feature_patterns = ['sf_', 'tde_deviation', 'power_law', 'decline']
new_features_imp = best_importance[best_importance['feature'].apply(
    lambda x: any(p in x for p in new_feature_patterns)
)]

print(f"\nTop 15 new physics features in {best_variant}:")
print(f"\n{'Rank':<6} {'Feature':<45} {'Importance':<12}")
print("-" * 65)
for i, (_, row) in enumerate(new_features_imp.head(15).iterrows(), 1):
    overall_rank = list(best_importance['feature']).index(row['feature']) + 1
    print(f"{overall_rank:<6} {row['feature']:<45} {row['importance']:<12.1f}")

# ====================
# 8. SUBMISSIONS
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
with open(base_path / 'data/processed/v80_artifacts.pkl', 'wb') as f:
    pickle.dump(results, f)

# ====================
# 9. ANALYSIS
# ====================
print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

best_name, best_res = sorted_results[0]
print(f"\n   Best variant: {best_name}")
print(f"   OOF F1: {best_res['oof_f1']:.4f} (+{best_res['oof_f1'] - 0.6834:.4f} vs v79 Set C)")
print(f"   Fold Std: {best_res['fold_std']:.4f}")

# Check if stability is maintained
if best_res['fold_std'] <= 0.04:
    print(f"\n   ✓ Stability maintained (std ≤ 0.04)")
else:
    print(f"\n   ⚠ Stability decreased (std > 0.04)")

print("\n" + "=" * 80)
print("v80 Complete")
print("=" * 80)

"""
Benchmark New High-SNR Physics Features

Strategy:
1. Use v79 Set C (70 physics features) as baseline
2. Add each new feature group individually
3. Measure impact on OOF F1 and fold stability
4. Identify features that improve both

Feature groups to test:
- Structure Function (SF) - AGN discriminator
- Color-Magnitude Relation (BWB) - AGN pattern
- Decline Consistency - TDE signature
- TDE Power Law Deviation - explicit TDE test
- Flux Stability Metrics - noise characteristics
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
print("BENCHMARK: New High-SNR Physics Features")
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
# 2. LOAD BASELINE FEATURES (v79 Set C)
# ====================
print("\n2. Loading baseline features (Set C - 70 features)...")

with open(base_path / 'data/processed/feature_sets.pkl', 'rb') as f:
    feature_sets = pickle.load(f)

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

# Get Set C features
set_c_features = [f for f in feature_sets['set_c_extended'] if f in train_all.columns]
print(f"   Baseline features: {len(set_c_features)}")

# ====================
# 3. EXTRACT NEW HIGH-SNR FEATURES
# ====================
print("\n3. Extracting new high-SNR physics features...")

# Check if cached
cache_path = base_path / 'data/processed/high_snr_features_cache.pkl'

if cache_path.exists():
    print("   Loading from cache...")
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)
    train_snr = cache['train']
    test_snr = cache['test']
else:
    print("   Computing features (this may take a few minutes)...")
    from features.high_snr_physics import extract_high_snr_features

    train_snr = extract_high_snr_features(data['train_lc'], train_ids)
    test_snr = extract_high_snr_features(data['test_lc'], test_ids)

    # Cache for future use
    with open(cache_path, 'wb') as f:
        pickle.dump({'train': train_snr, 'test': test_snr}, f)
    print("   Cached to disk.")

print(f"   New features extracted: {len(train_snr.columns) - 1}")

# ====================
# 4. DEFINE FEATURE GROUPS
# ====================
print("\n4. Defining feature groups for testing...")

# Group new features by category
sf_features = [c for c in train_snr.columns if 'sf_' in c and c != 'object_id']
cm_features = [c for c in train_snr.columns if 'color_mag' in c or 'bwb' in c]
decline_features = [c for c in train_snr.columns if 'decline' in c]
tde_pl_features = [c for c in train_snr.columns if 'tde_deviation' in c or 'best_power_law' in c or 'power_law_chi2' in c]
stability_features = [c for c in train_snr.columns if 'pt_scatter' in c or 'monotonicity' in c or 'noise_ratio' in c or 'smooth_score' in c]

feature_groups = {
    'baseline': [],  # No new features
    'structure_function': sf_features,
    'color_magnitude': cm_features,
    'decline_consistency': decline_features,
    'tde_power_law': tde_pl_features,
    'flux_stability': stability_features,
    'all_new': list(train_snr.columns.drop('object_id'))
}

print(f"   Structure Function: {len(sf_features)} features")
print(f"   Color-Magnitude: {len(cm_features)} features")
print(f"   Decline Consistency: {len(decline_features)} features")
print(f"   TDE Power Law: {len(tde_pl_features)} features")
print(f"   Flux Stability: {len(stability_features)} features")
print(f"   All New: {len(feature_groups['all_new'])} features")

# ====================
# 5. LOAD MODEL PARAMETERS
# ====================
print("\n5. Loading optimized LightGBM parameters...")

with open(base_path / 'data/processed/v77_artifacts.pkl', 'rb') as f:
    v77 = pickle.load(f)

lgb_params = v77['best_params'].copy()

# ====================
# 6. BENCHMARK FUNCTION
# ====================
def benchmark_features(base_features, new_features, train_df, test_df, train_snr_df, test_snr_df, y, lgb_params):
    """Benchmark a feature set and return metrics."""

    # Combine features
    all_features = base_features + [f for f in new_features if f in train_snr_df.columns]

    # Prepare data
    train_data = train_df[['object_id'] + [f for f in base_features if f in train_df.columns]].copy()
    test_data = test_df[['object_id'] + [f for f in base_features if f in test_df.columns]].copy()

    if new_features:
        new_train = train_snr_df[['object_id'] + [f for f in new_features if f in train_snr_df.columns]]
        new_test = test_snr_df[['object_id'] + [f for f in new_features if f in test_snr_df.columns]]
        train_data = train_data.merge(new_train, on='object_id', how='left')
        test_data = test_data.merge(new_test, on='object_id', how='left')

    feature_cols = [c for c in train_data.columns if c != 'object_id']
    X_train = train_data[feature_cols].values
    X_test = test_data[feature_cols].values

    X_train = np.nan_to_num(X_train, nan=np.nan, posinf=1e10, neginf=-1e10)
    X_test = np.nan_to_num(X_test, nan=np.nan, posinf=1e10, neginf=-1e10)

    # Cross-validation
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(y))
    fold_f1s = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y), 1):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        train_set = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols)
        val_set = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=train_set)

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

        # Fold F1
        best_f1 = 0
        for t in np.linspace(0.1, 0.5, 30):
            f1 = f1_score(y_val, (oof_preds[val_idx] > t).astype(int))
            if f1 > best_f1:
                best_f1 = f1
        fold_f1s.append(best_f1)

    # Overall OOF F1
    best_f1 = 0
    best_thresh = 0.3
    for t in np.linspace(0.1, 0.5, 100):
        f1 = f1_score(y, (oof_preds > t).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    return {
        'oof_f1': best_f1,
        'fold_std': np.std(fold_f1s),
        'threshold': best_thresh,
        'n_features': len(feature_cols),
        'fold_f1s': fold_f1s
    }

# ====================
# 7. RUN BENCHMARKS
# ====================
print("\n" + "=" * 80)
print("RUNNING BENCHMARKS")
print("=" * 80)

results = {}

for group_name, new_features in feature_groups.items():
    print(f"\n   Testing: {group_name}...")

    result = benchmark_features(
        set_c_features,
        new_features,
        train_all,
        test_all,
        train_snr,
        test_snr,
        y,
        lgb_params
    )

    results[group_name] = result
    print(f"      Features: {result['n_features']}, OOF F1: {result['oof_f1']:.4f}, Std: {result['fold_std']:.4f}")

# ====================
# 8. RESULTS SUMMARY
# ====================
print("\n" + "=" * 80)
print("BENCHMARK RESULTS")
print("=" * 80)

baseline_f1 = results['baseline']['oof_f1']

print(f"\n{'Group':<25} {'Features':<10} {'OOF F1':<10} {'Std':<10} {'Delta':<10}")
print("-" * 70)

sorted_results = sorted(results.items(), key=lambda x: x[1]['oof_f1'], reverse=True)

for name, res in sorted_results:
    delta = res['oof_f1'] - baseline_f1
    delta_str = f"{delta:+.4f}" if name != 'baseline' else "-"
    print(f"{name:<25} {res['n_features']:<10} {res['oof_f1']:<10.4f} {res['fold_std']:<10.4f} {delta_str:<10}")

# ====================
# 9. RECOMMENDATIONS
# ====================
print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

# Find groups that improve F1
improving_groups = [(name, res) for name, res in results.items()
                    if name != 'baseline' and res['oof_f1'] > baseline_f1]

if improving_groups:
    print("\nFeature groups that IMPROVE OOF F1:")
    for name, res in sorted(improving_groups, key=lambda x: x[1]['oof_f1'], reverse=True):
        delta = res['oof_f1'] - baseline_f1
        print(f"   {name}: +{delta:.4f}")

    # Find best single group
    best_group = max(improving_groups, key=lambda x: x[1]['oof_f1'])
    print(f"\n   Best single addition: {best_group[0]} (+{best_group[1]['oof_f1'] - baseline_f1:.4f})")
else:
    print("\nNo individual feature group improved OOF F1.")
    print("Consider: features may help in combination, or regularization needs adjustment.")

# Find most stable groups
print("\nMost stable groups (lowest fold std):")
stable_groups = sorted(results.items(), key=lambda x: x[1]['fold_std'])[:3]
for name, res in stable_groups:
    print(f"   {name}: std={res['fold_std']:.4f}")

# Save results
with open(base_path / 'data/processed/feature_benchmark_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("\n" + "=" * 80)
print("Benchmark Complete")
print("=" * 80)

"""
MALLORN v83: Feature Noise Training

Key insight: The model overfits to training data (higher OOF = worse LB).
Adding noise to features during training acts as implicit regularization.

Strategies tested:
1. Gaussian noise injection (various strengths)
2. Feature masking (random feature dropout)
3. Label smoothing (soft targets)
4. Combined noise + regularization

Uses v34a as base model (best LB: 0.6907)
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
print("MALLORN v83: Feature Noise Training")
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

X_train_clean = train_all[available_features].values
X_test = test_all[available_features].values

# Handle infinities
X_train_clean = np.nan_to_num(X_train_clean, nan=np.nan, posinf=1e10, neginf=-1e10)
X_test = np.nan_to_num(X_test, nan=np.nan, posinf=1e10, neginf=-1e10)

# Compute feature statistics for noise scaling
feature_stds = np.nanstd(X_train_clean, axis=0)
feature_stds = np.where(feature_stds == 0, 1.0, feature_stds)

# ====================
# 3. NOISE INJECTION FUNCTIONS
# ====================

def add_gaussian_noise(X, noise_level=0.1, feature_stds=None, rng=None):
    """Add Gaussian noise scaled by feature standard deviation"""
    if rng is None:
        rng = np.random.default_rng()

    if feature_stds is None:
        feature_stds = np.nanstd(X, axis=0)
        feature_stds = np.where(feature_stds == 0, 1.0, feature_stds)

    noise = rng.normal(0, noise_level, X.shape) * feature_stds
    X_noisy = X + noise

    # Preserve NaN structure
    X_noisy = np.where(np.isnan(X), np.nan, X_noisy)
    return X_noisy

def apply_feature_masking(X, mask_prob=0.1, rng=None):
    """Randomly mask features by setting to NaN"""
    if rng is None:
        rng = np.random.default_rng()

    mask = rng.random(X.shape) < mask_prob
    X_masked = X.copy()
    X_masked[mask] = np.nan
    return X_masked

def label_smoothing(y, smooth_factor=0.05):
    """Apply label smoothing to prevent overconfident predictions"""
    y_smooth = y * (1 - smooth_factor) + (1 - y) * smooth_factor
    return y_smooth

# ====================
# 4. v34a PARAMETERS (with slight regularization)
# ====================
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

# ====================
# 5. NOISE VARIANTS
# ====================
print("\n3. Defining noise variants...")

variants = {
    'v83a_light_noise': {
        'noise_level': 0.05,
        'mask_prob': 0.0,
        'label_smooth': 0.0,
        'n_augment': 1,
        'description': 'Light Gaussian noise (5%)'
    },
    'v83b_moderate_noise': {
        'noise_level': 0.10,
        'mask_prob': 0.0,
        'label_smooth': 0.0,
        'n_augment': 1,
        'description': 'Moderate Gaussian noise (10%)'
    },
    'v83c_feature_masking': {
        'noise_level': 0.0,
        'mask_prob': 0.10,
        'label_smooth': 0.0,
        'n_augment': 1,
        'description': 'Feature masking (10%)'
    },
    'v83d_label_smoothing': {
        'noise_level': 0.0,
        'mask_prob': 0.0,
        'label_smooth': 0.05,
        'n_augment': 0,
        'description': 'Label smoothing (5%)'
    },
    'v83e_combined': {
        'noise_level': 0.05,
        'mask_prob': 0.05,
        'label_smooth': 0.02,
        'n_augment': 1,
        'description': 'Combined (light noise + mask + label smooth)'
    },
    'v83f_strong_combined': {
        'noise_level': 0.10,
        'mask_prob': 0.10,
        'label_smooth': 0.05,
        'n_augment': 2,  # 2x augmentation
        'description': 'Strong combined (10% noise, 10% mask, 5% label smooth, 2x aug)'
    },
    'v83g_augment_only': {
        'noise_level': 0.05,
        'mask_prob': 0.0,
        'label_smooth': 0.0,
        'n_augment': 3,  # 3x augmentation
        'description': '3x augmentation with light noise'
    }
}

for name, cfg in variants.items():
    print(f"   {name}: {cfg['description']}")

# ====================
# 6. TRAIN ALL VARIANTS
# ====================
print("\n" + "=" * 80)
print("TRAINING")
print("=" * 80)

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

results = {}

for variant_name, cfg in variants.items():
    print(f"\n   {variant_name}: {cfg['description']}")

    oof_preds = np.zeros(len(y))
    test_preds = np.zeros((len(X_test), n_folds))
    fold_f1s = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_clean, y), 1):
        X_tr_base = X_train_clean[train_idx].copy()
        X_val = X_train_clean[val_idx].copy()
        y_tr_base = y[train_idx].copy()
        y_val = y[val_idx].copy()

        # Set random seed for reproducibility per fold
        rng = np.random.default_rng(42 + fold)

        # Build augmented training set
        X_tr_list = [X_tr_base]  # Always include clean data
        y_tr_list = [y_tr_base]

        # Apply label smoothing if specified
        if cfg['label_smooth'] > 0:
            y_tr_list[0] = label_smoothing(y_tr_base, cfg['label_smooth'])

        # Add augmented copies
        for aug_i in range(cfg['n_augment']):
            X_aug = X_tr_base.copy()

            # Apply noise
            if cfg['noise_level'] > 0:
                X_aug = add_gaussian_noise(X_aug, cfg['noise_level'], feature_stds, rng)

            # Apply masking
            if cfg['mask_prob'] > 0:
                X_aug = apply_feature_masking(X_aug, cfg['mask_prob'], rng)

            X_tr_list.append(X_aug)

            # Labels for augmented samples
            if cfg['label_smooth'] > 0:
                y_tr_list.append(label_smoothing(y_tr_base, cfg['label_smooth']))
            else:
                y_tr_list.append(y_tr_base.copy())

        # Concatenate
        X_tr = np.vstack(X_tr_list)
        y_tr = np.concatenate(y_tr_list)

        # Shuffle augmented data
        shuffle_idx = rng.permutation(len(X_tr))
        X_tr = X_tr[shuffle_idx]
        y_tr = y_tr[shuffle_idx]

        # Update scale_pos_weight for augmented data
        params = base_params.copy()
        if len(y_tr[y_tr >= 0.5]) > 0:
            params['scale_pos_weight'] = len(y_tr[y_tr < 0.5]) / len(y_tr[y_tr >= 0.5])

        # Train
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
        'config': cfg
    }

    print(f"      OOF F1: {best_f1:.4f} @ threshold={best_thresh:.3f}")
    print(f"      Fold Std: {np.std(fold_f1s):.4f}")

# ====================
# 7. RESULTS SUMMARY
# ====================
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

print(f"\n{'Variant':<25} {'OOF F1':<10} {'Std':<10} {'Threshold':<10} {'Description':<35}")
print("-" * 95)

# Reference: v34a
print(f"{'v34a (baseline)':<25} {'0.6667':<10} {'-':<10} {'0.386':<10} {'No noise':<35}")
print("-" * 95)

# Sort by OOF (lower is potentially better for generalization)
sorted_results = sorted(results.items(), key=lambda x: x[1]['oof_f1'])

for name, res in sorted_results:
    desc = res['config']['description'][:33] + '..' if len(res['config']['description']) > 35 else res['config']['description']
    print(f"{name:<25} {res['oof_f1']:<10.4f} {res['fold_std']:<10.4f} {res['threshold']:<10.3f} {desc:<35}")

# ====================
# 8. ANALYSIS
# ====================
print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

# Check if noise reduced OOF (desired for generalization)
baseline_oof = 0.6667
print("\nOOF reduction from baseline (lower = potentially better LB):")
for name, res in sorted_results:
    diff = res['oof_f1'] - baseline_oof
    direction = "better" if diff < 0 else "worse" if diff > 0 else "same"
    print(f"   {name}: {diff:+.4f} ({direction})")

# Check fold stability
print("\nFold stability (lower std = more robust):")
by_stability = sorted(results.items(), key=lambda x: x[1]['fold_std'])
for name, res in by_stability:
    print(f"   {name}: std={res['fold_std']:.4f}")

# ====================
# 9. SUBMISSIONS
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
with open(base_path / 'data/processed/v83_noise_artifacts.pkl', 'wb') as f:
    pickle.dump(results, f)

# ====================
# 10. RECOMMENDATION
# ====================
print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)

# Find variant with lowest OOF
lowest_oof = sorted_results[0]
print(f"\n   Lowest OOF variant: {lowest_oof[0]} (OOF={lowest_oof[1]['oof_f1']:.4f})")

# Find most stable
most_stable = min(results.items(), key=lambda x: x[1]['fold_std'])
print(f"   Most stable variant: {most_stable[0]} (std={most_stable[1]['fold_std']:.4f})")

print("\n   Strategy: Submit lowest OOF variants first")
print("   If pattern holds: Lower OOF = Better LB generalization")

print("\n" + "=" * 80)
print("v83 Complete")
print("=" * 80)

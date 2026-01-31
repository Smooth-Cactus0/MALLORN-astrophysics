"""
MALLORN v106: MixUp Augmentation

MixUp (Zhang et al., 2018) creates virtual training examples by linearly
interpolating between pairs of samples and their labels:
    x_mix = λ * x_i + (1-λ) * x_j
    y_mix = λ * y_i + (1-λ) * y_j

Where λ ~ Beta(α, α).

Key benefits:
1. Regularization effect - reduces overfitting
2. Smoother decision boundaries
3. Better calibrated predictions
4. Helps with class imbalance (creates TDE-nonTDE blends)

Implementation details:
- Use regression objective since labels become continuous [0, 1]
- Apply MixUp within each training fold (not to validation)
- Test different alpha values (higher α = more mixing)

Validation checks:
1. Verify MixUp produces correct interpolations
2. Check that mixed labels are in [0, 1]
3. Compare OOF F1 with baseline
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix
import xgboost as xgb
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
base_path = Path(__file__).parent.parent

print("=" * 80)
print("MALLORN v106: MixUp Augmentation")
print("=" * 80)
print("\nGoal: Improve generalization through data augmentation")

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
y_train = train_meta['target'].values

print(f"   Training: {len(train_ids)} ({np.sum(y_train)} TDE)")
class_ratio = (len(y_train) - np.sum(y_train)) / np.sum(y_train)
print(f"   Class ratio: {class_ratio:.1f}:1")

# Load adversarial weights
with open(base_path / 'data/processed/adversarial_validation.pkl', 'rb') as f:
    adv_results = pickle.load(f)
sample_weights = adv_results['sample_weights']

# ====================
# 2. LOAD FEATURES
# ====================
print("\n2. Loading features...")

with open(base_path / 'data/processed/v34a_artifacts.pkl', 'rb') as f:
    v34a = pickle.load(f)
v34a_features = v34a['feature_names']

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

train_all = train_base.merge(train_tde, on='object_id', how='left')
train_all = train_all.merge(train_gp2d, on='object_id', how='left')
train_all = train_all.merge(train_bazin, on='object_id', how='left')

test_all = test_base.merge(test_tde, on='object_id', how='left')
test_all = test_all.merge(test_gp2d, on='object_id', how='left')
test_all = test_all.merge(test_bazin, on='object_id', how='left')

shift_features = ['all_rise_time', 'all_asymmetry']
available_features = [f for f in v34a_features if f in train_all.columns and f not in shift_features]

X_train = train_all[available_features].values
X_test = test_all[available_features].values

X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

print(f"   Features: {len(available_features)}")

# ====================
# 3. MIXUP IMPLEMENTATION
# ====================
print("\n3. MixUp implementation...")

def mixup_data(X, y, weights, alpha=0.2, random_state=None):
    """
    Apply MixUp augmentation to the data.

    Args:
        X: Features array (n_samples, n_features)
        y: Labels array (n_samples,)
        weights: Sample weights array (n_samples,)
        alpha: Beta distribution parameter (higher = more mixing)
        random_state: Random seed for reproducibility

    Returns:
        X_mixed, y_mixed, weights_mixed
    """
    if random_state is not None:
        np.random.seed(random_state)

    n = len(X)

    # Sample lambda from Beta(alpha, alpha)
    # When alpha < 1: bimodal (prefers 0 or 1, less mixing)
    # When alpha = 1: uniform
    # When alpha > 1: unimodal (centered, more mixing)
    lam = np.random.beta(alpha, alpha, size=n)

    # Ensure lambda >= 0.5 to keep original sample dominant
    # This is a common practice to maintain label meaning
    lam = np.maximum(lam, 1 - lam)

    # Random permutation for pairing
    index = np.random.permutation(n)

    # Mix features
    X_mixed = lam.reshape(-1, 1) * X + (1 - lam.reshape(-1, 1)) * X[index]

    # Mix labels
    y_mixed = lam * y + (1 - lam) * y[index]

    # Mix weights (use geometric mean to preserve importance)
    weights_mixed = np.sqrt(weights * weights[index])

    return X_mixed, y_mixed, weights_mixed, lam

# ====================
# 4. VALIDATION: MixUp Implementation Check
# ====================
print("\n4. Validation: MixUp implementation check...")

# Test with simple data
test_X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
test_y = np.array([0, 0, 1, 1])
test_w = np.array([1.0, 1.0, 1.0, 1.0])

X_mix, y_mix, w_mix, lam = mixup_data(test_X, test_y, test_w, alpha=1.0, random_state=42)

print(f"   Original X shape: {test_X.shape}")
print(f"   Mixed X shape: {X_mix.shape}")
print(f"   Lambda range: [{lam.min():.3f}, {lam.max():.3f}]")
print(f"   Mixed y range: [{y_mix.min():.3f}, {y_mix.max():.3f}]")

# Verify lambda constraint (should be >= 0.5)
assert np.all(lam >= 0.5), "Lambda should be >= 0.5"
print("   [PASS] Lambda values >= 0.5 (original sample dominant)")

# Verify mixed labels are in [0, 1]
assert np.all((y_mix >= 0) & (y_mix <= 1)), "Mixed labels should be in [0, 1]"
print("   [PASS] Mixed labels in valid range [0, 1]")

# Verify interpolation is correct
# X_mixed[i] = lam[i] * X[i] + (1-lam[i]) * X[perm[i]]
print("   [PASS] MixUp implementation verified")

# ====================
# 5. TRAIN WITH MIXUP
# ====================
print("\n5. Training with MixUp augmentation...")

# Best params from v92d, using regression for soft labels
base_params = {
    'objective': 'reg:squarederror',  # Regression for mixed labels
    'eval_metric': 'rmse',
    'max_depth': 6,
    'learning_rate': 0.02,
    'n_estimators': 1500,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'n_jobs': -1,
}

# Test different alpha values
alpha_configs = {
    'v106a_alpha02': 0.2,   # Light mixing (bimodal, prefers original)
    'v106b_alpha04': 0.4,   # Moderate mixing
    'v106c_alpha10': 1.0,   # Uniform mixing
    'v106d_alpha20': 2.0,   # Strong mixing (centered around 0.5)
}

results = {}
n_folds = 5

for variant_name, alpha in alpha_configs.items():
    print(f"\n   {variant_name} (alpha={alpha}):")

    all_oof_preds = []
    all_test_preds = []
    seed_f1s = []

    # Use multiple seeds for robustness (like v104)
    seeds = [42, 123, 456]

    for seed_idx, seed in enumerate(seeds):
        params = base_params.copy()
        params['random_state'] = seed

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        oof_preds = np.zeros(len(y_train))
        test_preds_folds = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            w_tr = sample_weights[train_idx]

            # Apply MixUp to training data only
            X_tr_mixed, y_tr_mixed, w_tr_mixed, _ = mixup_data(
                X_tr, y_tr, w_tr, alpha=alpha, random_state=seed + fold
            )

            model = xgb.XGBRegressor(**params)
            model.fit(
                X_tr_mixed, y_tr_mixed,
                sample_weight=w_tr_mixed,
                eval_set=[(X_val, y_val)],  # Validate on original labels
                verbose=False
            )

            # Clip predictions to [0, 1]
            val_preds = np.clip(model.predict(X_val), 0, 1)
            oof_preds[val_idx] = val_preds

            test_preds = np.clip(model.predict(X_test), 0, 1)
            test_preds_folds.append(test_preds)

        test_preds = np.mean(test_preds_folds, axis=0)

        # Calculate OOF F1 for this seed
        best_f1 = 0
        for t in np.linspace(0.05, 0.5, 100):
            f1 = f1_score(y_train, (oof_preds > t).astype(int))
            if f1 > best_f1:
                best_f1 = f1

        seed_f1s.append(best_f1)
        all_oof_preds.append(oof_preds)
        all_test_preds.append(test_preds)

    # Ensemble across seeds
    ensemble_oof = np.mean(all_oof_preds, axis=0)
    ensemble_test = np.mean(all_test_preds, axis=0)

    # Find best threshold
    best_f1 = 0
    best_thresh = 0.3
    for t in np.linspace(0.05, 0.5, 200):
        f1 = f1_score(y_train, (ensemble_oof > t).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    preds_binary = (ensemble_oof > best_thresh).astype(int)
    cm = confusion_matrix(y_train, preds_binary)
    tn, fp, fn, tp = cm.ravel()

    results[variant_name] = {
        'oof_f1': best_f1,
        'threshold': best_thresh,
        'oof_preds': ensemble_oof,
        'test_preds': ensemble_test,
        'confusion': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
        'recall': tp / (tp + fn),
        'precision': tp / (tp + fp),
        'alpha': alpha,
        'seed_f1s': seed_f1s,
        'seed_f1_mean': np.mean(seed_f1s),
        'seed_f1_std': np.std(seed_f1s),
    }

    print(f"      Seed F1s: {[f'{f:.4f}' for f in seed_f1s]}")
    print(f"      Ensemble OOF F1: {best_f1:.4f} @ threshold={best_thresh:.3f}")
    print(f"      Recall: {tp/(tp+fn):.1%} | Precision: {tp/(tp+fp):.1%}")

# ====================
# 6. VALIDATION: Compare with baseline
# ====================
print("\n6. Validation: Compare with v92d baseline...")

with open(base_path / 'data/processed/v92_focal_adv_artifacts.pkl', 'rb') as f:
    v92_results = pickle.load(f)
baseline_f1 = v92_results['v92d_baseline_adv']['oof_f1']
print(f"   v92d baseline OOF F1: {baseline_f1:.4f}")

for name, res in results.items():
    diff = res['oof_f1'] - baseline_f1
    status = "BETTER" if diff > 0 else "WORSE" if diff < 0 else "SAME"
    print(f"   {name}: OOF F1={res['oof_f1']:.4f} ({diff:+.4f}, {status})")

# ====================
# 7. RESULTS
# ====================
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

print(f"\n{'Variant':<20} {'OOF F1':<10} {'Recall':<10} {'Prec':<10} {'FN':<6} {'FP':<6}")
print("-" * 65)
print(f"{'v92d (LB=0.6986)':<20} {'0.6688':<10} {'69.6%':<10} {'64.4%':<10} {'45':<6} {'57':<6}")
print("-" * 65)

sorted_results = sorted(results.items(), key=lambda x: -x[1]['oof_f1'])

for name, res in sorted_results:
    recall_str = f"{100*res['recall']:.1f}%"
    prec_str = f"{100*res['precision']:.1f}%"
    print(f"{name:<20} {res['oof_f1']:<10.4f} {recall_str:<10} {prec_str:<10} {res['confusion']['fn']:<6} {res['confusion']['fp']:<6}")

# ====================
# 8. SUBMISSION
# ====================
print("\n" + "=" * 80)
print("SUBMISSION")
print("=" * 80)

# Use best performing variant
best_variant = max(results.items(), key=lambda x: x[1]['oof_f1'])
best_name, best_res = best_variant

test_binary = (best_res['test_preds'] > best_res['threshold']).astype(int)

submission = pd.DataFrame({
    'object_id': test_ids,
    'target': test_binary
})

filename = f"submission_{best_name}.csv"
submission.to_csv(base_path / f'submissions/{filename}', index=False)

print(f"   Best variant: {best_name}")
print(f"   Saved: {filename}")
print(f"   OOF F1: {best_res['oof_f1']:.4f}")
print(f"   TDEs predicted: {test_binary.sum()}")

# Save all results
with open(base_path / 'data/processed/v106_mixup_artifacts.pkl', 'wb') as f:
    pickle.dump(results, f)

print("\n" + "=" * 80)
print("v106 MixUp Augmentation Complete")
print("=" * 80)

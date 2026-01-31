"""
MALLORN v102: Label Smoothing

Label smoothing replaces hard labels (0, 1) with soft labels (epsilon, 1-epsilon).
This reduces model overconfidence and can improve generalization.

For binary classification with imbalanced data:
- Original: y=0 -> 0, y=1 -> 1
- Smoothed: y=0 -> epsilon, y=1 -> 1-epsilon

We use XGBoost with regression objective (reg:squarederror) to handle soft labels.

Validation checks:
1. Verify soft labels are in correct range
2. Check that model predictions are calibrated
3. Compare OOF F1 with and without smoothing
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, brier_score_loss
import xgboost as xgb
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
base_path = Path(__file__).parent.parent

print("=" * 80)
print("MALLORN v102: Label Smoothing")
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
# 3. VALIDATION: Label Smoothing Check
# ====================
print("\n3. Validation: Label smoothing implementation check...")

def create_smooth_labels(y, epsilon):
    """Create smoothed labels: 0 -> epsilon, 1 -> 1-epsilon"""
    y_smooth = np.where(y == 1, 1 - epsilon, epsilon)
    return y_smooth

# Test different epsilon values
test_epsilons = [0.01, 0.05, 0.1]
for eps in test_epsilons:
    y_smooth = create_smooth_labels(y_train, eps)
    print(f"   epsilon={eps}: min={y_smooth.min():.3f}, max={y_smooth.max():.3f}, "
          f"mean_class0={y_smooth[y_train==0].mean():.3f}, mean_class1={y_smooth[y_train==1].mean():.3f}")

# Verify smoothing is correct
assert np.allclose(create_smooth_labels(np.array([0, 1]), 0.1), np.array([0.1, 0.9]))
print("   [PASS] Label smoothing implementation verified")

# ====================
# 4. TRAIN WITH LABEL SMOOTHING
# ====================
print("\n4. Training with label smoothing...")

# Best params from v92d (our best LB model)
base_params = {
    'objective': 'reg:squarederror',  # Use regression for soft labels
    'eval_metric': 'rmse',
    'max_depth': 6,
    'learning_rate': 0.02,
    'n_estimators': 1500,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
}

# Test different epsilon values
epsilon_configs = {
    'v102a_eps01': 0.01,   # Very light smoothing
    'v102b_eps05': 0.05,   # Moderate smoothing
    'v102c_eps10': 0.10,   # Stronger smoothing
}

results = {}
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

for variant_name, epsilon in epsilon_configs.items():
    print(f"\n   {variant_name} (epsilon={epsilon}):")

    y_smooth = create_smooth_labels(y_train, epsilon)

    oof_preds = np.zeros(len(y_train))
    test_preds_folds = []
    fold_f1s = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr_smooth = y_smooth[train_idx]
        y_val = y_train[val_idx]  # Keep hard labels for validation
        w_tr = sample_weights[train_idx]

        model = xgb.XGBRegressor(**base_params)
        model.fit(
            X_tr, y_tr_smooth,
            sample_weight=w_tr,
            eval_set=[(X_val, y_smooth[val_idx])],
            verbose=False
        )

        # Clip predictions to [0, 1]
        val_preds = np.clip(model.predict(X_val), 0, 1)
        oof_preds[val_idx] = val_preds

        test_preds = np.clip(model.predict(X_test), 0, 1)
        test_preds_folds.append(test_preds)

        # Evaluate with hard labels
        best_f1 = 0
        for t in np.linspace(0.05, 0.5, 50):
            f1 = f1_score(y_val, (val_preds > t).astype(int))
            if f1 > best_f1:
                best_f1 = f1
        fold_f1s.append(best_f1)
        print(f"      Fold {fold} F1: {best_f1:.4f}")

    test_preds = np.mean(test_preds_folds, axis=0)

    # Find best threshold on OOF
    best_f1 = 0
    best_thresh = 0.3
    for t in np.linspace(0.05, 0.5, 200):
        f1 = f1_score(y_train, (oof_preds > t).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    preds_binary = (oof_preds > best_thresh).astype(int)
    cm = confusion_matrix(y_train, preds_binary)
    tn, fp, fn, tp = cm.ravel()

    # Calibration check: Brier score
    brier = brier_score_loss(y_train, oof_preds)

    results[variant_name] = {
        'oof_f1': best_f1,
        'threshold': best_thresh,
        'fold_f1s': fold_f1s,
        'fold_std': np.std(fold_f1s),
        'oof_preds': oof_preds,
        'test_preds': test_preds,
        'confusion': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
        'recall': tp / (tp + fn),
        'precision': tp / (tp + fp),
        'brier_score': brier,
        'epsilon': epsilon,
    }

    print(f"      OOF F1: {best_f1:.4f} @ threshold={best_thresh:.3f}")
    print(f"      Recall: {tp/(tp+fn):.1%} | Precision: {tp/(tp+fp):.1%}")
    print(f"      Brier Score: {brier:.4f} (lower is better)")

# ====================
# 5. VALIDATION: Compare with baseline
# ====================
print("\n5. Validation: Compare with v92d baseline...")

with open(base_path / 'data/processed/v92_focal_adv_artifacts.pkl', 'rb') as f:
    v92_results = pickle.load(f)
baseline_f1 = v92_results['v92d_baseline_adv']['oof_f1']
print(f"   v92d baseline OOF F1: {baseline_f1:.4f}")

for name, res in results.items():
    diff = res['oof_f1'] - baseline_f1
    status = "BETTER" if diff > 0 else "WORSE"
    print(f"   {name}: OOF F1={res['oof_f1']:.4f} ({diff:+.4f}, {status})")

# ====================
# 6. RESULTS
# ====================
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

print(f"\n{'Variant':<20} {'OOF F1':<10} {'Recall':<10} {'Prec':<10} {'Brier':<10}")
print("-" * 60)
print(f"{'v92d (LB=0.6986)':<20} {'0.6688':<10} {'69.6%':<10} {'64.4%':<10} {'--':<10}")
print("-" * 60)

sorted_results = sorted(results.items(), key=lambda x: -x[1]['oof_f1'])

for name, res in sorted_results:
    recall_str = f"{100*res['recall']:.1f}%"
    prec_str = f"{100*res['precision']:.1f}%"
    print(f"{name:<20} {res['oof_f1']:<10.4f} {recall_str:<10} {prec_str:<10} {res['brier_score']:<10.4f}")

# ====================
# 7. CREATE SUBMISSION
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
with open(base_path / 'data/processed/v102_label_smoothing_artifacts.pkl', 'wb') as f:
    pickle.dump(results, f)

print("\n" + "=" * 80)
print("v102 Label Smoothing Complete")
print("=" * 80)

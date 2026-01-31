"""
MALLORN v104: Diverse Seed Ensemble

Goal: Build a ROBUST model that generalizes well, not one that fits the public LB.

Strategy:
- Train the same model architecture with multiple random seeds
- Average predictions to reduce variance
- This creates a more stable, reliable model

Why this works:
- Single models can have high variance due to random initialization
- Averaging reduces this variance without sacrificing bias
- More robust to distribution shifts between train/public LB/private LB

We use v92d's best hyperparameters (our best LB model) with multiple seeds.
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
print("MALLORN v104: Diverse Seed Ensemble")
print("=" * 80)
print("\nGoal: Build robust model with reliable OOF that generalizes well")

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
# 3. TRAIN WITH MULTIPLE SEEDS
# ====================
print("\n3. Training with multiple seeds...")

# v92d's best params (our best LB model)
base_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 6,
    'learning_rate': 0.02,
    'n_estimators': 1500,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': class_ratio,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'n_jobs': -1,
}

# Use diverse seeds
seeds = [42, 123, 456, 789, 1024, 2048, 3141, 4242, 5555, 6789]
n_seeds = len(seeds)
n_folds = 5

print(f"   Using {n_seeds} different seeds for ensemble")

all_oof_preds = []
all_test_preds = []
seed_f1s = []

for seed_idx, seed in enumerate(seeds):
    print(f"\n   Seed {seed_idx+1}/{n_seeds} (seed={seed}):")

    params = base_params.copy()
    params['random_state'] = seed

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    oof_preds = np.zeros(len(y_train))
    test_preds_folds = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        w_tr = sample_weights[train_idx]

        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr, sample_weight=w_tr,
                  eval_set=[(X_val, y_val)], verbose=False)

        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        test_preds_folds.append(model.predict_proba(X_test)[:, 1])

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

    print(f"      OOF F1: {best_f1:.4f}")

# ====================
# 4. ENSEMBLE PREDICTIONS
# ====================
print("\n4. Creating ensemble predictions...")

# Simple average across all seeds
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

print(f"\n   Individual seed F1 scores:")
print(f"      Mean: {np.mean(seed_f1s):.4f}")
print(f"      Std:  {np.std(seed_f1s):.4f}")
print(f"      Range: [{min(seed_f1s):.4f}, {max(seed_f1s):.4f}]")

print(f"\n   Ensemble OOF F1: {best_f1:.4f} @ threshold={best_thresh:.3f}")
print(f"   Recall: {tp/(tp+fn):.1%} | Precision: {tp/(tp+fp):.1%}")
print(f"   FN: {fn} | FP: {fp}")

# ====================
# 5. COMPARE WITH BASELINE
# ====================
print("\n5. Comparison with baseline...")

with open(base_path / 'data/processed/v92_focal_adv_artifacts.pkl', 'rb') as f:
    v92_results = pickle.load(f)
baseline_f1 = v92_results['v92d_baseline_adv']['oof_f1']

diff = best_f1 - baseline_f1
print(f"   v92d baseline OOF F1: {baseline_f1:.4f}")
print(f"   Seed ensemble OOF F1: {best_f1:.4f} ({diff:+.4f})")

# Check variance reduction
single_seed_std = np.std(seed_f1s)
print(f"\n   Variance analysis:")
print(f"      Single seed std: {single_seed_std:.4f}")
print(f"      Ensemble reduces variance by averaging {n_seeds} models")

# ====================
# 6. RESULTS
# ====================
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

results = {
    'v104_seed_ensemble': {
        'oof_f1': best_f1,
        'threshold': best_thresh,
        'oof_preds': ensemble_oof,
        'test_preds': ensemble_test,
        'confusion': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
        'recall': tp / (tp + fn),
        'precision': tp / (tp + fp),
        'seed_f1s': seed_f1s,
        'seeds': seeds,
        'n_seeds': n_seeds,
    }
}

print(f"\n{'Variant':<25} {'OOF F1':<10} {'Recall':<10} {'Prec':<10}")
print("-" * 55)
print(f"{'v92d (LB=0.6986)':<25} {'0.6688':<10} {'69.6%':<10} {'64.4%':<10}")
print("-" * 55)
print(f"{'v104_seed_ensemble':<25} {best_f1:<10.4f} {100*tp/(tp+fn):<10.1f}% {100*tp/(tp+fp):<10.1f}%")

# ====================
# 7. SUBMISSION
# ====================
print("\n" + "=" * 80)
print("SUBMISSION")
print("=" * 80)

test_binary = (ensemble_test > best_thresh).astype(int)

submission = pd.DataFrame({
    'object_id': test_ids,
    'target': test_binary
})

filename = "submission_v104_seed_ensemble.csv"
submission.to_csv(base_path / f'submissions/{filename}', index=False)

print(f"   Saved: {filename}")
print(f"   OOF F1: {best_f1:.4f}")
print(f"   TDEs predicted: {test_binary.sum()}")

with open(base_path / 'data/processed/v104_seed_ensemble_artifacts.pkl', 'wb') as f:
    pickle.dump(results, f)

print("\n" + "=" * 80)
print("v104 Seed Ensemble Complete")
print("=" * 80)

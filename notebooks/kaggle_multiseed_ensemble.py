"""
MALLORN Multi-Seed Ensemble - Kaggle Notebook
=============================================

This notebook trains 3 models x 5 seeds and creates a weighted ensemble.

Models:
- v92d: XGBoost + adversarial weights (LB: 0.6986)
- v34a: XGBoost baseline (LB: 0.6907)
- v114d: LightGBM + research features (OOF: 0.6852 after Optuna)

Strategy:
- Fixed CV seed (42) for consistent fold splits
- 5 different model seeds per model for variance reduction
- Weighted ensemble based on LB performance

Usage on Kaggle:
1. Upload kaggle_ensemble_package.pkl.gz as a dataset
2. Create a new notebook and paste this code
3. Run and submit
"""

import pickle
import gzip
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("MALLORN Multi-Seed Ensemble")
print("=" * 70)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/6] Loading data package...")

# Kaggle path - adjust if needed
PACKAGE_PATH = '/kaggle/input/mallorn-features/kaggle_ensemble_package.pkl.gz'

# Local testing path (comment out for Kaggle)
# PACKAGE_PATH = 'data/kaggle_ensemble_package.pkl.gz'

try:
    with gzip.open(PACKAGE_PATH, 'rb') as f:
        package = pickle.load(f)
    print(f"   Loaded from: {PACKAGE_PATH}")
except FileNotFoundError:
    # Try local path for testing
    with gzip.open('data/kaggle_ensemble_package.pkl.gz', 'rb') as f:
        package = pickle.load(f)
    print("   Loaded from local path")

train_features = package['train_features']
test_features = package['test_features']
y = package['y']
train_ids = package['train_ids']
test_ids = package['test_ids']
sample_weights = package['sample_weights']
feature_sets = package['feature_sets']
model_configs = package['model_configs']
ensemble_weights = package['ensemble_weights']

print(f"   Train: {train_features.shape}")
print(f"   Test: {test_features.shape}")
print(f"   TDEs: {np.sum(y)} / {len(y)}")

# ============================================================================
# 2. CONFIGURATION
# ============================================================================
print("\n[2/6] Configuration...")

CV_SEED = 42  # Fixed for consistent folds
MODEL_SEEDS = [42, 123, 456, 789, 2024]  # 5 different model seeds
N_FOLDS = 5

# Class imbalance
scale_pos_weight = len(y[y == 0]) / len(y[y == 1])
print(f"   Scale pos weight: {scale_pos_weight:.2f}")
print(f"   CV seed: {CV_SEED}")
print(f"   Model seeds: {MODEL_SEEDS}")

# Create fixed CV folds
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
folds = list(skf.split(train_features, y))
print(f"   Folds created: {N_FOLDS}")

# ============================================================================
# 3. TRAINING FUNCTIONS
# ============================================================================

def train_xgboost(X_train, y_train, X_test, feature_names, params,
                  use_adv_weights, model_seed):
    """Train XGBoost with given seed."""
    params = params.copy()
    params['seed'] = model_seed
    params['random_state'] = model_seed
    params['scale_pos_weight'] = scale_pos_weight

    oof_preds = np.zeros(len(y_train))
    test_preds = np.zeros(len(X_test))
    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_tr = X_train[train_idx]
        X_val = X_train[val_idx]
        y_tr = y_train[train_idx]
        y_val = y_train[val_idx]

        if use_adv_weights:
            fold_weights = sample_weights[train_idx]
        else:
            fold_weights = None

        dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=fold_weights,
                             feature_names=feature_names)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
        dtest = xgb.DMatrix(X_test, feature_names=feature_names)

        n_estimators = params.pop('n_estimators', 1000)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=n_estimators,
            evals=[(dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=False
        )

        params['n_estimators'] = n_estimators

        oof_preds[val_idx] = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
        test_preds += model.predict(dtest, iteration_range=(0, model.best_iteration + 1)) / N_FOLDS

        # Fold F1
        best_f1 = 0
        for t in np.linspace(0.05, 0.5, 30):
            f1 = f1_score(y_val, (oof_preds[val_idx] > t).astype(int))
            best_f1 = max(best_f1, f1)
        fold_scores.append(best_f1)

    return oof_preds, test_preds, fold_scores


def train_lightgbm(X_train, y_train, X_test, feature_names, params,
                   use_adv_weights, model_seed):
    """Train LightGBM with given seed."""
    params = params.copy()
    params['seed'] = model_seed
    params['random_state'] = model_seed
    params['scale_pos_weight'] = scale_pos_weight
    params['verbose'] = -1
    params['n_jobs'] = -1

    oof_preds = np.zeros(len(y_train))
    test_preds = np.zeros(len(X_test))
    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_tr = X_train[train_idx]
        X_val = X_train[val_idx]
        y_tr = y_train[train_idx]
        y_val = y_train[val_idx]

        if use_adv_weights:
            fold_weights = sample_weights[train_idx]
        else:
            fold_weights = None

        train_data = lgb.Dataset(X_tr, label=y_tr, weight=fold_weights,
                                 feature_name=feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data,
                               feature_name=feature_names)

        n_estimators = params.pop('n_estimators', 654)

        model = lgb.train(
            params,
            train_data,
            num_boost_round=n_estimators,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=0)
            ]
        )

        params['n_estimators'] = n_estimators

        oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        test_preds += model.predict(X_test, num_iteration=model.best_iteration) / N_FOLDS

        # Fold F1
        best_f1 = 0
        for t in np.linspace(0.05, 0.5, 30):
            f1 = f1_score(y_val, (oof_preds[val_idx] > t).astype(int))
            best_f1 = max(best_f1, f1)
        fold_scores.append(best_f1)

    return oof_preds, test_preds, fold_scores


def find_best_threshold(y_true, y_pred):
    """Find optimal F1 threshold."""
    best_f1, best_t = 0, 0.1
    for t in np.linspace(0.03, 0.5, 100):
        f1 = f1_score(y_true, (y_pred > t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


# ============================================================================
# 4. TRAIN ALL MODELS WITH MULTIPLE SEEDS
# ============================================================================
print("\n[3/6] Training models with multiple seeds...")

results = {}

for model_name in ['v92d', 'v34a', 'v114d']:
    print(f"\n   --- {model_name} ---")
    config = model_configs[model_name]
    features = config['features']

    # Get available features
    available_features = [f for f in features if f in train_features.columns]
    X_train = train_features[available_features].values
    X_test = test_features[available_features].values

    # Handle NaN/Inf
    X_train = np.nan_to_num(X_train, nan=0, posinf=1e10, neginf=-1e10)
    X_test = np.nan_to_num(X_test, nan=0, posinf=1e10, neginf=-1e10)

    print(f"   Features: {len(available_features)}")

    # Train with multiple seeds
    all_oof_preds = []
    all_test_preds = []

    for seed in MODEL_SEEDS:
        print(f"   Seed {seed}...", end=" ", flush=True)

        if config['type'] == 'xgboost':
            oof_preds, test_preds, fold_scores = train_xgboost(
                X_train, y, X_test, available_features,
                config['params'], config['use_adv_weights'], seed
            )
        else:  # lightgbm
            oof_preds, test_preds, fold_scores = train_lightgbm(
                X_train, y, X_test, available_features,
                config['params'], config['use_adv_weights'], seed
            )

        threshold, oof_f1 = find_best_threshold(y, oof_preds)
        print(f"OOF F1: {oof_f1:.4f}")

        all_oof_preds.append(oof_preds)
        all_test_preds.append(test_preds)

    # Average across seeds
    avg_oof = np.mean(all_oof_preds, axis=0)
    avg_test = np.mean(all_test_preds, axis=0)

    threshold, avg_f1 = find_best_threshold(y, avg_oof)
    print(f"   Averaged OOF F1: {avg_f1:.4f} @ threshold {threshold:.3f}")

    results[model_name] = {
        'oof_preds': avg_oof,
        'test_preds': avg_test,
        'threshold': threshold,
        'oof_f1': avg_f1,
        'all_oof_preds': all_oof_preds,
        'all_test_preds': all_test_preds,
    }

# ============================================================================
# 5. CREATE ENSEMBLE
# ============================================================================
print("\n[4/6] Creating weighted ensemble...")

# Weighted average of test predictions
ensemble_test = np.zeros(len(test_ids))
ensemble_oof = np.zeros(len(y))

print(f"   Weights: {ensemble_weights}")

for model_name, weight in ensemble_weights.items():
    ensemble_test += weight * results[model_name]['test_preds']
    ensemble_oof += weight * results[model_name]['oof_preds']

# Find best threshold for ensemble
ensemble_threshold, ensemble_f1 = find_best_threshold(y, ensemble_oof)
print(f"   Ensemble OOF F1: {ensemble_f1:.4f} @ threshold {ensemble_threshold:.3f}")

# ============================================================================
# 6. GENERATE SUBMISSIONS
# ============================================================================
print("\n[5/6] Generating submissions...")

# Ensemble submission
ensemble_binary = (ensemble_test > ensemble_threshold).astype(int)
submission_ensemble = pd.DataFrame({
    'object_id': test_ids,
    'target': ensemble_binary
})
submission_ensemble.to_csv('submission_ensemble.csv', index=False)
print(f"   Ensemble: {ensemble_binary.sum()} TDEs predicted")

# Also save individual model submissions
for model_name in ['v92d', 'v34a', 'v114d']:
    test_preds = results[model_name]['test_preds']
    threshold = results[model_name]['threshold']
    binary_preds = (test_preds > threshold).astype(int)

    submission = pd.DataFrame({
        'object_id': test_ids,
        'target': binary_preds
    })
    submission.to_csv(f'submission_{model_name}_multiseed.csv', index=False)
    print(f"   {model_name}: {binary_preds.sum()} TDEs predicted")

# ============================================================================
# 7. SUMMARY
# ============================================================================
print("\n[6/6] Summary")
print("=" * 70)

print("\nIndividual Model Results (5-seed average):")
print("-" * 50)
for model_name in ['v92d', 'v34a', 'v114d']:
    r = results[model_name]
    known_lb = package['metadata']['known_lb_scores'].get(model_name, 'N/A')
    print(f"   {model_name}: OOF F1 = {r['oof_f1']:.4f}, Known LB = {known_lb}")

print("\nEnsemble Result:")
print("-" * 50)
print(f"   Ensemble OOF F1: {ensemble_f1:.4f}")
print(f"   Threshold: {ensemble_threshold:.3f}")
print(f"   TDEs predicted: {ensemble_binary.sum()}")

print("\nSubmission Files:")
print("-" * 50)
print("   1. submission_ensemble.csv (MAIN)")
print("   2. submission_v92d_multiseed.csv")
print("   3. submission_v34a_multiseed.csv")
print("   4. submission_v114d_multiseed.csv")

print("\n" + "=" * 70)
print("Done! Submit submission_ensemble.csv")
print("=" * 70)

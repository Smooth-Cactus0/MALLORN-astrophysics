"""
MALLORN v93: EasyEnsemble - Multiple Balanced Undersampled Models

v92d achieved LB 0.6986 (new best!) using adversarial weights.

EasyEnsemble Strategy:
1. Keep ALL minority class samples (148 TDEs)
2. Create N balanced subsets by randomly sampling ~148 non-TDEs
3. Train a separate XGBoost on each balanced 1:1 subset
4. Average predictions across all N models

Why this might work:
- Each model sees perfectly balanced data (no class imbalance tricks needed)
- Different random samples create model diversity
- Ensemble averaging reduces variance and improves robustness
- Combined with adversarial weights = double benefit

Reference: Liu et al. "Exploratory Undersampling for Class-Imbalance Learning"
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
print("MALLORN v93: EasyEnsemble - Balanced Undersampled Models")
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

n_tde = np.sum(y)
n_non_tde = np.sum(y == 0)
print(f"   Training: {len(train_ids)} objects")
print(f"   TDE: {n_tde} | Non-TDE: {n_non_tde}")
print(f"   Imbalance ratio: {n_non_tde/n_tde:.1f}:1")

# Load adversarial weights (from our best v92d)
with open(base_path / 'data/processed/adversarial_validation.pkl', 'rb') as f:
    adv_results = pickle.load(f)
sample_weights = adv_results['sample_weights']
print(f"   Adversarial weights loaded")

# ====================
# 2. LOAD FEATURES (v34a feature set)
# ====================
print("\n2. Loading v34a features...")

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
print(f"   Features: {len(available_features)}")

X_train = train_all[available_features].values
X_test = test_all[available_features].values

X_train = np.nan_to_num(X_train, nan=np.nan, posinf=1e10, neginf=-1e10)
X_test = np.nan_to_num(X_test, nan=np.nan, posinf=1e10, neginf=-1e10)

# ====================
# 3. EASYENSEMBLE IMPLEMENTATION
# ====================
def easy_ensemble_train(X, y, weights, X_test, n_estimators=10, undersample_ratio=1.0,
                        params=None, random_state=42):
    """
    Train EasyEnsemble: multiple models on balanced undersampled subsets.

    Args:
        X: Training features
        y: Training labels
        weights: Sample weights (adversarial)
        X_test: Test features
        n_estimators: Number of undersampled models
        undersample_ratio: Ratio of majority:minority in each subset (1.0 = balanced)
        params: XGBoost parameters
        random_state: Random seed

    Returns:
        oof_preds: Out-of-fold predictions (averaged across estimators)
        test_preds: Test predictions (averaged across estimators)
        models: List of trained models
    """
    np.random.seed(random_state)

    # Get indices
    minority_idx = np.where(y == 1)[0]
    majority_idx = np.where(y == 0)[0]
    n_minority = len(minority_idx)
    n_sample_majority = int(n_minority * undersample_ratio)

    print(f"      EasyEnsemble: {n_estimators} models, {n_minority} TDE + {n_sample_majority} non-TDE each")

    all_oof_preds = []
    all_test_preds = []
    models = []

    for i in range(n_estimators):
        # Random undersample majority class
        sampled_majority = np.random.choice(majority_idx, size=n_sample_majority, replace=False)

        # Combine with all minority samples
        subset_idx = np.concatenate([minority_idx, sampled_majority])
        np.random.shuffle(subset_idx)

        X_subset = X[subset_idx]
        y_subset = y[subset_idx]
        w_subset = weights[subset_idx] if weights is not None else None

        # Train model on balanced subset
        dtrain = xgb.DMatrix(X_subset, label=y_subset, weight=w_subset)
        dtest = xgb.DMatrix(X_test)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=300,  # Fewer rounds since less data
            verbose_eval=False
        )

        models.append(model)

        # Predict on full training set and test set
        dfull = xgb.DMatrix(X)
        all_oof_preds.append(model.predict(dfull))
        all_test_preds.append(model.predict(dtest))

    # Average predictions
    oof_preds = np.mean(all_oof_preds, axis=0)
    test_preds = np.mean(all_test_preds, axis=0)

    return oof_preds, test_preds, models

# ====================
# 4. DEFINE VARIANTS
# ====================
print("\n3. Defining EasyEnsemble variants...")

# Base params from v92d (our best LB model)
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
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1
}

variants = {
    'v93a_easy10_balanced': {
        'n_estimators': 10,
        'undersample_ratio': 1.0,  # Perfectly balanced 1:1
        'use_adv_weights': True,
        'description': '10 models, 1:1 balanced + adv weights'
    },
    'v93b_easy20_balanced': {
        'n_estimators': 20,
        'undersample_ratio': 1.0,
        'use_adv_weights': True,
        'description': '20 models, 1:1 balanced + adv weights'
    },
    'v93c_easy10_ratio2': {
        'n_estimators': 10,
        'undersample_ratio': 2.0,  # 2:1 ratio
        'use_adv_weights': True,
        'description': '10 models, 2:1 ratio + adv weights'
    },
    'v93d_easy10_ratio3': {
        'n_estimators': 10,
        'undersample_ratio': 3.0,  # 3:1 ratio
        'use_adv_weights': True,
        'description': '10 models, 3:1 ratio + adv weights'
    },
    'v93e_easy30_balanced': {
        'n_estimators': 30,
        'undersample_ratio': 1.0,
        'use_adv_weights': True,
        'description': '30 models, 1:1 balanced + adv weights'
    },
}

for name, cfg in variants.items():
    print(f"   {name}: {cfg['description']}")

# ====================
# 5. TRAIN WITH CROSS-VALIDATION
# ====================
print("\n" + "=" * 80)
print("TRAINING EASYENSEMBLE MODELS")
print("=" * 80)

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

results = {}

for variant_name, cfg in variants.items():
    print(f"\n   {variant_name}: {cfg['description']}")

    oof_preds = np.zeros(len(y))
    test_preds_folds = []
    fold_f1s = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y), 1):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        w_tr = sample_weights[train_idx] if cfg['use_adv_weights'] else None

        # Train EasyEnsemble on this fold
        fold_oof, fold_test, _ = easy_ensemble_train(
            X_tr, y_tr, w_tr, X_test,
            n_estimators=cfg['n_estimators'],
            undersample_ratio=cfg['undersample_ratio'],
            params=base_params,
            random_state=42 + fold
        )

        # Get validation predictions (subset of fold_oof)
        # Need to retrain on train_idx and predict on val_idx
        # Actually, let's do proper OOF by training on train_idx

        # For proper OOF, we need to predict only on val_idx
        # Let's modify the approach
        val_preds_ensemble = []

        minority_idx_tr = np.where(y_tr == 1)[0]
        majority_idx_tr = np.where(y_tr == 0)[0]
        n_minority = len(minority_idx_tr)
        n_sample = int(n_minority * cfg['undersample_ratio'])

        for i in range(cfg['n_estimators']):
            np.random.seed(42 + fold * 100 + i)
            sampled_maj = np.random.choice(majority_idx_tr, size=min(n_sample, len(majority_idx_tr)), replace=False)
            subset_idx = np.concatenate([minority_idx_tr, sampled_maj])

            X_sub = X_tr[subset_idx]
            y_sub = y_tr[subset_idx]
            w_sub = w_tr[subset_idx] if w_tr is not None else None

            dtrain = xgb.DMatrix(X_sub, label=y_sub, weight=w_sub)
            dval = xgb.DMatrix(X_val)
            dtest = xgb.DMatrix(X_test)

            model = xgb.train(base_params, dtrain, num_boost_round=300, verbose_eval=False)
            val_preds_ensemble.append(model.predict(dval))

            if i == 0:
                test_preds_folds.append([model.predict(dtest)])
            else:
                test_preds_folds[-1].append(model.predict(dtest))

        # Average validation predictions
        oof_preds[val_idx] = np.mean(val_preds_ensemble, axis=0)

        # Fold F1
        best_f1 = 0
        for t in np.linspace(0.1, 0.6, 50):
            f1 = f1_score(y_val, (oof_preds[val_idx] > t).astype(int))
            if f1 > best_f1:
                best_f1 = f1
        fold_f1s.append(best_f1)
        print(f"      Fold {fold}: F1={best_f1:.4f}")

    # Average test predictions across folds and estimators
    test_preds = np.mean([np.mean(fp, axis=0) for fp in test_preds_folds], axis=0)

    # Overall OOF F1
    best_f1 = 0
    best_thresh = 0.3
    for t in np.linspace(0.1, 0.6, 200):
        f1 = f1_score(y, (oof_preds > t).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    preds_binary = (oof_preds > best_thresh).astype(int)
    cm = confusion_matrix(y, preds_binary)
    tn, fp, fn, tp = cm.ravel()

    tde_mask = y == 1
    hard_count = np.sum(oof_preds[tde_mask] < 0.1)

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
        'hard_tde_count': hard_count,
        'config': cfg
    }

    print(f"      OOF F1: {best_f1:.4f} @ threshold={best_thresh:.3f}")
    print(f"      Recall: {tp/(tp+fn):.4f} | Precision: {tp/(tp+fp):.4f}")
    print(f"      FN: {fn} | FP: {fp} | Hard TDEs: {hard_count}")

# ====================
# 6. RESULTS COMPARISON
# ====================
print("\n" + "=" * 80)
print("RESULTS COMPARISON")
print("=" * 80)

print(f"\n{'Variant':<25} {'OOF F1':<10} {'Recall':<10} {'Prec':<10} {'FN':<6} {'Hard':<6}")
print("-" * 75)
print(f"{'v92d (LB=0.6986) BEST':<25} {'0.6688':<10} {'69.6%':<10} {'64.4%':<10} {'45':<6} {'20':<6}")
print(f"{'v34a (LB=0.6907)':<25} {'0.6667':<10} {'-':<10} {'-':<10} {'-':<6} {'-':<6}")
print("-" * 75)

sorted_results = sorted(results.items(), key=lambda x: -x[1]['oof_f1'])

for name, res in sorted_results:
    print(f"{name:<25} {res['oof_f1']:<10.4f} {100*res['recall']:<10.1f}% {100*res['precision']:<10.1f}% {res['confusion']['fn']:<6} {res['hard_tde_count']:<6}")

# ====================
# 7. SUBMISSIONS
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

    # v92d: OOF 0.6688 -> LB 0.6986 (+0.03)
    expected_lb = res['oof_f1'] + 0.03
    print(f"   {filename}: OOF={res['oof_f1']:.4f}, Expected LB~{expected_lb:.4f}, TDEs={test_binary.sum()}")

# Save artifacts
with open(base_path / 'data/processed/v93_easy_ensemble_artifacts.pkl', 'wb') as f:
    pickle.dump(results, f)

print("\n" + "=" * 80)
print("v93 EasyEnsemble Complete")
print("=" * 80)

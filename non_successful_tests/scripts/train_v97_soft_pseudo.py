"""
MALLORN v97: Soft Pseudo-labels (Fix Confirmation Bias)

Previous pseudo-labeling (v94-v96) failed because:
- Hard labels (0 or 1) reinforce model errors
- Model becomes overconfident in wrong predictions

Soft pseudo-labels:
- Instead of y=1 for confident TDE, use y=0.9
- Instead of y=0 for confident non-TDE, use y=0.1
- This allows uncertainty and reduces confirmation bias

Based on: https://arxiv.org/pdf/1908.02983 (Pseudo-Labeling and Confirmation Bias)
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
print("MALLORN v97: Soft Pseudo-labels")
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

n_tde_orig = np.sum(y_train)
n_non_tde_orig = np.sum(y_train == 0)
original_ratio = n_non_tde_orig / n_tde_orig

print(f"   Training: {len(train_ids)} ({n_tde_orig} TDE, {n_non_tde_orig} non-TDE)")

# Load adversarial weights (key to v92d success)
with open(base_path / 'data/processed/adversarial_validation.pkl', 'rb') as f:
    adv_results = pickle.load(f)
train_sample_weights = adv_results['sample_weights']

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

X_train_orig = train_all[available_features].values
X_test = test_all[available_features].values

X_train_orig = np.nan_to_num(X_train_orig, nan=np.nan, posinf=1e10, neginf=-1e10)
X_test = np.nan_to_num(X_test, nan=np.nan, posinf=1e10, neginf=-1e10)

print(f"   Features: {len(available_features)}")

# ====================
# 3. LOAD v92d PREDICTIONS
# ====================
print("\n3. Loading v92d predictions...")

with open(base_path / 'data/processed/v92_focal_adv_artifacts.pkl', 'rb') as f:
    v92_results = pickle.load(f)

v92d_test_preds = v92_results['v92d_baseline_adv']['test_preds']
print(f"   v92d predictions: [{v92d_test_preds.min():.4f}, {v92d_test_preds.max():.4f}]")

# ====================
# 4. BASE PARAMS (from v92d - our best)
# ====================
base_params = {
    'objective': 'reg:squarederror',  # Use regression for soft labels!
    'eval_metric': 'rmse',
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

# ====================
# 5. DEFINE VARIANTS
# ====================
print("\n4. Defining soft pseudo-label variants...")

variants = {
    'v97a_soft_95': {
        'tde_thresh': 0.90,
        'non_tde_thresh': 0.10,
        'soft_tde': 0.95,      # Soft label for TDE
        'soft_non_tde': 0.05,  # Soft label for non-TDE
        'preserve_ratio': True,
        'description': 'Soft labels 0.95/0.05, ratio preserved'
    },
    'v97b_soft_90': {
        'tde_thresh': 0.90,
        'non_tde_thresh': 0.10,
        'soft_tde': 0.90,
        'soft_non_tde': 0.10,
        'preserve_ratio': True,
        'description': 'Soft labels 0.90/0.10, ratio preserved'
    },
    'v97c_soft_85': {
        'tde_thresh': 0.90,
        'non_tde_thresh': 0.10,
        'soft_tde': 0.85,
        'soft_non_tde': 0.15,
        'preserve_ratio': True,
        'description': 'Soft labels 0.85/0.15, ratio preserved'
    },
    'v97d_use_probs': {
        'tde_thresh': 0.90,
        'non_tde_thresh': 0.10,
        'use_actual_probs': True,  # Use v92d's actual probabilities as soft labels
        'preserve_ratio': True,
        'description': 'Use actual v92d probabilities as soft labels'
    },
    'v97e_no_pseudo': {
        'tde_thresh': None,
        'non_tde_thresh': None,
        'description': 'Baseline - no pseudo-labels'
    },
}

for name, cfg in variants.items():
    print(f"   {name}: {cfg['description']}")

# ====================
# 6. TRAIN
# ====================
print("\n" + "=" * 80)
print("TRAINING WITH SOFT PSEUDO-LABELS")
print("=" * 80)

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

results = {}

for variant_name, cfg in variants.items():
    print(f"\n   {variant_name}: {cfg['description']}")

    # Create soft pseudo-labels
    if cfg.get('tde_thresh') is not None:
        tde_mask = v92d_test_preds > cfg['tde_thresh']
        non_tde_mask = v92d_test_preds < cfg['non_tde_thresh']

        n_pseudo_tde = np.sum(tde_mask)
        n_pseudo_non_tde_available = np.sum(non_tde_mask)

        # Preserve ratio
        if cfg.get('preserve_ratio', False):
            n_target_non_tde = int(n_pseudo_tde * original_ratio)
            n_pseudo_non_tde = min(n_target_non_tde, n_pseudo_non_tde_available)

            if n_pseudo_non_tde < n_pseudo_non_tde_available:
                non_tde_indices = np.where(non_tde_mask)[0]
                np.random.seed(42)
                selected = np.random.choice(non_tde_indices, size=n_pseudo_non_tde, replace=False)
                non_tde_mask = np.zeros(len(v92d_test_preds), dtype=bool)
                non_tde_mask[selected] = True
        else:
            n_pseudo_non_tde = n_pseudo_non_tde_available

        n_pseudo_non_tde = np.sum(non_tde_mask)

        # Create SOFT labels
        if cfg.get('use_actual_probs', False):
            # Use actual probabilities
            y_pseudo_tde = v92d_test_preds[tde_mask]
            y_pseudo_non_tde = v92d_test_preds[non_tde_mask]
        else:
            # Use fixed soft labels
            y_pseudo_tde = np.full(n_pseudo_tde, cfg['soft_tde'])
            y_pseudo_non_tde = np.full(n_pseudo_non_tde, cfg['soft_non_tde'])

        X_pseudo = np.vstack([X_test[tde_mask], X_test[non_tde_mask]])
        y_pseudo = np.concatenate([y_pseudo_tde, y_pseudo_non_tde])

        print(f"      Pseudo TDEs: {n_pseudo_tde} (soft label: {y_pseudo_tde.mean():.3f})")
        print(f"      Pseudo non-TDEs: {n_pseudo_non_tde} (soft label: {y_pseudo_non_tde.mean():.3f})")
    else:
        X_pseudo = np.array([]).reshape(0, X_train_orig.shape[1])
        y_pseudo = np.array([])
        n_pseudo_tde = 0
        n_pseudo_non_tde = 0

    oof_preds = np.zeros(len(y_train))
    test_preds_folds = []
    fold_f1s = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_orig, y_train), 1):
        X_tr, X_val = X_train_orig[train_idx], X_train_orig[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        w_tr = train_sample_weights[train_idx]

        # Convert training labels to float (for soft label compatibility)
        y_tr_soft = y_tr.astype(float)

        if len(y_pseudo) > 0:
            X_combined = np.vstack([X_tr, X_pseudo])
            y_combined = np.concatenate([y_tr_soft, y_pseudo])
            # Pseudo-labels get weight 1.0 (same as real)
            w_combined = np.concatenate([w_tr, np.ones(len(y_pseudo))])
        else:
            X_combined = X_tr
            y_combined = y_tr_soft
            w_combined = w_tr

        params = base_params.copy()

        dtrain = xgb.DMatrix(X_combined, label=y_combined, weight=w_combined)
        dval = xgb.DMatrix(X_val, label=y_val.astype(float))
        dtest = xgb.DMatrix(X_test)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            evals=[(dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=False
        )

        # Predictions are soft (regression output), clip to [0, 1]
        val_preds = np.clip(model.predict(dval), 0, 1)
        test_preds = np.clip(model.predict(dtest), 0, 1)

        oof_preds[val_idx] = val_preds
        test_preds_folds.append(test_preds)

        # Fold F1
        best_f1 = 0
        for t in np.linspace(0.1, 0.6, 50):
            f1 = f1_score(y_val, (val_preds > t).astype(int))
            if f1 > best_f1:
                best_f1 = f1
        fold_f1s.append(best_f1)

    test_preds_avg = np.mean(test_preds_folds, axis=0)

    # OOF F1
    best_f1 = 0
    best_thresh = 0.3
    for t in np.linspace(0.1, 0.6, 200):
        f1 = f1_score(y_train, (oof_preds > t).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    preds_binary = (oof_preds > best_thresh).astype(int)
    cm = confusion_matrix(y_train, preds_binary)
    tn, fp, fn, tp = cm.ravel()

    results[variant_name] = {
        'oof_f1': best_f1,
        'threshold': best_thresh,
        'fold_f1s': fold_f1s,
        'fold_std': np.std(fold_f1s),
        'oof_preds': oof_preds,
        'test_preds': test_preds_avg,
        'confusion': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
        'recall': tp / (tp + fn),
        'precision': tp / (tp + fp),
        'n_pseudo_tde': n_pseudo_tde,
        'n_pseudo_non_tde': n_pseudo_non_tde,
        'config': cfg
    }

    print(f"      OOF F1: {best_f1:.4f} @ threshold={best_thresh:.3f}")
    print(f"      Recall: {tp/(tp+fn):.4f} | Precision: {tp/(tp+fp):.4f}")
    print(f"      FN: {fn} | FP: {fp}")

# ====================
# 7. RESULTS
# ====================
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

print(f"\n{'Variant':<20} {'OOF F1':<10} {'Recall':<10} {'Prec':<10} {'Thresh':<10}")
print("-" * 65)
print(f"{'v92d (LB=0.6986)':<20} {'0.6688':<10} {'69.6%':<10} {'64.4%':<10} {'0.414':<10}")
print("-" * 65)

sorted_results = sorted(results.items(), key=lambda x: -x[1]['oof_f1'])

for name, res in sorted_results:
    print(f"{name:<20} {res['oof_f1']:<10.4f} {100*res['recall']:<10.1f}% {100*res['precision']:<10.1f}% {res['threshold']:<10.3f}")

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

with open(base_path / 'data/processed/v97_soft_pseudo_artifacts.pkl', 'wb') as f:
    pickle.dump(results, f)

print("\n" + "=" * 80)
print("v97 Soft Pseudo-labels Complete")
print("=" * 80)

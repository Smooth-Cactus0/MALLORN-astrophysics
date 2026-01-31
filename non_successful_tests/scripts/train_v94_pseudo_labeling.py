"""
MALLORN v94: Pseudo-labeling to Break 0.71 Barrier

Current best: v92d with LB 0.6986
Goal: Break the 0.71 LB ceiling

Pseudo-labeling Strategy:
1. Use v92d's predictions on the 7,124 test samples
2. Select high-confidence predictions as pseudo-labels
3. Add pseudo-labeled samples to training data
4. Retrain with adversarial weights on expanded dataset

Why this might work:
- We're only using 30% of data for training (3,043 samples)
- Test set has 70% of data (7,124 samples)
- If v92d is confident about some test samples, those are likely correct
- Adding them expands training diversity without labeling noise

PLAsTiCC winner used pseudo-labeling to great effect.
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
print("MALLORN v94: Pseudo-labeling to Break 0.71 Barrier")
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

print(f"   Training: {len(train_ids)} objects ({np.sum(y_train)} TDE)")
print(f"   Test: {len(test_ids)} objects (unlabeled)")

# Load adversarial weights
with open(base_path / 'data/processed/adversarial_validation.pkl', 'rb') as f:
    adv_results = pickle.load(f)
train_sample_weights = adv_results['sample_weights']

# ====================
# 2. LOAD FEATURES
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

X_train_orig = train_all[available_features].values
X_test = test_all[available_features].values

X_train_orig = np.nan_to_num(X_train_orig, nan=np.nan, posinf=1e10, neginf=-1e10)
X_test = np.nan_to_num(X_test, nan=np.nan, posinf=1e10, neginf=-1e10)

# ====================
# 3. LOAD v92d PREDICTIONS FOR PSEUDO-LABELS
# ====================
print("\n3. Loading v92d predictions for pseudo-labeling...")

with open(base_path / 'data/processed/v92_focal_adv_artifacts.pkl', 'rb') as f:
    v92_results = pickle.load(f)

# v92d is our best model (standard XGB + adversarial weights)
v92d_test_preds = v92_results['v92d_baseline_adv']['test_preds']

print(f"   v92d test predictions range: [{v92d_test_preds.min():.4f}, {v92d_test_preds.max():.4f}]")
print(f"   Mean: {v92d_test_preds.mean():.4f}, Median: {np.median(v92d_test_preds):.4f}")

# Analyze confidence distribution
for thresh in [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]:
    if thresh < 0.5:
        count = np.sum(v92d_test_preds < thresh)
        print(f"   Confident non-TDE (p < {thresh}): {count}")
    else:
        count = np.sum(v92d_test_preds > thresh)
        print(f"   Confident TDE (p > {thresh}): {count}")

# ====================
# 4. BASE PARAMS (from v92d)
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
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1
}

# ====================
# 5. DEFINE PSEUDO-LABELING VARIANTS
# ====================
print("\n4. Defining pseudo-labeling variants...")

variants = {
    'v94a_strict': {
        'tde_thresh': 0.9,      # Only very confident TDEs
        'non_tde_thresh': 0.1,  # Only very confident non-TDEs
        'pseudo_weight': 0.5,   # Weight pseudo-labels at 50% of real labels
        'description': 'Strict: TDE>0.9, non-TDE<0.1, weight=0.5'
    },
    'v94b_moderate': {
        'tde_thresh': 0.8,
        'non_tde_thresh': 0.2,
        'pseudo_weight': 0.5,
        'description': 'Moderate: TDE>0.8, non-TDE<0.2, weight=0.5'
    },
    'v94c_relaxed': {
        'tde_thresh': 0.7,
        'non_tde_thresh': 0.3,
        'pseudo_weight': 0.5,
        'description': 'Relaxed: TDE>0.7, non-TDE<0.3, weight=0.5'
    },
    'v94d_strict_full': {
        'tde_thresh': 0.9,
        'non_tde_thresh': 0.1,
        'pseudo_weight': 1.0,   # Full weight on pseudo-labels
        'description': 'Strict thresholds, full weight'
    },
    'v94e_tde_only': {
        'tde_thresh': 0.8,
        'non_tde_thresh': 0.0,  # No non-TDE pseudo-labels
        'pseudo_weight': 0.5,
        'description': 'TDE-only pseudo-labels (no non-TDE)'
    },
    'v94f_iterative': {
        'tde_thresh': 0.85,
        'non_tde_thresh': 0.15,
        'pseudo_weight': 0.7,
        'iterative': True,      # Two rounds of pseudo-labeling
        'description': 'Iterative: 2 rounds, thresh 0.85/0.15'
    },
}

for name, cfg in variants.items():
    print(f"   {name}: {cfg['description']}")

# ====================
# 6. TRAIN WITH PSEUDO-LABELING
# ====================
print("\n" + "=" * 80)
print("TRAINING WITH PSEUDO-LABELING")
print("=" * 80)

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

results = {}

for variant_name, cfg in variants.items():
    print(f"\n   {variant_name}: {cfg['description']}")

    # Select pseudo-labeled samples from test set
    tde_mask = v92d_test_preds > cfg['tde_thresh']
    non_tde_mask = v92d_test_preds < cfg['non_tde_thresh']

    n_pseudo_tde = np.sum(tde_mask)
    n_pseudo_non_tde = np.sum(non_tde_mask)

    print(f"      Pseudo TDEs: {n_pseudo_tde} (p > {cfg['tde_thresh']})")
    print(f"      Pseudo non-TDEs: {n_pseudo_non_tde} (p < {cfg['non_tde_thresh']})")

    # Create pseudo-labeled dataset
    X_pseudo = np.vstack([
        X_test[tde_mask],
        X_test[non_tde_mask]
    ]) if n_pseudo_tde > 0 or n_pseudo_non_tde > 0 else np.array([]).reshape(0, X_train_orig.shape[1])

    y_pseudo = np.concatenate([
        np.ones(n_pseudo_tde),
        np.zeros(n_pseudo_non_tde)
    ]) if n_pseudo_tde > 0 or n_pseudo_non_tde > 0 else np.array([])

    # Pseudo-label weights (less than real labels)
    pseudo_weights = np.full(len(y_pseudo), cfg['pseudo_weight'])

    print(f"      Total pseudo-labels: {len(y_pseudo)}")

    oof_preds = np.zeros(len(y_train))
    test_preds_folds = []
    fold_f1s = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_orig, y_train), 1):
        X_tr, X_val = X_train_orig[train_idx], X_train_orig[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        w_tr = train_sample_weights[train_idx]

        # Combine with pseudo-labeled data
        if len(y_pseudo) > 0:
            X_combined = np.vstack([X_tr, X_pseudo])
            y_combined = np.concatenate([y_tr, y_pseudo])
            # Weights: real labels get adversarial weights, pseudo get pseudo_weight
            w_combined = np.concatenate([w_tr, pseudo_weights])
        else:
            X_combined = X_tr
            y_combined = y_tr
            w_combined = w_tr

        # Handle iterative pseudo-labeling
        if cfg.get('iterative', False):
            # First round: train on combined data
            params = base_params.copy()
            params['scale_pos_weight'] = len(y_combined[y_combined==0]) / max(len(y_combined[y_combined==1]), 1)

            dtrain = xgb.DMatrix(X_combined, label=y_combined, weight=w_combined)
            model1 = xgb.train(params, dtrain, num_boost_round=300, verbose_eval=False)

            # Get new predictions on test set
            dtest_full = xgb.DMatrix(X_test)
            round1_preds = model1.predict(dtest_full)

            # Second round: use round1 predictions for stricter pseudo-labels
            tde_mask2 = round1_preds > cfg['tde_thresh']
            non_tde_mask2 = round1_preds < cfg['non_tde_thresh']

            X_pseudo2 = np.vstack([X_test[tde_mask2], X_test[non_tde_mask2]])
            y_pseudo2 = np.concatenate([np.ones(np.sum(tde_mask2)), np.zeros(np.sum(non_tde_mask2))])
            w_pseudo2 = np.full(len(y_pseudo2), cfg['pseudo_weight'])

            X_combined2 = np.vstack([X_tr, X_pseudo2])
            y_combined2 = np.concatenate([y_tr, y_pseudo2])
            w_combined2 = np.concatenate([w_tr, w_pseudo2])

            dtrain2 = xgb.DMatrix(X_combined2, label=y_combined2, weight=w_combined2)
            model = xgb.train(params, dtrain2, num_boost_round=300, verbose_eval=False)
        else:
            # Single round
            params = base_params.copy()
            params['scale_pos_weight'] = len(y_combined[y_combined==0]) / max(len(y_combined[y_combined==1]), 1)

            dtrain = xgb.DMatrix(X_combined, label=y_combined, weight=w_combined)
            dval = xgb.DMatrix(X_val, label=y_val)

            model = xgb.train(
                params,
                dtrain,
                num_boost_round=500,
                evals=[(dval, 'val')],
                early_stopping_rounds=50,
                verbose_eval=False
            )

        # Predict
        dval = xgb.DMatrix(X_val)
        dtest = xgb.DMatrix(X_test)

        oof_preds[val_idx] = model.predict(dval)
        test_preds_folds.append(model.predict(dtest))

        # Fold F1
        best_f1 = 0
        for t in np.linspace(0.05, 0.5, 50):
            f1 = f1_score(y_val, (oof_preds[val_idx] > t).astype(int))
            if f1 > best_f1:
                best_f1 = f1
        fold_f1s.append(best_f1)

    # Average test predictions
    test_preds = np.mean(test_preds_folds, axis=0)

    # Overall OOF F1
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
        'n_pseudo_tde': n_pseudo_tde,
        'n_pseudo_non_tde': n_pseudo_non_tde,
        'config': cfg
    }

    print(f"      OOF F1: {best_f1:.4f} @ threshold={best_thresh:.3f}")
    print(f"      Recall: {tp/(tp+fn):.4f} | Precision: {tp/(tp+fp):.4f}")
    print(f"      FN: {fn} | FP: {fp}")

# ====================
# 7. RESULTS COMPARISON
# ====================
print("\n" + "=" * 80)
print("RESULTS COMPARISON")
print("=" * 80)

print(f"\n{'Variant':<20} {'OOF F1':<10} {'Recall':<10} {'Prec':<10} {'Pseudo TDE':<12} {'Pseudo nonTDE':<14}")
print("-" * 85)
print(f"{'v92d (LB=0.6986)':<20} {'0.6688':<10} {'69.6%':<10} {'64.4%':<10} {'-':<12} {'-':<14}")
print("-" * 85)

sorted_results = sorted(results.items(), key=lambda x: -x[1]['oof_f1'])

for name, res in sorted_results:
    print(f"{name:<20} {res['oof_f1']:<10.4f} {100*res['recall']:<10.1f}% {100*res['precision']:<10.1f}% {res['n_pseudo_tde']:<12} {res['n_pseudo_non_tde']:<14}")

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

    # v92d: OOF 0.6688 -> LB 0.6986 (+0.03)
    expected_lb = res['oof_f1'] + 0.03
    print(f"   {filename}: OOF={res['oof_f1']:.4f}, Expected LB~{expected_lb:.4f}, TDEs={test_binary.sum()}")

# Save artifacts
with open(base_path / 'data/processed/v94_pseudo_labeling_artifacts.pkl', 'wb') as f:
    pickle.dump(results, f)

print("\n" + "=" * 80)
print("v94 Pseudo-labeling Complete")
print("=" * 80)

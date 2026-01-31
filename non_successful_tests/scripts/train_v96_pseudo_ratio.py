"""
MALLORN v96: Pseudo-labeling with Preserved Ratio + Harsh Thresholds

v95 Results:
- v95c (balanced 124:124) got LB 0.6753
- v95b (TDE-only 193) got LB 0.6716
- Both worse than v92d (LB 0.6986)

User suggestions:
1. Preserve the original TDE:non-TDE ratio (1:19.6) in pseudo-labels
2. Use harsher confidence thresholds (p>0.95, p<0.05)

If we have 124 pseudo-TDEs, ratio-preserved means 124 * 19.6 = 2,430 pseudo-non-TDEs
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
print("MALLORN v96: Pseudo-labeling with Ratio Preservation")
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
print(f"   Original ratio: 1:{original_ratio:.1f}")

# Load adversarial weights
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

# ====================
# 3. LOAD v92d PREDICTIONS
# ====================
print("\n3. Loading v92d predictions...")

with open(base_path / 'data/processed/v92_focal_adv_artifacts.pkl', 'rb') as f:
    v92_results = pickle.load(f)

v92d_test_preds = v92_results['v92d_baseline_adv']['test_preds']

# Analyze confidence distribution
print(f"\n   Confidence distribution:")
for thresh in [0.99, 0.95, 0.90, 0.85]:
    print(f"   TDE (p > {thresh}): {np.sum(v92d_test_preds > thresh)}")
for thresh in [0.01, 0.05, 0.10, 0.15]:
    print(f"   non-TDE (p < {thresh}): {np.sum(v92d_test_preds < thresh)}")

# ====================
# 4. BASE PARAMS
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
    'scale_pos_weight': original_ratio,
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1
}

# ====================
# 5. DEFINE VARIANTS
# ====================
print("\n4. Defining variants...")

variants = {
    'v96a_ratio_preserved': {
        'tde_thresh': 0.90,
        'non_tde_thresh': 0.10,
        'preserve_ratio': True,  # Match original 1:19.6 ratio
        'pseudo_weight': 1.0,
        'description': 'Ratio preserved (1:19.6), thresh 0.90/0.10'
    },
    'v96b_harsh_ratio': {
        'tde_thresh': 0.95,
        'non_tde_thresh': 0.05,
        'preserve_ratio': True,
        'pseudo_weight': 1.0,
        'description': 'Ratio preserved, HARSH thresh 0.95/0.05'
    },
    'v96c_very_harsh': {
        'tde_thresh': 0.99,
        'non_tde_thresh': 0.01,
        'preserve_ratio': True,
        'pseudo_weight': 1.0,
        'description': 'Ratio preserved, VERY HARSH 0.99/0.01'
    },
    'v96d_harsh_balanced': {
        'tde_thresh': 0.95,
        'non_tde_thresh': 0.05,
        'preserve_ratio': False,  # Keep balanced 1:1
        'pseudo_weight': 1.0,
        'description': 'Balanced 1:1, HARSH thresh 0.95/0.05'
    },
    'v96e_no_pseudo': {
        'tde_thresh': None,
        'non_tde_thresh': None,
        'preserve_ratio': False,
        'pseudo_weight': 0,
        'description': 'Baseline - no pseudo-labels'
    },
}

for name, cfg in variants.items():
    print(f"   {name}: {cfg['description']}")

# ====================
# 6. TRAIN
# ====================
print("\n" + "=" * 80)
print("TRAINING")
print("=" * 80)

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

results = {}

for variant_name, cfg in variants.items():
    print(f"\n   {variant_name}: {cfg['description']}")

    # Select pseudo-labels
    if cfg['tde_thresh'] is not None:
        tde_mask = v92d_test_preds > cfg['tde_thresh']
        n_pseudo_tde = np.sum(tde_mask)
    else:
        tde_mask = np.zeros(len(v92d_test_preds), dtype=bool)
        n_pseudo_tde = 0

    if cfg.get('non_tde_thresh') is not None:
        non_tde_candidates = v92d_test_preds < cfg['non_tde_thresh']
        n_non_tde_candidates = np.sum(non_tde_candidates)

        if cfg.get('preserve_ratio', False):
            # Preserve original ratio: for every pseudo-TDE, add ratio * pseudo-non-TDEs
            n_target_non_tde = int(n_pseudo_tde * original_ratio)
            n_pseudo_non_tde = min(n_target_non_tde, n_non_tde_candidates)

            if n_pseudo_non_tde < n_non_tde_candidates:
                # Randomly sample
                non_tde_indices = np.where(non_tde_candidates)[0]
                np.random.seed(42)
                selected = np.random.choice(non_tde_indices, size=n_pseudo_non_tde, replace=False)
                non_tde_mask = np.zeros(len(v92d_test_preds), dtype=bool)
                non_tde_mask[selected] = True
            else:
                non_tde_mask = non_tde_candidates
        else:
            # Balanced 1:1
            n_pseudo_non_tde = min(n_pseudo_tde, n_non_tde_candidates)
            non_tde_indices = np.where(non_tde_candidates)[0]
            np.random.seed(42)
            selected = np.random.choice(non_tde_indices, size=n_pseudo_non_tde, replace=False)
            non_tde_mask = np.zeros(len(v92d_test_preds), dtype=bool)
            non_tde_mask[selected] = True
    else:
        non_tde_mask = np.zeros(len(v92d_test_preds), dtype=bool)
        n_pseudo_non_tde = 0

    n_pseudo_non_tde = np.sum(non_tde_mask)
    pseudo_ratio = n_pseudo_non_tde / max(n_pseudo_tde, 1)

    print(f"      Pseudo TDEs: {n_pseudo_tde}")
    print(f"      Pseudo non-TDEs: {n_pseudo_non_tde}")
    print(f"      Pseudo ratio: 1:{pseudo_ratio:.1f}")

    # Create pseudo dataset
    if n_pseudo_tde > 0 or n_pseudo_non_tde > 0:
        X_pseudo = np.vstack([X_test[tde_mask], X_test[non_tde_mask]])
        y_pseudo = np.concatenate([np.ones(n_pseudo_tde), np.zeros(n_pseudo_non_tde)])
        pseudo_weights = np.full(len(y_pseudo), cfg['pseudo_weight'])
    else:
        X_pseudo = np.array([]).reshape(0, X_train_orig.shape[1])
        y_pseudo = np.array([])
        pseudo_weights = np.array([])

    oof_preds = np.zeros(len(y_train))
    test_preds_folds = []
    fold_f1s = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_orig, y_train), 1):
        X_tr, X_val = X_train_orig[train_idx], X_train_orig[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        w_tr = train_sample_weights[train_idx]

        if len(y_pseudo) > 0:
            X_combined = np.vstack([X_tr, X_pseudo])
            y_combined = np.concatenate([y_tr, y_pseudo])
            w_combined = np.concatenate([w_tr, pseudo_weights])
        else:
            X_combined = X_tr
            y_combined = y_tr
            w_combined = w_tr

        params = base_params.copy()

        dtrain = xgb.DMatrix(X_combined, label=y_combined, weight=w_combined)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            evals=[(dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=False
        )

        oof_preds[val_idx] = model.predict(dval)
        test_preds_folds.append(model.predict(dtest))

        best_f1 = 0
        for t in np.linspace(0.05, 0.5, 50):
            f1 = f1_score(y_val, (oof_preds[val_idx] > t).astype(int))
            if f1 > best_f1:
                best_f1 = f1
        fold_f1s.append(best_f1)

    test_preds = np.mean(test_preds_folds, axis=0)

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
# 7. RESULTS
# ====================
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

print(f"\n{'Variant':<25} {'OOF F1':<10} {'Recall':<10} {'P-TDE':<8} {'P-nonTDE':<10} {'P-Ratio':<10}")
print("-" * 85)
print(f"{'v92d (LB=0.6986)':<25} {'0.6688':<10} {'69.6%':<10} {'0':<8} {'0':<10} {'-':<10}")
print("-" * 85)

sorted_results = sorted(results.items(), key=lambda x: -x[1]['oof_f1'])

for name, res in sorted_results:
    p_ratio = res['n_pseudo_non_tde'] / max(res['n_pseudo_tde'], 1)
    print(f"{name:<25} {res['oof_f1']:<10.4f} {100*res['recall']:<10.1f}% {res['n_pseudo_tde']:<8} {res['n_pseudo_non_tde']:<10} {'1:'+f'{p_ratio:.1f}':<10}")

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

with open(base_path / 'data/processed/v96_pseudo_ratio_artifacts.pkl', 'wb') as f:
    pickle.dump(results, f)

print("\n" + "=" * 80)
print("v96 Complete")
print("=" * 80)

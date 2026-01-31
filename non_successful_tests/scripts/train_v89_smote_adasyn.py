"""
MALLORN v89: SMOTE and ADASYN for Class Imbalance

Error Analysis Findings:
- 19.6:1 class imbalance (non-TDE : TDE)
- 25.7% False Negative Rate (missing TDEs)
- Missed TDEs have higher redshift (+17.8%) and amplitude (+19.1%)

SMOTE: Synthetic Minority Over-sampling Technique
- Creates synthetic samples by interpolating between existing minority samples
- Good for continuous features

ADASYN: Adaptive Synthetic Sampling
- Like SMOTE but focuses on harder-to-learn examples
- Generates more samples near decision boundary

Both should help the model learn TDE patterns better.
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek, SMOTEENN
import xgboost as xgb
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
base_path = Path(__file__).parent.parent

print("=" * 80)
print("MALLORN v89: SMOTE and ADASYN for Class Imbalance")
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

print(f"   Training: {len(train_ids)} objects")
print(f"   TDE: {np.sum(y)} ({100*np.mean(y):.2f}%)")
print(f"   Non-TDE: {np.sum(y==0)} ({100*np.mean(y==0):.2f}%)")
print(f"   Imbalance ratio: {np.sum(y==0)/np.sum(y):.1f}:1")

# ====================
# 2. LOAD FEATURES (v34a feature set)
# ====================
print("\n2. Loading features...")

with open(base_path / 'data/processed/v34a_artifacts.pkl', 'rb') as f:
    v34a = pickle.load(f)
v34a_features = v34a['feature_names']

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

# Remove shift features (from adversarial validation)
shift_features = ['all_rise_time', 'all_asymmetry']
available_features = [f for f in v34a_features if f in train_all.columns and f not in shift_features]
print(f"   Features: {len(available_features)}")

X_train_full = train_all[available_features].values
X_test = test_all[available_features].values

# Handle NaN and infinities for SMOTE (requires clean data)
X_train_full = np.nan_to_num(X_train_full, nan=0, posinf=1e10, neginf=-1e10)
X_test = np.nan_to_num(X_test, nan=0, posinf=1e10, neginf=-1e10)

# ====================
# 3. v34a PARAMETERS (baseline)
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
# 4. DEFINE OVERSAMPLING VARIANTS
# ====================
print("\n3. Defining oversampling variants...")

variants = {
    'v89a_baseline': {
        'sampler': None,
        'description': 'No oversampling (baseline)',
        'use_class_weight': True
    },
    'v89b_smote': {
        'sampler': SMOTE(random_state=42, k_neighbors=5),
        'description': 'SMOTE (k=5)',
        'use_class_weight': False
    },
    'v89c_smote_k3': {
        'sampler': SMOTE(random_state=42, k_neighbors=3),
        'description': 'SMOTE (k=3, for small minority)',
        'use_class_weight': False
    },
    'v89d_adasyn': {
        'sampler': ADASYN(random_state=42, n_neighbors=5),
        'description': 'ADASYN (adaptive)',
        'use_class_weight': False
    },
    'v89e_smote_tomek': {
        'sampler': SMOTETomek(random_state=42),
        'description': 'SMOTE + Tomek links (clean boundary)',
        'use_class_weight': False
    },
    'v89f_smote_enn': {
        'sampler': SMOTEENN(random_state=42),
        'description': 'SMOTE + ENN (clean noisy samples)',
        'use_class_weight': False
    },
    'v89g_smote_ratio': {
        'sampler': SMOTE(random_state=42, sampling_strategy=0.3),  # 30% of majority
        'description': 'SMOTE (ratio=0.3, partial balance)',
        'use_class_weight': True
    },
}

for name, cfg in variants.items():
    print(f"   {name}: {cfg['description']}")

# ====================
# 5. TRAIN ALL VARIANTS
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
    fold_fn_rates = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y), 1):
        X_tr, X_val = X_train_full[train_idx], X_train_full[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Apply oversampling to training data only
        if cfg['sampler'] is not None:
            try:
                X_tr_resampled, y_tr_resampled = cfg['sampler'].fit_resample(X_tr, y_tr)
            except Exception as e:
                print(f"      Fold {fold} sampler failed: {e}")
                X_tr_resampled, y_tr_resampled = X_tr, y_tr
        else:
            X_tr_resampled, y_tr_resampled = X_tr, y_tr

        # Set params
        params = base_params.copy()
        if cfg['use_class_weight']:
            params['scale_pos_weight'] = np.sum(y_tr_resampled == 0) / np.sum(y_tr_resampled == 1)
        else:
            params['scale_pos_weight'] = 1.0  # Balanced after oversampling

        dtrain = xgb.DMatrix(X_tr_resampled, label=y_tr_resampled, feature_names=available_features)
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

        # Fold metrics
        best_f1 = 0
        best_thresh = 0.3
        for t in np.linspace(0.05, 0.5, 50):
            f1 = f1_score(y_val, (oof_preds[val_idx] > t).astype(int))
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
        fold_f1s.append(best_f1)

        # FN rate
        preds_fold = (oof_preds[val_idx] > best_thresh).astype(int)
        cm = confusion_matrix(y_val, preds_fold)
        if cm.shape == (2, 2):
            fn = cm[1, 0]
            tp = cm[1, 1]
            fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
            fold_fn_rates.append(fn_rate)

    # Overall OOF F1
    best_f1 = 0
    best_thresh = 0.3
    for t in np.linspace(0.05, 0.5, 200):
        f1 = f1_score(y, (oof_preds > t).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    # Confusion matrix
    preds_binary = (oof_preds > best_thresh).astype(int)
    cm = confusion_matrix(y, preds_binary)
    tn, fp, fn, tp = cm.ravel()

    results[variant_name] = {
        'oof_f1': best_f1,
        'threshold': best_thresh,
        'fold_f1s': fold_f1s,
        'fold_std': np.std(fold_f1s),
        'oof_preds': oof_preds,
        'test_preds': test_preds.mean(axis=1),
        'confusion': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
        'fn_rate': fn / (fn + tp),
        'fp_rate': fp / (fp + tn),
        'recall': tp / (tp + fn),
        'precision': tp / (tp + fp),
        'config': cfg
    }

    print(f"      OOF F1: {best_f1:.4f} @ threshold={best_thresh:.3f}")
    print(f"      Recall: {tp/(tp+fn):.4f} | Precision: {tp/(tp+fp):.4f}")
    print(f"      FN: {fn} | FP: {fp}")

# ====================
# 6. RESULTS SUMMARY
# ====================
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

print(f"\n{'Variant':<20} {'OOF F1':<10} {'Recall':<10} {'FN':<8} {'FP':<8} {'FN Rate':<10}")
print("-" * 75)

# Sort by recall (we want to catch more TDEs)
sorted_results = sorted(results.items(), key=lambda x: -x[1]['recall'])

for name, res in sorted_results:
    print(f"{name:<20} {res['oof_f1']:<10.4f} {res['recall']:<10.4f} {res['confusion']['fn']:<8} {res['confusion']['fp']:<8} {res['fn_rate']:<10.4f}")

# ====================
# 7. ANALYSIS
# ====================
print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

baseline = results['v89a_baseline']
print(f"\nBaseline (v89a):")
print(f"   FN: {baseline['confusion']['fn']}, FP: {baseline['confusion']['fp']}")
print(f"   Recall: {baseline['recall']:.4f}")

print("\nImprovement over baseline:")
for name, res in sorted_results:
    if name == 'v89a_baseline':
        continue
    fn_change = res['confusion']['fn'] - baseline['confusion']['fn']
    fp_change = res['confusion']['fp'] - baseline['confusion']['fp']
    recall_change = res['recall'] - baseline['recall']
    print(f"   {name}: FN {fn_change:+d}, FP {fp_change:+d}, Recall {recall_change:+.4f}")

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
    print(f"   {filename}: OOF={res['oof_f1']:.4f}, Recall={res['recall']:.4f}, TDEs={test_binary.sum()}")

# Save artifacts
with open(base_path / 'data/processed/v89_smote_artifacts.pkl', 'wb') as f:
    pickle.dump(results, f)

print("\n" + "=" * 80)
print("v89 Complete")
print("=" * 80)

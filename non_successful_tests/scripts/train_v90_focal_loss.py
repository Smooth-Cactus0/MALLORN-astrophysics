"""
MALLORN v90: Focal Loss for Hard TDE Examples

Error Analysis Findings:
- 25.7% False Negative Rate (missing TDEs)
- 22 TDEs have probability < 0.1 (model completely wrong)
- Standard XGBoost ignores these hard examples after a few iterations

Focal Loss: FL(p) = -alpha(1-p)^gamma * log(p)
- gamma: focusing parameter - higher = more focus on hard examples
- alpha: class weighting factor

When model is confident but WRONG (TDE with p=0.05):
- Standard loss: -log(0.05) = 3.0
- Focal loss (gamma=2): -(0.95)^2 * log(0.05) = 2.71 * 3.0 = 8.1 (2.7x stronger!)

This should help the model focus on those 22 hard TDE cases.
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix
import xgboost as xgb
from imxgboost.focal_loss import Focal_Binary_Loss
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
base_path = Path(__file__).parent.parent

print("=" * 80)
print("MALLORN v90: Focal Loss for Hard TDE Examples")
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

# ====================
# 2. LOAD FEATURES (v34a feature set - our best LB model)
# ====================
print("\n2. Loading v34a features...")

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

X_train = train_all[available_features].values
X_test = test_all[available_features].values

# Handle infinities
X_train = np.nan_to_num(X_train, nan=np.nan, posinf=1e10, neginf=-1e10)
X_test = np.nan_to_num(X_test, nan=np.nan, posinf=1e10, neginf=-1e10)

# ====================
# 3. DEFINE FOCAL LOSS VARIANTS
# ====================
print("\n3. Defining Focal Loss variants...")

# Base v34a parameters (our best LB model)
base_params = {
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

# Note: With focal loss, we don't need scale_pos_weight - the loss itself handles imbalance

# Focal loss gamma values to test
# gamma=0: standard cross-entropy
# gamma=2: recommended default, 2.7x weight on hard examples
# gamma=5: aggressive focus on hard examples
variants = {
    'v90a_focal_g1': {
        'gamma': 1.0,
        'description': 'Focal Loss gamma=1 (mild focus)'
    },
    'v90b_focal_g2': {
        'gamma': 2.0,
        'description': 'Focal Loss gamma=2 (standard)'
    },
    'v90c_focal_g3': {
        'gamma': 3.0,
        'description': 'Focal Loss gamma=3 (stronger)'
    },
    'v90d_focal_g5': {
        'gamma': 5.0,
        'description': 'Focal Loss gamma=5 (aggressive)'
    },
    'v90e_focal_g2_reg': {
        'gamma': 2.0,
        'reg_alpha': 0.3,
        'reg_lambda': 2.0,
        'description': 'Focal gamma=2 + more regularization'
    },
}

for name, cfg in variants.items():
    print(f"   {name}: {cfg['description']}")

# ====================
# 4. TRAIN ALL VARIANTS
# ====================
print("\n" + "=" * 80)
print("TRAINING WITH FOCAL LOSS")
print("=" * 80)

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

results = {}

for variant_name, cfg in variants.items():
    print(f"\n   {variant_name}: {cfg['description']}")

    # Create focal loss object
    focal_loss = Focal_Binary_Loss(gamma_indct=cfg['gamma'])

    oof_preds = np.zeros(len(y))
    test_preds = np.zeros((len(X_test), n_folds))
    fold_f1s = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y), 1):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Build params for this variant
        params = base_params.copy()
        if 'reg_alpha' in cfg:
            params['reg_alpha'] = cfg['reg_alpha']
        if 'reg_lambda' in cfg:
            params['reg_lambda'] = cfg['reg_lambda']

        # Create DMatrix
        dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=available_features)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=available_features)
        dtest = xgb.DMatrix(X_test, feature_names=available_features)

        # Train with custom focal loss objective
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            evals=[(dval, 'val')],
            obj=focal_loss.focal_binary_object,
            early_stopping_rounds=50,
            verbose_eval=False
        )

        # Predict raw margin (logits), then apply sigmoid
        raw_val = model.predict(dval)
        raw_test = model.predict(dtest)

        # Convert to probabilities using sigmoid
        oof_preds[val_idx] = 1.0 / (1.0 + np.exp(-raw_val))
        test_preds[:, fold-1] = 1.0 / (1.0 + np.exp(-raw_test))

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
        'recall': tp / (tp + fn),
        'precision': tp / (tp + fp),
        'config': cfg
    }

    print(f"      OOF F1: {best_f1:.4f} @ threshold={best_thresh:.3f}")
    print(f"      Recall: {tp/(tp+fn):.4f} | Precision: {tp/(tp+fp):.4f}")
    print(f"      FN: {fn} ({100*fn/(fn+tp):.1f}%) | FP: {fp}")

# ====================
# 5. COMPARE TO v34a BASELINE
# ====================
print("\n" + "=" * 80)
print("RESULTS COMPARISON")
print("=" * 80)

print(f"\n{'Variant':<25} {'OOF F1':<10} {'Recall':<10} {'FN':<8} {'FN Rate':<10}")
print("-" * 70)
print(f"{'v34a (LB=0.6907)':<25} {'0.6667':<10} {'-':<10} {'-':<8} {'-':<10}")
print(f"{'v88 (LB=0.6876)':<25} {'0.6854':<10} {'74.3%':<10} {'38':<8} {'25.7%':<10}")
print("-" * 70)

sorted_results = sorted(results.items(), key=lambda x: -x[1]['oof_f1'])

for name, res in sorted_results:
    print(f"{name:<25} {res['oof_f1']:<10.4f} {100*res['recall']:<10.1f}% {res['confusion']['fn']:<8} {100*res['fn_rate']:<10.1f}%")

# ====================
# 6. ANALYZE HARD EXAMPLES
# ====================
print("\n" + "=" * 80)
print("HARD EXAMPLE ANALYSIS")
print("=" * 80)

# Compare probability distributions for TDEs between v88 and focal loss
best_variant = sorted_results[0][0]
best_preds = results[best_variant]['oof_preds']

# Load v88 preds for comparison
try:
    with open(base_path / 'data/processed/v88_artifacts.pkl', 'rb') as f:
        v88 = pickle.load(f)
    v88_preds = v88['oof_preds']

    # TDE samples
    tde_mask = y == 1

    print(f"\n   TDE probability comparison (n={np.sum(tde_mask)}):")
    print(f"   {'Metric':<20} {'v88':<12} {best_variant:<15}")
    print("-" * 50)
    print(f"   {'Mean prob':<20} {np.mean(v88_preds[tde_mask]):<12.4f} {np.mean(best_preds[tde_mask]):<15.4f}")
    print(f"   {'Median prob':<20} {np.median(v88_preds[tde_mask]):<12.4f} {np.median(best_preds[tde_mask]):<15.4f}")
    print(f"   {'Min prob':<20} {np.min(v88_preds[tde_mask]):<12.4f} {np.min(best_preds[tde_mask]):<15.4f}")
    print(f"   {'Prob < 0.1 count':<20} {np.sum(v88_preds[tde_mask] < 0.1):<12} {np.sum(best_preds[tde_mask] < 0.1):<15}")
    print(f"   {'Prob < 0.2 count':<20} {np.sum(v88_preds[tde_mask] < 0.2):<12} {np.sum(best_preds[tde_mask] < 0.2):<15}")

except FileNotFoundError:
    print("   v88 artifacts not found for comparison")

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

    # v34a had OOF 0.6667 -> LB 0.6907 (+0.024 boost)
    expected_lb = res['oof_f1'] + 0.024
    print(f"   {filename}: OOF={res['oof_f1']:.4f}, Expected LB~{expected_lb:.4f}, TDEs={test_binary.sum()}")

# Save artifacts
with open(base_path / 'data/processed/v90_focal_artifacts.pkl', 'wb') as f:
    pickle.dump(results, f)

print("\n" + "=" * 80)
print("v90 Focal Loss Complete")
print("=" * 80)

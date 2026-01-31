"""
MALLORN v91: Focal Loss + Class Weighting (Balanced Approach)

v90 Finding:
- Focal loss ELIMINATED the "completely wrong" problem (0 TDEs with prob<0.1)
- But OOF F1 dropped from 0.68 to 0.62 due to too many false positives
- v90e achieved 80.4% recall but 123 FP hurt precision

Solution: Combine focal loss with XGBoost's scale_pos_weight
- Focal loss: Focus on hard examples (gamma)
- scale_pos_weight: Penalize missing TDEs (alpha)

This should give us the best of both worlds:
- High recall (catch hard TDEs)
- Reasonable precision (don't over-predict)
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
print("MALLORN v91: Focal Loss + Class Weighting")
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

imbalance_ratio = np.sum(y==0) / np.sum(y)
print(f"   Imbalance ratio: {imbalance_ratio:.1f}:1")

# ====================
# 2. LOAD FEATURES (v34a feature set)
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

# Remove shift features
shift_features = ['all_rise_time', 'all_asymmetry']
available_features = [f for f in v34a_features if f in train_all.columns and f not in shift_features]
print(f"   Features: {len(available_features)}")

X_train = train_all[available_features].values
X_test = test_all[available_features].values

X_train = np.nan_to_num(X_train, nan=np.nan, posinf=1e10, neginf=-1e10)
X_test = np.nan_to_num(X_test, nan=np.nan, posinf=1e10, neginf=-1e10)

# ====================
# 3. WEIGHTED FOCAL LOSS CLASS
# ====================
class Weighted_Focal_Binary_Loss:
    """Focal loss with class weighting (alpha parameter)"""

    def __init__(self, gamma, alpha):
        """
        gamma: focusing parameter (higher = more focus on hard examples)
        alpha: class weight for positive class (higher = penalize missing TDEs more)
        """
        self.gamma = gamma
        self.alpha = alpha

    def focal_binary_object(self, pred, dtrain):
        label = dtrain.get_label()
        sigmoid_pred = 1.0 / (1.0 + np.exp(-pred))

        # For positive class (TDE): weight = alpha * (1-p)^gamma
        # For negative class: weight = (1-alpha) * p^gamma
        # This gives more weight to hard examples AND to the minority class

        # Standard focal loss gradient components
        g1 = sigmoid_pred * (1 - sigmoid_pred)
        g2 = label + ((-1) ** label) * sigmoid_pred
        g3 = sigmoid_pred + label - 1
        g4 = 1 - label - ((-1) ** label) * sigmoid_pred
        g5 = label + ((-1) ** label) * sigmoid_pred

        # Compute alpha weights per sample
        # alpha_t = alpha for positive class, (1-alpha) for negative
        alpha_t = label * self.alpha + (1 - label) * (1 - self.alpha)

        # Robust power function
        def robust_pow(num_base, num_pow):
            return np.sign(num_base) * (np.abs(num_base)) ** num_pow

        # Focal gradient with alpha weighting
        grad = alpha_t * (self.gamma * g3 * robust_pow(g2, self.gamma) * np.log(g4 + 1e-9) +
                         ((-1) ** label) * robust_pow(g5, (self.gamma + 1)))

        # Hessian
        hess_1 = robust_pow(g2, self.gamma) + \
                 self.gamma * ((-1) ** label) * g3 * robust_pow(g2, (self.gamma - 1))
        hess_2 = ((-1) ** label) * g3 * robust_pow(g2, self.gamma) / (g4 + 1e-9)

        hess = alpha_t * ((hess_1 * np.log(g4 + 1e-9) - hess_2) * self.gamma +
                          (self.gamma + 1) * robust_pow(g5, self.gamma)) * g1

        return grad, hess

# ====================
# 4. DEFINE VARIANTS
# ====================
print("\n3. Defining weighted focal loss variants...")

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

# Alpha = weight for positive class
# Higher alpha = penalize missing TDEs more
# Since we have ~20:1 imbalance, alpha should be close to 0.95
variants = {
    'v91a_g2_a75': {
        'gamma': 2.0,
        'alpha': 0.75,
        'description': 'Focal g=2, alpha=0.75'
    },
    'v91b_g2_a85': {
        'gamma': 2.0,
        'alpha': 0.85,
        'description': 'Focal g=2, alpha=0.85'
    },
    'v91c_g2_a90': {
        'gamma': 2.0,
        'alpha': 0.90,
        'description': 'Focal g=2, alpha=0.90'
    },
    'v91d_g2_a95': {
        'gamma': 2.0,
        'alpha': 0.95,
        'description': 'Focal g=2, alpha=0.95 (matches imbalance)'
    },
    'v91e_g1_a90': {
        'gamma': 1.0,
        'alpha': 0.90,
        'description': 'Focal g=1, alpha=0.90 (milder focus)'
    },
    'v91f_g3_a90': {
        'gamma': 3.0,
        'alpha': 0.90,
        'description': 'Focal g=3, alpha=0.90 (stronger focus)'
    },
}

for name, cfg in variants.items():
    print(f"   {name}: {cfg['description']}")

# ====================
# 5. TRAIN ALL VARIANTS
# ====================
print("\n" + "=" * 80)
print("TRAINING WITH WEIGHTED FOCAL LOSS")
print("=" * 80)

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

results = {}

for variant_name, cfg in variants.items():
    print(f"\n   {variant_name}: {cfg['description']}")

    focal_loss = Weighted_Focal_Binary_Loss(gamma=cfg['gamma'], alpha=cfg['alpha'])

    oof_preds = np.zeros(len(y))
    test_preds = np.zeros((len(X_test), n_folds))
    fold_f1s = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y), 1):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        params = base_params.copy()

        dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=available_features)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=available_features)
        dtest = xgb.DMatrix(X_test, feature_names=available_features)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            evals=[(dval, 'val')],
            obj=focal_loss.focal_binary_object,
            early_stopping_rounds=50,
            verbose_eval=False
        )

        raw_val = model.predict(dval)
        raw_test = model.predict(dtest)

        oof_preds[val_idx] = 1.0 / (1.0 + np.exp(-raw_val))
        test_preds[:, fold-1] = 1.0 / (1.0 + np.exp(-raw_test))

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

    # Count hard examples
    tde_mask = y == 1
    hard_count = np.sum(oof_preds[tde_mask] < 0.1)

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
        'hard_tde_count': hard_count,
        'config': cfg
    }

    print(f"      OOF F1: {best_f1:.4f} @ threshold={best_thresh:.3f}")
    print(f"      Recall: {tp/(tp+fn):.4f} | Precision: {tp/(tp+fp):.4f}")
    print(f"      FN: {fn} | FP: {fp} | Hard TDEs (p<0.1): {hard_count}")

# ====================
# 6. RESULTS COMPARISON
# ====================
print("\n" + "=" * 80)
print("RESULTS COMPARISON")
print("=" * 80)

print(f"\n{'Variant':<20} {'OOF F1':<10} {'Recall':<10} {'Precision':<12} {'FN':<6} {'Hard':<6}")
print("-" * 75)
print(f"{'v34a (LB=0.6907)':<20} {'0.6667':<10} {'-':<10} {'-':<12} {'-':<6} {'-':<6}")
print(f"{'v88 (LB=0.6876)':<20} {'0.6854':<10} {'74.3%':<10} {'63.6%':<12} {'38':<6} {'22':<6}")
print(f"{'v90b focal g2':<20} {'0.6222':<10} {'75.7%':<10} {'52.8%':<12} {'36':<6} {'0':<6}")
print("-" * 75)

sorted_results = sorted(results.items(), key=lambda x: -x[1]['oof_f1'])

for name, res in sorted_results:
    print(f"{name:<20} {res['oof_f1']:<10.4f} {100*res['recall']:<10.1f}% {100*res['precision']:<12.1f}% {res['confusion']['fn']:<6} {res['hard_tde_count']:<6}")

# ====================
# 7. BEST MODEL ANALYSIS
# ====================
print("\n" + "=" * 80)
print("BEST MODEL ANALYSIS")
print("=" * 80)

best_name, best_res = sorted_results[0]
print(f"\n   Best variant: {best_name}")
print(f"   OOF F1: {best_res['oof_f1']:.4f}")
print(f"   Recall: {100*best_res['recall']:.1f}%")
print(f"   Precision: {100*best_res['precision']:.1f}%")
print(f"   Hard TDEs (p<0.1): {best_res['hard_tde_count']}")

# TDE probability distribution
tde_preds = best_res['oof_preds'][y == 1]
print(f"\n   TDE probability distribution:")
print(f"      Mean: {np.mean(tde_preds):.4f}")
print(f"      Median: {np.median(tde_preds):.4f}")
print(f"      Min: {np.min(tde_preds):.4f}")
print(f"      Max: {np.max(tde_preds):.4f}")
print(f"      p < 0.2: {np.sum(tde_preds < 0.2)}")
print(f"      p < 0.3: {np.sum(tde_preds < 0.3)}")

# ====================
# 8. SUBMISSIONS
# ====================
print("\n" + "=" * 80)
print("SUBMISSIONS")
print("=" * 80)

for name, res in sorted_results[:3]:  # Top 3 only
    test_binary = (res['test_preds'] > res['threshold']).astype(int)

    submission = pd.DataFrame({
        'object_id': test_ids,
        'target': test_binary
    })

    filename = f"submission_{name}.csv"
    submission.to_csv(base_path / f'submissions/{filename}', index=False)

    expected_lb = res['oof_f1'] + 0.024
    print(f"   {filename}: OOF={res['oof_f1']:.4f}, Expected LB~{expected_lb:.4f}, TDEs={test_binary.sum()}")

# Save artifacts
with open(base_path / 'data/processed/v91_focal_weighted_artifacts.pkl', 'wb') as f:
    pickle.dump(results, f)

print("\n" + "=" * 80)
print("v91 Weighted Focal Loss Complete")
print("=" * 80)

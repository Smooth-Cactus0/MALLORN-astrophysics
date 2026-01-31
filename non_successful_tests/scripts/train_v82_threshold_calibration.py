"""
MALLORN v82: Threshold Calibration

Key insight: Optimal threshold is ~0.07-0.10, not 0.5, due to extreme class imbalance.
The model's raw probabilities may not reflect true probabilities.

Strategies tested:
1. Optimal threshold search on OOF predictions
2. Platt scaling (sigmoid calibration)
3. Isotonic regression calibration
4. Temperature scaling (single parameter)
5. Per-fold threshold optimization

Uses v34a as base model (best LB: 0.6907)
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
base_path = Path(__file__).parent.parent

print("=" * 80)
print("MALLORN v82: Threshold Calibration")
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

print(f"   Training: {len(train_ids)} objects ({np.sum(y)} TDE, {100*np.mean(y):.2f}%)")

# ====================
# 2. LOAD v34a FEATURES
# ====================
print("\n2. Loading v34a features...")

with open(base_path / 'data/processed/v34a_artifacts.pkl', 'rb') as f:
    v34a = pickle.load(f)

feature_names = v34a['feature_names']
print(f"   v34a features: {len(feature_names)}")

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

# Select v34a features
available_features = [f for f in feature_names if f in train_all.columns]
print(f"   Available features: {len(available_features)}")

X_train = train_all[available_features].values
X_test = test_all[available_features].values

# Handle infinities
X_train = np.nan_to_num(X_train, nan=np.nan, posinf=1e10, neginf=-1e10)
X_test = np.nan_to_num(X_test, nan=np.nan, posinf=1e10, neginf=-1e10)

# ====================
# 3. v34a PARAMETERS
# ====================
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 5,
    'learning_rate': 0.025,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'reg_alpha': 0.2,
    'reg_lambda': 1.5,
    'scale_pos_weight': len(y[y==0]) / len(y[y==1]),
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1
}

# ====================
# 4. CALIBRATION METHODS
# ====================

def platt_scaling(probs, labels, test_probs):
    """Platt scaling: fit sigmoid to probability outputs"""
    from sklearn.linear_model import LogisticRegression

    # Fit sigmoid
    lr = LogisticRegression(solver='lbfgs')
    lr.fit(probs.reshape(-1, 1), labels)

    # Transform
    calibrated = lr.predict_proba(test_probs.reshape(-1, 1))[:, 1]
    return calibrated

def isotonic_calibration(probs, labels, test_probs):
    """Isotonic regression: non-parametric monotonic calibration"""
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(probs, labels)

    calibrated = ir.predict(test_probs)
    return calibrated

def temperature_scaling(probs, labels):
    """Temperature scaling: find optimal T for softmax temperature"""
    best_t = 1.0
    best_brier = float('inf')

    for t in np.linspace(0.1, 5.0, 50):
        # Apply temperature: p_scaled = sigmoid(logit(p) / T)
        logits = np.log(probs / (1 - probs + 1e-10))
        scaled_logits = logits / t
        scaled_probs = 1 / (1 + np.exp(-scaled_logits))

        brier = brier_score_loss(labels, scaled_probs)
        if brier < best_brier:
            best_brier = brier
            best_t = t

    return best_t

def apply_temperature(probs, temperature):
    """Apply temperature scaling to probabilities"""
    logits = np.log(probs / (1 - probs + 1e-10))
    scaled_logits = logits / temperature
    scaled_probs = 1 / (1 + np.exp(-scaled_logits))
    return scaled_probs

def find_optimal_threshold(probs, labels, n_points=500):
    """Find threshold that maximizes F1"""
    best_f1 = 0
    best_thresh = 0.5

    for t in np.linspace(0.01, 0.5, n_points):
        preds = (probs > t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    return best_thresh, best_f1

# ====================
# 5. CROSS-VALIDATION WITH CALIBRATION
# ====================
print("\n" + "=" * 80)
print("TRAINING WITH CALIBRATION")
print("=" * 80)

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Store OOF predictions for each method
oof_raw = np.zeros(len(y))
oof_platt = np.zeros(len(y))
oof_isotonic = np.zeros(len(y))
oof_temperature = np.zeros(len(y))

test_raw = np.zeros((len(X_test), n_folds))
test_platt = np.zeros((len(X_test), n_folds))
test_isotonic = np.zeros((len(X_test), n_folds))
test_temperature = np.zeros((len(X_test), n_folds))

fold_thresholds = []
temperatures = []

print("\nTraining folds...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y), 1):
    print(f"\n   Fold {fold}/5...")

    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    # Split training into train/calibration (for Platt and Isotonic)
    n_calib = len(X_tr) // 5
    X_train_fold = X_tr[n_calib:]
    y_train_fold = y_tr[n_calib:]
    X_calib = X_tr[:n_calib]
    y_calib = y_tr[:n_calib]

    # Train XGBoost
    dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold, feature_names=available_features)
    dcalib = xgb.DMatrix(X_calib, feature_names=available_features)
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

    # Get raw predictions
    raw_calib = model.predict(dcalib)
    raw_val = model.predict(dval)
    raw_test = model.predict(dtest)

    # Store raw OOF
    oof_raw[val_idx] = raw_val
    test_raw[:, fold-1] = raw_test

    # Platt scaling (trained on calibration set)
    platt_val = platt_scaling(raw_calib, y_calib, raw_val)
    platt_test = platt_scaling(raw_calib, y_calib, raw_test)
    oof_platt[val_idx] = platt_val
    test_platt[:, fold-1] = platt_test

    # Isotonic calibration
    iso_val = isotonic_calibration(raw_calib, y_calib, raw_val)
    iso_test = isotonic_calibration(raw_calib, y_calib, raw_test)
    oof_isotonic[val_idx] = iso_val
    test_isotonic[:, fold-1] = iso_test

    # Temperature scaling
    temp = temperature_scaling(raw_calib, y_calib)
    temperatures.append(temp)
    temp_val = apply_temperature(raw_val, temp)
    temp_test = apply_temperature(raw_test, temp)
    oof_temperature[val_idx] = temp_val
    test_temperature[:, fold-1] = temp_test

    # Fold-specific optimal threshold
    thresh, f1 = find_optimal_threshold(raw_val, y_val)
    fold_thresholds.append(thresh)

    print(f"      Raw val F1: {f1:.4f} @ {thresh:.3f}")
    print(f"      Temperature: {temp:.2f}")

# ====================
# 6. RESULTS ANALYSIS
# ====================
print("\n" + "=" * 80)
print("CALIBRATION RESULTS")
print("=" * 80)

methods = {
    'Raw (v34a baseline)': (oof_raw, test_raw.mean(axis=1)),
    'Platt Scaling': (oof_platt, test_platt.mean(axis=1)),
    'Isotonic Regression': (oof_isotonic, test_isotonic.mean(axis=1)),
    'Temperature Scaling': (oof_temperature, test_temperature.mean(axis=1)),
}

print(f"\n{'Method':<25} {'Brier Score':<12} {'Best F1':<10} {'Threshold':<10}")
print("-" * 60)

results = {}
for name, (oof, test_preds) in methods.items():
    brier = brier_score_loss(y, oof)
    thresh, f1 = find_optimal_threshold(oof, y)

    results[name] = {
        'brier': brier,
        'f1': f1,
        'threshold': thresh,
        'oof': oof,
        'test_preds': test_preds
    }

    print(f"{name:<25} {brier:<12.4f} {f1:<10.4f} {thresh:<10.3f}")

# Fold threshold analysis
print(f"\nPer-fold threshold analysis (raw predictions):")
print(f"   Thresholds: {[f'{t:.3f}' for t in fold_thresholds]}")
print(f"   Mean: {np.mean(fold_thresholds):.3f}, Std: {np.std(fold_thresholds):.3f}")
print(f"   Temperatures: {[f'{t:.2f}' for t in temperatures]}")

# ====================
# 7. THRESHOLD STRATEGIES
# ====================
print("\n" + "=" * 80)
print("THRESHOLD STRATEGIES")
print("=" * 80)

# Strategy 1: Global optimal
global_thresh, global_f1 = find_optimal_threshold(oof_raw, y)
print(f"\n1. Global optimal threshold: {global_thresh:.3f} (OOF F1: {global_f1:.4f})")

# Strategy 2: Mean of fold thresholds
mean_thresh = np.mean(fold_thresholds)
mean_thresh_f1 = f1_score(y, (oof_raw > mean_thresh).astype(int))
print(f"2. Mean fold threshold: {mean_thresh:.3f} (OOF F1: {mean_thresh_f1:.4f})")

# Strategy 3: Conservative (lower recall, higher precision)
conservative_thresh = global_thresh * 1.5
conservative_f1 = f1_score(y, (oof_raw > conservative_thresh).astype(int))
print(f"3. Conservative threshold: {conservative_thresh:.3f} (OOF F1: {conservative_f1:.4f})")

# Strategy 4: Aggressive (higher recall, lower precision)
aggressive_thresh = global_thresh * 0.7
aggressive_f1 = f1_score(y, (oof_raw > aggressive_thresh).astype(int))
print(f"4. Aggressive threshold: {aggressive_thresh:.3f} (OOF F1: {aggressive_f1:.4f})")

# Strategy 5: Based on class ratio
class_ratio = np.mean(y)
ratio_thresh = class_ratio
ratio_f1 = f1_score(y, (oof_raw > ratio_thresh).astype(int))
print(f"5. Class ratio threshold: {ratio_thresh:.4f} (OOF F1: {ratio_f1:.4f})")

# ====================
# 8. SUBMISSIONS
# ====================
print("\n" + "=" * 80)
print("SUBMISSIONS")
print("=" * 80)

submissions = {
    'v82a_global_optimal': (test_raw.mean(axis=1), global_thresh),
    'v82b_platt': (test_platt.mean(axis=1), results['Platt Scaling']['threshold']),
    'v82c_isotonic': (test_isotonic.mean(axis=1), results['Isotonic Regression']['threshold']),
    'v82d_conservative': (test_raw.mean(axis=1), conservative_thresh),
    'v82e_aggressive': (test_raw.mean(axis=1), aggressive_thresh),
}

for name, (preds, thresh) in submissions.items():
    binary = (preds > thresh).astype(int)

    submission = pd.DataFrame({
        'object_id': test_ids,
        'target': binary
    })

    filename = f"submission_{name}.csv"
    submission.to_csv(base_path / f'submissions/{filename}', index=False)
    print(f"   {filename}: threshold={thresh:.3f}, TDEs={binary.sum()}")

# Save artifacts
with open(base_path / 'data/processed/v82_calibration_artifacts.pkl', 'wb') as f:
    pickle.dump({
        'results': results,
        'fold_thresholds': fold_thresholds,
        'temperatures': temperatures,
        'submissions': submissions
    }, f)

print("\n" + "=" * 80)
print("v82 Complete")
print("=" * 80)

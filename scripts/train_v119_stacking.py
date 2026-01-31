"""
MALLORN v119: Meta-Learner Stacking Ensemble
=============================================

Instead of fixed weights, train a meta-learner to optimally combine
predictions from Level 0 models (XGBoost, LightGBM, CatBoost).

Strategy:
1. Load OOF predictions from all models
2. Train meta-learner on OOF predictions (Level 1)
3. Apply to test predictions
4. Compare with weighted averaging

Level 0 Models:
- v92d XGBoost (5-seed avg)
- v34a XGBoost (5-seed avg)
- v114d LightGBM (5-seed avg)
- v118 CatBoost (5-seed avg)

Level 1 Meta-learners:
- Logistic Regression (simple, interpretable)
- Ridge Classifier (regularized)
- Small XGBoost (2-layer stacking)
"""

import sys
import pickle
import gzip
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import f1_score
import xgboost as xgb
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

base_path = Path(__file__).parent.parent

print("=" * 70)
print("MALLORN v119: Meta-Learner Stacking Ensemble")
print("=" * 70)

# ============================================================================
# 1. LOAD OOF AND TEST PREDICTIONS FROM ALL MODELS
# ============================================================================
print("\n[1/5] Loading predictions from all models...")

# Load package for basic info
with gzip.open(base_path / 'data/kaggle_ensemble_package.pkl.gz', 'rb') as f:
    package = pickle.load(f)
y = package['y']
test_ids = package['test_ids']

# Load v92d and v34a from the multi-seed run
# We need to reconstruct from the ensemble run or load from artifacts
# Let's check what we have

# Load from the ensemble artifacts if available, otherwise compute
try:
    # Try loading from a combined file
    with open(base_path / 'data/processed/multiseed_ensemble_artifacts.pkl', 'rb') as f:
        ensemble_arts = pickle.load(f)
    print("   Loaded from multiseed_ensemble_artifacts.pkl")
except FileNotFoundError:
    print("   Computing multi-seed predictions...")

    # We need to recompute the multi-seed predictions
    # Load original artifacts
    with open(base_path / 'data/processed/v34a_artifacts.pkl', 'rb') as f:
        v34a_arts = pickle.load(f)

    with open(base_path / 'data/processed/v92_focal_adv_artifacts.pkl', 'rb') as f:
        v92_arts = pickle.load(f)

    with open(base_path / 'data/processed/v114_optimized_artifacts.pkl', 'rb') as f:
        v114_arts = pickle.load(f)

    # For now, use single-seed OOF predictions
    # v92d
    v92d_oof = v92_arts['v92d_baseline_adv']['oof_preds']
    v92d_test = v92_arts['v92d_baseline_adv']['test_preds']

    # v34a
    v34a_oof = v34a_arts['oof_preds']
    v34a_test = v34a_arts['test_preds']

    # v114d
    v114d_oof = v114_arts['results']['v114d_minimal_research']['oof_preds']
    v114d_test = v114_arts['results']['v114d_minimal_research']['test_preds']

    ensemble_arts = {
        'v92d': {'oof': v92d_oof, 'test': v92d_test},
        'v34a': {'oof': v34a_oof, 'test': v34a_test},
        'v114d': {'oof': v114d_oof, 'test': v114d_test},
    }

# Load CatBoost
with open(base_path / 'data/processed/v118_catboost_artifacts.pkl', 'rb') as f:
    catboost_arts = pickle.load(f)

# Collect all predictions
models = {}
models['v92d'] = {
    'oof': ensemble_arts['v92d']['oof'] if 'v92d' in ensemble_arts else ensemble_arts.get('v92d_oof'),
    'test': ensemble_arts['v92d']['test'] if 'v92d' in ensemble_arts else ensemble_arts.get('v92d_test'),
    'lb_score': 0.6986
}
models['v34a'] = {
    'oof': ensemble_arts['v34a']['oof'] if 'v34a' in ensemble_arts else ensemble_arts.get('v34a_oof'),
    'test': ensemble_arts['v34a']['test'] if 'v34a' in ensemble_arts else ensemble_arts.get('v34a_test'),
    'lb_score': 0.6907
}
models['v114d'] = {
    'oof': ensemble_arts['v114d']['oof'] if 'v114d' in ensemble_arts else ensemble_arts.get('v114d_oof'),
    'test': ensemble_arts['v114d']['test'] if 'v114d' in ensemble_arts else ensemble_arts.get('v114d_test'),
    'lb_score': 0.6797
}
models['v118_catboost'] = {
    'oof': catboost_arts['avg_oof_preds'],
    'test': catboost_arts['avg_test_preds'],
    'lb_score': None  # Unknown
}

print(f"\n   Models loaded:")
for name, data in models.items():
    oof_f1 = 0
    for t in np.linspace(0.05, 0.5, 50):
        f1 = f1_score(y, (data['oof'] > t).astype(int))
        oof_f1 = max(oof_f1, f1)
    print(f"   - {name}: OOF shape={data['oof'].shape}, OOF F1={oof_f1:.4f}, LB={data['lb_score']}")

# ============================================================================
# 2. CREATE STACKING FEATURES
# ============================================================================
print("\n[2/5] Creating stacking features...")

# Stack OOF predictions as features for meta-learner
model_names = list(models.keys())
X_meta_train = np.column_stack([models[name]['oof'] for name in model_names])
X_meta_test = np.column_stack([models[name]['test'] for name in model_names])

print(f"   Meta-train shape: {X_meta_train.shape}")
print(f"   Meta-test shape: {X_meta_test.shape}")
print(f"   Feature names: {model_names}")

# Also create interaction features
print("   Adding interaction features...")
# Product of probabilities (agreement signal)
X_meta_train_extended = X_meta_train.copy()
X_meta_test_extended = X_meta_test.copy()

# Add mean and std
mean_pred_train = X_meta_train.mean(axis=1, keepdims=True)
std_pred_train = X_meta_train.std(axis=1, keepdims=True)
mean_pred_test = X_meta_test.mean(axis=1, keepdims=True)
std_pred_test = X_meta_test.std(axis=1, keepdims=True)

X_meta_train_extended = np.hstack([X_meta_train_extended, mean_pred_train, std_pred_train])
X_meta_test_extended = np.hstack([X_meta_test_extended, mean_pred_test, std_pred_test])

print(f"   Extended meta-train shape: {X_meta_train_extended.shape}")

# ============================================================================
# 3. TRAIN META-LEARNERS
# ============================================================================
print("\n[3/5] Training meta-learners...")

def find_best_threshold(y_true, y_pred):
    best_f1, best_t = 0, 0.1
    for t in np.linspace(0.03, 0.6, 100):
        f1 = f1_score(y_true, (y_pred > t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1

CV_SEED = 42
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)

results = {}

# --- Logistic Regression ---
print("\n   [A] Logistic Regression...")
lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)

# Get OOF predictions for meta-learner
lr_oof = cross_val_predict(lr, X_meta_train, y, cv=skf, method='predict_proba')[:, 1]
lr.fit(X_meta_train, y)
lr_test = lr.predict_proba(X_meta_test)[:, 1]

threshold, oof_f1 = find_best_threshold(y, lr_oof)
print(f"       OOF F1: {oof_f1:.4f} @ threshold {threshold:.3f}")
print(f"       Coefficients: {dict(zip(model_names, lr.coef_[0].round(3)))}")

results['lr'] = {'oof': lr_oof, 'test': lr_test, 'threshold': threshold, 'oof_f1': oof_f1}

# --- Logistic Regression with extended features ---
print("\n   [B] Logistic Regression (extended features)...")
lr_ext = LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=42)

lr_ext_oof = cross_val_predict(lr_ext, X_meta_train_extended, y, cv=skf, method='predict_proba')[:, 1]
lr_ext.fit(X_meta_train_extended, y)
lr_ext_test = lr_ext.predict_proba(X_meta_test_extended)[:, 1]

threshold, oof_f1 = find_best_threshold(y, lr_ext_oof)
print(f"       OOF F1: {oof_f1:.4f} @ threshold {threshold:.3f}")

results['lr_extended'] = {'oof': lr_ext_oof, 'test': lr_ext_test, 'threshold': threshold, 'oof_f1': oof_f1}

# --- Ridge Classifier ---
print("\n   [C] Ridge Classifier...")
ridge = RidgeClassifier(alpha=1.0, class_weight='balanced', random_state=42)

ridge_oof = cross_val_predict(ridge, X_meta_train, y, cv=skf, method='decision_function')
# Normalize to 0-1 range
ridge_oof = (ridge_oof - ridge_oof.min()) / (ridge_oof.max() - ridge_oof.min())

ridge.fit(X_meta_train, y)
ridge_test = ridge.decision_function(X_meta_test)
ridge_test = (ridge_test - ridge_test.min()) / (ridge_test.max() - ridge_test.min())

threshold, oof_f1 = find_best_threshold(y, ridge_oof)
print(f"       OOF F1: {oof_f1:.4f} @ threshold {threshold:.3f}")

results['ridge'] = {'oof': ridge_oof, 'test': ridge_test, 'threshold': threshold, 'oof_f1': oof_f1}

# --- Small XGBoost meta-learner ---
print("\n   [D] XGBoost meta-learner...")
xgb_meta_params = {
    'objective': 'binary:logistic',
    'max_depth': 2,
    'learning_rate': 0.05,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.5,
    'reg_lambda': 1.0,
    'scale_pos_weight': len(y[y==0]) / len(y[y==1]),
    'random_state': 42,
    'n_jobs': -1,
}

xgb_meta_oof = np.zeros(len(y))
xgb_meta_test = np.zeros(len(X_meta_test))

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_meta_train_extended, y)):
    X_tr = X_meta_train_extended[train_idx]
    X_val = X_meta_train_extended[val_idx]
    y_tr = y[train_idx]

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val)
    dtest = xgb.DMatrix(X_meta_test_extended)

    model = xgb.train(
        xgb_meta_params,
        dtrain,
        num_boost_round=100,
        verbose_eval=False
    )

    xgb_meta_oof[val_idx] = model.predict(dval)
    xgb_meta_test += model.predict(dtest) / N_FOLDS

threshold, oof_f1 = find_best_threshold(y, xgb_meta_oof)
print(f"       OOF F1: {oof_f1:.4f} @ threshold {threshold:.3f}")

results['xgb_meta'] = {'oof': xgb_meta_oof, 'test': xgb_meta_test, 'threshold': threshold, 'oof_f1': oof_f1}

# ============================================================================
# 4. COMPARE WITH SIMPLE WEIGHTED AVERAGE
# ============================================================================
print("\n[4/5] Comparing with weighted averages...")

# Simple average
simple_avg_oof = X_meta_train.mean(axis=1)
simple_avg_test = X_meta_test.mean(axis=1)
threshold, oof_f1 = find_best_threshold(y, simple_avg_oof)
print(f"\n   [E] Simple Average (equal weights):")
print(f"       OOF F1: {oof_f1:.4f} @ threshold {threshold:.3f}")
results['simple_avg'] = {'oof': simple_avg_oof, 'test': simple_avg_test, 'threshold': threshold, 'oof_f1': oof_f1}

# LB-weighted average
lb_weights = np.array([0.6986, 0.6907, 0.6797, 0.65])  # v92d, v34a, v114d, catboost
lb_weights = lb_weights / lb_weights.sum()
print(f"   LB weights: {dict(zip(model_names, lb_weights.round(3)))}")

lb_weighted_oof = np.average(X_meta_train, axis=1, weights=lb_weights)
lb_weighted_test = np.average(X_meta_test, axis=1, weights=lb_weights)
threshold, oof_f1 = find_best_threshold(y, lb_weighted_oof)
print(f"\n   [F] LB-Weighted Average:")
print(f"       OOF F1: {oof_f1:.4f} @ threshold {threshold:.3f}")
results['lb_weighted'] = {'oof': lb_weighted_oof, 'test': lb_weighted_test, 'threshold': threshold, 'oof_f1': oof_f1}

# Optimized weights (using Logistic Regression coefficients as proxy)
print(f"\n   [G] LR-Learned Weights (from coefficients):")
lr_weights = np.abs(lr.coef_[0])
lr_weights = lr_weights / lr_weights.sum()
print(f"       Learned weights: {dict(zip(model_names, lr_weights.round(3)))}")

lr_learned_oof = np.average(X_meta_train, axis=1, weights=lr_weights)
lr_learned_test = np.average(X_meta_test, axis=1, weights=lr_weights)
threshold, oof_f1 = find_best_threshold(y, lr_learned_oof)
print(f"       OOF F1: {oof_f1:.4f} @ threshold {threshold:.3f}")
results['lr_learned_weights'] = {'oof': lr_learned_oof, 'test': lr_learned_test, 'threshold': threshold, 'oof_f1': oof_f1}

# ============================================================================
# 5. SAVE RESULTS AND SUBMISSIONS
# ============================================================================
print("\n[5/5] Saving results...")

# Find best method
print("\n   === RESULTS SUMMARY ===")
print(f"   {'Method':<25} {'OOF F1':<10} {'Threshold':<10}")
print("   " + "-" * 45)

best_method = None
best_f1 = 0
for method, data in sorted(results.items(), key=lambda x: x[1]['oof_f1'], reverse=True):
    print(f"   {method:<25} {data['oof_f1']:.4f}     {data['threshold']:.3f}")
    if data['oof_f1'] > best_f1:
        best_f1 = data['oof_f1']
        best_method = method

print(f"\n   Best method: {best_method} (OOF F1: {best_f1:.4f})")

# Save artifacts
artifacts = {
    'results': results,
    'model_names': model_names,
    'X_meta_train': X_meta_train,
    'X_meta_test': X_meta_test,
    'best_method': best_method,
}

with open(base_path / 'data/processed/v119_stacking_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)
print(f"\n   Saved artifacts to data/processed/v119_stacking_artifacts.pkl")

# Save submissions for top methods
for method in ['lr', 'xgb_meta', 'lb_weighted', best_method]:
    if method in results:
        data = results[method]
        binary_preds = (data['test'] > data['threshold']).astype(int)
        submission = pd.DataFrame({
            'object_id': test_ids,
            'target': binary_preds
        })
        filename = f'submission_v119_stack_{method}.csv'
        submission.to_csv(base_path / 'submissions' / filename, index=False)
        print(f"   Saved {filename}: {binary_preds.sum()} TDEs")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("STACKING COMPLETE")
print("=" * 70)

print(f"""
Key Findings:
   Best stacking method: {best_method}
   Best OOF F1: {best_f1:.4f}

   Individual model OOF F1s:
""")
for name in model_names:
    oof_f1 = 0
    for t in np.linspace(0.05, 0.5, 50):
        f1 = f1_score(y, (models[name]['oof'] > t).astype(int))
        oof_f1 = max(oof_f1, f1)
    print(f"      {name}: {oof_f1:.4f}")

print(f"""
   Stacking improvement over best individual: {best_f1 - max([results[m]['oof_f1'] for m in ['simple_avg']]):.4f}

Next steps:
   1. Submit stacking results to get LB scores
   2. Compare with simple weighted average
   3. If stacking helps, use it in final ensemble
""")

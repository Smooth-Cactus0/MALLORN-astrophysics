"""
MALLORN v70: Rank Averaging Ensemble

Key insight: v34a generalizes well (OOF 0.6667 -> LB 0.6907).
Adding features to v34a hurts (v65: OOF 0.6780 -> LB 0.6344).

Solution: Keep v34a UNCHANGED. Train separate models. Rank average.

Ensemble members:
1. v34a (unchanged) - Our best generalizing model
2. MaxVar-only model - Captures transient amplitude
3. Bazin-only model - Captures lightcurve shape

Rank averaging: Convert predictions to ranks, average, use threshold.
This is more robust than probability averaging because it's scale-invariant.
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from scipy.stats import rankdata
import xgboost as xgb
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
base_path = Path(__file__).parent.parent

print("=" * 80, flush=True)
print("MALLORN v70: Rank Averaging Ensemble", flush=True)
print("=" * 80, flush=True)

# ====================
# 1. LOAD DATA
# ====================
print("\n1. Loading data...", flush=True)

from utils.data_loader import load_all_data
data = load_all_data()

train_meta = data['train_meta']
test_meta = data['test_meta']
train_ids = train_meta['object_id'].tolist()
test_ids = test_meta['object_id'].tolist()
y = train_meta['target'].values

print(f"   Training: {len(train_ids)} ({np.sum(y)} TDE)", flush=True)

# ====================
# 2. LOAD v34a PREDICTIONS (UNCHANGED)
# ====================
print("\n2. Loading v34a predictions...", flush=True)

with open(base_path / 'data/processed/v34a_artifacts.pkl', 'rb') as f:
    v34a = pickle.load(f)

v34a_oof = v34a['oof_preds']
v34a_test = v34a['test_preds']
v34a_f1 = v34a['oof_f1']
print(f"   v34a OOF F1: {v34a_f1:.4f}", flush=True)

# ====================
# 3. TRAIN MAXVAR-ONLY MODEL
# ====================
print("\n3. Training MaxVar-only model...", flush=True)

with open(base_path / 'data/processed/powerlaw_features_cache.pkl', 'rb') as f:
    pl_cache = pickle.load(f)
train_pl = pl_cache['train']
test_pl = pl_cache['test']

maxvar_cols = [c for c in train_pl.columns if c != 'object_id']
X_maxvar = train_pl[maxvar_cols].values
X_maxvar_test = test_pl[maxvar_cols].values
X_maxvar = np.nan_to_num(X_maxvar, nan=0, posinf=1e10, neginf=-1e10)
X_maxvar_test = np.nan_to_num(X_maxvar_test, nan=0, posinf=1e10, neginf=-1e10)

xgb_params = {
    'objective': 'binary:logistic',
    'max_depth': 4,
    'learning_rate': 0.03,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': len(y[y==0]) / len(y[y==1]),
    'random_state': 42
}

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

oof_maxvar = np.zeros(len(y))
test_maxvar = np.zeros(len(X_maxvar_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_maxvar, y), 1):
    X_tr, X_val = X_maxvar[train_idx], X_maxvar[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=maxvar_cols)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=maxvar_cols)
    dtest = xgb.DMatrix(X_maxvar_test, feature_names=maxvar_cols)

    model = xgb.train(xgb_params, dtrain, num_boost_round=200,
                      evals=[(dval, 'val')], early_stopping_rounds=30, verbose_eval=False)

    oof_maxvar[val_idx] = model.predict(dval)
    test_maxvar += model.predict(dtest) / n_folds

best_f1_maxvar = max(f1_score(y, (oof_maxvar > t).astype(int)) for t in np.linspace(0.05, 0.5, 50))
print(f"   MaxVar-only OOF F1: {best_f1_maxvar:.4f}", flush=True)

# ====================
# 4. TRAIN BAZIN-ONLY MODEL
# ====================
print("\n4. Training Bazin-only model...", flush=True)

with open(base_path / 'data/processed/bazin_features_cache.pkl', 'rb') as f:
    bazin_cache = pickle.load(f)
train_bazin = bazin_cache['train']
test_bazin = bazin_cache['test']

bazin_cols = [c for c in train_bazin.columns if c != 'object_id']
X_bazin = train_bazin[bazin_cols].values
X_bazin_test = test_bazin[bazin_cols].values
X_bazin = np.nan_to_num(X_bazin, nan=0, posinf=1e10, neginf=-1e10)
X_bazin_test = np.nan_to_num(X_bazin_test, nan=0, posinf=1e10, neginf=-1e10)

oof_bazin = np.zeros(len(y))
test_bazin_pred = np.zeros(len(X_bazin_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_bazin, y), 1):
    X_tr, X_val = X_bazin[train_idx], X_bazin[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=bazin_cols)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=bazin_cols)
    dtest = xgb.DMatrix(X_bazin_test, feature_names=bazin_cols)

    model = xgb.train(xgb_params, dtrain, num_boost_round=200,
                      evals=[(dval, 'val')], early_stopping_rounds=30, verbose_eval=False)

    oof_bazin[val_idx] = model.predict(dval)
    test_bazin_pred += model.predict(dtest) / n_folds

best_f1_bazin = max(f1_score(y, (oof_bazin > t).astype(int)) for t in np.linspace(0.05, 0.5, 50))
print(f"   Bazin-only OOF F1: {best_f1_bazin:.4f}", flush=True)

# ====================
# 5. RANK AVERAGING ENSEMBLE
# ====================
print("\n" + "=" * 80, flush=True)
print("RANK AVERAGING ENSEMBLE", flush=True)
print("=" * 80, flush=True)

def rank_average(preds_list, weights=None):
    """Rank average multiple prediction arrays."""
    n = len(preds_list[0])
    if weights is None:
        weights = [1.0] * len(preds_list)
    weights = np.array(weights) / sum(weights)

    ranks = np.zeros(n)
    for preds, w in zip(preds_list, weights):
        ranks += w * rankdata(preds) / n
    return ranks

# Try different weightings
print("\n   Testing ensemble weightings...", flush=True)

best_result = {'f1': 0, 'weights': None, 'thresh': 0.5}

# Various weighting schemes
weight_schemes = [
    [1.0, 0.0, 0.0],      # v34a only
    [0.8, 0.1, 0.1],      # v34a dominant
    [0.7, 0.15, 0.15],    # v34a heavy
    [0.6, 0.2, 0.2],      # Balanced-ish
    [0.5, 0.25, 0.25],    # Equal-ish
    [0.7, 0.2, 0.1],      # v34a + MaxVar
    [0.7, 0.1, 0.2],      # v34a + Bazin
    [0.8, 0.2, 0.0],      # v34a + MaxVar only
    [0.9, 0.1, 0.0],      # v34a heavy + MaxVar
    [0.85, 0.15, 0.0],    # v34a very heavy + MaxVar
]

for weights in weight_schemes:
    oof_rank = rank_average([v34a_oof, oof_maxvar, oof_bazin], weights)

    for thresh in np.linspace(0.3, 0.7, 50):
        pred = (oof_rank > thresh).astype(int)
        f1 = f1_score(y, pred)

        if f1 > best_result['f1']:
            best_result['f1'] = f1
            best_result['weights'] = weights
            best_result['thresh'] = thresh
            best_result['oof'] = oof_rank.copy()

print(f"\n   Best weights: v34a={best_result['weights'][0]:.2f}, "
      f"MaxVar={best_result['weights'][1]:.2f}, Bazin={best_result['weights'][2]:.2f}", flush=True)
print(f"   Best threshold: {best_result['thresh']:.3f}", flush=True)
print(f"   OOF F1: {best_result['f1']:.4f}", flush=True)

# Also try probability average (not rank)
print("\n   Also testing probability average...", flush=True)

best_prob_result = {'f1': 0}
for weights in weight_schemes:
    w = np.array(weights) / sum(weights)
    oof_prob = w[0]*v34a_oof + w[1]*oof_maxvar + w[2]*oof_bazin

    for thresh in np.linspace(0.05, 0.3, 50):
        pred = (oof_prob > thresh).astype(int)
        f1 = f1_score(y, pred)

        if f1 > best_prob_result['f1']:
            best_prob_result['f1'] = f1
            best_prob_result['weights'] = weights
            best_prob_result['thresh'] = thresh
            best_prob_result['oof'] = oof_prob.copy()

print(f"   Prob avg best: F1={best_prob_result['f1']:.4f} (weights={best_prob_result['weights']})", flush=True)

# Choose best overall
if best_prob_result['f1'] > best_result['f1']:
    print("\n   Probability average wins!", flush=True)
    final_method = 'prob_avg'
    final_result = best_prob_result
    test_final = (final_result['weights'][0]*v34a_test +
                  final_result['weights'][1]*test_maxvar +
                  final_result['weights'][2]*test_bazin_pred)
else:
    print("\n   Rank average wins!", flush=True)
    final_method = 'rank_avg'
    final_result = best_result
    test_final = rank_average([v34a_test, test_maxvar, test_bazin_pred], final_result['weights'])

# ====================
# 6. FINAL RESULTS
# ====================
print("\n" + "=" * 80, flush=True)
print("FINAL RESULTS", flush=True)
print("=" * 80, flush=True)

print(f"\n   Method: {final_method}", flush=True)
print(f"   Weights: {final_result['weights']}", flush=True)
print(f"   OOF F1: {final_result['f1']:.4f}", flush=True)

final_pred = (final_result['oof'] > final_result['thresh']).astype(int)
tp = np.sum((final_pred == 1) & (y == 1))
fp = np.sum((final_pred == 1) & (y == 0))
fn = np.sum((final_pred == 0) & (y == 1))
print(f"\n   TP={tp}, FP={fp}, FN={fn}", flush=True)
print(f"   Precision: {tp/(tp+fp):.4f}, Recall: {tp/(tp+fn):.4f}", flush=True)

# ====================
# 7. SUBMISSION
# ====================
print("\n" + "=" * 80, flush=True)
print("SUBMISSION", flush=True)
print("=" * 80, flush=True)

test_binary = (test_final > final_result['thresh']).astype(int)

submission = pd.DataFrame({
    'object_id': test_ids,
    'target': test_binary
})

submission_path = base_path / 'submissions/submission_v70_rank_ensemble.csv'
submission.to_csv(submission_path, index=False)

print(f"   Saved: {submission_path.name}", flush=True)
print(f"   Predicted TDEs: {test_binary.sum()}", flush=True)

# Save artifacts
artifacts = {
    'v34a_oof': v34a_oof, 'v34a_test': v34a_test,
    'maxvar_oof': oof_maxvar, 'maxvar_test': test_maxvar,
    'bazin_oof': oof_bazin, 'bazin_test': test_bazin_pred,
    'final_result': final_result, 'final_method': final_method
}

with open(base_path / 'data/processed/v70_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

# Comparison
print("\n" + "=" * 80, flush=True)
print("COMPARISON", flush=True)
print("=" * 80, flush=True)

print(f"""
   Model                OOF F1   LB F1
   -----                ------   -----
   v34a (baseline)      0.6667   0.6907  <-- Best LB
   v65 (MaxVar added)   0.6780   0.6344  (overfit!)
   v70 (rank ensemble)  {final_result['f1']:.4f}   ???

   Key: v70 uses v34a UNCHANGED + separate models via rank averaging.
   This should preserve v34a's generalization while adding diversity.
""", flush=True)

print("=" * 80, flush=True)
print("v70 Complete", flush=True)
print("=" * 80, flush=True)

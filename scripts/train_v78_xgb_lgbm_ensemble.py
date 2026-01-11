"""
MALLORN v78: XGBoost + LightGBM Ensemble

Combines v34a (Optuna-tuned XGBoost) with v77 (Optuna-tuned LightGBM).

Ensemble strategies:
1. Simple average of probabilities
2. Weighted average (optimized on OOF)
3. Rank averaging (robust to calibration differences)

Both models use the same v34a feature set (223 features).
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score
from scipy.stats import rankdata
from scipy.optimize import minimize_scalar
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
base_path = Path(__file__).parent.parent

print("=" * 80, flush=True)
print("MALLORN v78: XGBoost + LightGBM Ensemble", flush=True)
print("=" * 80, flush=True)

# ====================
# 1. LOAD ARTIFACTS
# ====================
print("\n1. Loading model artifacts...", flush=True)

# Load v34a XGBoost
with open(base_path / 'data/processed/v34a_artifacts.pkl', 'rb') as f:
    v34a = pickle.load(f)

print(f"   v34a XGBoost: OOF F1={v34a['oof_f1']:.4f}", flush=True)

# Load v77 LightGBM
with open(base_path / 'data/processed/v77_artifacts.pkl', 'rb') as f:
    v77 = pickle.load(f)

print(f"   v77 LightGBM: OOF F1={v77['oof_f1']:.4f}", flush=True)

# Load ground truth
from utils.data_loader import load_all_data
data = load_all_data()
train_meta = data['train_meta']
test_meta = data['test_meta']
test_ids = test_meta['object_id'].tolist()
y = train_meta['target'].values

print(f"   Training samples: {len(y)} ({np.sum(y)} TDE)", flush=True)

# ====================
# 2. ENSEMBLE METHODS
# ====================
print("\n2. Testing ensemble methods...", flush=True)

xgb_oof = v34a['oof_preds']
lgb_oof = v77['oof_preds']
xgb_test = v34a['test_preds']
lgb_test = v77['test_preds']

results = {}

# Method 1: Simple Average
print("\n   Method 1: Simple Average", flush=True)
avg_oof = (xgb_oof + lgb_oof) / 2
avg_test = (xgb_test + lgb_test) / 2

best_f1, best_thresh = 0, 0.1
for t in np.linspace(0.03, 0.5, 200):
    f1 = f1_score(y, (avg_oof > t).astype(int))
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

results['simple_avg'] = {
    'oof_f1': best_f1,
    'threshold': best_thresh,
    'oof_preds': avg_oof,
    'test_preds': avg_test
}
print(f"      OOF F1: {best_f1:.4f} @ threshold={best_thresh:.3f}", flush=True)

# Method 2: Weighted Average (optimized)
print("\n   Method 2: Weighted Average (optimized)", flush=True)

def weighted_f1(w):
    weighted_oof = w * xgb_oof + (1-w) * lgb_oof
    best_f1 = 0
    for t in np.linspace(0.03, 0.5, 100):
        f1 = f1_score(y, (weighted_oof > t).astype(int))
        if f1 > best_f1:
            best_f1 = f1
    return -best_f1  # Minimize negative F1

result = minimize_scalar(weighted_f1, bounds=(0, 1), method='bounded')
best_weight = result.x

weighted_oof = best_weight * xgb_oof + (1-best_weight) * lgb_oof
weighted_test = best_weight * xgb_test + (1-best_weight) * lgb_test

best_f1, best_thresh = 0, 0.1
for t in np.linspace(0.03, 0.5, 200):
    f1 = f1_score(y, (weighted_oof > t).astype(int))
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

results['weighted_avg'] = {
    'oof_f1': best_f1,
    'threshold': best_thresh,
    'weight_xgb': best_weight,
    'oof_preds': weighted_oof,
    'test_preds': weighted_test
}
print(f"      XGBoost weight: {best_weight:.3f}, LightGBM weight: {1-best_weight:.3f}", flush=True)
print(f"      OOF F1: {best_f1:.4f} @ threshold={best_thresh:.3f}", flush=True)

# Method 3: Rank Averaging
print("\n   Method 3: Rank Averaging", flush=True)

xgb_oof_rank = rankdata(xgb_oof) / len(xgb_oof)
lgb_oof_rank = rankdata(lgb_oof) / len(lgb_oof)
rank_avg_oof = (xgb_oof_rank + lgb_oof_rank) / 2

xgb_test_rank = rankdata(xgb_test) / len(xgb_test)
lgb_test_rank = rankdata(lgb_test) / len(lgb_test)
rank_avg_test = (xgb_test_rank + lgb_test_rank) / 2

best_f1, best_thresh = 0, 0.1
for t in np.linspace(0.3, 0.99, 200):  # Ranks are 0-1
    f1 = f1_score(y, (rank_avg_oof > t).astype(int))
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

results['rank_avg'] = {
    'oof_f1': best_f1,
    'threshold': best_thresh,
    'oof_preds': rank_avg_oof,
    'test_preds': rank_avg_test
}
print(f"      OOF F1: {best_f1:.4f} @ threshold={best_thresh:.3f}", flush=True)

# ====================
# 3. SELECT BEST METHOD
# ====================
print("\n" + "=" * 80, flush=True)
print("RESULTS SUMMARY", flush=True)
print("=" * 80, flush=True)

print(f"\n   {'Method':<25} {'OOF F1':<10} {'Threshold':<10}", flush=True)
print(f"   {'-'*25} {'-'*10} {'-'*10}", flush=True)
print(f"   {'v34a XGBoost (solo)':<25} {v34a['oof_f1']:<10.4f} {v34a['best_threshold']:<10.3f}", flush=True)
print(f"   {'v77 LightGBM (solo)':<25} {v77['oof_f1']:<10.4f} {v77['best_threshold']:<10.3f}", flush=True)
print(f"   {'Simple Average':<25} {results['simple_avg']['oof_f1']:<10.4f} {results['simple_avg']['threshold']:<10.3f}", flush=True)
print(f"   {'Weighted Average':<25} {results['weighted_avg']['oof_f1']:<10.4f} {results['weighted_avg']['threshold']:<10.3f}", flush=True)
print(f"   {'Rank Average':<25} {results['rank_avg']['oof_f1']:<10.4f} {results['rank_avg']['threshold']:<10.3f}", flush=True)

# Find best method
best_method = max(results.keys(), key=lambda k: results[k]['oof_f1'])
best_result = results[best_method]

print(f"\n   Best method: {best_method} (OOF F1={best_result['oof_f1']:.4f})", flush=True)

# ====================
# 4. CONFUSION MATRIX FOR BEST
# ====================
print("\n" + "=" * 80, flush=True)
print(f"BEST METHOD: {best_method.upper()}", flush=True)
print("=" * 80, flush=True)

final_preds = (best_result['oof_preds'] > best_result['threshold']).astype(int)
tp = np.sum((final_preds == 1) & (y == 1))
fp = np.sum((final_preds == 1) & (y == 0))
fn = np.sum((final_preds == 0) & (y == 1))

print(f"\n   TP={tp}, FP={fp}, FN={fn}", flush=True)
print(f"   Precision: {tp/(tp+fp):.4f}, Recall: {tp/(tp+fn):.4f}", flush=True)

# ====================
# 5. SUBMISSIONS
# ====================
print("\n" + "=" * 80, flush=True)
print("SUBMISSIONS", flush=True)
print("=" * 80, flush=True)

# Submit best method
test_binary = (best_result['test_preds'] > best_result['threshold']).astype(int)

submission = pd.DataFrame({
    'object_id': test_ids,
    'target': test_binary
})

submission_path = base_path / 'submissions/submission_v78_ensemble.csv'
submission.to_csv(submission_path, index=False)

print(f"   Saved: {submission_path.name} ({best_method})", flush=True)
print(f"   Predicted TDEs: {test_binary.sum()}", flush=True)

# Also save simple average as backup (often most robust)
if best_method != 'simple_avg':
    simple_result = results['simple_avg']
    simple_binary = (simple_result['test_preds'] > simple_result['threshold']).astype(int)

    simple_sub = pd.DataFrame({
        'object_id': test_ids,
        'target': simple_binary
    })

    simple_path = base_path / 'submissions/submission_v78_simple_avg.csv'
    simple_sub.to_csv(simple_path, index=False)
    print(f"   Saved: {simple_path.name} (backup)", flush=True)
    print(f"   Predicted TDEs: {simple_binary.sum()}", flush=True)

# Save artifacts
artifacts = {
    'results': results,
    'best_method': best_method,
    'v34a_oof_f1': v34a['oof_f1'],
    'v77_oof_f1': v77['oof_f1'],
}

with open(base_path / 'data/processed/v78_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

# ====================
# 6. COMPARISON
# ====================
print("\n" + "=" * 80, flush=True)
print("COMPARISON", flush=True)
print("=" * 80, flush=True)

print(f"""
   Model                       OOF F1   LB F1
   -----                       ------   -----
   v34a XGBoost (Optuna)       0.6667   0.6907  <-- Best LB
   v77 LightGBM (Optuna)       {v77['oof_f1']:.4f}   ???
   v78 Ensemble ({best_method})   {best_result['oof_f1']:.4f}   ???

   Ensemble improvement over best solo: {best_result['oof_f1'] - max(v34a['oof_f1'], v77['oof_f1']):+.4f}
""", flush=True)

print("=" * 80, flush=True)
print("v78 Complete", flush=True)
print("=" * 80, flush=True)

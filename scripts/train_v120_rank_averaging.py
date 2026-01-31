"""
MALLORN v120: Rank Averaging Ensemble
=====================================

Rank averaging is more robust than probability averaging when models
have different probability scales or calibrations.

Strategy:
1. Convert each model's predictions to ranks
2. Average the ranks
3. Use rank-based threshold selection

Benefits:
- Robust to different probability scales
- Less sensitive to outlier predictions
- Works well with diverse models
"""

import sys
import pickle
import gzip
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata
from sklearn.metrics import f1_score
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

base_path = Path(__file__).parent.parent

print("=" * 70)
print("MALLORN v120: Rank Averaging Ensemble")
print("=" * 70)

# ============================================================================
# 1. LOAD PREDICTIONS
# ============================================================================
print("\n[1/4] Loading predictions from all models...")

# Load package
with gzip.open(base_path / 'data/kaggle_ensemble_package.pkl.gz', 'rb') as f:
    package = pickle.load(f)
y = package['y']
test_ids = package['test_ids']

# Load individual model predictions
with open(base_path / 'data/processed/v34a_artifacts.pkl', 'rb') as f:
    v34a_arts = pickle.load(f)

with open(base_path / 'data/processed/v92_focal_adv_artifacts.pkl', 'rb') as f:
    v92_arts = pickle.load(f)

with open(base_path / 'data/processed/v114_optimized_artifacts.pkl', 'rb') as f:
    v114_arts = pickle.load(f)

with open(base_path / 'data/processed/v118_catboost_artifacts.pkl', 'rb') as f:
    catboost_arts = pickle.load(f)

# Collect predictions
models = {
    'v92d': {
        'oof': v92_arts['v92d_baseline_adv']['oof_preds'],
        'test': v92_arts['v92d_baseline_adv']['test_preds'],
        'lb_score': 0.6986
    },
    'v34a': {
        'oof': v34a_arts['oof_preds'],
        'test': v34a_arts['test_preds'],
        'lb_score': 0.6907
    },
    'v114d': {
        'oof': v114_arts['results']['v114d_minimal_research']['oof_preds'],
        'test': v114_arts['results']['v114d_minimal_research']['test_preds'],
        'lb_score': 0.6797
    },
    'v118_catboost': {
        'oof': catboost_arts['avg_oof_preds'],
        'test': catboost_arts['avg_test_preds'],
        'lb_score': None
    }
}

print(f"   Loaded {len(models)} models")

# ============================================================================
# 2. COMPUTE RANKS
# ============================================================================
print("\n[2/4] Computing ranks...")

def to_ranks(preds):
    """Convert predictions to normalized ranks (0-1 scale)."""
    ranks = rankdata(preds)
    return ranks / len(ranks)

# Compute ranks for each model
for name, data in models.items():
    data['oof_rank'] = to_ranks(data['oof'])
    data['test_rank'] = to_ranks(data['test'])
    print(f"   {name}: OOF rank range [{data['oof_rank'].min():.4f}, {data['oof_rank'].max():.4f}]")

# ============================================================================
# 3. RANK AVERAGING METHODS
# ============================================================================
print("\n[3/4] Computing rank averages...")

def find_best_threshold(y_true, y_pred, n_thresholds=100):
    best_f1, best_t = 0, 0.1
    for t in np.linspace(0.3, 0.99, n_thresholds):  # Ranks are 0-1 scale
        f1 = f1_score(y_true, (y_pred > t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1

results = {}
model_names = list(models.keys())

# --- Simple Rank Average ---
print("\n   [A] Simple Rank Average (equal weights)...")
oof_ranks = np.column_stack([models[name]['oof_rank'] for name in model_names])
test_ranks = np.column_stack([models[name]['test_rank'] for name in model_names])

simple_rank_avg_oof = oof_ranks.mean(axis=1)
simple_rank_avg_test = test_ranks.mean(axis=1)

threshold, oof_f1 = find_best_threshold(y, simple_rank_avg_oof)
print(f"       OOF F1: {oof_f1:.4f} @ threshold {threshold:.3f}")

results['simple_rank_avg'] = {
    'oof': simple_rank_avg_oof,
    'test': simple_rank_avg_test,
    'threshold': threshold,
    'oof_f1': oof_f1
}

# --- LB-Weighted Rank Average ---
print("\n   [B] LB-Weighted Rank Average...")
lb_weights = np.array([0.6986, 0.6907, 0.6797, 0.65])  # Estimate for catboost
lb_weights = lb_weights / lb_weights.sum()

lb_rank_avg_oof = np.average(oof_ranks, axis=1, weights=lb_weights)
lb_rank_avg_test = np.average(test_ranks, axis=1, weights=lb_weights)

threshold, oof_f1 = find_best_threshold(y, lb_rank_avg_oof)
print(f"       OOF F1: {oof_f1:.4f} @ threshold {threshold:.3f}")
print(f"       Weights: {dict(zip(model_names, lb_weights.round(3)))}")

results['lb_rank_avg'] = {
    'oof': lb_rank_avg_oof,
    'test': lb_rank_avg_test,
    'threshold': threshold,
    'oof_f1': oof_f1
}

# --- XGB-Heavy Rank Average (favor best LB model) ---
print("\n   [C] XGB-Heavy Rank Average (favor v92d)...")
xgb_heavy_weights = np.array([0.4, 0.25, 0.2, 0.15])  # v92d, v34a, v114d, catboost
xgb_heavy_weights = xgb_heavy_weights / xgb_heavy_weights.sum()

xgb_heavy_oof = np.average(oof_ranks, axis=1, weights=xgb_heavy_weights)
xgb_heavy_test = np.average(test_ranks, axis=1, weights=xgb_heavy_weights)

threshold, oof_f1 = find_best_threshold(y, xgb_heavy_oof)
print(f"       OOF F1: {oof_f1:.4f} @ threshold {threshold:.3f}")
print(f"       Weights: {dict(zip(model_names, xgb_heavy_weights.round(3)))}")

results['xgb_heavy_rank'] = {
    'oof': xgb_heavy_oof,
    'test': xgb_heavy_test,
    'threshold': threshold,
    'oof_f1': oof_f1
}

# --- Top 3 Only (exclude CatBoost) ---
print("\n   [D] Top 3 Rank Average (v92d + v34a + v114d only)...")
top3_names = ['v92d', 'v34a', 'v114d']
top3_oof_ranks = np.column_stack([models[name]['oof_rank'] for name in top3_names])
top3_test_ranks = np.column_stack([models[name]['test_rank'] for name in top3_names])

top3_weights = np.array([0.6986, 0.6907, 0.6797])
top3_weights = top3_weights / top3_weights.sum()

top3_rank_avg_oof = np.average(top3_oof_ranks, axis=1, weights=top3_weights)
top3_rank_avg_test = np.average(top3_test_ranks, axis=1, weights=top3_weights)

threshold, oof_f1 = find_best_threshold(y, top3_rank_avg_oof)
print(f"       OOF F1: {oof_f1:.4f} @ threshold {threshold:.3f}")

results['top3_rank_avg'] = {
    'oof': top3_rank_avg_oof,
    'test': top3_rank_avg_test,
    'threshold': threshold,
    'oof_f1': oof_f1
}

# --- Compare with Probability Averaging ---
print("\n   [E] Probability Average (for comparison)...")
oof_probs = np.column_stack([models[name]['oof'] for name in model_names])
test_probs = np.column_stack([models[name]['test'] for name in model_names])

prob_avg_oof = np.average(oof_probs, axis=1, weights=lb_weights)
prob_avg_test = np.average(test_probs, axis=1, weights=lb_weights)

# Use probability scale for thresholding
best_f1, best_t = 0, 0.1
for t in np.linspace(0.03, 0.5, 100):
    f1 = f1_score(y, (prob_avg_oof > t).astype(int))
    if f1 > best_f1:
        best_f1, best_t = f1, t

print(f"       OOF F1: {best_f1:.4f} @ threshold {best_t:.3f}")

results['prob_avg'] = {
    'oof': prob_avg_oof,
    'test': prob_avg_test,
    'threshold': best_t,
    'oof_f1': best_f1
}

# ============================================================================
# 4. SAVE RESULTS
# ============================================================================
print("\n[4/4] Saving results...")

# Summary
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
    'models': {name: {'oof_rank': models[name]['oof_rank'], 'test_rank': models[name]['test_rank']}
               for name in model_names},
    'best_method': best_method,
}

with open(base_path / 'data/processed/v120_rank_avg_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)
print(f"\n   Saved artifacts to data/processed/v120_rank_avg_artifacts.pkl")

# Save submissions
for method in results.keys():
    data = results[method]

    # For rank-based methods, threshold is in 0-1 rank scale
    if 'rank' in method:
        binary_preds = (data['test'] > data['threshold']).astype(int)
    else:
        binary_preds = (data['test'] > data['threshold']).astype(int)

    submission = pd.DataFrame({
        'object_id': test_ids,
        'target': binary_preds
    })
    filename = f'submission_v120_{method}.csv'
    submission.to_csv(base_path / 'submissions' / filename, index=False)
    print(f"   Saved {filename}: {binary_preds.sum()} TDEs")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("RANK AVERAGING COMPLETE")
print("=" * 70)

print(f"""
Key Findings:
   Best method: {best_method}
   Best OOF F1: {best_f1:.4f}

   Rank vs Probability Averaging:
   - Rank average:  {results['simple_rank_avg']['oof_f1']:.4f}
   - Prob average:  {results['prob_avg']['oof_f1']:.4f}
   - Difference:    {results['simple_rank_avg']['oof_f1'] - results['prob_avg']['oof_f1']:+.4f}

   Rank averaging is {'better' if results['simple_rank_avg']['oof_f1'] > results['prob_avg']['oof_f1'] else 'worse'} than probability averaging!

Submissions created:
   - submission_v120_simple_rank_avg.csv
   - submission_v120_lb_rank_avg.csv
   - submission_v120_xgb_heavy_rank.csv
   - submission_v120_top3_rank_avg.csv
   - submission_v120_prob_avg.csv
""")

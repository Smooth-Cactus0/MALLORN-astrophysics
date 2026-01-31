"""
MALLORN v126: Conservative Final Submission
============================================

v125_optimized overfit badly (OOF 0.7003 → LB 0.6618, delta -0.0385)
v124_conservative generalized well (OOF 0.6857 → LB 0.6976, delta +0.0119)

Strategy: Use optimized models with CONSERVATIVE weights that won't overfit.
- Avoid high CatBoost weight (60% was too aggressive)
- Favor models with proven LB stability (v92d, v34a)
- Test individual optimized models as alternatives
"""

import pickle
import gzip
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score

base_path = Path(__file__).parent.parent

print("=" * 70)
print("MALLORN v126: Conservative Final Submission")
print("=" * 70)

# Load v125 optimized artifacts
with open(base_path / 'data/processed/v125_optimized_ensemble_artifacts.pkl', 'rb') as f:
    v125 = pickle.load(f)

with gzip.open(base_path / 'data/kaggle_ensemble_package.pkl.gz', 'rb') as f:
    package = pickle.load(f)
    y = package['y']
    test_ids = package['test_ids']

# Extract individual model OOF/test predictions
v92d_oof = v125['individual_results']['v92d']['oof']
v92d_test = v125['individual_results']['v92d']['test']

v34a_oof = v125['individual_results']['v34a']['oof']
v34a_test = v125['individual_results']['v34a']['test']

lgb_oof = v125['individual_results']['lgb']['oof']
lgb_test = v125['individual_results']['lgb']['test']

cb_oof = v125['individual_results']['catboost']['oof']
cb_test = v125['individual_results']['catboost']['test']

def find_best_threshold(y_true, y_pred):
    best_f1, best_t = 0, 0.1
    for t in np.linspace(0.03, 0.7, 100):
        f1 = f1_score(y_true, (y_pred > t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1

print("\n" + "=" * 70)
print("Individual Model Performance:")
print("=" * 70)
for name, oof in [('v92d XGBoost (120 feat)', v92d_oof),
                   ('v34a XGBoost (80 feat)', v34a_oof),
                   ('LightGBM (120 feat)', lgb_oof),
                   ('CatBoost (75 feat)', cb_oof)]:
    _, f1 = find_best_threshold(y, oof)
    print(f"   {name}: OOF F1 = {f1:.4f}")

# ============================================================================
# STRATEGY 1: Individual Optimized CatBoost (safest bet)
# ============================================================================
print("\n" + "=" * 70)
print("Strategy 1: Individual Optimized CatBoost")
print("=" * 70)

threshold_cb, f1_cb = find_best_threshold(y, cb_oof)
binary_cb = (cb_test > threshold_cb).astype(int)
print(f"   OOF F1: {f1_cb:.4f}, Threshold: {threshold_cb:.3f}")
print(f"   Test TDEs: {binary_cb.sum()}")

submission_cb = pd.DataFrame({
    'object_id': test_ids,
    'target': binary_cb
})
submission_cb.to_csv(base_path / 'submissions/submission_v126_catboost_solo.csv', index=False)
print("   -> submission_v126_catboost_solo.csv")

# ============================================================================
# STRATEGY 2: Individual Optimized LightGBM
# ============================================================================
print("\n" + "=" * 70)
print("Strategy 2: Individual Optimized LightGBM")
print("=" * 70)

threshold_lgb, f1_lgb = find_best_threshold(y, lgb_oof)
binary_lgb = (lgb_test > threshold_lgb).astype(int)
print(f"   OOF F1: {f1_lgb:.4f}, Threshold: {threshold_lgb:.3f}")
print(f"   Test TDEs: {binary_lgb.sum()}")

submission_lgb = pd.DataFrame({
    'object_id': test_ids,
    'target': binary_lgb
})
submission_lgb.to_csv(base_path / 'submissions/submission_v126_lgb_solo.csv', index=False)
print("   -> submission_v126_lgb_solo.csv")

# ============================================================================
# STRATEGY 3: v92d-Heavy Ensemble (proven stability)
# ============================================================================
print("\n" + "=" * 70)
print("Strategy 3: v92d-Heavy Conservative Ensemble")
print("=" * 70)

# v92d has proven LB stability, give it highest weight
# Lower CatBoost weight to avoid overfitting
w_v92d_heavy = 0.45
w_v34a_heavy = 0.25
w_cb_heavy = 0.25
w_lgb_heavy = 0.05

oof_v92d_heavy = (w_v92d_heavy * v92d_oof + w_v34a_heavy * v34a_oof +
                   w_cb_heavy * cb_oof + w_lgb_heavy * lgb_oof)
test_v92d_heavy = (w_v92d_heavy * v92d_test + w_v34a_heavy * v34a_test +
                    w_cb_heavy * cb_test + w_lgb_heavy * lgb_test)

threshold_v92d_heavy, f1_v92d_heavy = find_best_threshold(y, oof_v92d_heavy)
binary_v92d_heavy = (test_v92d_heavy > threshold_v92d_heavy).astype(int)

print(f"   Weights: v92d=0.45, v34a=0.25, cb=0.25, lgb=0.05")
print(f"   OOF F1: {f1_v92d_heavy:.4f}, Threshold: {threshold_v92d_heavy:.3f}")
print(f"   Test TDEs: {binary_v92d_heavy.sum()}")

submission_v92d_heavy = pd.DataFrame({
    'object_id': test_ids,
    'target': binary_v92d_heavy
})
submission_v92d_heavy.to_csv(base_path / 'submissions/submission_v126_v92d_heavy.csv', index=False)
print("   -> submission_v126_v92d_heavy.csv")

# ============================================================================
# STRATEGY 4: Balanced Conservative (equal XGB + equal others)
# ============================================================================
print("\n" + "=" * 70)
print("Strategy 4: Balanced Conservative Ensemble")
print("=" * 70)

# Equal weight to both XGBoosts (proven stable), lower weight to newer models
w_xgb = 0.30  # each XGB
w_cb_bal = 0.25
w_lgb_bal = 0.15

oof_balanced = (w_xgb * v92d_oof + w_xgb * v34a_oof +
                w_cb_bal * cb_oof + w_lgb_bal * lgb_oof)
test_balanced = (w_xgb * v92d_test + w_xgb * v34a_test +
                 w_cb_bal * cb_test + w_lgb_bal * lgb_test)

threshold_balanced, f1_balanced = find_best_threshold(y, oof_balanced)
binary_balanced = (test_balanced > threshold_balanced).astype(int)

print(f"   Weights: v92d=0.30, v34a=0.30, cb=0.25, lgb=0.15")
print(f"   OOF F1: {f1_balanced:.4f}, Threshold: {threshold_balanced:.3f}")
print(f"   Test TDEs: {binary_balanced.sum()}")

submission_balanced = pd.DataFrame({
    'object_id': test_ids,
    'target': binary_balanced
})
submission_balanced.to_csv(base_path / 'submissions/submission_v126_balanced.csv', index=False)
print("   -> submission_v126_balanced.csv")

# ============================================================================
# STRATEGY 5: No-CatBoost Ensemble (XGB + LGB only)
# ============================================================================
print("\n" + "=" * 70)
print("Strategy 5: No-CatBoost (XGB+LGB only)")
print("=" * 70)

# Maybe CatBoost is causing overfitting - try without it
w_v92d_no_cb = 0.50
w_v34a_no_cb = 0.30
w_lgb_no_cb = 0.20

oof_no_cb = (w_v92d_no_cb * v92d_oof + w_v34a_no_cb * v34a_oof +
             w_lgb_no_cb * lgb_oof)
test_no_cb = (w_v92d_no_cb * v92d_test + w_v34a_no_cb * v34a_test +
              w_lgb_no_cb * lgb_test)

threshold_no_cb, f1_no_cb = find_best_threshold(y, oof_no_cb)
binary_no_cb = (test_no_cb > threshold_no_cb).astype(int)

print(f"   Weights: v92d=0.50, v34a=0.30, lgb=0.20")
print(f"   OOF F1: {f1_no_cb:.4f}, Threshold: {threshold_no_cb:.3f}")
print(f"   Test TDEs: {binary_no_cb.sum()}")

submission_no_cb = pd.DataFrame({
    'object_id': test_ids,
    'target': binary_no_cb
})
submission_no_cb.to_csv(base_path / 'submissions/submission_v126_no_catboost.csv', index=False)
print("   -> submission_v126_no_catboost.csv")

# ============================================================================
# SUMMARY & RECOMMENDATION
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY: v126 Conservative Strategies")
print("=" * 70)

results = [
    ("CatBoost Solo", f1_cb, binary_cb.sum()),
    ("LightGBM Solo", f1_lgb, binary_lgb.sum()),
    ("v92d-Heavy", f1_v92d_heavy, binary_v92d_heavy.sum()),
    ("Balanced", f1_balanced, binary_balanced.sum()),
    ("No-CatBoost", f1_no_cb, binary_no_cb.sum()),
]

results_sorted = sorted(results, key=lambda x: x[1], reverse=False)  # Sort by OOF F1 ascending

print("\nAll strategies (sorted by OOF F1, lower may be better for LB):")
for name, f1, n_tdes in results_sorted:
    print(f"   {name:20s}: OOF F1 = {f1:.4f}, Test TDEs = {n_tdes}")

print("\n" + "=" * 70)
print("RECOMMENDATION FOR FINAL SUBMISSION:")
print("=" * 70)
print("""
Based on the OOF-LB paradox pattern:
- v124_conservative (OOF 0.6857) -> LB 0.6976 (+0.0119) [GOOD]
- v125_optimized (OOF 0.7003) -> LB 0.6618 (-0.0385) [BAD]

The lower OOF generalized better! Therefore, consider:

1st choice: v126_v92d_heavy.csv
   - v92d has proven LB stability (v92d was used in best v124)
   - Conservative CatBoost weight (25% instead of 60%)
   - OOF F1 slightly lower = may generalize better

2nd choice: v126_no_catboost.csv
   - If CatBoost is causing overfitting issues
   - Pure XGB+LGB blend (all proven stable)

3rd choice: v126_balanced.csv
   - Equal weight to both XGBoosts (stability)
   - Moderate CatBoost/LGB (25%/15%)
""")

# Save recommendation
recommendation = {
    'strategies': results,
    'recommendation': 'v126_v92d_heavy or v126_no_catboost',
    'reasoning': 'Lower OOF may generalize better based on v124 vs v125 results'
}

with open(base_path / 'data/processed/v126_conservative_strategies.pkl', 'wb') as f:
    pickle.dump(recommendation, f)

print("\nFiles saved:")
print("   - submission_v126_catboost_solo.csv")
print("   - submission_v126_lgb_solo.csv")
print("   - submission_v126_v92d_heavy.csv")
print("   - submission_v126_balanced.csv")
print("   - submission_v126_no_catboost.csv")
print("   - data/processed/v126_conservative_strategies.pkl")

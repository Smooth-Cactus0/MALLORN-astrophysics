"""
MALLORN: Deep Analysis of CatBoost Integration
===============================================

Analyze exactly how CatBoost helps and design optimal ensemble strategies.

Questions to answer:
1. Which specific TDEs does CatBoost recover?
2. What's the correlation between model predictions?
3. What's the optimal CatBoost weight for ensemble?
4. Can we create a "smart" ensemble that uses CatBoost selectively?
"""

import sys
import pickle
import gzip
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import f1_score
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

base_path = Path(__file__).parent.parent

print("=" * 70)
print("MALLORN: Deep Analysis of CatBoost Integration")
print("=" * 70)

# ============================================================================
# 1. LOAD ALL MODEL PREDICTIONS
# ============================================================================
print("\n[1/6] Loading predictions...")

with gzip.open(base_path / 'data/kaggle_ensemble_package.pkl.gz', 'rb') as f:
    package = pickle.load(f)
y = package['y']
test_ids = package['test_ids']

# Load predictions
with open(base_path / 'data/processed/v92_focal_adv_artifacts.pkl', 'rb') as f:
    v92_arts = pickle.load(f)
with open(base_path / 'data/processed/v34a_artifacts.pkl', 'rb') as f:
    v34a_arts = pickle.load(f)
with open(base_path / 'data/processed/v114_optimized_artifacts.pkl', 'rb') as f:
    v114_arts = pickle.load(f)
with open(base_path / 'data/processed/v118_catboost_artifacts.pkl', 'rb') as f:
    cb_arts = pickle.load(f)

models = {
    'v92d': {'oof': v92_arts['v92d_baseline_adv']['oof_preds'],
             'test': v92_arts['v92d_baseline_adv']['test_preds']},
    'v34a': {'oof': v34a_arts['oof_preds'],
             'test': v34a_arts['test_preds']},
    'v114d': {'oof': v114_arts['results']['v114d_minimal_research']['oof_preds'],
              'test': v114_arts['results']['v114d_minimal_research']['test_preds']},
    'catboost': {'oof': cb_arts['avg_oof_preds'],
                 'test': cb_arts['avg_test_preds']},
}

print(f"   Loaded 4 models")

# ============================================================================
# 2. CORRELATION ANALYSIS
# ============================================================================
print("\n[2/6] Correlation Analysis...")

model_names = list(models.keys())
oof_preds = {name: models[name]['oof'] for name in model_names}

print("\n   Pearson Correlation Matrix (OOF predictions):")
print(f"   {'':12}", end="")
for name in model_names:
    print(f"{name:12}", end="")
print()

corr_matrix = {}
for name1 in model_names:
    corr_matrix[name1] = {}
    print(f"   {name1:12}", end="")
    for name2 in model_names:
        corr, _ = pearsonr(oof_preds[name1], oof_preds[name2])
        corr_matrix[name1][name2] = corr
        print(f"{corr:12.3f}", end="")
    print()

# Calculate average correlation for each model
print("\n   Average correlation with other models:")
for name in model_names:
    avg_corr = np.mean([corr_matrix[name][n] for n in model_names if n != name])
    print(f"   {name}: {avg_corr:.3f}")

# ============================================================================
# 3. ERROR COMPLEMENTARITY ANALYSIS
# ============================================================================
print("\n[3/6] Error Complementarity Analysis...")

def get_errors(oof_pred, y_true, threshold=None):
    """Get FP and FN indices."""
    if threshold is None:
        # Find optimal threshold
        best_t = 0.1
        best_f1 = 0
        for t in np.linspace(0.03, 0.6, 100):
            f1 = f1_score(y_true, (oof_pred > t).astype(int))
            if f1 > best_f1:
                best_f1, best_t = f1, t
        threshold = best_t

    pred = (oof_pred > threshold).astype(int)
    fp = set(np.where((pred == 1) & (y_true == 0))[0])
    fn = set(np.where((pred == 0) & (y_true == 1))[0])
    tp = set(np.where((pred == 1) & (y_true == 1))[0])
    return fp, fn, tp, threshold

errors = {}
for name in model_names:
    fp, fn, tp, thresh = get_errors(oof_preds[name], y)
    errors[name] = {'fp': fp, 'fn': fn, 'tp': tp, 'threshold': thresh}
    print(f"   {name}: FP={len(fp)}, FN={len(fn)}, TP={len(tp)}, threshold={thresh:.3f}")

# Analyze which FN CatBoost uniquely recovers
print("\n   CatBoost's Unique Contributions:")

# FN recovered by CatBoost but NOT by XGB models
xgb_fn = errors['v92d']['fn'] & errors['v34a']['fn']  # FN by both XGB
cb_tp = errors['catboost']['tp']
cb_recovers = xgb_fn & cb_tp
print(f"   FN missed by both XGB but found by CatBoost: {len(cb_recovers)}")

# FN recovered by CatBoost but NOT by LightGBM
lgb_fn = errors['v114d']['fn']
cb_recovers_vs_lgb = lgb_fn & cb_tp
print(f"   FN missed by LightGBM but found by CatBoost: {len(cb_recovers_vs_lgb)}")

# FN missed by ALL 3 (v92d, v34a, v114d) but found by CatBoost
all_other_fn = errors['v92d']['fn'] & errors['v34a']['fn'] & errors['v114d']['fn']
cb_unique_recovery = all_other_fn & cb_tp
print(f"   FN missed by ALL other models but found by CatBoost: {len(cb_unique_recovery)}")

# ============================================================================
# 4. OPTIMAL CATBOOST WEIGHT SEARCH
# ============================================================================
print("\n[4/6] Optimal CatBoost Weight Search...")

# Base ensemble: v92d + v34a + v114d
base_weights = np.array([0.45, 0.30, 0.25])  # Based on LB scores
base_oof = (base_weights[0] * oof_preds['v92d'] +
            base_weights[1] * oof_preds['v34a'] +
            base_weights[2] * oof_preds['v114d'])

def find_best_f1(pred, y_true):
    best_f1, best_t = 0, 0.1
    for t in np.linspace(0.03, 0.6, 100):
        f1 = f1_score(y_true, (pred > t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_f1, best_t

base_f1, base_t = find_best_f1(base_oof, y)
print(f"\n   Base ensemble (v92d+v34a+v114d): F1={base_f1:.4f}")

# Search for optimal CatBoost weight
print("\n   Searching optimal CatBoost integration weight...")
results = []

for cb_weight in np.linspace(0, 0.5, 51):
    # Blend: (1-cb_weight)*base + cb_weight*catboost
    blended = (1 - cb_weight) * base_oof + cb_weight * oof_preds['catboost']
    f1, threshold = find_best_f1(blended, y)
    results.append({
        'cb_weight': cb_weight,
        'f1': f1,
        'threshold': threshold
    })

results_df = pd.DataFrame(results)
best_idx = results_df['f1'].idxmax()
best_result = results_df.loc[best_idx]

print(f"\n   Best CatBoost weight: {best_result['cb_weight']:.2f}")
print(f"   Best F1: {best_result['f1']:.4f}")
print(f"   Improvement over base: {best_result['f1'] - base_f1:+.4f}")

# Show top 5 weights
print("\n   Top 5 CatBoost weights:")
print(f"   {'Weight':<10} {'F1':<10} {'Improvement':<12}")
for _, row in results_df.nlargest(5, 'f1').iterrows():
    print(f"   {row['cb_weight']:<10.2f} {row['f1']:<10.4f} {row['f1']-base_f1:+.4f}")

# ============================================================================
# 5. DESIGN NEW ENSEMBLE STRATEGIES
# ============================================================================
print("\n[5/6] Designing New Ensemble Strategies...")

# Strategy A: Optimal CatBoost blend
opt_cb_weight = best_result['cb_weight']
strategy_a_oof = (1 - opt_cb_weight) * base_oof + opt_cb_weight * oof_preds['catboost']
strategy_a_test = ((1 - opt_cb_weight) *
                   (base_weights[0] * models['v92d']['test'] +
                    base_weights[1] * models['v34a']['test'] +
                    base_weights[2] * models['v114d']['test']) +
                   opt_cb_weight * models['catboost']['test'])
f1_a, t_a = find_best_f1(strategy_a_oof, y)
print(f"\n   Strategy A: Optimal CatBoost Blend")
print(f"   Formula: {1-opt_cb_weight:.2f}*(0.45*v92d + 0.30*v34a + 0.25*v114d) + {opt_cb_weight:.2f}*catboost")
print(f"   OOF F1: {f1_a:.4f}")

# Strategy B: XGB-heavy with CatBoost for diversity
# Give more weight to v92d (best LB) but include CatBoost
xgb_heavy_weights = {'v92d': 0.50, 'v34a': 0.20, 'v114d': 0.15, 'catboost': 0.15}
strategy_b_oof = sum(w * oof_preds[name] for name, w in xgb_heavy_weights.items())
strategy_b_test = sum(w * models[name]['test'] for name, w in xgb_heavy_weights.items())
f1_b, t_b = find_best_f1(strategy_b_oof, y)
print(f"\n   Strategy B: XGB-Heavy with CatBoost Diversity")
print(f"   Weights: {xgb_heavy_weights}")
print(f"   OOF F1: {f1_b:.4f}")

# Strategy C: Balanced 4-model (equal weights)
equal_weights = {'v92d': 0.25, 'v34a': 0.25, 'v114d': 0.25, 'catboost': 0.25}
strategy_c_oof = sum(w * oof_preds[name] for name, w in equal_weights.items())
strategy_c_test = sum(w * models[name]['test'] for name, w in equal_weights.items())
f1_c, t_c = find_best_f1(strategy_c_oof, y)
print(f"\n   Strategy C: Balanced 4-Model")
print(f"   Weights: {equal_weights}")
print(f"   OOF F1: {f1_c:.4f}")

# Strategy D: LB-proportional weights
lb_scores = {'v92d': 0.6986, 'v34a': 0.6907, 'v114d': 0.6797, 'catboost': 0.65}  # estimate
total_lb = sum(lb_scores.values())
lb_weights = {name: score/total_lb for name, score in lb_scores.items()}
strategy_d_oof = sum(w * oof_preds[name] for name, w in lb_weights.items())
strategy_d_test = sum(w * models[name]['test'] for name, w in lb_weights.items())
f1_d, t_d = find_best_f1(strategy_d_oof, y)
print(f"\n   Strategy D: LB-Proportional Weights")
print(f"   Weights: {{{', '.join(f'{k}: {v:.3f}' for k, v in lb_weights.items())}}}")
print(f"   OOF F1: {f1_d:.4f}")

# Strategy E: Focus on uncorrelated models (v92d + catboost heavy)
uncorr_weights = {'v92d': 0.55, 'v34a': 0.10, 'v114d': 0.10, 'catboost': 0.25}
strategy_e_oof = sum(w * oof_preds[name] for name, w in uncorr_weights.items())
strategy_e_test = sum(w * models[name]['test'] for name, w in uncorr_weights.items())
f1_e, t_e = find_best_f1(strategy_e_oof, y)
print(f"\n   Strategy E: Uncorrelated Focus (v92d + CatBoost heavy)")
print(f"   Weights: {uncorr_weights}")
print(f"   OOF F1: {f1_e:.4f}")

# ============================================================================
# 6. SAVE BEST STRATEGIES
# ============================================================================
print("\n[6/6] Saving best ensemble strategies...")

strategies = {
    'strategy_a_optimal_cb': {
        'name': 'Optimal CatBoost Blend',
        'oof': strategy_a_oof, 'test': strategy_a_test,
        'f1': f1_a, 'threshold': t_a,
        'description': f'{1-opt_cb_weight:.2f}*base_ensemble + {opt_cb_weight:.2f}*catboost'
    },
    'strategy_b_xgb_heavy': {
        'name': 'XGB-Heavy with CatBoost',
        'oof': strategy_b_oof, 'test': strategy_b_test,
        'f1': f1_b, 'threshold': t_b,
        'weights': xgb_heavy_weights
    },
    'strategy_c_balanced': {
        'name': 'Balanced 4-Model',
        'oof': strategy_c_oof, 'test': strategy_c_test,
        'f1': f1_c, 'threshold': t_c,
        'weights': equal_weights
    },
    'strategy_d_lb_prop': {
        'name': 'LB-Proportional',
        'oof': strategy_d_oof, 'test': strategy_d_test,
        'f1': f1_d, 'threshold': t_d,
        'weights': lb_weights
    },
    'strategy_e_uncorr': {
        'name': 'Uncorrelated Focus',
        'oof': strategy_e_oof, 'test': strategy_e_test,
        'f1': f1_e, 'threshold': t_e,
        'weights': uncorr_weights
    },
}

# Rank strategies
print("\n   === STRATEGY RANKING ===")
print(f"   {'Rank':<6} {'Strategy':<30} {'OOF F1':<10} {'vs Base':<10}")
print("   " + "-" * 56)

sorted_strategies = sorted(strategies.items(), key=lambda x: x[1]['f1'], reverse=True)
for rank, (key, data) in enumerate(sorted_strategies, 1):
    improvement = data['f1'] - base_f1
    print(f"   {rank:<6} {data['name']:<30} {data['f1']:<10.4f} {improvement:+.4f}")

# Save top 2 as submissions
for i, (key, data) in enumerate(sorted_strategies[:2], 1):
    binary_preds = (data['test'] > data['threshold']).astype(int)
    submission = pd.DataFrame({
        'object_id': test_ids,
        'target': binary_preds
    })
    filename = f'submission_v122_{key}.csv'
    submission.to_csv(base_path / 'submissions' / filename, index=False)
    print(f"\n   Saved {filename}: {binary_preds.sum()} TDEs")

# Save artifacts
artifacts = {
    'strategies': strategies,
    'correlation_matrix': corr_matrix,
    'optimal_cb_weight': opt_cb_weight,
    'base_f1': base_f1,
    'cb_weight_search': results_df,
}

with open(base_path / 'data/processed/v122_catboost_integration.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("CATBOOST INTEGRATION ANALYSIS COMPLETE")
print("=" * 70)

print(f"""
Key Findings:

1. CORRELATION ANALYSIS:
   - CatBoost has lowest avg correlation ({np.mean([corr_matrix['catboost'][n] for n in model_names if n != 'catboost']):.3f}) with other models
   - This makes it valuable for ensemble diversity

2. ERROR COMPLEMENTARITY:
   - CatBoost finds {len(cb_unique_recovery)} TDEs that ALL other models miss
   - This is {100*len(cb_unique_recovery)/len(all_other_fn):.1f}% of "universally missed" TDEs

3. OPTIMAL CATBOOST WEIGHT:
   - Best weight: {opt_cb_weight:.2f}
   - Improves F1 by {best_result['f1'] - base_f1:+.4f}

4. BEST STRATEGIES FOR TOMORROW:
   1. {sorted_strategies[0][1]['name']}: F1={sorted_strategies[0][1]['f1']:.4f}
   2. {sorted_strategies[1][1]['name']}: F1={sorted_strategies[1][1]['f1']:.4f}

New Submissions Created:
   - submission_v122_{sorted_strategies[0][0]}.csv
   - submission_v122_{sorted_strategies[1][0]}.csv
""")

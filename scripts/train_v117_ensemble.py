"""
MALLORN v117: Final Ensemble

Combines best XGBoost and LightGBM models for final submission.

Ensemble strategies:
1. Simple average
2. Weighted average (optimized on OOF)
3. Rank average
4. Threshold optimization

Usage:
    python scripts/train_v117_ensemble.py
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, confusion_matrix
from scipy.optimize import minimize_scalar, minimize
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
base_path = Path(__file__).parent.parent

print("=" * 80)
print("MALLORN v117: Final Ensemble")
print("=" * 80)

# ====================
# CONFIGURATION - UPDATE THESE BASED ON LB RESULTS
# ====================

# Best models to ensemble (update after LB analysis)
MODELS_TO_ENSEMBLE = {
    # Format: 'name': {'artifact_file': 'path', 'key': 'result_key', 'lb_score': float}

    # XGBoost candidates
    'v92d_xgb': {
        'artifact_file': 'non_successful_tests/data/processed/v92_focal_adv_artifacts.pkl',
        'key': 'v92d_baseline_adv',
        'lb_score': 0.6986,  # BEST known XGBoost
        'type': 'xgb'
    },
    'v115a_xgb': {
        'artifact_file': 'data/processed/v115_xgb_research_artifacts.pkl',
        'key': 'v115a_baseline_adv',
        'lb_score': None,  # Fill after submission
        'type': 'xgb'
    },

    # LightGBM candidates
    'v114d_lgb': {
        'artifact_file': 'data/processed/v114_optimized_artifacts.pkl',
        'key': 'v114d_minimal_research',
        'lb_score': None,  # Fill after submission
        'type': 'lgb'
    },
    'v113b_lgb': {
        'artifact_file': 'data/processed/v113_research_artifacts.pkl',
        'key': 'v113b_nuclear',
        'lb_score': 0.6717,
        'type': 'lgb'
    },
}

# ====================
# 1. LOAD MODEL PREDICTIONS
# ====================
print("\n1. Loading model predictions...")

def load_model_predictions(model_config):
    """Load OOF and test predictions from artifact file."""
    try:
        artifact_path = base_path / model_config['artifact_file']
        with open(artifact_path, 'rb') as f:
            artifacts = pickle.load(f)

        if 'results' in artifacts:
            result = artifacts['results'][model_config['key']]
        else:
            result = artifacts[model_config['key']]

        return {
            'oof_preds': result['oof_preds'],
            'test_preds': result['test_preds'],
            'threshold': result['threshold'],
            'oof_f1': result['oof_f1']
        }
    except Exception as e:
        print(f"   Warning: Could not load {model_config['key']}: {e}")
        return None


# Load all models
models = {}
for name, config in MODELS_TO_ENSEMBLE.items():
    preds = load_model_predictions(config)
    if preds is not None:
        models[name] = {
            **preds,
            'lb_score': config['lb_score'],
            'type': config['type']
        }
        lb_str = f"{config['lb_score']:.4f}" if config['lb_score'] else "TBD"
        print(f"   Loaded {name}: OOF F1={preds['oof_f1']:.4f}, LB={lb_str}")

if len(models) < 2:
    print("\n   ERROR: Need at least 2 models for ensemble!")
    print("   Please check artifact files exist.")
    sys.exit(1)

# Load ground truth
from utils.data_loader import load_all_data
data = load_all_data()
y = data['train_meta']['target'].values
test_ids = data['test_meta']['object_id'].tolist()

print(f"\n   Total models loaded: {len(models)}")

# ====================
# 2. ANALYZE PREDICTION DIVERSITY
# ====================
print("\n2. Analyzing prediction diversity...")

model_names = list(models.keys())
n_models = len(model_names)

# Correlation matrix
print("\n   OOF Prediction Correlations:")
corr_matrix = np.zeros((n_models, n_models))
for i, name1 in enumerate(model_names):
    for j, name2 in enumerate(model_names):
        corr = np.corrcoef(models[name1]['oof_preds'], models[name2]['oof_preds'])[0, 1]
        corr_matrix[i, j] = corr

print(f"   {'':15}", end='')
for name in model_names:
    print(f"{name[:10]:>12}", end='')
print()

for i, name1 in enumerate(model_names):
    print(f"   {name1[:15]:15}", end='')
    for j, name2 in enumerate(model_names):
        print(f"{corr_matrix[i,j]:12.3f}", end='')
    print()

# Prediction agreement
print("\n   Binary Prediction Agreement (at individual thresholds):")
for i, name1 in enumerate(model_names):
    for j, name2 in enumerate(model_names):
        if i < j:
            preds1 = (models[name1]['oof_preds'] > models[name1]['threshold']).astype(int)
            preds2 = (models[name2]['oof_preds'] > models[name2]['threshold']).astype(int)
            agreement = np.mean(preds1 == preds2)
            print(f"      {name1} vs {name2}: {agreement:.1%} agreement")

# ====================
# 3. ENSEMBLE STRATEGIES
# ====================
print("\n3. Testing ensemble strategies...")

ensemble_results = {}

# Strategy 1: Simple Average
print("\n   Strategy 1: Simple Average")
avg_oof = np.mean([models[name]['oof_preds'] for name in model_names], axis=0)
avg_test = np.mean([models[name]['test_preds'] for name in model_names], axis=0)

best_f1, best_thresh = 0, 0.1
for t in np.linspace(0.03, 0.5, 200):
    f1 = f1_score(y, (avg_oof > t).astype(int))
    if f1 > best_f1:
        best_f1, best_thresh = f1, t

ensemble_results['simple_avg'] = {
    'oof_f1': best_f1,
    'threshold': best_thresh,
    'oof_preds': avg_oof,
    'test_preds': avg_test
}
print(f"      OOF F1: {best_f1:.4f} @ threshold={best_thresh:.3f}")


# Strategy 2: Weighted Average (optimize weights on OOF)
print("\n   Strategy 2: Weighted Average (optimized)")

def optimize_weights(weights):
    """Negative F1 score for minimization."""
    weights = np.array(weights)
    weights = weights / weights.sum()  # Normalize

    blended = np.zeros_like(y, dtype=float)
    for i, name in enumerate(model_names):
        blended += weights[i] * models[name]['oof_preds']

    best_f1 = 0
    for t in np.linspace(0.05, 0.4, 50):
        f1 = f1_score(y, (blended > t).astype(int))
        if f1 > best_f1:
            best_f1 = f1

    return -best_f1  # Negative for minimization

# Grid search for 2-3 models (faster than scipy optimize)
best_weighted_f1 = 0
best_weights = None

if n_models == 2:
    for w1 in np.linspace(0.1, 0.9, 17):
        w2 = 1 - w1
        f1 = -optimize_weights([w1, w2])
        if f1 > best_weighted_f1:
            best_weighted_f1 = f1
            best_weights = [w1, w2]
elif n_models == 3:
    for w1 in np.linspace(0.1, 0.8, 8):
        for w2 in np.linspace(0.1, 0.8 - w1, 8):
            w3 = 1 - w1 - w2
            if w3 > 0.05:
                f1 = -optimize_weights([w1, w2, w3])
                if f1 > best_weighted_f1:
                    best_weighted_f1 = f1
                    best_weights = [w1, w2, w3]
else:
    # For more models, use scipy optimize
    from scipy.optimize import minimize
    x0 = np.ones(n_models) / n_models
    bounds = [(0.05, 0.95) for _ in range(n_models)]
    result = minimize(optimize_weights, x0, bounds=bounds, method='L-BFGS-B')
    best_weights = result.x / result.x.sum()
    best_weighted_f1 = -result.fun

# Apply best weights
weighted_oof = np.zeros_like(y, dtype=float)
weighted_test = np.zeros(len(test_ids), dtype=float)
for i, name in enumerate(model_names):
    weighted_oof += best_weights[i] * models[name]['oof_preds']
    weighted_test += best_weights[i] * models[name]['test_preds']

best_f1, best_thresh = 0, 0.1
for t in np.linspace(0.03, 0.5, 200):
    f1 = f1_score(y, (weighted_oof > t).astype(int))
    if f1 > best_f1:
        best_f1, best_thresh = f1, t

ensemble_results['weighted_avg'] = {
    'oof_f1': best_f1,
    'threshold': best_thresh,
    'weights': dict(zip(model_names, best_weights)),
    'oof_preds': weighted_oof,
    'test_preds': weighted_test
}

print(f"      Optimal weights: {dict(zip(model_names, [f'{w:.3f}' for w in best_weights]))}")
print(f"      OOF F1: {best_f1:.4f} @ threshold={best_thresh:.3f}")


# Strategy 3: Rank Average
print("\n   Strategy 3: Rank Average")

from scipy.stats import rankdata

rank_oof = np.zeros_like(y, dtype=float)
rank_test = np.zeros(len(test_ids), dtype=float)

for name in model_names:
    rank_oof += rankdata(models[name]['oof_preds'])
    rank_test += rankdata(models[name]['test_preds'])

rank_oof /= n_models
rank_test /= n_models

# Normalize to 0-1 range
rank_oof = (rank_oof - rank_oof.min()) / (rank_oof.max() - rank_oof.min())
rank_test = (rank_test - rank_test.min()) / (rank_test.max() - rank_test.min())

best_f1, best_thresh = 0, 0.5
for t in np.linspace(0.3, 0.8, 200):
    f1 = f1_score(y, (rank_oof > t).astype(int))
    if f1 > best_f1:
        best_f1, best_thresh = f1, t

ensemble_results['rank_avg'] = {
    'oof_f1': best_f1,
    'threshold': best_thresh,
    'oof_preds': rank_oof,
    'test_preds': rank_test
}
print(f"      OOF F1: {best_f1:.4f} @ threshold={best_thresh:.3f}")


# Strategy 4: Best XGB + Best LGB weighted
print("\n   Strategy 4: Best XGB + Best LGB (50/50)")

xgb_models = [name for name in model_names if models[name]['type'] == 'xgb']
lgb_models = [name for name in model_names if models[name]['type'] == 'lgb']

if xgb_models and lgb_models:
    # Pick best of each type by OOF F1
    best_xgb = max(xgb_models, key=lambda x: models[x]['oof_f1'])
    best_lgb = max(lgb_models, key=lambda x: models[x]['oof_f1'])

    xgb_lgb_oof = 0.5 * models[best_xgb]['oof_preds'] + 0.5 * models[best_lgb]['oof_preds']
    xgb_lgb_test = 0.5 * models[best_xgb]['test_preds'] + 0.5 * models[best_lgb]['test_preds']

    best_f1, best_thresh = 0, 0.1
    for t in np.linspace(0.03, 0.5, 200):
        f1 = f1_score(y, (xgb_lgb_oof > t).astype(int))
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    ensemble_results['xgb_lgb_50_50'] = {
        'oof_f1': best_f1,
        'threshold': best_thresh,
        'models': f"{best_xgb} + {best_lgb}",
        'oof_preds': xgb_lgb_oof,
        'test_preds': xgb_lgb_test
    }
    print(f"      Using: {best_xgb} + {best_lgb}")
    print(f"      OOF F1: {best_f1:.4f} @ threshold={best_thresh:.3f}")


# ====================
# 4. RESULTS COMPARISON
# ====================
print("\n" + "=" * 60)
print("ENSEMBLE RESULTS")
print("=" * 60)

print(f"\n{'Strategy':<25} {'OOF F1':<10} {'Threshold':<10}")
print("-" * 50)

# Individual models first
for name in model_names:
    lb_str = f"(LB={models[name]['lb_score']:.4f})" if models[name]['lb_score'] else ""
    print(f"{name:<25} {models[name]['oof_f1']:<10.4f} {models[name]['threshold']:<10.3f} {lb_str}")

print("-" * 50)

# Ensemble strategies
sorted_ensembles = sorted(ensemble_results.items(), key=lambda x: -x[1]['oof_f1'])
for name, result in sorted_ensembles:
    print(f"{name:<25} {result['oof_f1']:<10.4f} {result['threshold']:<10.3f}")


# ====================
# 5. CREATE SUBMISSIONS
# ====================
print("\n" + "=" * 60)
print("CREATING SUBMISSIONS")
print("=" * 60)

for name, result in sorted_ensembles:
    test_binary = (result['test_preds'] > result['threshold']).astype(int)

    submission = pd.DataFrame({
        'object_id': test_ids,
        'target': test_binary
    })

    filename = f"submission_v117_{name}.csv"
    submission.to_csv(base_path / f'submissions/{filename}', index=False)

    print(f"   {filename}: OOF F1={result['oof_f1']:.4f}, TDEs={test_binary.sum()}")


# ====================
# 6. SAVE ARTIFACTS
# ====================
print("\n6. Saving artifacts...")

artifacts = {
    'models': {name: {'oof_f1': m['oof_f1'], 'lb_score': m['lb_score'], 'type': m['type']}
               for name, m in models.items()},
    'ensemble_results': {name: {'oof_f1': r['oof_f1'], 'threshold': r['threshold']}
                         for name, r in ensemble_results.items()},
    'correlation_matrix': corr_matrix,
    'model_names': model_names
}

# Save full predictions for potential further analysis
artifacts['predictions'] = {
    name: {'oof': result['oof_preds'], 'test': result['test_preds']}
    for name, result in ensemble_results.items()
}

with open(base_path / 'data/processed/v117_ensemble_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print("   Saved: v117_ensemble_artifacts.pkl")


# ====================
# 7. FINAL RECOMMENDATIONS
# ====================
print("\n" + "=" * 60)
print("RECOMMENDATIONS")
print("=" * 60)

best_ensemble = sorted_ensembles[0]
print(f"""
   Best ensemble strategy: {best_ensemble[0]}
   Best OOF F1: {best_ensemble[1]['oof_f1']:.4f}

   Recommended submissions (in order of priority):
   1. {best_ensemble[0]} - Best OOF ensemble
   2. Best individual model with highest known LB
   3. weighted_avg - Often generalizes well

   Note: OOF performance doesn't always match LB!
   The weighted average often generalizes better than it appears.
""")

print("=" * 80)
print("v117 Ensemble Complete!")
print("=" * 80)

"""
MALLORN v107: Platt Scaling Calibration

Platt scaling (Platt, 1999) calibrates classifier outputs by fitting
a logistic regression on the model's predictions:
    P(y=1|s) = 1 / (1 + exp(A*s + B))

Where s is the original model score and A, B are learned parameters.

This can help when:
1. Model probabilities are poorly calibrated (overconfident/underconfident)
2. The decision boundary isn't at 0.5
3. Probability estimates need to be well-calibrated for threshold tuning

We apply Platt scaling to our two best models:
- v104: Seed Ensemble (OOF 0.6866)
- v105: Cross-Feature Interactions (OOF 0.6832)

Key implementation details:
- Use cross-validation to fit calibration parameters (avoid overfitting)
- Compare calibrated vs uncalibrated F1 scores
- Check reliability diagrams for calibration quality
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, brier_score_loss
from sklearn.isotonic import IsotonicRegression
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
base_path = Path(__file__).parent.parent

print("=" * 80)
print("MALLORN v107: Platt Scaling Calibration")
print("=" * 80)
print("\nGoal: Improve probability calibration for better threshold selection")

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
y_train = train_meta['target'].values

print(f"   Training: {len(train_ids)} ({np.sum(y_train)} TDE)")

# ====================
# 2. LOAD MODEL PREDICTIONS
# ====================
print("\n2. Loading model predictions...")

# v104: Seed Ensemble
with open(base_path / 'data/processed/v104_seed_ensemble_artifacts.pkl', 'rb') as f:
    v104_results = pickle.load(f)
v104_oof = v104_results['v104_seed_ensemble']['oof_preds']
v104_test = v104_results['v104_seed_ensemble']['test_preds']
v104_f1 = v104_results['v104_seed_ensemble']['oof_f1']
v104_thresh = v104_results['v104_seed_ensemble']['threshold']
print(f"   v104 Seed Ensemble: OOF F1={v104_f1:.4f}, threshold={v104_thresh:.3f}")

# v105: Cross-Feature Interactions
with open(base_path / 'data/processed/v105_interactions_artifacts.pkl', 'rb') as f:
    v105_results = pickle.load(f)
v105_oof = v105_results['v105b_interactions']['oof_preds']
v105_test = v105_results['v105b_interactions']['test_preds']
v105_f1 = v105_results['v105b_interactions']['oof_f1']
v105_thresh = v105_results['v105b_interactions']['threshold']
print(f"   v105 Interactions: OOF F1={v105_f1:.4f}, threshold={v105_thresh:.3f}")

# ====================
# 3. CALIBRATION ANALYSIS (Before)
# ====================
print("\n3. Calibration analysis (before Platt scaling)...")

def analyze_calibration(preds, y_true, name, n_bins=10):
    """Analyze probability calibration using binned statistics."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(preds, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_means = []
    bin_true_fracs = []
    bin_counts = []

    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_means.append(np.mean(preds[mask]))
            bin_true_fracs.append(np.mean(y_true[mask]))
            bin_counts.append(np.sum(mask))
        else:
            bin_means.append((bins[i] + bins[i+1]) / 2)
            bin_true_fracs.append(0)
            bin_counts.append(0)

    # Expected Calibration Error (ECE)
    ece = 0
    total = len(preds)
    for i in range(n_bins):
        if bin_counts[i] > 0:
            ece += (bin_counts[i] / total) * abs(bin_true_fracs[i] - bin_means[i])

    brier = brier_score_loss(y_true, preds)

    return {
        'ece': ece,
        'brier': brier,
        'bin_means': bin_means,
        'bin_true_fracs': bin_true_fracs,
        'bin_counts': bin_counts
    }

v104_cal = analyze_calibration(v104_oof, y_train, 'v104')
v105_cal = analyze_calibration(v105_oof, y_train, 'v105')

print(f"   v104 Seed Ensemble:")
print(f"      ECE (Expected Calibration Error): {v104_cal['ece']:.4f}")
print(f"      Brier Score: {v104_cal['brier']:.4f}")
print(f"      Prediction range: [{v104_oof.min():.4f}, {v104_oof.max():.4f}]")

print(f"   v105 Interactions:")
print(f"      ECE (Expected Calibration Error): {v105_cal['ece']:.4f}")
print(f"      Brier Score: {v105_cal['brier']:.4f}")
print(f"      Prediction range: [{v105_oof.min():.4f}, {v105_oof.max():.4f}]")

# ====================
# 4. PLATT SCALING IMPLEMENTATION
# ====================
print("\n4. Implementing Platt scaling...")

def platt_scale_cv(oof_preds, test_preds, y_train, n_folds=5, random_state=42):
    """
    Apply Platt scaling using cross-validation to avoid overfitting.

    For OOF predictions:
    - Split into folds
    - For each fold, fit calibrator on other folds, predict on held-out fold

    For test predictions:
    - Fit calibrator on all training data, apply to test
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # Reshape for sklearn
    X_oof = oof_preds.reshape(-1, 1)
    X_test = test_preds.reshape(-1, 1)

    # CV calibration for OOF
    calibrated_oof = np.zeros_like(oof_preds)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_oof, y_train)):
        # Fit logistic regression on training fold
        lr = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
        lr.fit(X_oof[train_idx], y_train[train_idx])

        # Predict probabilities on validation fold
        calibrated_oof[val_idx] = lr.predict_proba(X_oof[val_idx])[:, 1]

    # Fit final calibrator on all data for test predictions
    lr_final = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
    lr_final.fit(X_oof, y_train)
    calibrated_test = lr_final.predict_proba(X_test)[:, 1]

    # Get calibration parameters
    A = lr_final.coef_[0][0]
    B = lr_final.intercept_[0]

    return calibrated_oof, calibrated_test, {'A': A, 'B': B}

def isotonic_calibrate_cv(oof_preds, test_preds, y_train, n_folds=5, random_state=42):
    """
    Apply isotonic regression calibration using cross-validation.
    More flexible than Platt scaling but can overfit with small data.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # CV calibration for OOF
    calibrated_oof = np.zeros_like(oof_preds)

    for fold, (train_idx, val_idx) in enumerate(skf.split(oof_preds, y_train)):
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(oof_preds[train_idx], y_train[train_idx])
        calibrated_oof[val_idx] = iso.predict(oof_preds[val_idx])

    # Fit final calibrator on all data for test predictions
    iso_final = IsotonicRegression(out_of_bounds='clip')
    iso_final.fit(oof_preds, y_train)
    calibrated_test = iso_final.predict(test_preds)

    return calibrated_oof, calibrated_test

# ====================
# 5. VALIDATION: Platt Scaling Check
# ====================
print("\n5. Validation: Platt scaling implementation check...")

# Test with simple synthetic data
test_preds_simple = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 0.95])
test_y_simple = np.array([0, 0, 0, 1, 1, 0, 0, 1, 1, 1])
test_test_simple = np.array([0.15, 0.85])

cal_oof, cal_test, params = platt_scale_cv(
    test_preds_simple, test_test_simple, test_y_simple, n_folds=2
)

print(f"   Original predictions range: [{test_preds_simple.min():.3f}, {test_preds_simple.max():.3f}]")
print(f"   Calibrated predictions range: [{cal_oof.min():.3f}, {cal_oof.max():.3f}]")
print(f"   Platt parameters: A={params['A']:.4f}, B={params['B']:.4f}")

# Verify calibrated predictions are valid probabilities
assert np.all((cal_oof >= 0) & (cal_oof <= 1)), "Calibrated OOF not in [0,1]"
assert np.all((cal_test >= 0) & (cal_test <= 1)), "Calibrated test not in [0,1]"
print("   [PASS] Calibrated predictions are valid probabilities")

# ====================
# 6. APPLY PLATT SCALING TO BEST MODELS
# ====================
print("\n6. Applying Platt scaling to best models...")

results = {}

# Helper function
def evaluate_and_store(name, oof_preds, test_preds, y_train):
    """Evaluate calibrated predictions and return results dict."""
    best_f1 = 0
    best_thresh = 0.3
    for t in np.linspace(0.05, 0.5, 200):
        f1 = f1_score(y_train, (oof_preds > t).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    preds_binary = (oof_preds > best_thresh).astype(int)
    cm = confusion_matrix(y_train, preds_binary)
    tn, fp, fn, tp = cm.ravel()

    cal_metrics = analyze_calibration(oof_preds, y_train, name)

    return {
        'oof_f1': best_f1,
        'threshold': best_thresh,
        'oof_preds': oof_preds,
        'test_preds': test_preds,
        'confusion': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
        'recall': tp / (tp + fn),
        'precision': tp / (tp + fp),
        'ece': cal_metrics['ece'],
        'brier': cal_metrics['brier'],
    }

# v104 Seed Ensemble
print("\n   v104 Seed Ensemble:")

# Platt scaling
v104_platt_oof, v104_platt_test, v104_params = platt_scale_cv(
    v104_oof, v104_test, y_train
)
results['v107a_v104_platt'] = evaluate_and_store(
    'v104_platt', v104_platt_oof, v104_platt_test, y_train
)
results['v107a_v104_platt']['calibration_params'] = v104_params
print(f"      Platt: OOF F1={results['v107a_v104_platt']['oof_f1']:.4f}, "
      f"ECE={results['v107a_v104_platt']['ece']:.4f}")

# Isotonic regression
v104_iso_oof, v104_iso_test = isotonic_calibrate_cv(
    v104_oof, v104_test, y_train
)
results['v107b_v104_isotonic'] = evaluate_and_store(
    'v104_isotonic', v104_iso_oof, v104_iso_test, y_train
)
print(f"      Isotonic: OOF F1={results['v107b_v104_isotonic']['oof_f1']:.4f}, "
      f"ECE={results['v107b_v104_isotonic']['ece']:.4f}")

# v105 Interactions
print("\n   v105 Interactions:")

# Platt scaling
v105_platt_oof, v105_platt_test, v105_params = platt_scale_cv(
    v105_oof, v105_test, y_train
)
results['v107c_v105_platt'] = evaluate_and_store(
    'v105_platt', v105_platt_oof, v105_platt_test, y_train
)
results['v107c_v105_platt']['calibration_params'] = v105_params
print(f"      Platt: OOF F1={results['v107c_v105_platt']['oof_f1']:.4f}, "
      f"ECE={results['v107c_v105_platt']['ece']:.4f}")

# Isotonic regression
v105_iso_oof, v105_iso_test = isotonic_calibrate_cv(
    v105_oof, v105_test, y_train
)
results['v107d_v105_isotonic'] = evaluate_and_store(
    'v105_isotonic', v105_iso_oof, v105_iso_test, y_train
)
print(f"      Isotonic: OOF F1={results['v107d_v105_isotonic']['oof_f1']:.4f}, "
      f"ECE={results['v107d_v105_isotonic']['ece']:.4f}")

# ====================
# 7. CALIBRATION ANALYSIS (After)
# ====================
print("\n7. Calibration comparison (before vs after)...")

print("\n   v104 Seed Ensemble:")
print(f"      Before: ECE={v104_cal['ece']:.4f}, Brier={v104_cal['brier']:.4f}, F1={v104_f1:.4f}")
print(f"      Platt:  ECE={results['v107a_v104_platt']['ece']:.4f}, "
      f"Brier={results['v107a_v104_platt']['brier']:.4f}, "
      f"F1={results['v107a_v104_platt']['oof_f1']:.4f}")
print(f"      Isoton: ECE={results['v107b_v104_isotonic']['ece']:.4f}, "
      f"Brier={results['v107b_v104_isotonic']['brier']:.4f}, "
      f"F1={results['v107b_v104_isotonic']['oof_f1']:.4f}")

print("\n   v105 Interactions:")
print(f"      Before: ECE={v105_cal['ece']:.4f}, Brier={v105_cal['brier']:.4f}, F1={v105_f1:.4f}")
print(f"      Platt:  ECE={results['v107c_v105_platt']['ece']:.4f}, "
      f"Brier={results['v107c_v105_platt']['brier']:.4f}, "
      f"F1={results['v107c_v105_platt']['oof_f1']:.4f}")
print(f"      Isoton: ECE={results['v107d_v105_isotonic']['ece']:.4f}, "
      f"Brier={results['v107d_v105_isotonic']['brier']:.4f}, "
      f"F1={results['v107d_v105_isotonic']['oof_f1']:.4f}")

# ====================
# 8. RESULTS
# ====================
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

print(f"\n{'Variant':<25} {'OOF F1':<10} {'Recall':<10} {'Prec':<10} {'ECE':<10} {'Brier':<10}")
print("-" * 75)
print(f"{'v104 Original':<25} {v104_f1:<10.4f} {'77.7%':<10} {'61.5%':<10} {v104_cal['ece']:<10.4f} {v104_cal['brier']:<10.4f}")
print(f"{'v105 Original':<25} {v105_f1:<10.4f} {'74.3%':<10} {'63.2%':<10} {v105_cal['ece']:<10.4f} {v105_cal['brier']:<10.4f}")
print("-" * 75)

sorted_results = sorted(results.items(), key=lambda x: -x[1]['oof_f1'])

for name, res in sorted_results:
    recall_str = f"{100*res['recall']:.1f}%"
    prec_str = f"{100*res['precision']:.1f}%"
    print(f"{name:<25} {res['oof_f1']:<10.4f} {recall_str:<10} {prec_str:<10} {res['ece']:<10.4f} {res['brier']:<10.4f}")

# ====================
# 9. SUBMISSION
# ====================
print("\n" + "=" * 80)
print("SUBMISSION")
print("=" * 80)

# Create submissions for all variants
for name, res in sorted_results:
    test_binary = (res['test_preds'] > res['threshold']).astype(int)

    submission = pd.DataFrame({
        'object_id': test_ids,
        'target': test_binary
    })

    filename = f"submission_{name}.csv"
    submission.to_csv(base_path / f'submissions/{filename}', index=False)

    print(f"   {filename}: OOF={res['oof_f1']:.4f}, TDEs={test_binary.sum()}")

# Save all results
with open(base_path / 'data/processed/v107_platt_scaling_artifacts.pkl', 'wb') as f:
    pickle.dump(results, f)

# Also save original model info for comparison
results['originals'] = {
    'v104': {'oof_f1': v104_f1, 'ece': v104_cal['ece'], 'brier': v104_cal['brier']},
    'v105': {'oof_f1': v105_f1, 'ece': v105_cal['ece'], 'brier': v105_cal['brier']},
}

with open(base_path / 'data/processed/v107_platt_scaling_artifacts.pkl', 'wb') as f:
    pickle.dump(results, f)

print("\n" + "=" * 80)
print("v107 Platt Scaling Complete")
print("=" * 80)

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
best_v104 = max([v104_f1, results['v107a_v104_platt']['oof_f1'], results['v107b_v104_isotonic']['oof_f1']])
best_v105 = max([v105_f1, results['v107c_v105_platt']['oof_f1'], results['v107d_v105_isotonic']['oof_f1']])

v104_improved = best_v104 > v104_f1
v105_improved = best_v105 > v105_f1

print(f"\n   v104 Seed Ensemble: {'IMPROVED' if v104_improved else 'NO IMPROVEMENT'}")
print(f"      Original F1: {v104_f1:.4f}")
print(f"      Best calibrated F1: {best_v104:.4f} ({best_v104 - v104_f1:+.4f})")

print(f"\n   v105 Interactions: {'IMPROVED' if v105_improved else 'NO IMPROVEMENT'}")
print(f"      Original F1: {v105_f1:.4f}")
print(f"      Best calibrated F1: {best_v105:.4f} ({best_v105 - v105_f1:+.4f})")

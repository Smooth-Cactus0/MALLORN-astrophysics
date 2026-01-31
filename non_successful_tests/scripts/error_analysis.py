"""
MALLORN: Deep Error Analysis

Analyze what v34a and v88 are doing right and wrong.
Understand systematic errors to plan corrections.
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, classification_report, f1_score,
    precision_recall_curve, roc_curve, auc
)
from collections import Counter
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
base_path = Path(__file__).parent.parent

print("=" * 80)
print("MALLORN: Deep Error Analysis")
print("=" * 80)

# ====================
# 1. LOAD DATA AND PREDICTIONS
# ====================
print("\n1. Loading data and model predictions...")

from utils.data_loader import load_all_data
data = load_all_data()

train_meta = data['train_meta']
y_true = train_meta['target'].values
object_ids = train_meta['object_id'].tolist()

# Get SpecType for analysis
spec_types = train_meta['SpecType'].values if 'SpecType' in train_meta.columns else None

print(f"   Total training samples: {len(y_true)}")
print(f"   TDE (positive class): {np.sum(y_true)} ({100*np.mean(y_true):.2f}%)")
print(f"   Non-TDE (negative class): {np.sum(y_true==0)} ({100*np.mean(y_true==0):.2f}%)")

# Load v34a predictions
with open(base_path / 'data/processed/v34a_artifacts.pkl', 'rb') as f:
    v34a = pickle.load(f)

# Check what's in v34a
print(f"\n   v34a keys: {list(v34a.keys())}")

# Load v88 predictions
with open(base_path / 'data/processed/v88_artifacts.pkl', 'rb') as f:
    v88 = pickle.load(f)

oof_v88 = v88['oof_preds']
thresh_v88 = v88['threshold']

print(f"   v88 OOF predictions loaded, threshold: {thresh_v88:.3f}")

# Try to get v34a OOF predictions - might be stored differently
if 'oof_preds' in v34a:
    oof_v34a = v34a['oof_preds']
    thresh_v34a = v34a.get('threshold', 0.386)
else:
    # Reconstruct from artifacts or use a default
    print("   v34a OOF predictions not directly available, will compute from model")
    oof_v34a = None
    thresh_v34a = 0.386

# ====================
# 2. COMPUTE PREDICTIONS AT OPTIMAL THRESHOLDS
# ====================
print("\n2. Computing predictions at optimal thresholds...")

def find_optimal_threshold(y_true, y_proba):
    best_f1 = 0
    best_thresh = 0.5
    for t in np.linspace(0.01, 0.5, 200):
        preds = (y_proba > t).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return best_thresh, best_f1

# v88 analysis
thresh_v88_opt, f1_v88 = find_optimal_threshold(y_true, oof_v88)
preds_v88 = (oof_v88 > thresh_v88_opt).astype(int)

print(f"   v88: Optimal threshold={thresh_v88_opt:.3f}, F1={f1_v88:.4f}")

# ====================
# 3. CONFUSION MATRIX ANALYSIS
# ====================
print("\n" + "=" * 80)
print("3. CONFUSION MATRIX ANALYSIS")
print("=" * 80)

def analyze_confusion(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n{model_name}:")
    print(f"   Confusion Matrix:")
    print(f"                    Predicted")
    print(f"                  Non-TDE   TDE")
    print(f"   Actual Non-TDE   {tn:5d}   {fp:5d}")
    print(f"   Actual TDE       {fn:5d}   {tp:5d}")
    print(f"\n   True Negatives (TN): {tn} - Correctly identified non-TDE")
    print(f"   True Positives (TP): {tp} - Correctly identified TDE")
    print(f"   False Positives (FP): {fp} - Non-TDE wrongly classified as TDE")
    print(f"   False Negatives (FN): {fn} - TDE wrongly classified as non-TDE")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n   Precision: {precision:.4f} (of predicted TDE, how many are correct)")
    print(f"   Recall: {recall:.4f} (of actual TDE, how many did we find)")
    print(f"   F1 Score: {f1:.4f}")

    return {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 'precision': precision, 'recall': recall, 'f1': f1}

metrics_v88 = analyze_confusion(y_true, preds_v88, "v88 (OOF F1=0.6854, LB=0.6876)")

# ====================
# 4. PER-CLASS ANALYSIS (SpecType breakdown)
# ====================
print("\n" + "=" * 80)
print("4. PER-CLASS ANALYSIS (by SpecType)")
print("=" * 80)

if spec_types is not None:
    print("\n   Class distribution in training data:")
    spec_counts = Counter(spec_types)
    for spec, count in sorted(spec_counts.items(), key=lambda x: -x[1]):
        is_tde = 'TDE' in spec
        print(f"      {spec}: {count} samples ({'TDE' if is_tde else 'non-TDE'})")

    print("\n   v88 Performance by SpecType:")
    print(f"   {'SpecType':<15} {'Count':<8} {'Target':<8} {'Correct':<10} {'Accuracy':<10}")
    print("-" * 55)

    for spec in sorted(set(spec_types)):
        mask = spec_types == spec
        count = np.sum(mask)
        actual = y_true[mask]
        predicted = preds_v88[mask]
        correct = np.sum(actual == predicted)
        accuracy = correct / count if count > 0 else 0
        target = 'TDE' if np.mean(actual) > 0.5 else 'non-TDE'
        print(f"   {spec:<15} {count:<8} {target:<8} {correct:<10} {accuracy:.4f}")

# ====================
# 5. FALSE POSITIVE ANALYSIS
# ====================
print("\n" + "=" * 80)
print("5. FALSE POSITIVE ANALYSIS")
print("=" * 80)

fp_mask = (y_true == 0) & (preds_v88 == 1)
fp_indices = np.where(fp_mask)[0]
fp_ids = [object_ids[i] for i in fp_indices]
fp_probs = oof_v88[fp_mask]

print(f"\n   Total False Positives: {len(fp_indices)}")
print(f"   These are NON-TDE objects incorrectly classified as TDE")

if spec_types is not None:
    fp_specs = spec_types[fp_mask]
    print(f"\n   False Positives by SpecType:")
    fp_spec_counts = Counter(fp_specs)
    for spec, count in sorted(fp_spec_counts.items(), key=lambda x: -x[1]):
        print(f"      {spec}: {count} ({100*count/len(fp_indices):.1f}%)")

print(f"\n   Probability distribution of False Positives:")
print(f"      Mean: {np.mean(fp_probs):.4f}")
print(f"      Median: {np.median(fp_probs):.4f}")
print(f"      Min: {np.min(fp_probs):.4f}, Max: {np.max(fp_probs):.4f}")

# High-confidence false positives (most concerning)
high_conf_fp = fp_probs > 0.5
print(f"\n   High-confidence FP (prob > 0.5): {np.sum(high_conf_fp)}")

# ====================
# 6. FALSE NEGATIVE ANALYSIS
# ====================
print("\n" + "=" * 80)
print("6. FALSE NEGATIVE ANALYSIS")
print("=" * 80)

fn_mask = (y_true == 1) & (preds_v88 == 0)
fn_indices = np.where(fn_mask)[0]
fn_ids = [object_ids[i] for i in fn_indices]
fn_probs = oof_v88[fn_mask]

print(f"\n   Total False Negatives: {len(fn_indices)}")
print(f"   These are TDE objects incorrectly classified as non-TDE")

if spec_types is not None:
    fn_specs = spec_types[fn_mask]
    print(f"\n   False Negatives by SpecType:")
    fn_spec_counts = Counter(fn_specs)
    for spec, count in sorted(fn_spec_counts.items(), key=lambda x: -x[1]):
        print(f"      {spec}: {count} ({100*count/len(fn_indices):.1f}%)")

print(f"\n   Probability distribution of False Negatives:")
print(f"      Mean: {np.mean(fn_probs):.4f}")
print(f"      Median: {np.median(fn_probs):.4f}")
print(f"      Min: {np.min(fn_probs):.4f}, Max: {np.max(fn_probs):.4f}")

# Very low probability FN (model is very wrong)
very_low_fn = fn_probs < 0.1
print(f"\n   Very confident wrong FN (prob < 0.1): {np.sum(very_low_fn)}")

# ====================
# 7. FEATURE ANALYSIS OF MISCLASSIFIED SAMPLES
# ====================
print("\n" + "=" * 80)
print("7. FEATURE ANALYSIS OF MISCLASSIFIED SAMPLES")
print("=" * 80)

# Load features
cached = pd.read_pickle(base_path / 'data/processed/features_v4_cache.pkl')
train_features = cached['train_features']

# Key features to analyze
key_features = [
    'g_mean_flux', 'r_mean_flux', 'i_mean_flux',
    'g_minus_r_at_peak', 'r_minus_i_at_peak',
    'all_rise_time', 'all_fade_time', 'all_amplitude',
    'Z'  # redshift
]

available_key_features = [f for f in key_features if f in train_features.columns]

print(f"\n   Analyzing key features: {available_key_features}")

# Compare correctly vs incorrectly classified TDEs
correct_tde_mask = (y_true == 1) & (preds_v88 == 1)
wrong_tde_mask = (y_true == 1) & (preds_v88 == 0)

print(f"\n   Correctly classified TDEs: {np.sum(correct_tde_mask)}")
print(f"   Incorrectly classified TDEs (FN): {np.sum(wrong_tde_mask)}")

print(f"\n   Feature comparison (Correct TDE vs Missed TDE):")
print(f"   {'Feature':<25} {'Correct TDE':<15} {'Missed TDE':<15} {'Difference':<15}")
print("-" * 75)

for feat in available_key_features:
    if feat in train_features.columns:
        correct_vals = train_features.loc[correct_tde_mask, feat].dropna()
        wrong_vals = train_features.loc[wrong_tde_mask, feat].dropna()

        if len(correct_vals) > 0 and len(wrong_vals) > 0:
            correct_mean = np.mean(correct_vals)
            wrong_mean = np.mean(wrong_vals)
            diff_pct = 100 * (wrong_mean - correct_mean) / (correct_mean + 1e-9)
            print(f"   {feat:<25} {correct_mean:<15.4f} {wrong_mean:<15.4f} {diff_pct:+.1f}%")

# ====================
# 8. CLASS IMBALANCE IMPACT
# ====================
print("\n" + "=" * 80)
print("8. CLASS IMBALANCE ANALYSIS")
print("=" * 80)

n_tde = np.sum(y_true == 1)
n_non_tde = np.sum(y_true == 0)
imbalance_ratio = n_non_tde / n_tde

print(f"\n   TDE samples: {n_tde}")
print(f"   Non-TDE samples: {n_non_tde}")
print(f"   Imbalance ratio: {imbalance_ratio:.1f}:1 (non-TDE : TDE)")

print(f"\n   Current model behavior:")
print(f"   - Predicting {np.sum(preds_v88 == 1)} samples as TDE")
print(f"   - Predicting {np.sum(preds_v88 == 0)} samples as non-TDE")
print(f"   - Ratio of predictions: {np.sum(preds_v88 == 0) / max(np.sum(preds_v88 == 1), 1):.1f}:1")

# ====================
# 9. RECOMMENDATIONS
# ====================
print("\n" + "=" * 80)
print("9. DIAGNOSIS AND RECOMMENDATIONS")
print("=" * 80)

fp_rate = metrics_v88['fp'] / (metrics_v88['fp'] + metrics_v88['tn'])
fn_rate = metrics_v88['fn'] / (metrics_v88['fn'] + metrics_v88['tp'])

print(f"\n   False Positive Rate: {fp_rate:.4f} ({100*fp_rate:.2f}%)")
print(f"   False Negative Rate: {fn_rate:.4f} ({100*fn_rate:.2f}%)")

print("\n   DIAGNOSIS:")
if fn_rate > fp_rate:
    print(f"   - Model is UNDER-PREDICTING TDEs (missing {metrics_v88['fn']} of {n_tde} = {100*fn_rate:.1f}%)")
    print(f"   - This suggests the minority class (TDE) is underrepresented in training")
    print(f"   - SMOTE/ADASYN could help by synthesizing more TDE examples")
else:
    print(f"   - Model is OVER-PREDICTING TDEs (false alarm rate: {100*fp_rate:.1f}%)")
    print(f"   - This suggests the model is too sensitive to TDE-like patterns")

print("\n   RECOMMENDED ACTIONS:")
print("   1. SMOTE/ADASYN: Oversample minority class (TDE) to balance training")
print("   2. Class weights: Increase penalty for missing TDEs")
print("   3. Threshold tuning: May need different threshold for train vs test")
print("   4. Feature engineering: Focus on features that separate FN from TP")

# Save analysis results
analysis_results = {
    'metrics_v88': metrics_v88,
    'fp_indices': fp_indices,
    'fn_indices': fn_indices,
    'fp_ids': fp_ids,
    'fn_ids': fn_ids,
    'fp_probs': fp_probs,
    'fn_probs': fn_probs,
    'imbalance_ratio': imbalance_ratio
}

with open(base_path / 'data/processed/error_analysis.pkl', 'wb') as f:
    pickle.dump(analysis_results, f)

print("\n   Analysis saved to: data/processed/error_analysis.pkl")

print("\n" + "=" * 80)
print("Error Analysis Complete")
print("=" * 80)

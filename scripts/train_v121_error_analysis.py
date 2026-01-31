"""
MALLORN v121: Error Analysis
============================

Analyze false positives and false negatives to understand:
1. What types of TDEs are we missing?
2. What non-TDEs are we incorrectly classifying?
3. Are there patterns we can exploit?

This analysis can inform:
- Feature engineering improvements
- Threshold tuning
- Sample weighting adjustments
"""

import sys
import pickle
import gzip
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

base_path = Path(__file__).parent.parent

print("=" * 70)
print("MALLORN v121: Error Analysis")
print("=" * 70)

# ============================================================================
# 1. LOAD DATA AND PREDICTIONS
# ============================================================================
print("\n[1/5] Loading data and predictions...")

# Load feature package
with gzip.open(base_path / 'data/kaggle_ensemble_package.pkl.gz', 'rb') as f:
    package = pickle.load(f)

train_features = package['train_features']
y = package['y']
train_ids = package['train_ids']

# Load best model predictions (v92d)
with open(base_path / 'data/processed/v92_focal_adv_artifacts.pkl', 'rb') as f:
    v92_arts = pickle.load(f)

v92d_oof = v92_arts['v92d_baseline_adv']['oof_preds']
v92d_threshold = v92_arts['v92d_baseline_adv']['threshold']

# Load all model predictions for comparison
with open(base_path / 'data/processed/v34a_artifacts.pkl', 'rb') as f:
    v34a_arts = pickle.load(f)

with open(base_path / 'data/processed/v114_optimized_artifacts.pkl', 'rb') as f:
    v114_arts = pickle.load(f)

with open(base_path / 'data/processed/v118_catboost_artifacts.pkl', 'rb') as f:
    catboost_arts = pickle.load(f)

models = {
    'v92d': v92d_oof,
    'v34a': v34a_arts['oof_preds'],
    'v114d': v114_arts['results']['v114d_minimal_research']['oof_preds'],
    'v118_catboost': catboost_arts['avg_oof_preds'],
}

print(f"   Training samples: {len(y)}")
print(f"   TDEs: {np.sum(y)} ({100*np.sum(y)/len(y):.1f}%)")
print(f"   Non-TDEs: {len(y) - np.sum(y)} ({100*(len(y)-np.sum(y))/len(y):.1f}%)")

# ============================================================================
# 2. CLASSIFY ERRORS
# ============================================================================
print("\n[2/5] Classifying errors for v92d (best model)...")

# Use optimal threshold
best_threshold = v92d_threshold
preds = (v92d_oof > best_threshold).astype(int)

# Find optimal threshold
best_f1, opt_threshold = 0, 0.1
for t in np.linspace(0.03, 0.5, 100):
    f1 = f1_score(y, (v92d_oof > t).astype(int))
    if f1 > best_f1:
        best_f1, opt_threshold = f1, t

preds = (v92d_oof > opt_threshold).astype(int)

print(f"   Optimal threshold: {opt_threshold:.3f}")
print(f"   OOF F1: {best_f1:.4f}")

# Confusion matrix
tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
print(f"\n   Confusion Matrix:")
print(f"   TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"   Precision: {tp/(tp+fp):.3f}")
print(f"   Recall: {tp/(tp+fn):.3f}")

# Identify error indices
fp_idx = np.where((preds == 1) & (y == 0))[0]
fn_idx = np.where((preds == 0) & (y == 1))[0]
tp_idx = np.where((preds == 1) & (y == 1))[0]
tn_idx = np.where((preds == 0) & (y == 0))[0]

print(f"\n   False Positives (non-TDE predicted as TDE): {len(fp_idx)}")
print(f"   False Negatives (TDE missed): {len(fn_idx)}")
print(f"   True Positives (TDE correctly found): {len(tp_idx)}")

# ============================================================================
# 3. ANALYZE ERROR PATTERNS
# ============================================================================
print("\n[3/5] Analyzing error patterns...")

# Get important features from v34a
with open(base_path / 'data/processed/v34a_artifacts.pkl', 'rb') as f:
    v34a = pickle.load(f)

# Handle feature importance - could be DataFrame or array
fi = v34a['feature_importance']
if isinstance(fi, pd.DataFrame):
    feature_importance = fi.sort_values('importance', ascending=False)
else:
    # It's an array
    feature_importance = pd.DataFrame({
        'feature': v34a['feature_names'],
        'importance': fi.flatten() if hasattr(fi, 'flatten') else fi
    }).sort_values('importance', ascending=False)

top_features = feature_importance.head(20)['feature'].tolist()

# Add research features
research_features = [
    'nuclear_concentration', 'nuclear_smoothness',
    'g_r_color_at_peak', 'r_i_color_at_peak',
    'mhps_10_100_ratio', 'mhps_30_100_ratio'
]
analysis_features = top_features + [f for f in research_features if f in train_features.columns and f not in top_features]

# Compare statistics for each group
print("\n   Feature Statistics by Group:")
print("   " + "=" * 80)

df = train_features.copy()
df['prediction'] = preds
df['target'] = y
df['error_type'] = 'TN'  # Default
df.loc[tp_idx, 'error_type'] = 'TP'
df.loc[fp_idx, 'error_type'] = 'FP'
df.loc[fn_idx, 'error_type'] = 'FN'

print(f"\n   {'Feature':<30} {'TP':<12} {'FN':<12} {'FP':<12} {'TN':<12}")
print("   " + "-" * 78)

feature_diffs = []
for feat in analysis_features[:15]:
    if feat not in df.columns:
        continue

    tp_mean = df.loc[tp_idx, feat].mean()
    fn_mean = df.loc[fn_idx, feat].mean()
    fp_mean = df.loc[fp_idx, feat].mean()
    tn_mean = df.loc[tn_idx, feat].mean()

    # Calculate how different FN is from TP (missed TDEs vs found TDEs)
    fn_tp_diff = abs(fn_mean - tp_mean) / (abs(tp_mean) + 1e-10) if not np.isnan(fn_mean) else 0

    feature_diffs.append({
        'feature': feat,
        'tp_mean': tp_mean,
        'fn_mean': fn_mean,
        'fp_mean': fp_mean,
        'tn_mean': tn_mean,
        'fn_tp_diff': fn_tp_diff
    })

    print(f"   {feat:<30} {tp_mean:<12.3f} {fn_mean:<12.3f} {fp_mean:<12.3f} {tn_mean:<12.3f}")

# ============================================================================
# 4. ANALYZE MODEL AGREEMENT ON ERRORS
# ============================================================================
print("\n[4/5] Analyzing model agreement on errors...")

# For each error, check if other models agree or disagree
print("\n   Model predictions on False Negatives (missed TDEs):")
print(f"   Total FN: {len(fn_idx)}")

model_fn_preds = {}
for name, preds_model in models.items():
    # Find threshold for this model
    best_t = 0.1
    best_f1 = 0
    for t in np.linspace(0.03, 0.5, 50):
        f1 = f1_score(y, (preds_model > t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t

    fn_preds = preds_model[fn_idx]
    fn_pred_binary = (fn_preds > best_t).astype(int)
    model_fn_preds[name] = fn_pred_binary

    recovered = fn_pred_binary.sum()
    print(f"   {name}: {recovered}/{len(fn_idx)} would be recovered ({100*recovered/len(fn_idx):.1f}%)")

# Check how many FN are missed by ALL models
all_miss_count = 0
for i, idx in enumerate(fn_idx):
    all_miss = all(model_fn_preds[name][i] == 0 for name in models.keys())
    if all_miss:
        all_miss_count += 1

print(f"\n   Missed by ALL models: {all_miss_count}/{len(fn_idx)} ({100*all_miss_count/len(fn_idx):.1f}%)")
print(f"   Recoverable by at least one model: {len(fn_idx) - all_miss_count}")

# Analyze the "hard" false negatives
print("\n   Analyzing 'hard' false negatives (missed by all models)...")

hard_fn_idx = []
for i, idx in enumerate(fn_idx):
    all_miss = all(model_fn_preds[name][i] == 0 for name in models.keys())
    if all_miss:
        hard_fn_idx.append(idx)

if len(hard_fn_idx) > 0:
    print(f"\n   Top features that differ for hard FN vs TP:")
    for feat in analysis_features[:10]:
        if feat not in df.columns:
            continue
        hard_fn_mean = df.loc[hard_fn_idx, feat].mean()
        tp_mean = df.loc[tp_idx, feat].mean()
        diff_pct = 100 * (hard_fn_mean - tp_mean) / (abs(tp_mean) + 1e-10)
        if abs(diff_pct) > 20:  # Only show significant differences
            print(f"      {feat}: TP={tp_mean:.3f}, Hard FN={hard_fn_mean:.3f} ({diff_pct:+.1f}%)")

# ============================================================================
# 5. PREDICTION CONFIDENCE ANALYSIS
# ============================================================================
print("\n[5/5] Prediction confidence analysis...")

print("\n   v92d Prediction Distribution by Group:")
for group, idx in [('TP', tp_idx), ('FN', fn_idx), ('FP', fp_idx), ('TN', tn_idx)]:
    if len(idx) > 0:
        probs = v92d_oof[idx]
        print(f"   {group}: mean={probs.mean():.3f}, std={probs.std():.3f}, "
              f"min={probs.min():.3f}, max={probs.max():.3f}")

# Find "borderline" predictions (close to threshold)
borderline_margin = 0.1
borderline_idx = np.where(np.abs(v92d_oof - opt_threshold) < borderline_margin)[0]
borderline_tde = np.sum(y[borderline_idx])
print(f"\n   Borderline predictions (within {borderline_margin} of threshold):")
print(f"   Total: {len(borderline_idx)}")
print(f"   TDEs in borderline: {borderline_tde} ({100*borderline_tde/len(borderline_idx):.1f}%)")

# ============================================================================
# SUMMARY & RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 70)
print("ERROR ANALYSIS SUMMARY")
print("=" * 70)

# Save analysis
analysis_results = {
    'fp_idx': fp_idx,
    'fn_idx': fn_idx,
    'tp_idx': tp_idx,
    'tn_idx': tn_idx,
    'hard_fn_idx': hard_fn_idx,
    'feature_diffs': feature_diffs,
    'model_fn_recovery': model_fn_preds,
    'borderline_idx': borderline_idx,
}

with open(base_path / 'data/processed/v121_error_analysis.pkl', 'wb') as f:
    pickle.dump(analysis_results, f)

print(f"""
Key Findings:

1. ERROR RATES (v92d @ threshold {opt_threshold:.3f}):
   - False Negatives (missed TDEs): {len(fn_idx)}/{np.sum(y)} = {100*len(fn_idx)/np.sum(y):.1f}%
   - False Positives: {len(fp_idx)}/{len(y)-np.sum(y)} = {100*len(fp_idx)/(len(y)-np.sum(y)):.2f}%
   - OOF F1: {best_f1:.4f}

2. MODEL AGREEMENT:
   - {len(fn_idx) - all_miss_count} FN could be recovered by ensemble diversity
   - {all_miss_count} FN are "hard" cases missed by all models

3. ENSEMBLE POTENTIAL:
   - Different models disagree on some errors
   - This is why ensembling can help!

Recommendations:
   1. Focus on borderline predictions for threshold tuning
   2. Use model disagreement as a signal for uncertainty
   3. Hard FN cases may need physics-based features to capture

Saved analysis to: data/processed/v121_error_analysis.pkl
""")

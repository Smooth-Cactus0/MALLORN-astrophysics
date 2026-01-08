"""
MALLORN: Quick Distribution Check (No sklearn/xgboost dependencies)

Simple statistical comparison of train vs test distributions.
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

base_path = Path(__file__).parent.parent

print("=" * 80)
print("MALLORN: Train vs Test Distribution Analysis")
print("=" * 80)

# Load metadata
print("\n1. Loading metadata...")
train_meta = pd.read_csv(base_path / 'data/raw/train_log.csv')
test_meta = pd.read_csv(base_path / 'data/raw/test_log.csv')

print(f"   Train: {len(train_meta)} samples")
print(f"   Test: {len(test_meta)} samples")

# Analyze redshift
print("\n2. Redshift (Z) Distribution:")
train_z = train_meta['Z']
test_z = test_meta['Z']

print(f"   Train: mean={train_z.mean():.4f}, std={train_z.std():.4f}, median={train_z.median():.4f}")
print(f"   Test:  mean={test_z.mean():.4f}, std={test_z.std():.4f}, median={test_z.median():.4f}")

z_mean_diff = abs(test_z.mean() - train_z.mean()) / train_z.std()
print(f"   Mean difference: {z_mean_diff:.3f} std devs")

if z_mean_diff < 0.1:
    print("   --> IDENTICAL distributions")
elif z_mean_diff < 0.3:
    print("   --> VERY SIMILAR distributions")
else:
    print("   --> DIFFERENT distributions (concern!)")

# Analyze extinction
print("\n3. Extinction (EBV) Distribution:")
train_ebv = train_meta['EBV']
test_ebv = test_meta['EBV']

print(f"   Train: mean={train_ebv.mean():.4f}, std={train_ebv.std():.4f}, median={train_ebv.median():.4f}")
print(f"   Test:  mean={test_ebv.mean():.4f}, std={test_ebv.std():.4f}, median={test_ebv.median():.4f}")

ebv_mean_diff = abs(test_ebv.mean() - train_ebv.mean()) / (train_ebv.std() + 1e-9)
print(f"   Mean difference: {ebv_mean_diff:.3f} std devs")

if ebv_mean_diff < 0.1:
    print("   --> IDENTICAL distributions")
elif ebv_mean_diff < 0.3:
    print("   --> VERY SIMILAR distributions")
else:
    print("   --> DIFFERENT distributions (concern!)")

# Load v34a predictions
print("\n4. Analyzing v34a (Bazin) predictions...")
try:
    with open(base_path / 'data/processed/v34a_artifacts.pkl', 'rb') as f:
        v34a = pickle.load(f)

    oof_preds = v34a['oof_preds']
    test_preds = v34a['test_preds']

    print(f"\n   OOF predictions (train set):")
    print(f"      Mean: {oof_preds.mean():.4f}")
    print(f"      Std: {oof_preds.std():.4f}")
    print(f"      Median: {np.median(oof_preds):.4f}")
    print(f"      Min: {oof_preds.min():.4f}, Max: {oof_preds.max():.4f}")

    print(f"\n   Test predictions:")
    print(f"      Mean: {test_preds.mean():.4f}")
    print(f"      Std: {test_preds.std():.4f}")
    print(f"      Median: {np.median(test_preds):.4f}")
    print(f"      Min: {test_preds.min():.4f}, Max: {test_preds.max():.4f}")

    pred_mean_diff_pct = 100 * (test_preds.mean() - oof_preds.mean()) / oof_preds.mean()
    print(f"\n   Mean prediction difference: {pred_mean_diff_pct:+.2f}%")

    if abs(pred_mean_diff_pct) < 5:
        print("   --> Predictions are CONSISTENT between train and test")
    elif abs(pred_mean_diff_pct) < 15:
        print("   --> Moderate prediction shift")
    else:
        print("   --> LARGE prediction shift (concern!)")

    # Check if test predictions are more confident or less confident
    if test_preds.mean() > oof_preds.mean():
        print("   --> Model predicts MORE TDEs on test set")
    else:
        print("   --> Model predicts FEWER TDEs on test set")

    # Load true labels
    train_labels = train_meta['target'].values
    tde_rate_train = train_labels.mean()

    oof_pred_rate = (oof_preds > v34a['best_threshold']).mean()
    test_pred_rate = (test_preds > v34a['best_threshold']).mean()

    print(f"\n   Prediction rates:")
    print(f"      True TDE rate (train): {100*tde_rate_train:.2f}%")
    print(f"      OOF predicted rate: {100*oof_pred_rate:.2f}%")
    print(f"      Test predicted rate: {100*test_pred_rate:.2f}%")

except Exception as e:
    print(f"   Error loading v34a: {e}")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

overall_shift = max(z_mean_diff, ebv_mean_diff)

if overall_shift < 0.1:
    print("CONCLUSION: NO distribution shift detected")
    print("Train and test are drawn from the same distribution.")
    print("\nWhy OOF-LB gaps exist:")
    print("1. Random variance (small test set split)")
    print("2. Overfitting to training patterns")
    print("3. Threshold optimization on OOF")
    print("\nRECOMMENDATION: Skip adversarial reweighting")
    print("Move to Technique #6 (Focal Loss) or #8 (Fourier Features)")

elif overall_shift < 0.3:
    print("CONCLUSION: MINIMAL distribution shift")
    print("Train and test are very similar but not identical.")
    print("\nRECOMMENDATION: Adversarial reweighting may help but gains will be small")
    print("Better to move to Technique #6 (Focal Loss) or #8 (Fourier Features)")

else:
    print("CONCLUSION: SIGNIFICANT distribution shift detected!")
    print("Train and test have different distributions.")
    print("\nRECOMMENDATION: Consider adversarial reweighting")
    print("But also try Technique #6 (Focal Loss) - more robust to distribution shift")

print("=" * 80)

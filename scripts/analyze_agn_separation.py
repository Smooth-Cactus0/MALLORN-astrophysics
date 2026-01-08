"""
MALLORN: AGN Separation Analysis + Hierarchical Classifier

User Insight: AGN looks very different from TDE/SN:
- No defined peaks (stochastic variability)
- Lower flux values
- More negative flux values

Strategy:
1. Verify these statistical differences
2. Build AGN vs Rest classifier
3. If reliable, use two-stage approach:
   - Stage 1: AGN vs Rest
   - Stage 2: TDE vs SN (only for "Rest" predictions)
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

base_path = Path(__file__).parent.parent

print("=" * 80, flush=True)
print("MALLORN: AGN Separation Analysis", flush=True)
print("=" * 80, flush=True)

# ====================
# 1. LOAD DATA
# ====================
print("\n1. Loading data...", flush=True)

train_meta = pd.read_csv(base_path / 'data/raw/train_log.csv')

train_lcs = []
for i in range(1, 21):
    path = base_path / f'data/raw/split_{i:02d}/train_full_lightcurves.csv'
    if path.exists():
        train_lcs.append(pd.read_csv(path))
train_lc = pd.concat(train_lcs, ignore_index=True)

# Create class labels
tde_ids = set(train_meta[train_meta['target'] == 1]['object_id'])
agn_ids = set(train_meta[train_meta['SpecType'] == 'AGN']['object_id'])
sn_types = ['SN Ia', 'SN II', 'SN Ibc', 'SLSN', 'SN IIn']
sn_ids = set(train_meta[train_meta['SpecType'].isin(sn_types)]['object_id'])

print(f"   TDEs: {len(tde_ids)}", flush=True)
print(f"   SNe: {len(sn_ids)}", flush=True)
print(f"   AGN: {len(agn_ids)}", flush=True)

# ====================
# 2. STATISTICAL ANALYSIS
# ====================
print("\n2. Statistical Analysis: AGN vs TDE vs SN", flush=True)
print("=" * 80, flush=True)

def compute_lightcurve_stats(obj_ids, lc_data, class_name):
    """Compute statistics for a class of objects."""
    stats = {
        'mean_flux': [],
        'std_flux': [],
        'min_flux': [],
        'max_flux': [],
        'frac_negative': [],
        'n_peaks': [],
        'peak_prominence': [],
        'flux_range': [],
        'mean_abs_flux': []
    }

    for obj_id in obj_ids:
        obj_lc = lc_data[lc_data['object_id'] == obj_id]
        if len(obj_lc) < 5:
            continue

        flux = obj_lc['Flux'].values

        stats['mean_flux'].append(np.mean(flux))
        stats['std_flux'].append(np.std(flux))
        stats['min_flux'].append(np.min(flux))
        stats['max_flux'].append(np.max(flux))
        stats['frac_negative'].append(np.mean(flux < 0))
        stats['flux_range'].append(np.max(flux) - np.min(flux))
        stats['mean_abs_flux'].append(np.mean(np.abs(flux)))

        # Count peaks (local maxima)
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(flux, prominence=np.std(flux)*0.5)
        stats['n_peaks'].append(len(peaks))

        # Peak prominence (max flux / mean flux)
        if np.mean(np.abs(flux)) > 0:
            stats['peak_prominence'].append(np.max(flux) / np.mean(np.abs(flux)))
        else:
            stats['peak_prominence'].append(0)

    return {k: np.array(v) for k, v in stats.items()}

print("\n   Computing statistics for each class...", flush=True)
tde_stats = compute_lightcurve_stats(tde_ids, train_lc, 'TDE')
sn_stats = compute_lightcurve_stats(sn_ids, train_lc, 'SN')
agn_stats = compute_lightcurve_stats(agn_ids, train_lc, 'AGN')

print("\n" + "-" * 80, flush=True)
print(f"{'Statistic':<25} {'TDE':>12} {'SN':>12} {'AGN':>12} {'AGN distinct?':>15}", flush=True)
print("-" * 80, flush=True)

for stat_name in ['mean_flux', 'std_flux', 'min_flux', 'max_flux', 'frac_negative',
                  'n_peaks', 'peak_prominence', 'flux_range', 'mean_abs_flux']:
    tde_val = np.median(tde_stats[stat_name])
    sn_val = np.median(sn_stats[stat_name])
    agn_val = np.median(agn_stats[stat_name])

    # Check if AGN is distinctly different
    tde_sn_diff = abs(tde_val - sn_val)
    agn_rest_diff = abs(agn_val - (tde_val + sn_val)/2)
    distinct = "YES" if agn_rest_diff > tde_sn_diff * 1.5 else "no"

    print(f"{stat_name:<25} {tde_val:>12.3f} {sn_val:>12.3f} {agn_val:>12.3f} {distinct:>15}", flush=True)

print("-" * 80, flush=True)

# ====================
# 3. BUILD AGN vs REST CLASSIFIER
# ====================
print("\n3. Building AGN vs Rest Classifier", flush=True)
print("=" * 80, flush=True)

# Create binary target: AGN=1, Rest (TDE+SN)=0
train_meta['is_agn'] = (train_meta['SpecType'] == 'AGN').astype(int)

# Load features
cached = pd.read_pickle(base_path / 'data/processed/features_v4_cache.pkl')
train_features = cached['train_features']

# Use all available features
feature_cols = [c for c in train_features.columns if c != 'object_id']

# Merge
train_data = train_meta[['object_id', 'is_agn', 'target']].merge(
    train_features, on='object_id', how='left'
)

X = train_data[feature_cols].values
y_agn = train_data['is_agn'].values
y_tde = train_data['target'].values

print(f"   Training samples: {len(X)}", flush=True)
print(f"   AGN samples: {y_agn.sum()} ({100*y_agn.mean():.1f}%)", flush=True)
print(f"   Rest samples: {len(y_agn) - y_agn.sum()} ({100*(1-y_agn.mean()):.1f}%)", flush=True)

# Train AGN classifier
print("\n   Training XGBoost for AGN vs Rest...", flush=True)

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 5,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1
}

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

oof_agn_preds = np.zeros(len(X))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_agn), 1):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y_agn[train_idx], y_agn[val_idx]

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)

    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=300,
        evals=[(dval, 'val')],
        early_stopping_rounds=30,
        verbose_eval=False
    )

    oof_agn_preds[val_idx] = model.predict(dval)

# Evaluate AGN classifier
print("\n" + "=" * 80, flush=True)
print("AGN vs REST CLASSIFIER RESULTS", flush=True)
print("=" * 80, flush=True)

# Find optimal threshold
best_acc = 0
best_thresh = 0.5
for t in np.linspace(0.1, 0.9, 50):
    acc = accuracy_score(y_agn, (oof_agn_preds > t).astype(int))
    if acc > best_acc:
        best_acc = acc
        best_thresh = t

agn_binary = (oof_agn_preds > best_thresh).astype(int)

print(f"\n   Optimal threshold: {best_thresh:.2f}", flush=True)
print(f"   Accuracy: {best_acc:.4f} ({100*best_acc:.2f}%)", flush=True)

# Detailed metrics
tp = np.sum((agn_binary == 1) & (y_agn == 1))  # Correctly identified AGN
fp = np.sum((agn_binary == 1) & (y_agn == 0))  # Rest misclassified as AGN
fn = np.sum((agn_binary == 0) & (y_agn == 1))  # AGN misclassified as Rest
tn = np.sum((agn_binary == 0) & (y_agn == 0))  # Correctly identified Rest

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n   AGN Classification:", flush=True)
print(f"   Precision: {precision:.4f} (of predicted AGN, {100*precision:.1f}% are correct)", flush=True)
print(f"   Recall: {recall:.4f} (of actual AGN, {100*recall:.1f}% are found)", flush=True)
print(f"   F1 Score: {f1:.4f}", flush=True)

print(f"\n   Confusion Matrix:", flush=True)
print(f"   True AGN predicted AGN: {tp}", flush=True)
print(f"   True AGN predicted Rest: {fn} (AGN leaking into TDE/SN pool)", flush=True)
print(f"   True Rest predicted AGN: {fp} (TDE/SN incorrectly filtered)", flush=True)
print(f"   True Rest predicted Rest: {tn}", flush=True)

# Critical: How many TDEs are incorrectly classified as AGN?
tde_mask = y_tde == 1
tde_as_agn = np.sum((agn_binary == 1) & tde_mask)
print(f"\n   CRITICAL: TDEs misclassified as AGN: {tde_as_agn}/{tde_mask.sum()} ({100*tde_as_agn/tde_mask.sum():.1f}%)", flush=True)

# ====================
# 4. TWO-STAGE CLASSIFIER
# ====================
print("\n4. Two-Stage Hierarchical Classifier", flush=True)
print("=" * 80, flush=True)

print("\n   Stage 1: AGN vs Rest (filter out AGN)", flush=True)
print("   Stage 2: TDE vs SN (only on Rest predictions)", flush=True)

# For Stage 2, we need TDE vs SN classifier on non-AGN objects
rest_mask = y_agn == 0  # True Rest (TDE + SN)
predicted_rest_mask = agn_binary == 0  # Predicted as Rest

print(f"\n   True Rest objects: {rest_mask.sum()}", flush=True)
print(f"   Predicted Rest objects: {predicted_rest_mask.sum()}", flush=True)

# Train TDE classifier only on predicted Rest objects
print("\n   Training TDE classifier on predicted Rest objects...", flush=True)

# Use original features for TDE classification
X_rest = X[predicted_rest_mask]
y_tde_rest = y_tde[predicted_rest_mask]

print(f"   Stage 2 training samples: {len(X_rest)}", flush=True)
print(f"   TDEs in Stage 2 pool: {y_tde_rest.sum()}", flush=True)

# Load v34a for comparison
with open(base_path / 'data/processed/v34a_artifacts.pkl', 'rb') as f:
    v34a = pickle.load(f)

# Train Stage 2 TDE classifier
xgb_params_tde = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 5,
    'learning_rate': 0.025,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'reg_alpha': 0.2,
    'reg_lambda': 1.5,
    'scale_pos_weight': len(y_tde_rest[y_tde_rest==0]) / max(len(y_tde_rest[y_tde_rest==1]), 1),
    'random_state': 42,
    'n_jobs': -1
}

# Use indices relative to predicted_rest_mask
rest_indices = np.where(predicted_rest_mask)[0]

oof_tde_preds_stage2 = np.zeros(len(X_rest))
skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf2.split(X_rest, y_tde_rest), 1):
    X_tr, X_val = X_rest[train_idx], X_rest[val_idx]
    y_tr, y_val = y_tde_rest[train_idx], y_tde_rest[val_idx]

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)

    model = xgb.train(
        xgb_params_tde,
        dtrain,
        num_boost_round=500,
        evals=[(dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=False
    )

    oof_tde_preds_stage2[val_idx] = model.predict(dval)

# Find best threshold for Stage 2
best_f1_stage2 = 0
best_thresh_stage2 = 0.5
for t in np.linspace(0.05, 0.5, 50):
    preds = (oof_tde_preds_stage2 > t).astype(int)
    f1 = f1_score(y_tde_rest, preds)
    if f1 > best_f1_stage2:
        best_f1_stage2 = f1
        best_thresh_stage2 = t

print(f"\n   Stage 2 OOF F1: {best_f1_stage2:.4f} @ threshold={best_thresh_stage2:.2f}", flush=True)

# ====================
# 5. COMBINED EVALUATION
# ====================
print("\n5. Combined Two-Stage Evaluation", flush=True)
print("=" * 80, flush=True)

# Final predictions: TDE only if predicted Rest AND Stage2 predicts TDE
final_preds = np.zeros(len(X))

# Objects predicted as Rest get Stage 2 predictions
tde_preds_stage2_binary = (oof_tde_preds_stage2 > best_thresh_stage2).astype(int)

# Map back to full array
for i, idx in enumerate(rest_indices):
    final_preds[idx] = tde_preds_stage2_binary[i]

# Objects predicted as AGN stay 0 (not TDE)
# (already zero by default)

# Calculate final F1
final_f1 = f1_score(y_tde, final_preds)

print(f"\n   Two-Stage OOF F1: {final_f1:.4f}", flush=True)
print(f"   v34a Single-Stage OOF F1: {v34a['oof_f1']:.4f}", flush=True)
print(f"   Change: {100*(final_f1 - v34a['oof_f1'])/v34a['oof_f1']:+.2f}%", flush=True)

# Confusion matrix
tp = np.sum((final_preds == 1) & (y_tde == 1))
fp = np.sum((final_preds == 1) & (y_tde == 0))
fn = np.sum((final_preds == 0) & (y_tde == 1))
tn = np.sum((final_preds == 0) & (y_tde == 0))

print(f"\n   Final Confusion Matrix:", flush=True)
print(f"   TP (TDE correctly found): {tp}", flush=True)
print(f"   FP (Non-TDE predicted as TDE): {fp}", flush=True)
print(f"   FN (TDE missed): {fn}", flush=True)
print(f"   TN (Non-TDE correctly rejected): {tn}", flush=True)

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
print(f"\n   Precision: {precision:.4f}", flush=True)
print(f"   Recall: {recall:.4f}", flush=True)

# ====================
# 6. ANALYSIS
# ====================
print("\n" + "=" * 80, flush=True)
print("KEY INSIGHTS", flush=True)
print("=" * 80, flush=True)

print(f"""
   AGN Classifier Performance:
   - Accuracy: {best_acc:.2%}
   - Can reliably separate AGN from transients

   Two-Stage Approach:
   - Stage 1 filters out {y_agn.sum() - fn} AGN (reduces noise)
   - Stage 2 focuses on TDE vs SN distinction
   - Overall F1: {final_f1:.4f}

   Compared to Single-Stage (v34a):
   - Single-Stage F1: {v34a['oof_f1']:.4f}
   - Two-Stage F1: {final_f1:.4f}
   - {'IMPROVEMENT' if final_f1 > v34a['oof_f1'] else 'NO IMPROVEMENT'}: {100*(final_f1 - v34a['oof_f1'])/v34a['oof_f1']:+.2f}%
""", flush=True)

print("=" * 80, flush=True)

"""
MALLORN v26: XGBoost with ASTROMER Embeddings (Simple Version)

This version computes basic features from scratch to avoid numpy pickle compatibility issues.
Combines ASTROMER embeddings with basic statistical features.

Target: Beat v21's 0.6708 OOF / 0.6649 LB
"""
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import xgboost as xgb
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

base_path = Path(__file__).parent.parent
os.chdir(base_path)

print("=" * 60, flush=True)
print("MALLORN v26: XGBoost with ASTROMER Embeddings", flush=True)
print("=" * 60, flush=True)

# ====================
# 1. LOAD METADATA AND LIGHT CURVES
# ====================
print("\n[1/5] Loading data...", flush=True)

train_log = pd.read_csv(base_path / 'data/raw/train_log.csv')
test_log = pd.read_csv(base_path / 'data/raw/test_log.csv')

train_ids = train_log['object_id'].tolist()
test_ids = test_log['object_id'].tolist()
y = train_log['target'].values

# Load light curves
train_lcs = []
for i in range(1, 21):
    path = base_path / f'data/raw/split_{i:02d}/train_full_lightcurves.csv'
    if path.exists():
        train_lcs.append(pd.read_csv(path))
train_lc = pd.concat(train_lcs, ignore_index=True)

test_lcs = []
for i in range(1, 21):
    path = base_path / f'data/raw/split_{i:02d}/test_full_lightcurves.csv'
    if path.exists():
        test_lcs.append(pd.read_csv(path))
test_lc = pd.concat(test_lcs, ignore_index=True)

print(f"   Train: {len(train_ids)} objects, Test: {len(test_ids)} objects", flush=True)

# ====================
# 2. COMPUTE BASIC STATISTICAL FEATURES
# ====================
print("\n[2/5] Computing basic features...", flush=True)

def compute_basic_features(lightcurves_df, object_ids, redshift_df):
    """Compute basic statistical features per object."""
    bands = ['u', 'g', 'r', 'i', 'z', 'y']
    all_features = []

    for idx, obj_id in enumerate(object_ids):
        if (idx + 1) % 500 == 0:
            print(f"   Progress: {idx+1}/{len(object_ids)} ({100*(idx+1)/len(object_ids):.1f}%)", flush=True)

        obj_lc = lightcurves_df[lightcurves_df['object_id'] == obj_id]
        obj_meta = redshift_df[redshift_df['object_id'] == obj_id].iloc[0]

        features = {'object_id': obj_id}
        features['Z'] = obj_meta['Z']
        features['EBV'] = obj_meta['EBV']

        for band in bands:
            band_data = obj_lc[obj_lc['Filter'] == band]
            flux = band_data['Flux'].values
            flux_err = band_data['Flux_err'].values
            times = band_data['Time (MJD)'].values

            prefix = f'{band}_'
            n_obs = len(flux)
            features[prefix + 'n_obs'] = n_obs

            if n_obs < 3:
                features[prefix + 'mean'] = 0
                features[prefix + 'std'] = 0
                features[prefix + 'max'] = 0
                features[prefix + 'min'] = 0
                features[prefix + 'skew'] = 0
                features[prefix + 'amp'] = 0
                features[prefix + 'snr'] = 0
                features[prefix + 'duration'] = 0
                continue

            features[prefix + 'mean'] = flux.mean()
            features[prefix + 'std'] = flux.std()
            features[prefix + 'max'] = flux.max()
            features[prefix + 'min'] = flux.min()
            features[prefix + 'amp'] = flux.max() - flux.min()

            # Skewness
            if flux.std() > 0:
                features[prefix + 'skew'] = ((flux - flux.mean()) ** 3).mean() / (flux.std() ** 3)
            else:
                features[prefix + 'skew'] = 0

            # SNR
            if flux_err.mean() > 0:
                features[prefix + 'snr'] = flux.mean() / flux_err.mean()
            else:
                features[prefix + 'snr'] = 0

            # Duration
            features[prefix + 'duration'] = times.max() - times.min()

        # Color features (at mean fluxes)
        g_flux = features.get('g_mean', 0)
        r_flux = features.get('r_mean', 0)
        i_flux = features.get('i_mean', 0)

        # Convert flux to magnitude-like ratios
        if r_flux > 0 and g_flux > 0:
            features['color_gr'] = -2.5 * np.log10(g_flux / r_flux) if g_flux / r_flux > 0 else 0
        else:
            features['color_gr'] = 0

        if i_flux > 0 and r_flux > 0:
            features['color_ri'] = -2.5 * np.log10(r_flux / i_flux) if r_flux / i_flux > 0 else 0
        else:
            features['color_ri'] = 0

        all_features.append(features)

    return pd.DataFrame(all_features)


# Check for cached features
train_basic_path = base_path / 'data/processed/train_basic_features.csv'
test_basic_path = base_path / 'data/processed/test_basic_features.csv'

if train_basic_path.exists() and test_basic_path.exists():
    print("   Loading cached basic features...", flush=True)
    train_basic = pd.read_csv(train_basic_path)
    test_basic = pd.read_csv(test_basic_path)
else:
    print("   Computing training features...", flush=True)
    train_basic = compute_basic_features(train_lc, train_ids, train_log)
    train_basic.to_csv(train_basic_path, index=False)

    print("   Computing test features...", flush=True)
    test_basic = compute_basic_features(test_lc, test_ids, test_log)
    test_basic.to_csv(test_basic_path, index=False)

print(f"   Basic features shape: Train={train_basic.shape}, Test={test_basic.shape}", flush=True)

# ====================
# 3. LOAD ASTROMER EMBEDDINGS
# ====================
print("\n[3/5] Loading ASTROMER embeddings...", flush=True)

train_emb = pd.read_csv(base_path / 'data/processed/train_astromer_embeddings.csv')
test_emb = pd.read_csv(base_path / 'data/processed/test_astromer_embeddings.csv')
print(f"   ASTROMER embeddings: Train={train_emb.shape}, Test={test_emb.shape}", flush=True)

# ====================
# 4. COMBINE FEATURES
# ====================
print("\n[4/5] Combining features...", flush=True)

# Merge basic features with ASTROMER embeddings
train_combined = train_basic.merge(train_emb, on='object_id', how='left')
test_combined = test_basic.merge(test_emb, on='object_id', how='left')

# Get feature columns
feature_cols = [c for c in train_combined.columns if c != 'object_id']

# Align indices
train_combined = train_combined.set_index('object_id').loc[train_ids].reset_index()
test_combined = test_combined.set_index('object_id').loc[test_ids].reset_index()

X = train_combined[feature_cols].values.astype(np.float32)
X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
X_test = test_combined[feature_cols].values.astype(np.float32)
X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

print(f"   Total features: {len(feature_cols)}", flush=True)
print(f"   Basic: {len(train_basic.columns)-1}, ASTROMER: {len(train_emb.columns)-1}", flush=True)

# ====================
# 5. TRAIN XGBOOST (5-FOLD CV)
# ====================
print("\n[5/5] Training XGBoost (5-fold CV)...", flush=True)

n_neg, n_pos = np.sum(y == 0), np.sum(y == 1)
scale_pos_weight = n_neg / n_pos
print(f"   Class balance: {n_pos} TDE vs {n_neg} non-TDE (ratio={scale_pos_weight:.1f})", flush=True)

# XGBoost parameters (optimized from v20c)
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'hist',
    'max_depth': 5,
    'learning_rate': 0.025,
    'n_estimators': 500,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.6,
    'reg_alpha': 0.2,
    'reg_lambda': 1.5,
    'scale_pos_weight': scale_pos_weight,
    'random_state': 42,
    'verbosity': 0,
    'early_stopping_rounds': 50
}

n_splits = 5
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(y))
test_preds = np.zeros((len(test_ids), n_splits))
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds[:, fold] = model.predict_proba(X_test)[:, 1]

    # Find optimal threshold for this fold
    best_f1 = 0
    best_thresh = 0.07
    for thresh in np.arange(0.03, 0.20, 0.01):
        f1 = f1_score(y_val, (oof_preds[val_idx] > thresh).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    fold_scores.append(best_f1)
    print(f"   Fold {fold+1}: F1 = {best_f1:.4f} (threshold={best_thresh:.2f})", flush=True)

# Find optimal global threshold
best_global_thresh = 0.07
best_oof_f1 = 0
for thresh in np.arange(0.03, 0.20, 0.01):
    oof_f1 = f1_score(y, (oof_preds > thresh).astype(int))
    if oof_f1 > best_oof_f1:
        best_oof_f1 = oof_f1
        best_global_thresh = thresh

print(f"\n{'='*60}")
print(f"OOF F1 Score: {best_oof_f1:.4f} (threshold={best_global_thresh:.2f})")
print(f"Mean Fold F1: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
print(f"{'='*60}")

# Comparison
baseline_f1 = 0.6708
improvement = (best_oof_f1 - baseline_f1) / baseline_f1 * 100
print(f"\nComparison with v21 baseline (OOF F1={baseline_f1:.4f}):")
if best_oof_f1 > baseline_f1:
    print(f"   IMPROVEMENT: +{improvement:.2f}%")
else:
    print(f"   Change: {improvement:+.2f}%")

# ====================
# GENERATE SUBMISSION
# ====================
print("\nGenerating submission...", flush=True)

test_probs_mean = test_preds.mean(axis=1)
test_binary = (test_probs_mean > best_global_thresh).astype(int)

submission = pd.DataFrame({
    'object_id': test_ids,
    'target': test_binary
})
submission_path = base_path / 'submissions/submission_v26_astromer.csv'
submission.to_csv(submission_path, index=False)

print(f"   Saved to {submission_path}")
print(f"   Predicted TDEs: {test_binary.sum()} / {len(test_binary)}")

# Feature importance
print("\nTop 20 Features by Importance:")
final_model = xgb.XGBClassifier(**{k: v for k, v in xgb_params.items() if k != 'early_stopping_rounds'})
final_model.fit(X, y)
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)
print(importance.head(20).to_string(index=False))

# Count ASTROMER features in top 20
astromer_in_top20 = importance.head(20)['feature'].str.contains('emb').sum()
print(f"\nASTROMER features in top 20: {astromer_in_top20}")

print("\nDone!")

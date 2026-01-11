"""
MALLORN v71: PLAsTiCC-Style Data Augmentation

Based on winning techniques from PLAsTiCC Kaggle competition:
1. Redshift augmentation (1st place - Kyle Boone)
   - Time dilation: t_obs = t_rest * (1+z)
   - Flux scaling: F proportional to 1/d_L^2
2. Per-band skew (3rd place)
3. Quality degradation to match test distribution

Key insight: Match TRAINING to TEST distribution, not just balance classes.

Strategy:
1. Analyze train/test distribution shift (redshift, flux levels)
2. Augment TDE samples toward test distribution
3. Extract features from augmented samples
4. Train v34a-style model on augmented data
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import xgboost as xgb
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
base_path = Path(__file__).parent.parent

print("=" * 80, flush=True)
print("MALLORN v71: PLAsTiCC-Style Augmentation", flush=True)
print("=" * 80, flush=True)

# ====================
# 1. LOAD DATA
# ====================
print("\n1. Loading data...", flush=True)

from utils.data_loader import load_all_data
from features.plasticc_augmentation import (
    augment_for_test_distribution,
    analyze_distribution_shift
)

data = load_all_data()
train_meta = data['train_meta']
test_meta = data['test_meta']
train_lc = data['train_lc']
test_lc = data['test_lc']
train_ids = train_meta['object_id'].tolist()
test_ids = test_meta['object_id'].tolist()
y = train_meta['target'].values

print(f"   Training: {len(train_ids)} objects ({np.sum(y)} TDE)", flush=True)
print(f"   Test: {len(test_ids)} objects", flush=True)

# ====================
# 2. ANALYZE DISTRIBUTION SHIFT
# ====================
print("\n2. Analyzing train/test distribution shift...", flush=True)

shift = analyze_distribution_shift(train_meta, test_meta)
print(f"   Train Z: mean={shift['train_z_mean']:.3f}, median={shift['train_z_median']:.3f}", flush=True)
print(f"   Test Z:  mean={shift['test_z_mean']:.3f}, median={shift['test_z_median']:.3f}", flush=True)
print(f"   Z shift: {shift['z_shift']:+.3f}", flush=True)

# ====================
# 3. GENERATE AUGMENTED DATA
# ====================
print("\n3. Generating PLAsTiCC-style augmented data...", flush=True)

# Augment TDEs to match test distribution
aug_lc, aug_meta = augment_for_test_distribution(
    train_lc,
    train_meta,
    test_meta,
    augmentations_per_sample=5,  # 5x augmentation
    augment_all=False,  # Only augment TDEs
    random_state=42
)

n_aug_tde = len(aug_meta)
print(f"   Generated {n_aug_tde} augmented TDE samples", flush=True)

# Combine original + augmented
combined_lc = pd.concat([train_lc, aug_lc], ignore_index=True)
combined_meta = pd.concat([train_meta, aug_meta], ignore_index=True)
combined_ids = combined_meta['object_id'].tolist()
y_combined = combined_meta['target'].values

print(f"   Combined training: {len(combined_ids)} samples", flush=True)
print(f"   TDEs: {np.sum(y_combined)} ({100*np.sum(y_combined)/len(y_combined):.1f}%)", flush=True)

# ====================
# 4. EXTRACT FEATURES
# ====================
print("\n4. Extracting features from augmented data...", flush=True)

from features.statistical import extract_statistical_features
from features.colors import extract_color_features
from features.lightcurve_shape import extract_shape_features

# Check if we have cached augmented features
aug_cache_path = base_path / 'data/processed/v71_aug_features_cache.pkl'

if aug_cache_path.exists():
    print("   Loading cached augmented features...", flush=True)
    with open(aug_cache_path, 'rb') as f:
        aug_cache = pickle.load(f)
    combined_features = aug_cache['combined_features']
else:
    print("   Extracting features (this takes a while)...", flush=True)

    # Extract features for combined dataset
    print("      Statistical features...", flush=True)
    stat_features = extract_statistical_features(combined_lc, combined_ids)

    print("      Color features...", flush=True)
    color_features = extract_color_features(combined_lc, combined_ids)

    print("      Shape features...", flush=True)
    shape_features = extract_shape_features(combined_lc, combined_ids)

    # Merge all features
    combined_features = stat_features.merge(color_features, on='object_id', how='left')
    combined_features = combined_features.merge(shape_features, on='object_id', how='left')

    # Cache
    with open(aug_cache_path, 'wb') as f:
        pickle.dump({'combined_features': combined_features}, f)

print(f"   Extracted {len(combined_features.columns)-1} features", flush=True)

# ====================
# 5. LOAD v34a SELECTED FEATURES
# ====================
print("\n5. Loading v34a feature selection...", flush=True)

# Use same feature selection as v34a for fair comparison
selection = pd.read_pickle(base_path / 'data/processed/selected_features.pkl')
importance_df = selection['importance_df']
high_corr_df = selection['high_corr_df']

corr_to_drop = set()
for _, row in high_corr_df.iterrows():
    if row['feature_1'] not in corr_to_drop:
        corr_to_drop.add(row['feature_2'])
clean_features = importance_df[~importance_df['feature'].isin(corr_to_drop)]
selected_features = clean_features.head(100)['feature'].tolist()

# Filter to available features
available_features = [f for f in selected_features if f in combined_features.columns]
print(f"   Using {len(available_features)} features", flush=True)

# Prepare training data
X_combined = combined_features[['object_id'] + available_features].copy()

# Also need test features (from original v34a cache)
cached = pd.read_pickle(base_path / 'data/processed/features_v4_cache.pkl')
test_base = cached['test_features']
X_test = test_base[['object_id'] + [f for f in available_features if f in test_base.columns]]

# Align columns
for col in available_features:
    if col not in X_combined.columns:
        X_combined[col] = np.nan
    if col not in X_test.columns:
        X_test[col] = np.nan

X_train = X_combined.drop(columns=['object_id']).values
X_test_arr = X_test.drop(columns=['object_id']).values

print(f"   Train shape: {X_train.shape}", flush=True)
print(f"   Test shape: {X_test_arr.shape}", flush=True)

# ====================
# 6. TRAIN MODEL
# ====================
print("\n6. Training XGBoost on augmented data...", flush=True)

# v34a parameters
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 5,
    'learning_rate': 0.025,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'reg_alpha': 0.2,
    'reg_lambda': 1.5,
    'scale_pos_weight': len(y_combined[y_combined==0]) / max(1, len(y_combined[y_combined==1])),
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1
}

# Use stratified K-fold but only evaluate on ORIGINAL samples
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Track which samples are original (not augmented)
is_original = ~combined_meta['object_id'].str.contains('_zaug')
original_idx = np.where(is_original)[0]

oof_preds = np.zeros(len(y_combined))
test_preds = np.zeros((len(X_test_arr), n_folds))
fold_f1s = []

print(f"\n   Training with {len(y_combined)} samples (including augmented)...", flush=True)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_combined), 1):
    print(f"\n   Fold {fold}/{n_folds}:", flush=True)

    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_combined[train_idx], y_combined[val_idx]

    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=available_features)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=available_features)
    dtest = xgb.DMatrix(X_test_arr, feature_names=available_features)

    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=False
    )

    oof_preds[val_idx] = model.predict(dval)
    test_preds[:, fold-1] = model.predict(dtest)

    # Evaluate on original samples only
    val_original = [i for i in val_idx if i in original_idx]
    if len(val_original) > 0:
        val_orig_preds = oof_preds[val_original]
        val_orig_y = y_combined[val_original]

        best_f1 = 0
        for t in np.linspace(0.03, 0.3, 50):
            f1 = f1_score(val_orig_y, (val_orig_preds > t).astype(int))
            if f1 > best_f1:
                best_f1 = f1
        fold_f1s.append(best_f1)
        print(f"      Original-only F1: {best_f1:.4f}", flush=True)

# ====================
# 7. EVALUATE ON ORIGINAL DATA ONLY
# ====================
print("\n" + "=" * 80, flush=True)
print("RESULTS (evaluated on original data only)", flush=True)
print("=" * 80, flush=True)

# Get OOF predictions for original samples only
oof_original = oof_preds[original_idx]
y_original = y_combined[original_idx]

best_f1 = 0
best_thresh = 0.1
for t in np.linspace(0.03, 0.3, 200):
    preds = (oof_original > t).astype(int)
    f1 = f1_score(y_original, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"\n   OOF F1 (original only): {best_f1:.4f} @ threshold={best_thresh:.3f}", flush=True)
print(f"   Fold F1s: {[f'{f:.4f}' for f in fold_f1s]}", flush=True)

final_preds = (oof_original > best_thresh).astype(int)
tp = np.sum((final_preds == 1) & (y_original == 1))
fp = np.sum((final_preds == 1) & (y_original == 0))
fn = np.sum((final_preds == 0) & (y_original == 1))
print(f"\n   TP={tp}, FP={fp}, FN={fn}", flush=True)
print(f"   Precision: {tp/(tp+fp):.4f}, Recall: {tp/(tp+fn):.4f}", flush=True)

# ====================
# 8. SUBMISSION
# ====================
print("\n" + "=" * 80, flush=True)
print("SUBMISSION", flush=True)
print("=" * 80, flush=True)

test_avg = test_preds.mean(axis=1)
test_binary = (test_avg > best_thresh).astype(int)

submission = pd.DataFrame({
    'object_id': test_ids,
    'target': test_binary
})

submission_path = base_path / 'submissions/submission_v71_plasticc_aug.csv'
submission.to_csv(submission_path, index=False)

print(f"   Saved: {submission_path.name}", flush=True)
print(f"   Predicted TDEs: {test_binary.sum()}", flush=True)

# Save artifacts
artifacts = {
    'oof_preds': oof_preds,
    'oof_original': oof_original,
    'test_preds': test_avg,
    'best_threshold': best_thresh,
    'oof_f1': best_f1,
    'n_augmented': n_aug_tde,
    'distribution_shift': shift
}

with open(base_path / 'data/processed/v71_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

# ====================
# 9. COMPARISON
# ====================
print("\n" + "=" * 80, flush=True)
print("COMPARISON", flush=True)
print("=" * 80, flush=True)

print(f"""
   Model                          OOF F1   LB F1
   -----                          ------   -----
   v34a (no augmentation)         0.6667   0.6907  <-- Current best
   v71 (PLAsTiCC augmentation)    {best_f1:.4f}   ???

   Augmentation details:
   - TDE samples augmented: {n_aug_tde} (5x original)
   - Redshift shift: {shift['z_shift']:+.3f}
   - Test z distribution matched

   Key insight: Matching train to test distribution should improve LB.
""", flush=True)

print("=" * 80, flush=True)
print("v71 Complete", flush=True)
print("=" * 80, flush=True)

"""
MALLORN v52: XGBoost + TDE-Only GP Data Augmentation

Strategy revision based on v50 results:
- v50 (augment all): OOF F1=0.6388, LB F1=0.6721
- Issue: Augmenting all classes dilutes discriminative features
- Solution: Only augment minority class (TDEs)

TDE-Only Augmentation:
- Filter to TDEs only (target==1): ~64 objects
- Apply GP augmentation: 64 -> 256 objects (4x)
- Combine with original full dataset: 3,043 + 192 = 3,235 objects
- Better class balance without diluting SNe/AGN signatures

Expected benefit:
- Balanced training (~8% TDEs vs 2% originally)
- Preserve discriminative features for majority classes
- More robust TDE detection

Target: Push toward LB 0.70+
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
import xgboost as xgb
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
base_path = Path(__file__).parent.parent

print("=" * 80, flush=True)
print("MALLORN v52: XGBoost + TDE-Only GP Data Augmentation", flush=True)
print("=" * 80, flush=True)

# ====================
# 1. LOAD ORIGINAL DATA
# ====================
print("\n1. Loading original data...", flush=True)

from utils.data_loader import load_all_data
data = load_all_data()

train_meta = data['train_meta']
test_meta = data['test_meta']
train_lc = data['train_lc']
test_lc = data['test_lc']

test_ids = test_meta['object_id'].tolist()

# Load v34a for reference
with open(base_path / 'data/processed/v34a_artifacts.pkl', 'rb') as f:
    v34a = pickle.load(f)

n_tdes = (train_meta['target'] == 1).sum()
print(f"   Original training: {len(train_meta)} objects", flush=True)
print(f"   TDEs: {n_tdes} ({100*n_tdes/len(train_meta):.1f}%)", flush=True)
print(f"   v34a baseline: OOF F1=0.6667, LB F1=0.6907", flush=True)
print(f"   v50 (augment all): OOF F1=0.6388, LB F1=0.6721", flush=True)

# ====================
# 2. CREATE TDE-ONLY AUGMENTED DATASET
# ====================
print("\n2. Creating TDE-only augmented training dataset...", flush=True)

from features.gp_augmentation import create_augmented_dataset

# Filter to only TDE objects for augmentation
tde_meta = train_meta[train_meta['target'] == 1].copy()
tde_ids = tde_meta['object_id'].tolist()
tde_lc = train_lc[train_lc['object_id'].isin(tde_ids)].copy()

print(f"   Filtering to TDEs only: {len(tde_meta)} objects", flush=True)

# Create 3 augmented copies per TDE (64 -> 256 objects)
aug_lc, aug_meta = create_augmented_dataset(
    tde_lc, tde_meta,
    n_augmentations_per_object=3,
    random_seed=42
)

# Combine: original_all + augmented_TDEs_only
# CRITICAL: Add original_id to train_meta before concatenating
# Otherwise, GroupKFold will group all originals together causing data leakage!
train_meta_with_orig = train_meta.copy()
train_meta_with_orig['original_id'] = train_meta_with_orig['object_id']  # Originals point to themselves
train_meta_with_orig['augmentation_type'] = 'original'

combined_meta = pd.concat([train_meta_with_orig, aug_meta], ignore_index=True)
combined_lc = pd.concat([train_lc, aug_lc], ignore_index=True)

n_tdes_combined = (combined_meta['target'] == 1).sum()
print(f"   TDE augmentation: {len(tde_meta)} -> {len(aug_meta)} objects", flush=True)
print(f"   Combined training: {len(combined_meta)} objects", flush=True)
print(f"   TDEs after augmentation: {n_tdes_combined} ({100*n_tdes_combined/len(combined_meta):.1f}%)", flush=True)

# ====================
# 3. EXTRACT FEATURES FROM COMBINED DATA
# ====================
print("\n3. Extracting features from combined dataset...", flush=True)

# Load feature extraction modules
cached = pd.read_pickle(base_path / 'data/processed/features_v4_cache.pkl')
train_base = cached['train_features']
test_base = cached['test_features']

selection = pd.read_pickle(base_path / 'data/processed/selected_features.pkl')
importance_df = selection['importance_df']
high_corr_df = selection['high_corr_df']

corr_to_drop = set()
for _, row in high_corr_df.iterrows():
    if row['feature_1'] not in corr_to_drop:
        corr_to_drop.add(row['feature_2'])

clean_features = importance_df[~importance_df['feature'].isin(corr_to_drop)]
selected_120 = clean_features.head(120)['feature'].tolist()

print("   Extracting base features for combined objects...", flush=True)

# Build IDs and targets for combined dataset
combined_train_ids = combined_meta['object_id'].tolist()
combined_targets = combined_meta['target'].values

# Extract features: use existing for originals, recompute for augmented
print("   Processing Bazin features for combined data...", flush=True)

from features.bazin_fitting import extract_bazin_features

# Extract Bazin for augmented lightcurves only (originals already cached)
augmented_ids = aug_meta['object_id'].tolist()
train_bazin_aug = extract_bazin_features(aug_lc, augmented_ids)
print(f"      Extracted {len(train_bazin_aug.columns)-1} Bazin features for augmented TDEs", flush=True)

# Load TDE and GP features for original objects
tde_cached = pd.read_pickle(base_path / 'data/processed/tde_physics_cache.pkl')
train_tde = tde_cached['train']
test_tde = tde_cached['test']

with open(base_path / 'data/processed/multiband_gp_cache.pkl', 'rb') as f:
    gp2d_data = pickle.load(f)
train_gp2d = gp2d_data['train']
test_gp2d = gp2d_data['test']
gp2d_cols = [c for c in train_gp2d.columns if c != 'object_id']

# For augmented objects, use features from original object
print("   Mapping features from original to augmented TDEs...", flush=True)

# Create mapping of augmented -> original
aug_to_orig = dict(zip(aug_meta['object_id'], aug_meta['original_id']))

# Deduplicate IDs
print(f"   Original combined_train_ids: {len(combined_train_ids)}, Unique: {len(set(combined_train_ids))}", flush=True)

# Create mapping before deduplication
id_to_target = dict(zip(combined_meta['object_id'], combined_meta['target']))

# Deduplicate IDs
combined_train_ids = list(dict.fromkeys(combined_train_ids))  # Remove duplicates while preserving order
print(f"   After deduplication: {len(combined_train_ids)}", flush=True)

# Filter targets to match deduplicated IDs
combined_targets = np.array([id_to_target[comb_id] for comb_id in combined_train_ids])
print(f"   Filtered combined_targets: {len(combined_targets)} labels", flush=True)

# Build feature DataFrame for combined data
train_base_full = train_base[['object_id'] + selected_120].copy()

# For augmented objects, map to original features
def map_to_original_features(comb_id):
    if comb_id in train_base_full['object_id'].values:
        return comb_id  # Original object
    else:
        return aug_to_orig.get(comb_id, comb_id)  # Augmented -> original

combined_features_list = []
for comb_id in combined_train_ids:
    orig_id = map_to_original_features(comb_id)

    # Get original features
    orig_features = train_base_full[train_base_full['object_id'] == orig_id]

    if len(orig_features) > 0:
        new_row = orig_features.iloc[0].copy()
        new_row['object_id'] = comb_id
        combined_features_list.append(new_row)
    else:
        # Fallback: NaN features
        new_row = pd.Series([comb_id] + [np.nan] * len(selected_120),
                           index=['object_id'] + selected_120)
        combined_features_list.append(new_row)

train_base_combined = pd.DataFrame(combined_features_list)

# Merge TDE and GP features (mapped from originals)
train_tde_mapped = []
for comb_id in combined_train_ids:
    orig_id = map_to_original_features(comb_id)
    orig_tde = train_tde[train_tde['object_id'] == orig_id]
    if len(orig_tde) > 0:
        new_row = orig_tde.iloc[0].copy()
        new_row['object_id'] = comb_id
        train_tde_mapped.append(new_row)

train_tde_combined = pd.DataFrame(train_tde_mapped)

train_gp2d_mapped = []
for comb_id in combined_train_ids:
    orig_id = map_to_original_features(comb_id)
    orig_gp = train_gp2d[train_gp2d['object_id'] == orig_id]
    if len(orig_gp) > 0:
        new_row = orig_gp.iloc[0].copy()
        new_row['object_id'] = comb_id
        train_gp2d_mapped.append(new_row)

train_gp2d_combined = pd.DataFrame(train_gp2d_mapped)

# For Bazin features: use cached for originals, augmented for new TDEs
# Load original Bazin features
train_bazin_orig = pd.read_pickle(base_path / 'data/processed/bazin_features_cache.pkl')['train']

# Combine original + augmented Bazin
train_bazin_combined = pd.concat([train_bazin_orig, train_bazin_aug], ignore_index=True)

# Combine all features
print("   Combining features...", flush=True)

# Ensure all dataframes have same object_ids and no duplicates
print(f"   train_base_combined: {len(train_base_combined)} rows", flush=True)
print(f"   train_tde_combined: {len(train_tde_combined)} rows", flush=True)
print(f"   train_gp2d_combined: {len(train_gp2d_combined)} rows", flush=True)
print(f"   train_bazin_combined: {len(train_bazin_combined)} rows", flush=True)

# Filter train_bazin_combined to only keep deduplicated IDs
train_bazin_combined = train_bazin_combined[train_bazin_combined['object_id'].isin(combined_train_ids)]
train_bazin_combined = train_bazin_combined.drop_duplicates(subset=['object_id'], keep='first')
print(f"   train_bazin_combined after filtering: {len(train_bazin_combined)} rows", flush=True)

train_combined_final = train_base_combined.merge(train_tde_combined, on='object_id', how='left')
print(f"   After TDE merge: {len(train_combined_final)} rows", flush=True)

train_combined_final = train_combined_final.merge(train_gp2d_combined[['object_id'] + gp2d_cols], on='object_id', how='left')
print(f"   After GP merge: {len(train_combined_final)} rows", flush=True)

train_combined_final = train_combined_final.merge(train_bazin_combined, on='object_id', how='left')
print(f"   After Bazin merge: {len(train_combined_final)} rows", flush=True)

# Remove any duplicate rows
train_combined_final = train_combined_final.drop_duplicates(subset=['object_id'], keep='first')
print(f"   After deduplication: {len(train_combined_final)} rows", flush=True)

# Build test features (same as v34a)
test_v21 = test_base[['object_id'] + selected_120].copy()
test_v21 = test_v21.merge(test_tde, on='object_id', how='left')
test_v21 = test_v21.merge(test_gp2d[['object_id'] + gp2d_cols], on='object_id', how='left')

test_bazin = extract_bazin_features(test_lc, test_ids)
test_combined = test_v21.merge(test_bazin, on='object_id', how='left')

X_train = train_combined_final.drop(columns=['object_id']).values
X_test = test_combined.drop(columns=['object_id']).values
feature_names = [c for c in train_combined_final.columns if c != 'object_id']

print(f"   Training shape: {X_train.shape}", flush=True)
print(f"   Features: {len(feature_names)}", flush=True)

# ====================
# 4. TRAIN XGBOOST ON TDE-AUGMENTED DATA
# ====================
print("\n4. Training XGBoost with 5-fold CV on TDE-augmented data...", flush=True)

y_combined = combined_targets

# Same hyperparameters as v34a
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
    'scale_pos_weight': len(y_combined[y_combined==0]) / len(y_combined[y_combined==1]),
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1
}

n_folds = 5

# Use GroupKFold to prevent data leakage from augmented TDE copies
# Group by original_id so all augmented versions stay in same fold
unique_originals = list(set(combined_meta['original_id']))
original_to_group = {orig: i for i, orig in enumerate(unique_originals)}
groups = np.array([original_to_group.get(combined_meta[combined_meta['object_id']==comb_id]['original_id'].iloc[0], 0)
                   for comb_id in train_combined_final['object_id']])

print(f"   GroupKFold: {len(unique_originals)} unique objects (groups)", flush=True)
print(f"   Each TDE group has ~{len([m for m in combined_meta['original_id'] if m in tde_ids])/n_tdes:.1f} samples (original + augmented)", flush=True)

gkf = GroupKFold(n_splits=n_folds)

oof_preds = np.zeros(len(X_train))
test_preds = np.zeros((len(X_test), n_folds))
feature_importance = np.zeros(len(feature_names))

for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_combined, groups), 1):
    print(f"\n   Fold {fold}/{n_folds}:", flush=True)

    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_combined[train_idx], y_combined[val_idx]

    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test, feature_names=feature_names)

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

    importance = model.get_score(importance_type='gain')
    for feat, gain in importance.items():
        if feat in feature_names:
            idx = feature_names.index(feat)
            feature_importance[idx] += gain

    best_f1 = 0
    best_thresh = 0.5
    for t in np.linspace(0.05, 0.5, 50):
        preds_binary = (oof_preds[val_idx] > t).astype(int)
        f1 = f1_score(y_val, preds_binary)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    print(f"      Best threshold: {best_thresh:.3f}, F1: {best_f1:.4f}", flush=True)

print("\n" + "=" * 80, flush=True)
print("CROSS-VALIDATION RESULTS (TDE-Augmented Data)", flush=True)
print("=" * 80, flush=True)

best_f1 = 0
best_thresh = 0.5
for t in np.linspace(0.05, 0.5, 100):
    preds_binary = (oof_preds > t).astype(int)
    f1 = f1_score(y_combined, preds_binary)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"   OOF F1: {best_f1:.4f} @ threshold={best_thresh:.2f}", flush=True)

final_preds = (oof_preds > best_thresh).astype(int)
tp = np.sum((final_preds == 1) & (y_combined == 1))
fp = np.sum((final_preds == 1) & (y_combined == 0))
fn = np.sum((final_preds == 0) & (y_combined == 1))
tn = np.sum((final_preds == 0) & (y_combined == 0))

print(f"   Confusion: TP={tp}, FP={fp}, FN={fn}, TN={tn}", flush=True)
print(f"   Precision: {tp/(tp+fp):.4f}", flush=True)
print(f"   Recall: {tp/(tp+fn):.4f}", flush=True)

# ====================
# 5. CREATE SUBMISSION
# ====================
print("\n5. Creating submission...", flush=True)

test_avg = test_preds.mean(axis=1)
test_final = (test_avg > best_thresh).astype(int)

submission = pd.DataFrame({
    'object_id': test_ids,
    'target': test_final
})

submission_path = base_path / 'submissions/submission_v52_xgb_tde_aug.csv'
submission.to_csv(submission_path, index=False)

print(f"   Submission saved: {submission_path.name}", flush=True)
print(f"   Predicted TDEs: {test_final.sum()} / {len(test_final)}", flush=True)

# Save artifacts
artifacts = {
    'oof_preds': oof_preds,
    'test_preds': test_avg,
    'best_threshold': best_thresh,
    'oof_f1': best_f1,
    'n_training_samples': len(X_train),
    'n_tdes_original': n_tdes,
    'n_tdes_augmented': n_tdes_combined
}

with open(base_path / 'data/processed/v52_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print("\n" + "=" * 80, flush=True)
print(f"MALLORN v52 (XGBoost + TDE-Only Augmentation) Complete: OOF F1 = {best_f1:.4f}", flush=True)
print(f"v34a (no augmentation): OOF F1 = {v34a['oof_f1']:.4f}, LB F1 = 0.6907", flush=True)
print(f"v50 (augment all): OOF F1 = 0.6388, LB F1 = 0.6721", flush=True)
print(f"Training samples: {len(train_meta)} -> {len(combined_meta)} (TDEs: {n_tdes} -> {n_tdes_combined})", flush=True)

print("\nKey Insight: TDE-only data augmentation", flush=True)
print("  - Only augment minority class (TDEs)", flush=True)
print(f"  - Better class balance: {100*n_tdes/len(train_meta):.1f}% -> {100*n_tdes_combined/len(combined_meta):.1f}%", flush=True)
print("  - Preserve discriminative features for SNe/AGN", flush=True)
print("  - More robust TDE detection without diluting signal", flush=True)

print("\nNEXT STEP: Test v52 on leaderboard", flush=True)
print("Then proceed to v53 (TDE-Only Augmentation for LightGBM)", flush=True)
print("=" * 80, flush=True)

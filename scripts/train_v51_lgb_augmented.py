"""
MALLORN v51: LightGBM + GP-Based Data Augmentation

Same augmentation strategy as v50, but with LightGBM.

v46 (LightGBM baseline): OOF 0.6120, LB 0.60
v50 (XGBoost + augmentation): OOF pending

Features:
- v34a baseline (224 Bazin features)
- GP augmentation: 3,043 → 12,172 samples (4x expansion)

LightGBM advantages:
- Leaf-wise growth (more aggressive)
- Different splitting = good ensemble with XGBoost
- May benefit more from augmentation than XGBoost

Expected: Improved over v46, ensemble potential with v50
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
base_path = Path(__file__).parent.parent

print("=" * 80, flush=True)
print("MALLORN v51: LightGBM + GP Data Augmentation", flush=True)
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

# Load references
with open(base_path / 'data/processed/v34a_artifacts.pkl', 'rb') as f:
    v34a = pickle.load(f)

with open(base_path / 'data/processed/v46_artifacts.pkl', 'rb') as f:
    v46 = pickle.load(f)

print(f"   Original training: {len(train_meta)} objects", flush=True)
print(f"   v46 (LightGBM baseline): OOF F1={v46['oof_f1']:.4f}, LB F1=0.60", flush=True)

# ====================
# 2. CREATE AUGMENTED DATASET
# ====================
print("\n2. Creating augmented training dataset...", flush=True)

from features.gp_augmentation import create_augmented_dataset

aug_lc, aug_meta = create_augmented_dataset(
    train_lc, train_meta,
    n_augmentations_per_object=3,
    random_seed=42
)

print(f"   Augmented training: {len(aug_meta)} objects", flush=True)

# ====================
# 3. EXTRACT FEATURES
# ====================
print("\n3. Extracting features from augmented dataset...", flush=True)

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

print("   Processing Bazin features for augmented data...", flush=True)

from features.bazin_fitting import extract_bazin_features

aug_train_ids = aug_meta['object_id'].tolist()
aug_targets = aug_meta['target'].values

train_bazin_aug = extract_bazin_features(aug_lc, aug_train_ids)

# Load other features
tde_cached = pd.read_pickle(base_path / 'data/processed/tde_physics_cache.pkl')
train_tde = tde_cached['train']
test_tde = tde_cached['test']

with open(base_path / 'data/processed/multiband_gp_cache.pkl', 'rb') as f:
    gp2d_data = pickle.load(f)
train_gp2d = gp2d_data['train']
test_gp2d = gp2d_data['test']
gp2d_cols = [c for c in train_gp2d.columns if c != 'object_id']

# Map features from original to augmented
print("   Mapping features from original to augmented objects...", flush=True)

aug_to_orig = dict(zip(aug_meta['object_id'], aug_meta['original_id']))

train_base_full = train_base[['object_id'] + selected_120].copy()

def map_to_original_features(aug_id):
    if aug_id in train_base_full['object_id'].values:
        return aug_id
    else:
        return aug_to_orig.get(aug_id, aug_id)

# Ensure unique aug_train_ids first and filter targets accordingly
print(f"   Original aug_train_ids: {len(aug_train_ids)}, Unique: {len(set(aug_train_ids))}", flush=True)

# Create mapping before deduplication
id_to_target = dict(zip(aug_meta['object_id'], aug_meta['target']))

# Deduplicate IDs
aug_train_ids = list(dict.fromkeys(aug_train_ids))  # Remove duplicates while preserving order
print(f"   After deduplication: {len(aug_train_ids)}", flush=True)

# Filter targets to match deduplicated IDs
aug_targets = np.array([id_to_target[aug_id] for aug_id in aug_train_ids])
print(f"   Filtered aug_targets: {len(aug_targets)} labels", flush=True)

# Build feature DataFrames
aug_features_list = []
for aug_id in aug_train_ids:
    orig_id = map_to_original_features(aug_id)
    orig_features = train_base_full[train_base_full['object_id'] == orig_id]
    if len(orig_features) > 0:
        new_row = orig_features.iloc[0].copy()
        new_row['object_id'] = aug_id
        aug_features_list.append(new_row)

train_base_aug = pd.DataFrame(aug_features_list)

train_tde_mapped = []
for aug_id in aug_train_ids:
    orig_id = map_to_original_features(aug_id)
    orig_tde = train_tde[train_tde['object_id'] == orig_id]
    if len(orig_tde) > 0:
        new_row = orig_tde.iloc[0].copy()
        new_row['object_id'] = aug_id
        train_tde_mapped.append(new_row)

train_tde_aug = pd.DataFrame(train_tde_mapped)

train_gp2d_mapped = []
for aug_id in aug_train_ids:
    orig_id = map_to_original_features(aug_id)
    orig_gp = train_gp2d[train_gp2d['object_id'] == orig_id]
    if len(orig_gp) > 0:
        new_row = orig_gp.iloc[0].copy()
        new_row['object_id'] = aug_id
        train_gp2d_mapped.append(new_row)

train_gp2d_aug = pd.DataFrame(train_gp2d_mapped)

print("   Combining features...", flush=True)

# Ensure all dataframes have same object_ids and no duplicates
print(f"   train_base_aug: {len(train_base_aug)} rows", flush=True)
print(f"   train_tde_aug: {len(train_tde_aug)} rows", flush=True)
print(f"   train_gp2d_aug: {len(train_gp2d_aug)} rows", flush=True)
print(f"   train_bazin_aug: {len(train_bazin_aug)} rows", flush=True)

# Verify object_id uniqueness and filter Bazin to match deduplicated IDs
assert train_base_aug['object_id'].nunique() == len(train_base_aug), "Duplicate IDs in train_base_aug"

# Filter train_bazin_aug to only keep deduplicated IDs
train_bazin_aug = train_bazin_aug[train_bazin_aug['object_id'].isin(aug_train_ids)]
train_bazin_aug = train_bazin_aug.drop_duplicates(subset=['object_id'], keep='first')
print(f"   train_bazin_aug after filtering: {len(train_bazin_aug)} rows", flush=True)

assert train_bazin_aug['object_id'].nunique() == len(train_bazin_aug), "Duplicate IDs in train_bazin_aug"

train_combined_aug = train_base_aug.merge(train_tde_aug, on='object_id', how='left')
print(f"   After TDE merge: {len(train_combined_aug)} rows", flush=True)

train_combined_aug = train_combined_aug.merge(train_gp2d_aug[['object_id'] + gp2d_cols], on='object_id', how='left')
print(f"   After GP merge: {len(train_combined_aug)} rows", flush=True)

train_combined_aug = train_combined_aug.merge(train_bazin_aug, on='object_id', how='left')
print(f"   After Bazin merge: {len(train_combined_aug)} rows", flush=True)

# Remove any duplicate rows
train_combined_aug = train_combined_aug.drop_duplicates(subset=['object_id'], keep='first')
print(f"   After deduplication: {len(train_combined_aug)} rows", flush=True)

test_v21 = test_base[['object_id'] + selected_120].copy()
test_v21 = test_v21.merge(test_tde, on='object_id', how='left')
test_v21 = test_v21.merge(test_gp2d[['object_id'] + gp2d_cols], on='object_id', how='left')

test_bazin = extract_bazin_features(test_lc, test_ids)
test_combined = test_v21.merge(test_bazin, on='object_id', how='left')

X_train = train_combined_aug.drop(columns=['object_id']).values
X_test = test_combined.drop(columns=['object_id']).values
feature_names = [c for c in train_combined_aug.columns if c != 'object_id']

print(f"   Training shape: {X_train.shape}", flush=True)

# ====================
# 4. TRAIN LIGHTGBM ON AUGMENTED DATA
# ====================
print("\n4. Training LightGBM with 5-fold CV on augmented data...", flush=True)

y_aug = aug_targets

lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.025,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'min_child_samples': 3,
    'lambda_l1': 0.2,
    'lambda_l2': 1.5,
    'scale_pos_weight': len(y_aug[y_aug==0]) / len(y_aug[y_aug==1]),
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

n_folds = 5

# Use GroupKFold to prevent data leakage from augmented copies
# Group by original_id so all augmented versions stay in same fold
unique_originals = list(set(aug_meta['original_id']))
original_to_group = {orig: i for i, orig in enumerate(unique_originals)}
groups = np.array([original_to_group.get(aug_meta[aug_meta['object_id']==aug_id]['original_id'].iloc[0], 0)
                   for aug_id in train_combined_aug['object_id']])

print(f"   GroupKFold: {len(unique_originals)} unique objects (groups)", flush=True)
print(f"   Each group has ~{len(train_combined_aug)/len(unique_originals):.1f} samples (original + augmented)", flush=True)

gkf = GroupKFold(n_splits=n_folds)

oof_preds = np.zeros(len(X_train))
test_preds = np.zeros((len(X_test), n_folds))

for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_aug, groups), 1):
    print(f"\n   Fold {fold}/{n_folds}:", flush=True)

    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_aug[train_idx], y_aug[val_idx]

    train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_names)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_data)

    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=0)
        ]
    )

    oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    test_preds[:, fold-1] = model.predict(X_test, num_iteration=model.best_iteration)

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
print("CROSS-VALIDATION RESULTS (Augmented Data)", flush=True)
print("=" * 80, flush=True)

best_f1 = 0
best_thresh = 0.5
for t in np.linspace(0.05, 0.5, 100):
    preds_binary = (oof_preds > t).astype(int)
    f1 = f1_score(y_aug, preds_binary)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"   OOF F1: {best_f1:.4f} @ threshold={best_thresh:.2f}", flush=True)

final_preds = (oof_preds > best_thresh).astype(int)
tp = np.sum((final_preds == 1) & (y_aug == 1))
fp = np.sum((final_preds == 1) & (y_aug == 0))
fn = np.sum((final_preds == 0) & (y_aug == 1))
tn = np.sum((final_preds == 0) & (y_aug == 0))

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

submission_path = base_path / 'submissions/submission_v51_lgb_augmented.csv'
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
    'model_type': 'LightGBM'
}

with open(base_path / 'data/processed/v51_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print("\n" + "=" * 80, flush=True)
print(f"MALLORN v51 (LightGBM + GP Augmentation) Complete: OOF F1 = {best_f1:.4f}", flush=True)
print(f"v46 (LightGBM baseline): OOF F1 = {v46['oof_f1']:.4f}, LB F1 = 0.60", flush=True)
print(f"v34a (XGBoost): OOF F1 = {v34a['oof_f1']:.4f}, LB F1 = 0.6907", flush=True)

change_vs_v46 = (best_f1 - v46['oof_f1']) * 100 / v46['oof_f1']
print(f"Change vs v46 (OOF): {change_vs_v46:+.2f}% ({best_f1 - v46['oof_f1']:+.4f})", flush=True)

print("\nKey Insight: LightGBM + GP Augmentation", flush=True)
print("  - Leaf-wise growth may benefit more from augmentation", flush=True)
print("  - 4x training data: more robust model", flush=True)
print("  - Good ensemble potential with v50 (XGBoost)", flush=True)

print("\nNEXT STEP: Test v51 on leaderboard", flush=True)
print("Then analyze cumulative improvement from all techniques", flush=True)
print("=" * 80, flush=True)

"""
MALLORN v28b: Pseudo-labeling to Augment Training Data

Strategy:
1. Train initial model on labeled data
2. Predict on test set
3. Add high-confidence predictions as pseudo-labels
4. Retrain on augmented dataset

Target: Increase effective training size for rare TDE class
"""
import sys
import os
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

base_path = Path(__file__).parent.parent
os.chdir(base_path)

print("=" * 60)
print("MALLORN v28b: Pseudo-labeling")
print("=" * 60)

# ====================
# 1. LOAD DATA
# ====================
print("\n[1/4] Loading data...", flush=True)

train_log = pd.read_csv(base_path / 'data/raw/train_log.csv')
test_log = pd.read_csv(base_path / 'data/raw/test_log.csv')

train_ids = train_log['object_id'].tolist()
test_ids = test_log['object_id'].tolist()
y_train = train_log['target'].values

# Load features
cached = pd.read_pickle(base_path / 'data/processed/features_v4_cache.pkl')
train_features = cached['train_features']
test_features = cached['test_features']

selection = pd.read_pickle(base_path / 'data/processed/selected_features.pkl')
importance_df = selection['importance_df']
high_corr_df = selection['high_corr_df']

corr_to_drop = set()
for _, row in high_corr_df.iterrows():
    if row['feature_1'] not in corr_to_drop:
        corr_to_drop.add(row['feature_2'])
clean_features = importance_df[~importance_df['feature'].isin(corr_to_drop)]
selected_120 = clean_features.head(120)['feature'].tolist()

# GP2D features
with open(base_path / 'data/processed/multiband_gp_cache.pkl', 'rb') as f:
    gp2d_data = pickle.load(f)
train_gp2d = gp2d_data['train']
test_gp2d = gp2d_data['test']
gp2d_cols = [c for c in train_gp2d.columns if c != 'object_id']

# Merge
train_combined = train_features.merge(train_gp2d, on='object_id', how='left')
test_combined = test_features.merge(test_gp2d, on='object_id', how='left')

base_cols = [c for c in selected_120 if c in train_combined.columns]
all_feature_cols = base_cols + gp2d_cols
all_feature_cols = list(dict.fromkeys(all_feature_cols))

train_combined = train_combined.set_index('object_id').loc[train_ids].reset_index()
test_combined = test_combined.set_index('object_id').loc[test_ids].reset_index()

X_train = train_combined[all_feature_cols].values.astype(np.float32)
X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
X_test = test_combined[all_feature_cols].values.astype(np.float32)
X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

print(f"   Train: {len(y_train)}, Test: {len(test_ids)}")
print(f"   Features: {len(all_feature_cols)}")

# ====================
# 2. INITIAL MODEL FOR PSEUDO-LABELS
# ====================
print("\n[2/4] Training initial model...", flush=True)

n_neg, n_pos = np.sum(y_train == 0), np.sum(y_train == 1)
scale_pos_weight = n_neg / n_pos

# Load Optuna params
with open(base_path / 'data/processed/optuna_v20c_results.pkl', 'rb') as f:
    optuna_data = pickle.load(f)
xgb_params = optuna_data['xgb_best_params']

# Train on all labeled data to get pseudo-labels
initial_model = xgb.XGBClassifier(
    **xgb_params,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    verbosity=0
)
initial_model.fit(X_train, y_train)

# Predict on test set
test_probs = initial_model.predict_proba(X_test)[:, 1]

# Select high-confidence predictions
HIGH_CONF_POS = 0.85  # Confident TDE
HIGH_CONF_NEG = 0.02  # Confident non-TDE

pseudo_pos_mask = test_probs > HIGH_CONF_POS
pseudo_neg_mask = test_probs < HIGH_CONF_NEG

n_pseudo_pos = pseudo_pos_mask.sum()
n_pseudo_neg = pseudo_neg_mask.sum()

print(f"   High-conf TDE (>{HIGH_CONF_POS}): {n_pseudo_pos}")
print(f"   High-conf non-TDE (<{HIGH_CONF_NEG}): {n_pseudo_neg}")

# Create pseudo-labeled data (limit non-TDE to maintain balance)
max_neg_pseudo = min(n_pseudo_neg, n_pseudo_pos * 5)  # Max 5:1 ratio

pseudo_pos_idx = np.where(pseudo_pos_mask)[0]
pseudo_neg_idx = np.where(pseudo_neg_mask)[0]
if len(pseudo_neg_idx) > max_neg_pseudo:
    pseudo_neg_idx = np.random.choice(pseudo_neg_idx, max_neg_pseudo, replace=False)

pseudo_idx = np.concatenate([pseudo_pos_idx, pseudo_neg_idx])
X_pseudo = X_test[pseudo_idx]
y_pseudo = np.concatenate([
    np.ones(len(pseudo_pos_idx)),
    np.zeros(len(pseudo_neg_idx))
]).astype(int)

print(f"   Pseudo-labeled samples: {len(y_pseudo)} ({y_pseudo.sum()} TDE, {len(y_pseudo)-y_pseudo.sum()} non-TDE)")

# ====================
# 3. RETRAIN WITH PSEUDO-LABELS
# ====================
print("\n[3/4] Retraining with pseudo-labels...", flush=True)

# Combine original + pseudo data
X_augmented = np.vstack([X_train, X_pseudo])
y_augmented = np.concatenate([y_train, y_pseudo])

# Shuffle
shuffle_idx = np.random.permutation(len(y_augmented))
X_augmented = X_augmented[shuffle_idx]
y_augmented = y_augmented[shuffle_idx]

print(f"   Augmented dataset: {len(y_augmented)} ({y_augmented.sum()} TDE)")

# Cross-validation on ORIGINAL labels only
n_splits = 5
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(y_train))
test_preds_final = np.zeros((len(test_ids), n_splits))
fold_scores = []

# Update class weights for augmented data
n_neg_aug, n_pos_aug = np.sum(y_augmented == 0), np.sum(y_augmented == 1)
scale_pos_weight_aug = n_neg_aug / n_pos_aug

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
    # Training: use augmented data BUT validation on original only
    # Find which augmented samples to use (original train samples + all pseudo)
    orig_train_mask = np.zeros(len(y_augmented), dtype=bool)
    orig_train_mask[:len(y_train)] = True
    orig_train_mask[train_idx] = True  # Original train fold
    orig_train_mask[len(y_train):] = True  # All pseudo samples

    X_tr = X_augmented[orig_train_mask[:len(X_augmented)]]
    y_tr = y_augmented[orig_train_mask[:len(y_augmented)]]

    # Validation on original data only
    X_val = X_train[val_idx]
    y_val = y_train[val_idx]

    model = xgb.XGBClassifier(
        **xgb_params,
        scale_pos_weight=scale_pos_weight_aug,
        random_state=42,
        verbosity=0
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds_final[:, fold] = model.predict_proba(X_test)[:, 1]

    # Fold score
    best_f1 = 0
    for thresh in np.arange(0.03, 0.20, 0.01):
        f1 = f1_score(y_val, (oof_preds[val_idx] > thresh).astype(int))
        if f1 > best_f1:
            best_f1 = f1

    fold_scores.append(best_f1)
    print(f"   Fold {fold+1}: F1 = {best_f1:.4f}")

# Global threshold
best_oof_f1 = 0
best_thresh = 0.07
for thresh in np.arange(0.03, 0.20, 0.01):
    f1 = f1_score(y_train, (oof_preds > thresh).astype(int))
    if f1 > best_oof_f1:
        best_oof_f1 = f1
        best_thresh = thresh

print(f"\n{'='*60}")
print(f"OOF F1: {best_oof_f1:.4f} (threshold={best_thresh:.2f})")
print(f"Mean Fold F1: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
print(f"{'='*60}")

baseline = 0.6708
print(f"\nComparison with v21 ({baseline:.4f}): {(best_oof_f1-baseline)/baseline*100:+.2f}%")

# ====================
# 4. SUBMISSION
# ====================
print("\n[4/4] Generating submission...", flush=True)

test_probs_final = test_preds_final.mean(axis=1)
test_binary = (test_probs_final > best_thresh).astype(int)

submission = pd.DataFrame({'object_id': test_ids, 'target': test_binary})
submission.to_csv(base_path / 'submissions/submission_v28b_pseudolabel.csv', index=False)

print(f"   Predicted TDEs: {test_binary.sum()} / {len(test_binary)}")

# Save cache
with open(base_path / 'data/processed/v28b_cache.pkl', 'wb') as f:
    pickle.dump({'oof_preds': oof_preds, 'test_preds': test_preds_final, 'threshold': best_thresh}, f)

print("Done!")

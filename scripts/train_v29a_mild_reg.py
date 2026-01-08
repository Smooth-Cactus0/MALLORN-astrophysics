"""
MALLORN v29a: Mild Regularization (100 features)

Between v21 (147 features) and v28a (64 features)
Slight regularization increase without being too aggressive.
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
print("MALLORN v29a: Mild Regularization (100 features)")
print("=" * 60)

# Load data
train_log = pd.read_csv(base_path / 'data/raw/train_log.csv')
test_log = pd.read_csv(base_path / 'data/raw/test_log.csv')

train_ids = train_log['object_id'].tolist()
test_ids = test_log['object_id'].tolist()
y = train_log['target'].values

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

# KEY: Use 100 features (between 147 and 64)
n_features = 100
selected_features = clean_features.head(n_features)['feature'].tolist()

# GP2D features (all of them - they're important)
with open(base_path / 'data/processed/multiband_gp_cache.pkl', 'rb') as f:
    gp2d_data = pickle.load(f)
train_gp2d = gp2d_data['train']
test_gp2d = gp2d_data['test']
gp2d_cols = [c for c in train_gp2d.columns if c != 'object_id']

# Merge
train_combined = train_features.merge(train_gp2d, on='object_id', how='left')
test_combined = test_features.merge(test_gp2d, on='object_id', how='left')

base_cols = [c for c in selected_features if c in train_combined.columns]
all_feature_cols = base_cols + gp2d_cols
all_feature_cols = list(dict.fromkeys(all_feature_cols))

train_combined = train_combined.set_index('object_id').loc[train_ids].reset_index()
test_combined = test_combined.set_index('object_id').loc[test_ids].reset_index()

X = train_combined[all_feature_cols].values.astype(np.float32)
X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
X_test = test_combined[all_feature_cols].values.astype(np.float32)
X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

print(f"Features: {len(all_feature_cols)}")

# Mild regularization params (between v21 and v28a)
n_neg, n_pos = np.sum(y == 0), np.sum(y == 1)
scale_pos_weight = n_neg / n_pos

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'hist',
    'max_depth': 5,           # Same as v21
    'learning_rate': 0.02,    # Slightly lower than v21 (0.025)
    'n_estimators': 600,      # More trees
    'min_child_weight': 4,    # Slightly higher than v21 (3)
    'subsample': 0.75,        # Slightly lower than v21 (0.8)
    'colsample_bytree': 0.55, # Slightly lower than v21 (0.6)
    'reg_alpha': 0.3,         # Slightly higher than v21 (0.2)
    'reg_lambda': 2.0,        # Slightly higher than v21 (1.5)
    'scale_pos_weight': scale_pos_weight,
    'random_state': 42,
    'verbosity': 0,
    'early_stopping_rounds': 80
}

# Train
n_splits = 5
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(y))
test_preds = np.zeros((len(test_ids), n_splits))

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X[train_idx], y[train_idx], eval_set=[(X[val_idx], y[val_idx])], verbose=False)

    oof_preds[val_idx] = model.predict_proba(X[val_idx])[:, 1]
    test_preds[:, fold] = model.predict_proba(X_test)[:, 1]

    best_f1 = max(f1_score(y[val_idx], (oof_preds[val_idx] > t).astype(int))
                  for t in np.arange(0.03, 0.20, 0.01))
    print(f"Fold {fold+1}: F1 = {best_f1:.4f}")

# Best threshold
best_oof_f1 = 0
best_thresh = 0.07
for thresh in np.arange(0.03, 0.20, 0.01):
    f1 = f1_score(y, (oof_preds > thresh).astype(int))
    if f1 > best_oof_f1:
        best_oof_f1 = f1
        best_thresh = thresh

print(f"\nOOF F1: {best_oof_f1:.4f} (threshold={best_thresh:.2f})")
print(f"Comparison with v21 (0.6708): {(best_oof_f1-0.6708)/0.6708*100:+.2f}%")

# Submission
test_probs = test_preds.mean(axis=1)
test_binary = (test_probs > best_thresh).astype(int)

submission = pd.DataFrame({'object_id': test_ids, 'target': test_binary})
submission.to_csv(base_path / 'submissions/submission_v29a_mild_reg.csv', index=False)
print(f"Predicted TDEs: {test_binary.sum()}")

# Save predictions for threshold tuning
np.save(base_path / 'data/processed/v29a_test_probs.npy', test_probs)
np.save(base_path / 'data/processed/v29a_oof_preds.npy', oof_preds)

print("Done!")

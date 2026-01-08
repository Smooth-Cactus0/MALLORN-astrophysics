"""
MALLORN v28a: Regularized XGBoost with Fewer Features

Key changes from v21:
- Use only top 60 features (instead of 120)
- Stronger L1/L2 regularization
- More aggressive early stopping
- Lower learning rate

Target: Reduce overfitting (v21: OOF 0.6708 → LB 0.6649)
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
print("MALLORN v28a: Regularized XGBoost")
print("=" * 60)

# ====================
# 1. LOAD DATA
# ====================
print("\n[1/3] Loading data...", flush=True)

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

# Remove correlated features
corr_to_drop = set()
for _, row in high_corr_df.iterrows():
    if row['feature_1'] not in corr_to_drop:
        corr_to_drop.add(row['feature_2'])
clean_features = importance_df[~importance_df['feature'].isin(corr_to_drop)]

# KEY CHANGE: Use only top 60 features (more aggressive selection)
n_features = 60
selected_features = clean_features.head(n_features)['feature'].tolist()

# GP2D features (keep only most important)
with open(base_path / 'data/processed/multiband_gp_cache.pkl', 'rb') as f:
    gp2d_data = pickle.load(f)
train_gp2d = gp2d_data['train']
test_gp2d = gp2d_data['test']

# Only keep top GP features
gp_importance = [
    'gp2d_wave_scale', 'gp2d_time_wave_ratio', 'gp2d_log_likelihood',
    'gp2d_time_scale', 'gp2d_residual_std'
]
gp_cols = [c for c in gp_importance if c in train_gp2d.columns]

# Merge
train_combined = train_features.merge(train_gp2d[['object_id'] + gp_cols], on='object_id', how='left')
test_combined = test_features.merge(test_gp2d[['object_id'] + gp_cols], on='object_id', how='left')

# Select features
base_cols = [c for c in selected_features if c in train_combined.columns]
all_feature_cols = base_cols + gp_cols
all_feature_cols = list(dict.fromkeys(all_feature_cols))

train_combined = train_combined.set_index('object_id').loc[train_ids].reset_index()
test_combined = test_combined.set_index('object_id').loc[test_ids].reset_index()

X = train_combined[all_feature_cols].values.astype(np.float32)
X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
X_test = test_combined[all_feature_cols].values.astype(np.float32)
X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

print(f"   Features: {len(all_feature_cols)} (reduced from 147)")
print(f"   Samples: {len(y)} ({np.sum(y==1)} TDE, {np.sum(y==0)} non-TDE)")

# ====================
# 2. TRAIN WITH STRONGER REGULARIZATION
# ====================
print("\n[2/3] Training with regularization...", flush=True)

n_neg, n_pos = np.sum(y == 0), np.sum(y == 1)
scale_pos_weight = n_neg / n_pos

# KEY CHANGES: Stronger regularization
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'hist',
    'max_depth': 4,          # Reduced from 5
    'learning_rate': 0.015,   # Reduced from 0.025
    'n_estimators': 800,      # More trees with lower LR
    'min_child_weight': 5,    # Increased from 3
    'subsample': 0.7,         # Reduced from 0.8
    'colsample_bytree': 0.5,  # Reduced from 0.6
    'reg_alpha': 0.5,         # Increased from 0.2
    'reg_lambda': 3.0,        # Increased from 1.5
    'scale_pos_weight': scale_pos_weight,
    'random_state': 42,
    'verbosity': 0,
    'early_stopping_rounds': 100  # More patience
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

    # Find optimal threshold
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
    f1 = f1_score(y, (oof_preds > thresh).astype(int))
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
# 3. SUBMISSION
# ====================
print("\n[3/3] Generating submission...", flush=True)

test_probs = test_preds.mean(axis=1)
test_binary = (test_probs > best_thresh).astype(int)

submission = pd.DataFrame({'object_id': test_ids, 'target': test_binary})
submission.to_csv(base_path / 'submissions/submission_v28a_regularized.csv', index=False)

print(f"   Predicted TDEs: {test_binary.sum()} / {len(test_binary)}")

# Save for ensemble
with open(base_path / 'data/processed/v28a_cache.pkl', 'wb') as f:
    pickle.dump({'oof_preds': oof_preds, 'test_preds': test_preds, 'threshold': best_thresh}, f)

print("Done!")

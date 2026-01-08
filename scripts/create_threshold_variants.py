"""
MALLORN: Create Multiple Threshold Variants

Regenerate v21 predictions and create submissions at different thresholds.
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
print("Creating Threshold Variants (v21 base)")
print("=" * 60)

# Load data
train_log = pd.read_csv(base_path / 'data/raw/train_log.csv')
test_log = pd.read_csv(base_path / 'data/raw/test_log.csv')

train_ids = train_log['object_id'].tolist()
test_ids = test_log['object_id'].tolist()
y = train_log['target'].values

# Load v21 features
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

with open(base_path / 'data/processed/multiband_gp_cache.pkl', 'rb') as f:
    gp2d_data = pickle.load(f)
train_gp2d = gp2d_data['train']
test_gp2d = gp2d_data['test']
gp2d_cols = [c for c in train_gp2d.columns if c != 'object_id']

train_combined = train_features.merge(train_gp2d, on='object_id', how='left')
test_combined = test_features.merge(test_gp2d, on='object_id', how='left')

base_cols = [c for c in selected_120 if c in train_combined.columns]
all_feature_cols = base_cols + gp2d_cols
all_feature_cols = list(dict.fromkeys(all_feature_cols))

train_combined = train_combined.set_index('object_id').loc[train_ids].reset_index()
test_combined = test_combined.set_index('object_id').loc[test_ids].reset_index()

X = train_combined[all_feature_cols].values.astype(np.float32)
X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
X_test = test_combined[all_feature_cols].values.astype(np.float32)
X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

print(f"Features: {len(all_feature_cols)}")

# Load Optuna params (v21 settings)
with open(base_path / 'data/processed/optuna_v20c_results.pkl', 'rb') as f:
    optuna_data = pickle.load(f)
xgb_params = optuna_data['xgb_best_params']

n_neg, n_pos = np.sum(y == 0), np.sum(y == 1)
scale_pos_weight = n_neg / n_pos

# Train and predict
print("\nTraining v21 model...")
n_splits = 5
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(y))
test_preds = np.zeros((len(test_ids), n_splits))

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    model = xgb.XGBClassifier(**xgb_params, scale_pos_weight=scale_pos_weight, random_state=42, verbosity=0)
    model.fit(X[train_idx], y[train_idx], eval_set=[(X[val_idx], y[val_idx])], verbose=False)

    oof_preds[val_idx] = model.predict_proba(X[val_idx])[:, 1]
    test_preds[:, fold] = model.predict_proba(X_test)[:, 1]

test_probs = test_preds.mean(axis=1)

print(f"Test probs range: [{test_probs.min():.4f}, {test_probs.max():.4f}]")

# Generate submissions at different thresholds
thresholds = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.15]

print("\nThreshold variants:")
print("-" * 50)
print(f"{'Threshold':<12} {'OOF F1':<10} {'# TDEs':<10} {'File'}")
print("-" * 50)

for thresh in thresholds:
    # OOF F1
    oof_f1 = f1_score(y, (oof_preds > thresh).astype(int))

    # Test predictions
    preds = (test_probs > thresh).astype(int)
    n_tde = preds.sum()

    submission = pd.DataFrame({'object_id': test_ids, 'target': preds})
    filename = f'submission_v21_thresh{int(thresh*100):02d}.csv'
    submission.to_csv(base_path / 'submissions' / filename, index=False)

    print(f"{thresh:<12.2f} {oof_f1:<10.4f} {n_tde:<10} {filename}")

print("-" * 50)
print("\nRecommendation: Start with thresh=0.07 (best OOF), then try 0.06 and 0.08")
print("Done!")

"""
MALLORN v26: XGBoost with v21 Features + ASTROMER Embeddings

Combines:
- Full v21 feature set (274 statistical + color + shape features)
- Selected top 120 features (reduced correlation)
- Multi-band GP features (2D kernel)
- ASTROMER pre-trained embeddings (384 features)

Target: Beat v21's 0.6708 OOF / 0.6649 LB
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

print("=" * 60, flush=True)
print("MALLORN v26: Full Features + ASTROMER Embeddings", flush=True)
print("=" * 60, flush=True)

# ====================
# 1. LOAD DATA AND V21 FEATURES
# ====================
print("\n[1/5] Loading v21 features...", flush=True)

# Load metadata
train_log = pd.read_csv(base_path / 'data/raw/train_log.csv')
test_log = pd.read_csv(base_path / 'data/raw/test_log.csv')
train_ids = train_log['object_id'].tolist()
test_ids = test_log['object_id'].tolist()
y = train_log['target'].values

# Load cached features
cached = pd.read_pickle(base_path / 'data/processed/features_v4_cache.pkl')
train_features = cached['train_features']
test_features = cached['test_features']
print(f"   Base features: {train_features.shape}", flush=True)

# Load feature selection
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
print(f"   GP2D features: {len(gp2d_cols)}", flush=True)

# Merge base features with GP2D
train_combined = train_features.merge(train_gp2d, on='object_id', how='left')
test_combined = test_features.merge(test_gp2d, on='object_id', how='left')

# Get v21 feature columns (selected 120 + GP2D)
base_cols = [c for c in selected_120 if c in train_combined.columns]
v21_feature_cols = base_cols + gp2d_cols
v21_feature_cols = list(dict.fromkeys(v21_feature_cols))
v21_feature_cols = [c for c in v21_feature_cols if c in train_combined.columns]
print(f"   V21 total features: {len(v21_feature_cols)}", flush=True)

# ====================
# 2. LOAD ASTROMER EMBEDDINGS
# ====================
print("\n[2/5] Loading ASTROMER embeddings...", flush=True)

train_emb = pd.read_csv(base_path / 'data/processed/train_astromer_embeddings.csv')
test_emb = pd.read_csv(base_path / 'data/processed/test_astromer_embeddings.csv')
emb_cols = [c for c in train_emb.columns if c != 'object_id']
print(f"   ASTROMER features: {len(emb_cols)}", flush=True)

# ====================
# 3. COMBINE ALL FEATURES
# ====================
print("\n[3/5] Combining features...", flush=True)

# Merge ASTROMER embeddings
train_combined = train_combined.merge(train_emb, on='object_id', how='left')
test_combined = test_combined.merge(test_emb, on='object_id', how='left')

# All feature columns = v21 features + ASTROMER embeddings
all_feature_cols = v21_feature_cols + emb_cols
all_feature_cols = [c for c in all_feature_cols if c in train_combined.columns]

# Align indices
train_combined = train_combined.set_index('object_id').loc[train_ids].reset_index()
test_combined = test_combined.set_index('object_id').loc[test_ids].reset_index()

X = train_combined[all_feature_cols].values.astype(np.float32)
X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
X_test = test_combined[all_feature_cols].values.astype(np.float32)
X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

print(f"   Total features: {len(all_feature_cols)}", flush=True)
print(f"   V21: {len(v21_feature_cols)}, ASTROMER: {len(emb_cols)}", flush=True)

# ====================
# 4. LOAD OPTUNA PARAMS
# ====================
print("\n[4/5] Loading Optuna parameters...", flush=True)

with open(base_path / 'data/processed/optuna_v20c_results.pkl', 'rb') as f:
    optuna_data = pickle.load(f)

xgb_params = optuna_data['xgb_best_params']
print(f"   XGB params loaded", flush=True)

# ====================
# 5. TRAIN XGBOOST (5-FOLD CV)
# ====================
print("\n[5/5] Training XGBoost (5-fold CV)...", flush=True)

n_neg, n_pos = np.sum(y == 0), np.sum(y == 1)
scale_pos_weight = n_neg / n_pos
print(f"   Class balance: {n_pos} TDE vs {n_neg} non-TDE", flush=True)

n_splits = 5
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(y))
test_preds = np.zeros((len(test_ids), n_splits))
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    model = xgb.XGBClassifier(
        **xgb_params,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbosity=0
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds[:, fold] = model.predict_proba(X_test)[:, 1]

    # Find optimal threshold
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
    print(f"   🎉 IMPROVEMENT: +{improvement:.2f}%")
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
submission_path = base_path / 'submissions/submission_v26_full.csv'
submission.to_csv(submission_path, index=False)

print(f"   Saved to {submission_path}")
print(f"   Predicted TDEs: {test_binary.sum()} / {len(test_binary)}")

# Feature importance
print("\nTop 30 Features by Importance:")
final_model = xgb.XGBClassifier(**{k: v for k, v in xgb_params.items() if k != 'early_stopping_rounds'})
final_model.fit(X, y)
importance = pd.DataFrame({
    'feature': all_feature_cols,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)
print(importance.head(30).to_string(index=False))

# Count ASTROMER features in top 30
astromer_in_top30 = importance.head(30)['feature'].str.contains('emb').sum()
print(f"\nASTROMER features in top 30: {astromer_in_top30}")

# Save cache
cache_path = base_path / 'data/processed/v26_full_cache.pkl'
with open(cache_path, 'wb') as f:
    pickle.dump({
        'oof_preds': oof_preds,
        'test_preds': test_preds,
        'best_threshold': best_global_thresh,
        'oof_f1': best_oof_f1,
        'feature_cols': all_feature_cols
    }, f)
print(f"\nCache saved to {cache_path}")

print("\nDone!")

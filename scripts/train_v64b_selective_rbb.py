"""
MALLORN v64b: Selective R_bb Features

v64 showed that R_bb features have potential (12.3% importance) but
full set hurt performance due to noise and low coverage.

This version uses ONLY the most discriminative R_bb features:
1. R_ratio_peak_50d - Radius ratio (TDEs decrease, SNe increase early)
2. R_ratio_10d_30d - Early radius evolution
3. dRdt_early - Direction of radius change (THE key discriminator)
4. dRdt_late - Late-phase behavior
5. R_increasing_early - Binary: is R increasing? (SN signature)
6. R_monotonic_decrease - Binary: always decreasing? (TDE signature)
7. T_drop_peak_50d - Temperature drop (SNe cool faster)
8. T_constancy - Temperature stability (TDEs stay hot)
9. R_direction_score - Combined direction metric

Target: Beat v34a OOF F1=0.6667
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
print("MALLORN v64b: Selective R_bb Features", flush=True)
print("=" * 80, flush=True)

# ====================
# 1. LOAD DATA
# ====================
print("\n1. Loading data...", flush=True)

from utils.data_loader import load_all_data
data = load_all_data()

train_meta = data['train_meta']
test_meta = data['test_meta']
train_ids = train_meta['object_id'].tolist()
test_ids = test_meta['object_id'].tolist()
y = train_meta['target'].values

print(f"   Training: {len(train_ids)} objects ({np.sum(y)} TDEs)", flush=True)

# Load v34a features (v21 + Bazin)
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

tde_cached = pd.read_pickle(base_path / 'data/processed/tde_physics_cache.pkl')
train_tde = tde_cached['train']
test_tde = tde_cached['test']

train_base = train_base.merge(train_tde, on='object_id', how='left')
test_base = test_base.merge(test_tde, on='object_id', how='left')

with open(base_path / 'data/processed/multiband_gp_cache.pkl', 'rb') as f:
    gp2d_data = pickle.load(f)
train_gp2d = gp2d_data['train']
test_gp2d = gp2d_data['test']
gp2d_cols = [c for c in train_gp2d.columns if c != 'object_id']

train_v21 = train_base[['object_id'] + selected_120].copy()
train_v21 = train_v21.merge(train_tde, on='object_id', how='left')
train_v21 = train_v21.merge(train_gp2d[['object_id'] + gp2d_cols], on='object_id', how='left')

test_v21 = test_base[['object_id'] + selected_120].copy()
test_v21 = test_v21.merge(test_tde, on='object_id', how='left')
test_v21 = test_v21.merge(test_gp2d[['object_id'] + gp2d_cols], on='object_id', how='left')

# Bazin
bazin_cache_path = base_path / 'data/processed/bazin_features_cache.pkl'
with open(bazin_cache_path, 'rb') as f:
    bazin_cache = pickle.load(f)
train_bazin = bazin_cache['train']
test_bazin = bazin_cache['test']

# v34a base
train_v34a = train_v21.merge(train_bazin, on='object_id', how='left')
test_v34a = test_v21.merge(test_bazin, on='object_id', how='left')

print(f"   v34a features: {len(train_v34a.columns)-1}", flush=True)

# ====================
# 2. LOAD R_bb FEATURES (SELECTIVE)
# ====================
print("\n2. Loading selective R_bb features...", flush=True)

rbb_cache_path = base_path / 'data/processed/rbb_features_cache.pkl'
with open(rbb_cache_path, 'rb') as f:
    rbb_cache = pickle.load(f)
train_rbb = rbb_cache['train']
test_rbb = rbb_cache['test']

# SELECT ONLY THE MOST DISCRIMINATIVE FEATURES
selected_rbb = [
    # Core discriminators
    'R_ratio_peak_50d',      # TDEs: R decreases → ratio > 1
    'R_ratio_10d_30d',       # Early evolution pattern
    'dRdt_early',            # CRITICAL: positive = SN, negative = TDE
    'dRdt_late',             # Late-phase behavior
    'dRdt_overall',          # Overall trend

    # Binary indicators
    'R_increasing_early',    # 1 = SN signature (expanding)
    'R_monotonic_decrease',  # 1 = TDE signature (contracting)
    'R_frac_decreasing',     # What fraction of epochs show decrease

    # Temperature features
    'T_drop_peak_50d',       # SNe cool faster
    'T_constancy',           # TDEs stay hot (high = TDE)
    'T_variance',            # Low variance = TDE

    # Combined score
    'R_direction_score',     # Positive = SN, Negative = TDE
]

# Keep only selected features
available_rbb = [c for c in selected_rbb if c in train_rbb.columns]
train_rbb_sel = train_rbb[['object_id'] + available_rbb].copy()
test_rbb_sel = test_rbb[['object_id'] + available_rbb].copy()

print(f"   Selected R_bb features: {len(available_rbb)}", flush=True)
print(f"   Features: {available_rbb}", flush=True)

# Coverage check
for feat in available_rbb[:5]:
    coverage = train_rbb_sel[feat].notna().sum() / len(train_rbb_sel)
    print(f"      {feat}: {100*coverage:.1f}% coverage", flush=True)

# ====================
# 3. COMBINE FEATURES
# ====================
print("\n3. Combining v34a + selective R_bb...", flush=True)

train_combined = train_v34a.merge(train_rbb_sel, on='object_id', how='left')
test_combined = test_v34a.merge(test_rbb_sel, on='object_id', how='left')

feature_names = [c for c in train_combined.columns if c != 'object_id']

print(f"   Total features: {len(feature_names)}", flush=True)
print(f"   (v34a: {len(train_v34a.columns)-1} + R_bb: {len(available_rbb)})", flush=True)

X_train = train_combined.drop(columns=['object_id']).values
X_test = test_combined.drop(columns=['object_id']).values

X_train = np.nan_to_num(X_train, nan=np.nan, posinf=1e10, neginf=-1e10)
X_test = np.nan_to_num(X_test, nan=np.nan, posinf=1e10, neginf=-1e10)

# ====================
# 4. TRAIN XGBOOST
# ====================
print("\n4. Training XGBoost with 5-fold CV...", flush=True)

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
    'scale_pos_weight': len(y[y==0]) / len(y[y==1]),
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1
}

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X_train))
test_preds = np.zeros((len(X_test), n_folds))
feature_importance = np.zeros(len(feature_names))
fold_f1s = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y), 1):
    print(f"\n   Fold {fold}/{n_folds}:", flush=True)

    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

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
    for t in np.linspace(0.03, 0.3, 50):
        preds_binary = (oof_preds[val_idx] > t).astype(int)
        f1 = f1_score(y_val, preds_binary)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    fold_f1s.append(best_f1)
    print(f"      Fold F1: {best_f1:.4f} @ threshold={best_thresh:.3f}", flush=True)

# ====================
# 5. RESULTS
# ====================
print("\n" + "=" * 80, flush=True)
print("RESULTS", flush=True)
print("=" * 80, flush=True)

best_f1 = 0
best_thresh = 0.1
for t in np.linspace(0.03, 0.3, 200):
    preds_binary = (oof_preds > t).astype(int)
    f1 = f1_score(y, preds_binary)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"\n   OOF F1: {best_f1:.4f} @ threshold={best_thresh:.3f}", flush=True)
print(f"   Fold F1s: {[f'{f:.4f}' for f in fold_f1s]}", flush=True)
print(f"   Mean Fold F1: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}", flush=True)

final_preds = (oof_preds > best_thresh).astype(int)
tp = np.sum((final_preds == 1) & (y == 1))
fp = np.sum((final_preds == 1) & (y == 0))
fn = np.sum((final_preds == 0) & (y == 1))

print(f"\n   TP={tp}, FP={fp}, FN={fn}", flush=True)
print(f"   Precision: {tp/(tp+fp):.4f}", flush=True)
print(f"   Recall: {tp/(tp+fn):.4f}", flush=True)

# Feature importance
feature_importance = feature_importance / n_folds
importance_df_result = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("\n   Top 20 Features:", flush=True)
print(importance_df_result.head(20).to_string(index=False), flush=True)

# R_bb analysis
print("\n   R_bb Feature Rankings:", flush=True)
for feat in available_rbb:
    if feat in importance_df_result['feature'].values:
        rank = list(importance_df_result['feature']).index(feat) + 1
        imp = importance_df_result[importance_df_result['feature'] == feat]['importance'].values[0]
        print(f"      {rank:3d}. {feat:25s} {imp:8.1f}", flush=True)

# ====================
# 6. SUBMISSION
# ====================
print("\n" + "=" * 80, flush=True)
print("SUBMISSION", flush=True)
print("=" * 80, flush=True)

test_avg = test_preds.mean(axis=1)
test_final = (test_avg > best_thresh).astype(int)

submission = pd.DataFrame({
    'object_id': test_ids,
    'target': test_final
})

submission_path = base_path / 'submissions/submission_v64b_selective_rbb.csv'
submission.to_csv(submission_path, index=False)

print(f"   Saved: {submission_path.name}", flush=True)
print(f"   Predicted TDEs: {test_final.sum()}", flush=True)

# Artifacts
artifacts = {
    'oof_preds': oof_preds,
    'test_preds': test_avg,
    'feature_importance': importance_df_result,
    'best_threshold': best_thresh,
    'oof_f1': best_f1,
    'selected_rbb': available_rbb,
    'fold_f1s': fold_f1s
}

with open(base_path / 'data/processed/v64b_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

# ====================
# 7. COMPARISON
# ====================
print("\n" + "=" * 80, flush=True)
print("COMPARISON", flush=True)
print("=" * 80, flush=True)

v34a_f1 = 0.6667
improvement = (best_f1 - v34a_f1) / v34a_f1 * 100

print(f"\n   v34a baseline: {v34a_f1:.4f}", flush=True)
print(f"   v64 (all R_bb): 0.6609 (-0.87%)", flush=True)
print(f"   v64b (selective): {best_f1:.4f} ({improvement:+.2f}%)", flush=True)

if best_f1 > v34a_f1:
    print("\n   SUCCESS: Selective R_bb features improved performance!", flush=True)
elif best_f1 > 0.6609:
    print("\n   PARTIAL SUCCESS: Better than full R_bb but not baseline", flush=True)
else:
    print("\n   No improvement from R_bb features", flush=True)

print("\n" + "=" * 80, flush=True)
print("v64b Complete", flush=True)
print("=" * 80, flush=True)

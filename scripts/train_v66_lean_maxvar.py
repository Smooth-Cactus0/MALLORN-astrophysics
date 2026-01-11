"""
MALLORN v66: Lean Feature Set + MaxVar

PROBLEM: v65 overfit (OOF 0.6780 but LB 0.6344)
- v34a: 224 features → LB 0.6907
- v65: 245 features → LB 0.6344 (worse!)

SOLUTION:
1. Start with v34a's feature importance ranking
2. Keep only TOP features (reduce from 224)
3. Add ONLY the most valuable MaxVar features
4. Target: ~150-180 total features (leaner than v34a)

This should improve generalization while keeping MaxVar's discriminative power.
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
print("MALLORN v66: Lean Feature Set + MaxVar", flush=True)
print("=" * 80, flush=True)
print("Reducing features to prevent overfitting", flush=True)
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

# ====================
# 2. LOAD v34a ARTIFACTS FOR FEATURE IMPORTANCE
# ====================
print("\n2. Loading v34a feature importance...", flush=True)

with open(base_path / 'data/processed/v34a_artifacts.pkl', 'rb') as f:
    v34a_artifacts = pickle.load(f)

v34a_importance = v34a_artifacts['feature_importance']
v34a_features = v34a_artifacts['feature_names']

print(f"   v34a had {len(v34a_features)} features", flush=True)
print(f"   v34a OOF: {v34a_artifacts['oof_f1']:.4f}", flush=True)

# Top features from v34a
top_v34a = v34a_importance.head(50)['feature'].tolist()
print(f"\n   Top 20 v34a features:", flush=True)
for i, row in v34a_importance.head(20).iterrows():
    print(f"      {row['feature']:35s} {row['importance']:8.1f}", flush=True)

# ====================
# 3. BUILD LEAN FEATURE SET
# ====================
print("\n3. Building lean feature set...", flush=True)

# Load base features
cached = pd.read_pickle(base_path / 'data/processed/features_v4_cache.pkl')
train_base = cached['train_features']
test_base = cached['test_features']

# Load feature selection from v21
selection = pd.read_pickle(base_path / 'data/processed/selected_features.pkl')
importance_df = selection['importance_df']
high_corr_df = selection['high_corr_df']

corr_to_drop = set()
for _, row in high_corr_df.iterrows():
    if row['feature_1'] not in corr_to_drop:
        corr_to_drop.add(row['feature_2'])
clean_features = importance_df[~importance_df['feature'].isin(corr_to_drop)]

# LEAN: Take only top 80 base features (was 120)
selected_80 = clean_features.head(80)['feature'].tolist()
print(f"   Base features (reduced): {len(selected_80)} (was 120)", flush=True)

# TDE physics - keep all (they're important)
tde_cached = pd.read_pickle(base_path / 'data/processed/tde_physics_cache.pkl')
train_tde = tde_cached['train']
test_tde = tde_cached['test']
tde_cols = [c for c in train_tde.columns if c != 'object_id']
print(f"   TDE physics features: {len(tde_cols)}", flush=True)

# GP2D - keep only most important ones
with open(base_path / 'data/processed/multiband_gp_cache.pkl', 'rb') as f:
    gp2d_data = pickle.load(f)
train_gp2d = gp2d_data['train']
test_gp2d = gp2d_data['test']

# Select only top GP2D features based on v34a importance
gp2d_cols_all = [c for c in train_gp2d.columns if c != 'object_id']
gp2d_in_top = [c for c in gp2d_cols_all if c in top_v34a]
gp2d_cols = gp2d_in_top if len(gp2d_in_top) > 5 else gp2d_cols_all[:10]
print(f"   GP2D features (reduced): {len(gp2d_cols)} (was {len(gp2d_cols_all)})", flush=True)

# Bazin - keep only most important ones
bazin_cache_path = base_path / 'data/processed/bazin_features_cache.pkl'
with open(bazin_cache_path, 'rb') as f:
    bazin_cache = pickle.load(f)
train_bazin = bazin_cache['train']
test_bazin = bazin_cache['test']

bazin_cols_all = [c for c in train_bazin.columns if c != 'object_id']
bazin_in_top = [c for c in bazin_cols_all if c in v34a_importance.head(100)['feature'].tolist()]
bazin_cols = bazin_in_top if len(bazin_in_top) > 10 else bazin_cols_all[:20]
print(f"   Bazin features (reduced): {len(bazin_cols)} (was {len(bazin_cols_all)})", flush=True)

# Merge base features
train_base = train_base.merge(train_tde, on='object_id', how='left')
test_base = test_base.merge(test_tde, on='object_id', how='left')

train_lean = train_base[['object_id'] + selected_80].copy()
train_lean = train_lean.merge(train_tde, on='object_id', how='left')
train_lean = train_lean.merge(train_gp2d[['object_id'] + gp2d_cols], on='object_id', how='left')
train_lean = train_lean.merge(train_bazin[['object_id'] + bazin_cols], on='object_id', how='left')

test_lean = test_base[['object_id'] + selected_80].copy()
test_lean = test_lean.merge(test_tde, on='object_id', how='left')
test_lean = test_lean.merge(test_gp2d[['object_id'] + gp2d_cols], on='object_id', how='left')
test_lean = test_lean.merge(test_bazin[['object_id'] + bazin_cols], on='object_id', how='left')

print(f"\n   Lean base features: {len(train_lean.columns)-1}", flush=True)

# ====================
# 4. ADD SELECTIVE MAXVAR FEATURES
# ====================
print("\n4. Adding selective MaxVar features...", flush=True)

# Load v65 power-law features
pl_cache_path = base_path / 'data/processed/powerlaw_features_cache.pkl'
with open(pl_cache_path, 'rb') as f:
    pl_cache = pickle.load(f)
train_pl = pl_cache['train']
test_pl = pl_cache['test']

# Select ONLY the most important MaxVar features from v65
# (r_maxvar was #1, maxvar_mean was #2)
maxvar_select = [
    'r_maxvar',           # #1 in v65 - THE key feature
    'maxvar_mean',        # #2 in v65
    'maxvar_max',         # #8 in v65
    'r_late_frac',        # #13 in v65
    'g_maxvar',           # Useful
]

available_maxvar = [c for c in maxvar_select if c in train_pl.columns]
train_maxvar = train_pl[['object_id'] + available_maxvar]
test_maxvar = test_pl[['object_id'] + available_maxvar]

print(f"   Adding {len(available_maxvar)} MaxVar features: {available_maxvar}", flush=True)

# Combine
train_combined = train_lean.merge(train_maxvar, on='object_id', how='left')
test_combined = test_lean.merge(test_maxvar, on='object_id', how='left')

feature_names = [c for c in train_combined.columns if c != 'object_id']
print(f"\n   Total features: {len(feature_names)} (v34a had 224)", flush=True)

X_train = train_combined.drop(columns=['object_id']).values
X_test = test_combined.drop(columns=['object_id']).values

X_train = np.nan_to_num(X_train, nan=np.nan, posinf=1e10, neginf=-1e10)
X_test = np.nan_to_num(X_test, nan=np.nan, posinf=1e10, neginf=-1e10)

# ====================
# 5. TRAIN WITH MORE REGULARIZATION
# ====================
print("\n5. Training XGBoost (increased regularization)...", flush=True)

# Stronger regularization to prevent overfitting
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 4,           # Reduced from 5
    'learning_rate': 0.02,    # Reduced from 0.025
    'subsample': 0.7,         # Reduced from 0.8
    'colsample_bytree': 0.7,  # Reduced from 0.8
    'min_child_weight': 5,    # Increased from 3
    'reg_alpha': 0.5,         # Increased from 0.2
    'reg_lambda': 2.0,        # Increased from 1.5
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
    for t in np.linspace(0.03, 0.3, 50):
        preds_binary = (oof_preds[val_idx] > t).astype(int)
        f1 = f1_score(y_val, preds_binary)
        if f1 > best_f1:
            best_f1 = f1

    fold_f1s.append(best_f1)
    print(f"      Fold F1: {best_f1:.4f}", flush=True)

# ====================
# 6. RESULTS
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
print(f"   Std: {np.std(fold_f1s):.4f}", flush=True)

final_preds = (oof_preds > best_thresh).astype(int)
tp = np.sum((final_preds == 1) & (y == 1))
fp = np.sum((final_preds == 1) & (y == 0))
fn = np.sum((final_preds == 0) & (y == 1))
print(f"\n   TP={tp}, FP={fp}, FN={fn}", flush=True)
print(f"   Precision: {tp/(tp+fp):.4f}, Recall: {tp/(tp+fn):.4f}", flush=True)

# Feature importance
feature_importance = feature_importance / n_folds
importance_df_result = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("\n   Top 20 Features:", flush=True)
print(importance_df_result.head(20).to_string(index=False), flush=True)

# MaxVar rankings
print("\n   MaxVar Feature Rankings:", flush=True)
for col in available_maxvar:
    if col in importance_df_result['feature'].values:
        rank = list(importance_df_result['feature']).index(col) + 1
        imp = importance_df_result[importance_df_result['feature'] == col]['importance'].values[0]
        print(f"      {rank:3d}. {col:20s} {imp:8.1f}", flush=True)

# ====================
# 7. SUBMISSION
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

submission_path = base_path / 'submissions/submission_v66_lean_maxvar.csv'
submission.to_csv(submission_path, index=False)

print(f"   Saved: {submission_path.name}", flush=True)
print(f"   Predicted TDEs: {test_final.sum()}", flush=True)
print(f"   Features: {len(feature_names)} (vs v34a: 224, v65: 245)", flush=True)

# Artifacts
artifacts = {
    'oof_preds': oof_preds,
    'test_preds': test_avg,
    'feature_importance': importance_df_result,
    'best_threshold': best_thresh,
    'oof_f1': best_f1,
    'feature_names': feature_names,
    'fold_f1s': fold_f1s
}

with open(base_path / 'data/processed/v66_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

# Comparison
print("\n" + "=" * 80, flush=True)
print("COMPARISON (OOF → LB correlation is key)", flush=True)
print("=" * 80, flush=True)

print(f"""
   Version  Features  OOF F1   LB F1    Notes
   -------  --------  ------   -----    -----
   v34a     224       0.6667   0.6907   Baseline (LB > OOF ✓)
   v65      245       0.6780   0.6344   OVERFIT (LB << OOF ✗)
   v66      {len(feature_names):3d}       {best_f1:.4f}   ???      Lean + MaxVar

   Goal: OOF closer to v34a, hope for better LB generalization
""", flush=True)

print("=" * 80, flush=True)
print("v66 Complete", flush=True)
print("=" * 80, flush=True)

"""
MALLORN v30b: Selective Advanced Physics Features

v30 failed (-11.63%) due to feature explosion (314 features).
v30b uses v21's proven 147 features + only TOP advanced physics features.

Strategy:
1. Load v21's exact feature set (147 features, OOF F1=0.6708)
2. Add only the most promising advanced physics features (10-15)
3. Target: ~160 total features for balanced signal-to-noise

Top advanced features from v30:
- sed_quality_mean (rank 18, 31.3 importance)
- cooling_rate_overall (rank 22, 30.8 importance)
- temp_epoch features, late-time colors, asymmetry metrics
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
print("MALLORN v30b: Selective Advanced Physics Features", flush=True)
print("=" * 80, flush=True)

# ====================
# 1. LOAD v21 FEATURES (EXACT REPLICATION)
# ====================
print("\n1. Loading v21 feature set (147 features)...", flush=True)

from utils.data_loader import load_all_data
data = load_all_data()

train_meta = data['train_meta']
test_meta = data['test_meta']
train_lc = data['train_lc']
test_lc = data['test_lc']

train_ids = train_meta['object_id'].tolist()
test_ids = test_meta['object_id'].tolist()
y = train_meta['target'].values

# Load v21 base features
cached = pd.read_pickle(base_path / 'data/processed/features_v4_cache.pkl')
train_base = cached['train_features']
test_base = cached['test_features']

# Feature selection (v21 method)
selection = pd.read_pickle(base_path / 'data/processed/selected_features.pkl')
importance_df = selection['importance_df']
high_corr_df = selection['high_corr_df']

corr_to_drop = set()
for _, row in high_corr_df.iterrows():
    if row['feature_1'] not in corr_to_drop:
        corr_to_drop.add(row['feature_2'])
clean_features = importance_df[~importance_df['feature'].isin(corr_to_drop)]
selected_120 = clean_features.head(120)['feature'].tolist()

# TDE physics
tde_cached = pd.read_pickle(base_path / 'data/processed/tde_physics_cache.pkl')
train_tde = tde_cached['train']
test_tde = tde_cached['test']
tde_cols = [c for c in train_tde.columns if c != 'object_id']

train_base = train_base.merge(train_tde, on='object_id', how='left')
test_base = test_base.merge(test_tde, on='object_id', how='left')

# GP2D
with open(base_path / 'data/processed/multiband_gp_cache.pkl', 'rb') as f:
    gp2d_data = pickle.load(f)
train_gp2d = gp2d_data['train']
test_gp2d = gp2d_data['test']
gp2d_cols = [c for c in train_gp2d.columns if c != 'object_id']

# Combine v21 core features (base + TDE + GP2D)
train_v21 = train_base[['object_id'] + selected_120].copy()
train_v21 = train_v21.merge(train_tde, on='object_id', how='left')
train_v21 = train_v21.merge(train_gp2d[['object_id'] + gp2d_cols], on='object_id', how='left')

test_v21 = test_base[['object_id'] + selected_120].copy()
test_v21 = test_v21.merge(test_tde, on='object_id', how='left')
test_v21 = test_v21.merge(test_gp2d[['object_id'] + gp2d_cols], on='object_id', how='left')

print(f"   v21 features loaded: {len(train_v21.columns)-1}", flush=True)

# ====================
# 2. EXTRACT NEW ADVANCED PHYSICS FEATURES
# ====================
print("\n2. Extracting NEW advanced physics features...", flush=True)

from features.advanced_physics import extract_advanced_physics_features

print("   Training set...", flush=True)
train_adv_new = extract_advanced_physics_features(train_lc, train_ids)
print(f"   Test set...", flush=True)
test_adv_new = extract_advanced_physics_features(test_lc, test_ids)

# Select TOP advanced physics features based on v30 results
# These showed promise (ranked 18-30 in v30 with 314 features)
top_new_features = [
    'sed_quality_mean',           # Rank 18 in v30
    'cooling_rate_overall',       # Rank 22 in v30
    'temp_epoch_0d',              # Temperature at peak
    'temp_epoch_50d',             # Temperature 50d post-peak
    'temp_epoch_100d',            # Temperature 100d post-peak
    'cooling_rate_early',         # Early cooling rate
    'temp_dispersion_early',      # Temperature variability
    'g_r_late_100d',              # Late-time g-r color
    'g_r_late_150d',              # Very late g-r
    'g_r_late_slope',             # Late-time reddening rate
    'g_r_late_dispersion',        # Late color stability
    'asymmetry_dispersion',       # Cross-band asymmetry consistency
    'asymmetry_diff_g_r',         # g vs r asymmetry
    'peak_time_dispersion',       # Peak synchronization
    'rise_time_dispersion'        # Rise synchronization
]

# Filter to available features
available_new = [f for f in top_new_features if f in train_adv_new.columns]
print(f"   Selected {len(available_new)} new advanced features", flush=True)

# ====================
# 3. COMBINE FEATURES
# ====================
print("\n3. Combining v21 + new advanced features...", flush=True)

train_combined = train_v21.merge(train_adv_new[['object_id'] + available_new], on='object_id', how='left')
test_combined = test_v21.merge(test_adv_new[['object_id'] + available_new], on='object_id', how='left')

X_train = train_combined.drop(columns=['object_id']).values
X_test = test_combined.drop(columns=['object_id']).values
feature_names = [c for c in train_combined.columns if c != 'object_id']

print(f"   Total features: {len(feature_names)}", flush=True)
print(f"   v21 baseline: 147, new advanced: {len(available_new)}", flush=True)
print(f"   Training shape: {X_train.shape}", flush=True)

# ====================
# 4. TRAIN XGBoost
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
    for t in np.linspace(0.05, 0.5, 50):
        preds_binary = (oof_preds[val_idx] > t).astype(int)
        f1 = f1_score(y_val, preds_binary)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    print(f"      Best threshold: {best_thresh:.3f}, F1: {best_f1:.4f}", flush=True)

print("\n" + "=" * 80, flush=True)
print("CROSS-VALIDATION RESULTS", flush=True)
print("=" * 80, flush=True)

best_f1 = 0
best_thresh = 0.5
for t in np.linspace(0.05, 0.5, 100):
    preds_binary = (oof_preds > t).astype(int)
    f1 = f1_score(y, preds_binary)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"   OOF F1: {best_f1:.4f} @ threshold={best_thresh:.2f}", flush=True)

final_preds = (oof_preds > best_thresh).astype(int)
tp = np.sum((final_preds == 1) & (y == 1))
fp = np.sum((final_preds == 1) & (y == 0))
fn = np.sum((final_preds == 0) & (y == 1))
tn = np.sum((final_preds == 0) & (y == 0))

print(f"   Confusion: TP={tp}, FP={fp}, FN={fn}, TN={tn}", flush=True)
print(f"   Precision: {tp/(tp+fp):.4f}", flush=True)
print(f"   Recall: {tp/(tp+fn):.4f}", flush=True)

# ====================
# 5. FEATURE IMPORTANCE
# ====================
print("\n5. Top 30 Features by Importance:", flush=True)

feature_importance = feature_importance / n_folds
importance_df_result = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(importance_df_result.head(30).to_string(index=False), flush=True)

# Highlight NEW advanced features
new_importance = importance_df_result[importance_df_result['feature'].isin(available_new)]
if len(new_importance) > 0:
    print(f"\n   {len(new_importance)} new advanced features in model", flush=True)
    top_new = new_importance.head(5)
    print("\n   Top 5 new advanced features:")
    for idx, row in top_new.iterrows():
        rank = list(importance_df_result.index).index(idx) + 1
        print(f"      {rank:3d}. {row['feature']:30s} {row['importance']:8.1f}", flush=True)

# ====================
# 6. CREATE SUBMISSION
# ====================
print("\n6. Creating submission...", flush=True)

test_avg = test_preds.mean(axis=1)
test_final = (test_avg > best_thresh).astype(int)

submission = pd.DataFrame({
    'object_id': test_ids,
    'target': test_final
})

submission_path = base_path / 'submissions/submission_v30b_selective.csv'
submission.to_csv(submission_path, index=False)

print(f"   Submission saved: {submission_path.name}", flush=True)
print(f"   Predicted TDEs: {test_final.sum()} / {len(test_final)}", flush=True)

# Save artifacts
artifacts = {
    'oof_preds': oof_preds,
    'test_preds': test_avg,
    'feature_importance': importance_df_result,
    'best_threshold': best_thresh,
    'oof_f1': best_f1,
    'feature_names': feature_names
}

with open(base_path / 'data/processed/v30b_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print("\n" + "=" * 80, flush=True)
print(f"MALLORN v30b Complete: OOF F1 = {best_f1:.4f}", flush=True)
print(f"Baseline v21: OOF F1 = 0.6708", flush=True)
change_pct = (best_f1 - 0.6708) * 100
change_abs = best_f1 - 0.6708
print(f"Change: {change_pct:+.2f}% ({change_abs:+.4f})", flush=True)

if best_f1 > 0.6708:
    print("SUCCESS: New physics features improved performance!", flush=True)
else:
    print("Physics features did not improve beyond v21 baseline", flush=True)
print("=" * 80, flush=True)

"""
MALLORN v20b: Selective Feature Addition

Key insight from v20 benchmark:
- BASE + GP2D (0.6467) > ALL (0.6341)
- Adding all ADV features hurts performance (noise)

Strategy:
1. Start with v19's proven feature set (BASE + TDE + GP2D)
2. Incrementally add only the BEST ADV features
3. Stop when adding more features hurts performance

This should beat v19's 0.6626.
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

base_path = Path(__file__).parent.parent

print("=" * 60, flush=True)
print("MALLORN v20b: Selective Feature Addition", flush=True)
print("=" * 60, flush=True)

# ====================
# 1. LOAD ALL DATA
# ====================
print("\n1. Loading data...", flush=True)

from utils.data_loader import load_all_data
data = load_all_data()

train_lc = data['train_lc']
test_lc = data['test_lc']
train_meta = data['train_meta']
test_meta = data['test_meta']

train_ids = train_meta['object_id'].tolist()
test_ids = test_meta['object_id'].tolist()

y = train_meta['target'].values
print(f"   Train: {len(train_ids)} objects ({y.sum()} TDE)", flush=True)
print(f"   Test: {len(test_ids)} objects", flush=True)

# ====================
# 2. LOAD BASE FEATURES (v19 style)
# ====================
print("\n2. Loading base features (v19 style)...", flush=True)

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

# TDE physics
tde_cached = pd.read_pickle(base_path / 'data/processed/tde_physics_cache.pkl')
train_tde = tde_cached['train']
test_tde = tde_cached['test']
tde_cols = [c for c in train_tde.columns if c != 'object_id']

train_base = train_base.merge(train_tde, on='object_id', how='left')
test_base = test_base.merge(test_tde, on='object_id', how='left')

print(f"   BASE: {len(selected_120)} features", flush=True)
print(f"   TDE: {len(tde_cols)} features", flush=True)

# ====================
# 3. LOAD GP2D FEATURES
# ====================
print("\n3. Loading multi-band GP features...", flush=True)

with open(base_path / 'data/processed/multiband_gp_cache.pkl', 'rb') as f:
    gp2d_data = pickle.load(f)
train_gp2d = gp2d_data['train']
test_gp2d = gp2d_data['test']
gp2d_cols = [c for c in train_gp2d.columns if c != 'object_id']

print(f"   GP2D: {len(gp2d_cols)} features", flush=True)

# ====================
# 4. LOAD ADVANCED FEATURES
# ====================
print("\n4. Loading advanced features...", flush=True)

with open(base_path / 'data/processed/advanced_features_cache.pkl', 'rb') as f:
    adv_data = pickle.load(f)
train_adv = adv_data['train']
test_adv = adv_data['test']
adv_cols = [c for c in train_adv.columns if c != 'object_id']

print(f"   ADV: {len(adv_cols)} features", flush=True)

# ====================
# 5. IDENTIFY BEST ADV FEATURES
# ====================
print("\n5. Selecting best ADV features...", flush=True)

# These are the ADV features that showed up in top 30 importance in v20
top_adv_features = [
    'g_mhps_100',           # rank 7 - MHPS variability at 100 days
    'pre_peak_r_i_slope',   # rank 20 - pre-peak color evolution
    'r_mhps_30',            # MHPS at 30 days
    'r_mhps_ratio_30_365',  # ratio of short to long variability
    'r_abs_mag_peak',       # absolute magnitude at peak
    'g_abs_mag_peak',       # g-band absolute magnitude
    'r_fleet_width',        # FLEET model width
    'r_fleet_asymmetry',    # FLEET rise/decline asymmetry
    'peak_lag_g_r',         # time lag between g and r peak
    'flux_skewness',        # distribution shape
    'r_acf_10d',            # short-term autocorrelation
]

# Filter to features that exist
available_adv = [c for c in top_adv_features if c in adv_cols]
print(f"   Selected {len(available_adv)} ADV features:", flush=True)
for f in available_adv:
    print(f"     - {f}", flush=True)

# ====================
# 6. COMBINE FEATURES (v19 + selected ADV)
# ====================
print("\n6. Combining features...", flush=True)

# Start with base
train_combined = train_base.copy()
test_combined = test_base.copy()

# Add GP2D
train_combined = train_combined.merge(train_gp2d, on='object_id', how='left')
test_combined = test_combined.merge(test_gp2d, on='object_id', how='left')

# Add selected ADV features only
train_adv_selected = train_adv[['object_id'] + available_adv]
test_adv_selected = test_adv[['object_id'] + available_adv]

train_combined = train_combined.merge(train_adv_selected, on='object_id', how='left')
test_combined = test_combined.merge(test_adv_selected, on='object_id', how='left')

# Build feature list
base_cols = [c for c in selected_120 if c in train_combined.columns]
all_feature_cols = base_cols + tde_cols + gp2d_cols + available_adv

# De-duplicate
all_feature_cols = list(dict.fromkeys(all_feature_cols))
all_feature_cols = [c for c in all_feature_cols if c in train_combined.columns]

print(f"\n   Total features: {len(all_feature_cols)}", flush=True)
print(f"     BASE: {len(base_cols)}", flush=True)
print(f"     TDE: {len(tde_cols)}", flush=True)
print(f"     GP2D: {len(gp2d_cols)}", flush=True)
print(f"     ADV (selected): {len(available_adv)}", flush=True)

# Prepare data
train_combined = train_combined.fillna(-999)
test_combined = test_combined.fillna(-999)

train_combined = train_combined.set_index('object_id').loc[train_ids].reset_index()

X_all = train_combined[all_feature_cols].values.astype(np.float32)
X_test_all = test_combined[all_feature_cols].values.astype(np.float32)

# ====================
# 7. TRAIN ENSEMBLE
# ====================
print("\n7. Training 3-model ensemble...", flush=True)

n_splits = 5
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
scale_pos_weight = len(y[y==0]) / max(1, len(y[y==1]))

# Load Optuna params
optuna_path = base_path / 'data/processed/optuna_results.pkl'
if optuna_path.exists():
    with open(optuna_path, 'rb') as f:
        optuna_data = pickle.load(f)
    best_params = optuna_data.get('best_params', {})
    print("   Loaded Optuna params", flush=True)
else:
    best_params = {}
    print("   Using default params", flush=True)

# Default params
xgb_params = best_params.get('xgb', {
    'max_depth': 5, 'learning_rate': 0.03, 'n_estimators': 500,
    'min_child_weight': 3, 'subsample': 0.85, 'colsample_bytree': 0.85,
    'gamma': 0.1, 'reg_alpha': 0.3, 'reg_lambda': 1.5
})
lgb_params = best_params.get('lgb', {
    'max_depth': 6, 'learning_rate': 0.03, 'n_estimators': 500,
    'num_leaves': 31, 'min_child_samples': 25, 'subsample': 0.85,
    'colsample_bytree': 0.85, 'reg_alpha': 0.3, 'reg_lambda': 1.5
})
cat_params = best_params.get('cat', {
    'depth': 6, 'learning_rate': 0.03, 'iterations': 500,
    'l2_leaf_reg': 3.0
})

oof_xgb = np.zeros(len(y))
oof_lgb = np.zeros(len(y))
oof_cat = np.zeros(len(y))

test_preds_xgb = np.zeros((len(test_ids), n_splits))
test_preds_lgb = np.zeros((len(test_ids), n_splits))
test_preds_cat = np.zeros((len(test_ids), n_splits))

models = {'xgb': [], 'lgb': [], 'cat': []}

for fold, (train_idx, val_idx) in enumerate(cv.split(X_all, y)):
    X_tr, X_val = X_all[train_idx], X_all[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    # XGBoost
    xgb_model = xgb.XGBClassifier(
        **xgb_params,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_idx] = xgb_model.predict_proba(X_val)[:, 1]
    test_preds_xgb[:, fold] = xgb_model.predict_proba(X_test_all)[:, 1]
    models['xgb'].append(xgb_model)

    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        **lgb_params,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    oof_lgb[val_idx] = lgb_model.predict_proba(X_val)[:, 1]
    test_preds_lgb[:, fold] = lgb_model.predict_proba(X_test_all)[:, 1]
    models['lgb'].append(lgb_model)

    # CatBoost
    cat_model = CatBoostClassifier(
        **cat_params,
        auto_class_weights='Balanced',
        random_state=42,
        verbose=False
    )
    cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)
    oof_cat[val_idx] = cat_model.predict_proba(X_val)[:, 1]
    test_preds_cat[:, fold] = cat_model.predict_proba(X_test_all)[:, 1]
    models['cat'].append(cat_model)

    fold_preds = (0.33*oof_xgb[val_idx] + 0.33*oof_lgb[val_idx] + 0.34*oof_cat[val_idx] > 0.3).astype(int)
    fold_f1 = f1_score(y_val, fold_preds)
    print(f"   Fold {fold+1}: F1={fold_f1:.4f}", flush=True)

# ====================
# 8. OPTIMIZE WEIGHTS
# ====================
print("\n8. Optimizing ensemble weights...", flush=True)

best_f1 = 0
best_weights = (0.33, 0.33, 0.34)
best_thresh = 0.3

for w1 in np.arange(0.1, 0.6, 0.05):
    for w2 in np.arange(0.1, 0.6, 0.05):
        w3 = 1 - w1 - w2
        if w3 < 0.1 or w3 > 0.6:
            continue

        oof_blend = w1 * oof_xgb + w2 * oof_lgb + w3 * oof_cat

        for thresh in np.arange(0.05, 0.6, 0.02):
            preds = (oof_blend > thresh).astype(int)
            f1 = f1_score(y, preds)

            if f1 > best_f1:
                best_f1 = f1
                best_weights = (w1, w2, w3)
                best_thresh = thresh

print(f"   Best weights: XGB={best_weights[0]:.2f}, LGB={best_weights[1]:.2f}, CAT={best_weights[2]:.2f}", flush=True)
print(f"   OOF F1: {best_f1:.4f} @ threshold={best_thresh:.2f}", flush=True)

# Individual scores
xgb_best = max([f1_score(y, (oof_xgb > t).astype(int)) for t in np.arange(0.05, 0.6, 0.01)])
lgb_best = max([f1_score(y, (oof_lgb > t).astype(int)) for t in np.arange(0.05, 0.6, 0.01)])
cat_best = max([f1_score(y, (oof_cat > t).astype(int)) for t in np.arange(0.05, 0.6, 0.01)])
print(f"\n   Individual OOF F1: XGB={xgb_best:.4f}, LGB={lgb_best:.4f}, CAT={cat_best:.4f}", flush=True)

# Confusion matrix
oof_blend = best_weights[0] * oof_xgb + best_weights[1] * oof_lgb + best_weights[2] * oof_cat
final_preds = (oof_blend > best_thresh).astype(int)

tp = np.sum((final_preds == 1) & (y == 1))
fp = np.sum((final_preds == 1) & (y == 0))
fn = np.sum((final_preds == 0) & (y == 1))
tn = np.sum((final_preds == 0) & (y == 0))

print(f"\n   Confusion: TP={tp}, FP={fp}, FN={fn}, TN={tn}", flush=True)
print(f"   Precision: {precision_score(y, final_preds):.4f}", flush=True)
print(f"   Recall: {recall_score(y, final_preds):.4f}", flush=True)

# ====================
# 9. FEATURE IMPORTANCE
# ====================
print("\n9. Top features by importance...", flush=True)

importance = np.zeros(len(all_feature_cols))
for model in models['xgb']:
    importance += model.feature_importances_
for model in models['lgb']:
    importance += model.feature_importances_
for model in models['cat']:
    importance += model.feature_importances_

importance /= (3 * n_splits)

importance_df = pd.DataFrame({
    'feature': all_feature_cols,
    'importance': importance
}).sort_values('importance', ascending=False)

def categorize(f):
    if f in gp2d_cols:
        return 'GP2D'
    elif f in available_adv:
        return 'ADV'
    elif f in tde_cols:
        return 'TDE'
    else:
        return 'BASE'

importance_df['group'] = importance_df['feature'].apply(categorize)

print("\n   Top 20 features:", flush=True)
for i, (_, row) in enumerate(importance_df.head(20).iterrows()):
    print(f"     {row['feature']}: {row['importance']:.0f} [{row['group']}]", flush=True)

# ====================
# 10. CREATE SUBMISSION
# ====================
print("\n10. Creating submission...", flush=True)

test_blend = (
    best_weights[0] * test_preds_xgb.mean(axis=1) +
    best_weights[1] * test_preds_lgb.mean(axis=1) +
    best_weights[2] * test_preds_cat.mean(axis=1)
)

test_preds_final = (test_blend > best_thresh).astype(int)

submission = pd.DataFrame({
    'object_id': test_ids,
    'target': test_preds_final
})

submission_path = base_path / 'submissions/submission_v20b_selective.csv'
submission.to_csv(submission_path, index=False)

print(f"   Saved to {submission_path}", flush=True)
print(f"   Predictions: {test_preds_final.sum()} TDEs / {len(test_preds_final)} total ({100*test_preds_final.mean():.1f}%)", flush=True)

# Save
with open(base_path / 'data/processed/models_v20b.pkl', 'wb') as f:
    pickle.dump({
        'models': models,
        'best_weights': best_weights,
        'best_thresh': best_thresh,
        'feature_cols': all_feature_cols,
        'importance_df': importance_df
    }, f)

# ====================
# SUMMARY
# ====================
print("\n" + "=" * 60, flush=True)
print("TRAINING COMPLETE!", flush=True)
print("=" * 60, flush=True)

print(f"\nVersion Comparison:", flush=True)
print(f"  v8 (Baseline):         OOF F1 = 0.6262, LB = 0.6481", flush=True)
print(f"  v19 (Multi-band GP):   OOF F1 = 0.6626, LB = 0.6649 (Rank 23)", flush=True)
print(f"  v20 (All features):    OOF F1 = 0.6432", flush=True)
print(f"  v20b (Selective):      OOF F1 = {best_f1:.4f}", flush=True)

improvement = (best_f1 - 0.6626) / 0.6626 * 100
if improvement > 0:
    print(f"\n  Improvement over v19: +{improvement:.2f}%", flush=True)
else:
    print(f"\n  Difference from v19: {improvement:.2f}%", flush=True)

print(f"\nTotal features: {len(all_feature_cols)}", flush=True)
print("=" * 60, flush=True)

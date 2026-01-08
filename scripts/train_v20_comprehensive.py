"""
MALLORN v20: Comprehensive Feature Engineering + Benchmarking

This version:
1. Adds advanced features: MHPS, FLEET, absolute magnitude, pre-peak colors
2. Combines with v19 multi-band GP features
3. Benchmarks each feature group independently
4. Trains optimized ensemble with all features

Feature Groups:
- BASE: v8 120 statistical + lightcurve shape features
- TDE: 25 TDE-specific physics features
- GP2D: 27 multi-band GP features (from v19)
- ADV: ~50 new advanced features (MHPS, FLEET, abs mag, etc.)
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

# Unbuffered output
sys.stdout.reconfigure(line_buffering=True)

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

base_path = Path(__file__).parent.parent

print("=" * 60, flush=True)
print("MALLORN v20: Comprehensive Features + Benchmarking", flush=True)
print("=" * 60, flush=True)

# ====================
# 1. LOAD DATA
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
print(f"   Train: {len(train_ids)} objects ({y.sum()} TDE, {len(y)-y.sum()} non-TDE)", flush=True)
print(f"   Test: {len(test_ids)} objects", flush=True)

# ====================
# 2. LOAD BASE FEATURES (from v4 cache + feature selection)
# ====================
print("\n2. Loading base features...", flush=True)

# Load v4 cached features (used by v8, v19)
cached = pd.read_pickle(base_path / 'data/processed/features_v4_cache.pkl')
train_base = cached['train_features']
test_base = cached['test_features']

# Load feature selection results
selection = pd.read_pickle(base_path / 'data/processed/selected_features.pkl')
importance_df = selection['importance_df']
high_corr_df = selection['high_corr_df']

# Get top 120 non-correlated features
corr_to_drop = set()
for _, row in high_corr_df.iterrows():
    if row['feature_1'] not in corr_to_drop:
        corr_to_drop.add(row['feature_2'])
clean_features = importance_df[~importance_df['feature'].isin(corr_to_drop)]
selected_120 = clean_features.head(120)['feature'].tolist()

# Load TDE physics features
tde_cached = pd.read_pickle(base_path / 'data/processed/tde_physics_cache.pkl')
train_tde = tde_cached['train']
test_tde = tde_cached['test']
tde_cols = [c for c in train_tde.columns if c != 'object_id']

# Merge TDE features into base
train_base = train_base.merge(train_tde, on='object_id', how='left')
test_base = test_base.merge(test_tde, on='object_id', how='left')

print(f"   Loaded base: {len(selected_120)} + {len(tde_cols)} TDE physics", flush=True)

# ====================
# 3. LOAD v19 MULTI-BAND GP FEATURES
# ====================
print("\n3. Loading multi-band GP features...", flush=True)

gp2d_cache = base_path / 'data/processed/multiband_gp_cache.pkl'
if gp2d_cache.exists():
    with open(gp2d_cache, 'rb') as f:
        gp2d_data = pickle.load(f)
    train_gp2d = gp2d_data['train']
    test_gp2d = gp2d_data['test']
    gp2d_cols = [c for c in train_gp2d.columns if c != 'object_id']
    print(f"   Loaded GP2D features: {len(gp2d_cols)}", flush=True)
else:
    print("   Computing multi-band GP features...", flush=True)
    from features.multiband_gp import extract_multiband_gp_features

    train_gp2d = extract_multiband_gp_features(train_lc, train_meta, train_ids, verbose=True)
    test_gp2d = extract_multiband_gp_features(test_lc, test_meta, test_ids, verbose=True)

    gp2d_cols = [c for c in train_gp2d.columns if c != 'object_id']

    with open(gp2d_cache, 'wb') as f:
        pickle.dump({'train': train_gp2d, 'test': test_gp2d}, f)
    print(f"   Computed GP2D features: {len(gp2d_cols)}", flush=True)

# ====================
# 4. COMPUTE ADVANCED FEATURES
# ====================
print("\n4. Computing advanced features (MHPS, FLEET, abs mag)...", flush=True)

adv_cache = base_path / 'data/processed/advanced_features_cache.pkl'
if adv_cache.exists():
    with open(adv_cache, 'rb') as f:
        adv_data = pickle.load(f)
    train_adv = adv_data['train']
    test_adv = adv_data['test']
    adv_cols = [c for c in train_adv.columns if c != 'object_id']
    print(f"   Loaded advanced features: {len(adv_cols)}", flush=True)
else:
    from features.advanced_features import extract_advanced_features

    print("   Computing train advanced features...", flush=True)
    train_adv = extract_advanced_features(train_lc, train_meta, train_ids, verbose=True)
    adv_cols = [c for c in train_adv.columns if c != 'object_id']
    print(f"   Train advanced: {len(adv_cols)} features", flush=True)

    print("   Computing test advanced features...", flush=True)
    test_adv = extract_advanced_features(test_lc, test_meta, test_ids, verbose=True)
    print(f"   Test advanced: {len([c for c in test_adv.columns if c != 'object_id'])} features", flush=True)

    with open(adv_cache, 'wb') as f:
        pickle.dump({'train': train_adv, 'test': test_adv}, f)

# ====================
# 5. COMBINE ALL FEATURES
# ====================
print("\n5. Combining all features...", flush=True)

# Start with base features
train_combined = train_base.copy()
test_combined = test_base.copy()

# Add GP2D features
train_combined = train_combined.merge(train_gp2d, on='object_id', how='left')
test_combined = test_combined.merge(test_gp2d, on='object_id', how='left')

# Add advanced features
train_combined = train_combined.merge(train_adv, on='object_id', how='left')
test_combined = test_combined.merge(test_adv, on='object_id', how='left')

# Identify feature groups
base_cols = [c for c in selected_120 if c in train_combined.columns]
tde_cols = [c for c in train_combined.columns if any(x in c for x in ['late_slope', 'late_flux', 'rebrightening', 'color_var', 'color_range', 'color_trend', 'temp_stability', 'temp_trend', 'decay_alpha', 'rise_shape', 'rise_rate'])]
tde_cols = [c for c in tde_cols if c not in base_cols]

all_feature_cols = [c for c in train_combined.columns if c not in ['object_id', 'target']]

print(f"   Feature counts:", flush=True)
print(f"     BASE: {len(base_cols)}", flush=True)
print(f"     TDE: {len(tde_cols)}", flush=True)
print(f"     GP2D: {len(gp2d_cols)}", flush=True)
print(f"     ADV: {len(adv_cols)}", flush=True)
print(f"     TOTAL: {len(all_feature_cols)}", flush=True)

# Fill NaN and prepare data
train_combined = train_combined.fillna(-999)
test_combined = test_combined.fillna(-999)

# Ensure order matches
train_combined = train_combined.set_index('object_id').loc[train_ids].reset_index()

X_all = train_combined[all_feature_cols].values.astype(np.float32)
X_test_all = test_combined[all_feature_cols].values.astype(np.float32)

# ====================
# 6. BENCHMARK FEATURE GROUPS
# ====================
print("\n6. Benchmarking feature groups...", flush=True)

def benchmark_features(feature_cols, X_full, y, cv, name):
    """Quick benchmark using single LightGBM model."""
    # Get indices of these features
    all_cols = all_feature_cols
    indices = [all_cols.index(c) for c in feature_cols if c in all_cols]

    if not indices:
        return np.nan

    X = X_full[:, indices]

    oof_preds = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=len(y_tr[y_tr==0])/max(1, len(y_tr[y_tr==1])),
            random_state=42,
            verbose=-1
        )

        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

    # Find best threshold
    best_f1 = 0
    for thresh in np.arange(0.05, 0.95, 0.01):
        preds = (oof_preds > thresh).astype(int)
        f1 = f1_score(y, preds)
        if f1 > best_f1:
            best_f1 = f1

    return best_f1

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("   Running feature group benchmarks...", flush=True)

# Benchmark individual groups
groups = {
    'BASE only': base_cols,
    'GP2D only': gp2d_cols,
    'ADV only': adv_cols,
    'BASE + GP2D': base_cols + gp2d_cols,
    'BASE + ADV': base_cols + adv_cols,
    'GP2D + ADV': gp2d_cols + adv_cols,
    'ALL': all_feature_cols
}

benchmark_results = {}
for name, cols in groups.items():
    valid_cols = [c for c in cols if c in all_feature_cols]
    f1 = benchmark_features(valid_cols, X_all, y, cv, name)
    benchmark_results[name] = f1
    print(f"     {name}: OOF F1 = {f1:.4f} ({len(valid_cols)} features)", flush=True)

# ====================
# 7. LOAD OPTUNA HYPERPARAMETERS
# ====================
print("\n7. Loading Optuna-tuned hyperparameters...", flush=True)

optuna_path = base_path / 'data/processed/optuna_best_params.pkl'
if optuna_path.exists():
    with open(optuna_path, 'rb') as f:
        best_params = pickle.load(f)
    print("   Loaded Optuna params", flush=True)
else:
    best_params = {
        'xgb': {
            'max_depth': 5, 'learning_rate': 0.03, 'n_estimators': 500,
            'min_child_weight': 3, 'subsample': 0.85, 'colsample_bytree': 0.85,
            'gamma': 0.1, 'reg_alpha': 0.3, 'reg_lambda': 1.5
        },
        'lgb': {
            'max_depth': 6, 'learning_rate': 0.03, 'n_estimators': 500,
            'num_leaves': 31, 'min_child_samples': 25, 'subsample': 0.85,
            'colsample_bytree': 0.85, 'reg_alpha': 0.3, 'reg_lambda': 1.5
        },
        'cat': {
            'depth': 6, 'learning_rate': 0.03, 'iterations': 500,
            'l2_leaf_reg': 3.0
        }
    }
    print("   Using default params", flush=True)

# ====================
# 8. TRAIN 3-MODEL ENSEMBLE
# ====================
print("\n8. Training 3-model ensemble with ALL features...", flush=True)

n_splits = 5
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

scale_pos_weight = len(y[y==0]) / max(1, len(y[y==1]))

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
        **best_params['xgb'],
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
        **best_params['lgb'],
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
        **best_params['cat'],
        auto_class_weights='Balanced',
        random_state=42,
        verbose=False
    )
    cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)
    oof_cat[val_idx] = cat_model.predict_proba(X_val)[:, 1]
    test_preds_cat[:, fold] = cat_model.predict_proba(X_test_all)[:, 1]
    models['cat'].append(cat_model)

    # Fold F1
    fold_preds = (0.33*oof_xgb[val_idx] + 0.33*oof_lgb[val_idx] + 0.34*oof_cat[val_idx] > 0.3).astype(int)
    fold_f1 = f1_score(y_val, fold_preds)
    print(f"   Fold {fold+1}: F1={fold_f1:.4f}", flush=True)

# ====================
# 9. OPTIMIZE ENSEMBLE WEIGHTS
# ====================
print("\n9. Optimizing ensemble weights...", flush=True)

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

# Individual model OOF scores
for thresh_search in np.arange(0.05, 0.6, 0.01):
    xgb_f1 = f1_score(y, (oof_xgb > thresh_search).astype(int))
    lgb_f1 = f1_score(y, (oof_lgb > thresh_search).astype(int))
    cat_f1 = f1_score(y, (oof_cat > thresh_search).astype(int))

# Find individual best thresholds
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
# 10. FEATURE IMPORTANCE BY GROUP
# ====================
print("\n10. Analyzing feature importance by group...", flush=True)

# Aggregate importance from all models
importance = np.zeros(len(all_feature_cols))
for model in models['xgb']:
    importance += model.feature_importances_
for model in models['lgb']:
    importance += model.feature_importances_
for model in models['cat']:
    importance += model.feature_importances_

importance /= (3 * n_splits)

# Create importance DataFrame
importance_df = pd.DataFrame({
    'feature': all_feature_cols,
    'importance': importance
}).sort_values('importance', ascending=False)

# Categorize features
def categorize_feature(f):
    if f in gp2d_cols:
        return 'GP2D'
    elif f in adv_cols:
        return 'ADV'
    elif f in tde_cols:
        return 'TDE'
    else:
        return 'BASE'

importance_df['group'] = importance_df['feature'].apply(categorize_feature)

# Group importance
group_importance = importance_df.groupby('group')['importance'].sum()
group_importance = group_importance / group_importance.sum() * 100

print("\n   Feature importance by group:", flush=True)
for group, pct in group_importance.sort_values(ascending=False).items():
    print(f"     {group}: {pct:.1f}%", flush=True)

print("\n   Top 30 features:", flush=True)
for i, (_, row) in enumerate(importance_df.head(30).iterrows()):
    print(f"     {row['feature']}: {row['importance']:.0f} [{row['group']}]", flush=True)

# ====================
# 11. CREATE SUBMISSION
# ====================
print("\n11. Creating submission...", flush=True)

# Average test predictions across folds
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

submission_path = base_path / 'submissions/submission_v20_comprehensive.csv'
submission.to_csv(submission_path, index=False)

print(f"   Saved to {submission_path}", flush=True)
print(f"   Predictions: {test_preds_final.sum()} TDEs / {len(test_preds_final)} total ({100*test_preds_final.mean():.1f}%)", flush=True)

# Save models and features
models_path = base_path / 'data/processed/models_v20.pkl'
with open(models_path, 'wb') as f:
    pickle.dump({
        'models': models,
        'best_weights': best_weights,
        'best_thresh': best_thresh,
        'feature_cols': all_feature_cols,
        'importance_df': importance_df
    }, f)
print(f"   Models saved to {models_path}", flush=True)

# ====================
# SUMMARY
# ====================
print("\n" + "=" * 60, flush=True)
print("TRAINING COMPLETE!", flush=True)
print("=" * 60, flush=True)

print("\nBenchmark Summary:", flush=True)
for name, f1 in benchmark_results.items():
    marker = " <-- BEST" if name == 'ALL' else ""
    print(f"  {name}: {f1:.4f}{marker}", flush=True)

print(f"\nVersion Comparison:", flush=True)
print(f"  v8 (Baseline):        OOF F1 = 0.6262, LB = 0.6481", flush=True)
print(f"  v19 (Multi-band GP):  OOF F1 = 0.6626, LB = 0.6649 (Rank 23/496)", flush=True)
print(f"  v20 (Comprehensive):  OOF F1 = {best_f1:.4f}", flush=True)

print(f"\nTotal features: {len(all_feature_cols)}", flush=True)
print(f"  BASE: {len(base_cols)}", flush=True)
print(f"  TDE: {len(tde_cols)}", flush=True)
print(f"  GP2D: {len(gp2d_cols)}", flush=True)
print(f"  ADV: {len(adv_cols)}", flush=True)
print("=" * 60, flush=True)

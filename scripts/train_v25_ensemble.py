"""
MALLORN v25: Advanced Ensembling Strategy

Combines multiple models with optimized weights and stacking:
1. XGBoost (Optuna-tuned from v20c)
2. LightGBM (Optuna-tuned)
3. CatBoost (Optuna-tuned)
4. Stacking meta-learner
5. Rank averaging

Target: Beat v21's 0.6708 OOF / 0.6649 LB
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
base_path = Path(__file__).parent.parent

print("=" * 60, flush=True)
print("MALLORN v25: Advanced Ensemble Strategy", flush=True)
print("=" * 60, flush=True)

# ====================
# 1. LOAD DATA AND FEATURES
# ====================
print("\n1. Loading data and features...", flush=True)

from utils.data_loader import load_all_data
data = load_all_data()

train_meta = data['train_meta']
test_meta = data['test_meta']

train_ids = train_meta['object_id'].tolist()
test_ids = test_meta['object_id'].tolist()
y = train_meta['target'].values

# Load cached features
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

# Select columns
base_cols = [c for c in selected_120 if c in train_combined.columns]
all_feature_cols = base_cols + gp2d_cols
all_feature_cols = list(dict.fromkeys(all_feature_cols))
all_feature_cols = [c for c in all_feature_cols if c in train_combined.columns]

train_combined = train_combined.set_index('object_id').loc[train_ids].reset_index()
test_combined = test_combined.set_index('object_id').loc[test_ids].reset_index()

X = train_combined[all_feature_cols].values.astype(np.float32)
X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
X_test = test_combined[all_feature_cols].values.astype(np.float32)
X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

n_neg, n_pos = np.sum(y == 0), np.sum(y == 1)
scale_pos_weight = n_neg / n_pos

print(f"   Features: {len(all_feature_cols)}", flush=True)
print(f"   Samples: {len(y)} ({n_pos} TDE, {n_neg} non-TDE)", flush=True)

# ====================
# 2. LOAD OPTUNA PARAMS
# ====================
print("\n2. Loading Optuna parameters...", flush=True)

with open(base_path / 'data/processed/optuna_v20c_results.pkl', 'rb') as f:
    optuna_data = pickle.load(f)

xgb_params = optuna_data['xgb_best_params']
lgb_params = optuna_data['lgb_best_params']
cat_params = optuna_data['cat_best_params']

print(f"   XGB params loaded", flush=True)
print(f"   LGB params loaded", flush=True)
print(f"   CAT params loaded", flush=True)

# ====================
# 3. TRAIN ALL MODELS (5-FOLD CV)
# ====================
print("\n3. Training all models (5-fold CV)...", flush=True)

n_splits = 5
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Storage for OOF predictions (for stacking)
oof_xgb = np.zeros(len(y))
oof_lgb = np.zeros(len(y))
oof_cat = np.zeros(len(y))

# Storage for test predictions
test_xgb = np.zeros((len(test_ids), n_splits))
test_lgb = np.zeros((len(test_ids), n_splits))
test_cat = np.zeros((len(test_ids), n_splits))

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    print(f"\n   Fold {fold+1}/{n_splits}:", flush=True)

    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    # --- XGBoost ---
    xgb_model = xgb.XGBClassifier(
        **xgb_params,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbosity=0
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_idx] = xgb_model.predict_proba(X_val)[:, 1]
    test_xgb[:, fold] = xgb_model.predict_proba(X_test)[:, 1]

    # --- LightGBM ---
    lgb_model = lgb.LGBMClassifier(
        **lgb_params,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbosity=-1
    )
    lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
    oof_lgb[val_idx] = lgb_model.predict_proba(X_val)[:, 1]
    test_lgb[:, fold] = lgb_model.predict_proba(X_test)[:, 1]

    # --- CatBoost ---
    cat_model = CatBoostClassifier(
        **cat_params,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbose=False
    )
    cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
    oof_cat[val_idx] = cat_model.predict_proba(X_val)[:, 1]
    test_cat[:, fold] = cat_model.predict_proba(X_test)[:, 1]

    # Print fold results
    xgb_f1 = max(f1_score(y_val, (oof_xgb[val_idx] > t).astype(int)) for t in np.arange(0.05, 0.5, 0.01))
    lgb_f1 = max(f1_score(y_val, (oof_lgb[val_idx] > t).astype(int)) for t in np.arange(0.05, 0.5, 0.01))
    cat_f1 = max(f1_score(y_val, (oof_cat[val_idx] > t).astype(int)) for t in np.arange(0.05, 0.5, 0.01))

    print(f"      XGB F1={xgb_f1:.4f}, LGB F1={lgb_f1:.4f}, CAT F1={cat_f1:.4f}", flush=True)

# Average test predictions across folds
test_xgb_avg = test_xgb.mean(axis=1)
test_lgb_avg = test_lgb.mean(axis=1)
test_cat_avg = test_cat.mean(axis=1)

# ====================
# 4. EVALUATE INDIVIDUAL MODELS
# ====================
print("\n4. Evaluating individual models (OOF)...", flush=True)

def find_best_f1(y_true, y_pred):
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.01, 0.5, 0.005):
        f1 = f1_score(y_true, (y_pred > t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_f1, best_t

xgb_f1, xgb_t = find_best_f1(y, oof_xgb)
lgb_f1, lgb_t = find_best_f1(y, oof_lgb)
cat_f1, cat_t = find_best_f1(y, oof_cat)

print(f"   XGBoost:   OOF F1 = {xgb_f1:.4f} @ t={xgb_t:.3f}", flush=True)
print(f"   LightGBM:  OOF F1 = {lgb_f1:.4f} @ t={lgb_t:.3f}", flush=True)
print(f"   CatBoost:  OOF F1 = {cat_f1:.4f} @ t={cat_t:.3f}", flush=True)

# ====================
# 5. ENSEMBLE STRATEGIES
# ====================
print("\n5. Testing ensemble strategies...", flush=True)

# Strategy A: Simple Average
oof_avg = (oof_xgb + oof_lgb + oof_cat) / 3
avg_f1, avg_t = find_best_f1(y, oof_avg)
print(f"   A. Simple Average:    OOF F1 = {avg_f1:.4f} @ t={avg_t:.3f}", flush=True)

# Strategy B: Weighted Average (based on individual F1 scores)
weights = np.array([xgb_f1, lgb_f1, cat_f1])
weights = weights / weights.sum()
oof_weighted = weights[0] * oof_xgb + weights[1] * oof_lgb + weights[2] * oof_cat
weighted_f1, weighted_t = find_best_f1(y, oof_weighted)
print(f"   B. Weighted Average:  OOF F1 = {weighted_f1:.4f} @ t={weighted_t:.3f} (w={weights.round(3)})", flush=True)

# Strategy C: Rank Average
from scipy.stats import rankdata
oof_rank = (rankdata(oof_xgb) + rankdata(oof_lgb) + rankdata(oof_cat)) / 3
oof_rank = oof_rank / len(oof_rank)  # Normalize to [0, 1]
rank_f1, rank_t = find_best_f1(y, oof_rank)
print(f"   C. Rank Average:      OOF F1 = {rank_f1:.4f} @ t={rank_t:.3f}", flush=True)

# Strategy D: Stacking with Logistic Regression
print("   D. Stacking (Logistic Regression)...", flush=True)
oof_stack = np.zeros(len(y))
test_stack = np.zeros(len(test_ids))

stack_features = np.column_stack([oof_xgb, oof_lgb, oof_cat])
test_stack_features = np.column_stack([test_xgb_avg, test_lgb_avg, test_cat_avg])

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    stack_tr = stack_features[train_idx]
    stack_val = stack_features[val_idx]
    y_tr = y[train_idx]

    meta_model = LogisticRegression(class_weight='balanced', max_iter=1000, C=0.1)
    meta_model.fit(stack_tr, y_tr)

    oof_stack[val_idx] = meta_model.predict_proba(stack_val)[:, 1]
    test_stack += meta_model.predict_proba(test_stack_features)[:, 1] / n_splits

stack_f1, stack_t = find_best_f1(y, oof_stack)
print(f"      Stacking:          OOF F1 = {stack_f1:.4f} @ t={stack_t:.3f}", flush=True)

# Strategy E: Optimized Weights (Grid Search)
print("   E. Grid search for optimal weights...", flush=True)
best_grid_f1 = 0
best_grid_weights = (1/3, 1/3, 1/3)

for w1 in np.arange(0.2, 0.6, 0.05):
    for w2 in np.arange(0.2, 0.6, 0.05):
        w3 = 1 - w1 - w2
        if w3 < 0.1 or w3 > 0.6:
            continue

        oof_grid = w1 * oof_xgb + w2 * oof_lgb + w3 * oof_cat
        grid_f1, _ = find_best_f1(y, oof_grid)

        if grid_f1 > best_grid_f1:
            best_grid_f1 = grid_f1
            best_grid_weights = (w1, w2, w3)

oof_optimal = best_grid_weights[0] * oof_xgb + best_grid_weights[1] * oof_lgb + best_grid_weights[2] * oof_cat
optimal_f1, optimal_t = find_best_f1(y, oof_optimal)
print(f"      Optimal Weights:   OOF F1 = {optimal_f1:.4f} @ t={optimal_t:.3f}", flush=True)
print(f"      Weights: XGB={best_grid_weights[0]:.2f}, LGB={best_grid_weights[1]:.2f}, CAT={best_grid_weights[2]:.2f}", flush=True)

# ====================
# 6. SELECT BEST STRATEGY
# ====================
print("\n6. Selecting best ensemble strategy...", flush=True)

strategies = {
    'xgb_only': (xgb_f1, oof_xgb, test_xgb_avg, xgb_t),
    'lgb_only': (lgb_f1, oof_lgb, test_lgb_avg, lgb_t),
    'cat_only': (cat_f1, oof_cat, test_cat_avg, cat_t),
    'simple_avg': (avg_f1, oof_avg, (test_xgb_avg + test_lgb_avg + test_cat_avg) / 3, avg_t),
    'weighted_avg': (weighted_f1, oof_weighted,
                     weights[0] * test_xgb_avg + weights[1] * test_lgb_avg + weights[2] * test_cat_avg, weighted_t),
    'rank_avg': (rank_f1, oof_rank,
                 (rankdata(test_xgb_avg) + rankdata(test_lgb_avg) + rankdata(test_cat_avg)) / 3 / len(test_ids), rank_t),
    'stacking': (stack_f1, oof_stack, test_stack, stack_t),
    'optimal_weights': (optimal_f1, oof_optimal,
                        best_grid_weights[0] * test_xgb_avg + best_grid_weights[1] * test_lgb_avg + best_grid_weights[2] * test_cat_avg,
                        optimal_t),
}

best_strategy = max(strategies.items(), key=lambda x: x[1][0])
print(f"   Best strategy: {best_strategy[0]} with OOF F1 = {best_strategy[1][0]:.4f}", flush=True)

best_name = best_strategy[0]
best_f1_final = best_strategy[1][0]
best_oof = best_strategy[1][1]
best_test = best_strategy[1][2]
best_thresh = best_strategy[1][3]

# ====================
# 7. CREATE SUBMISSIONS
# ====================
print("\n7. Creating submissions...", flush=True)

# Best ensemble submission
test_preds_best = (best_test > best_thresh).astype(int)
submission_best = pd.DataFrame({
    'object_id': test_ids,
    'target': test_preds_best
})
submission_best.to_csv(base_path / 'submissions/submission_v25_ensemble.csv', index=False)
print(f"   Best ensemble: {test_preds_best.sum()} TDEs ({best_name})", flush=True)

# Also save XGBoost-only for comparison (since it was best in v21)
test_preds_xgb = (test_xgb_avg > xgb_t).astype(int)
submission_xgb = pd.DataFrame({
    'object_id': test_ids,
    'target': test_preds_xgb
})
submission_xgb.to_csv(base_path / 'submissions/submission_v25_xgb.csv', index=False)
print(f"   XGBoost only: {test_preds_xgb.sum()} TDEs", flush=True)

# Save all predictions for future use
with open(base_path / 'data/processed/ensemble_v25_cache.pkl', 'wb') as f:
    pickle.dump({
        'oof_xgb': oof_xgb, 'oof_lgb': oof_lgb, 'oof_cat': oof_cat,
        'test_xgb': test_xgb_avg, 'test_lgb': test_lgb_avg, 'test_cat': test_cat_avg,
        'oof_stack': oof_stack, 'test_stack': test_stack,
        'best_strategy': best_name,
        'best_weights': best_grid_weights,
        'thresholds': {'xgb': xgb_t, 'lgb': lgb_t, 'cat': cat_t, 'best': best_thresh}
    }, f)

# ====================
# SUMMARY
# ====================
print("\n" + "=" * 60, flush=True)
print("v25 Advanced Ensemble Complete!", flush=True)
print("=" * 60, flush=True)

print(f"\nIndividual Model Results:", flush=True)
print(f"  XGBoost:   OOF F1 = {xgb_f1:.4f}", flush=True)
print(f"  LightGBM:  OOF F1 = {lgb_f1:.4f}", flush=True)
print(f"  CatBoost:  OOF F1 = {cat_f1:.4f}", flush=True)

print(f"\nEnsemble Results:", flush=True)
print(f"  Simple Average:    OOF F1 = {avg_f1:.4f}", flush=True)
print(f"  Weighted Average:  OOF F1 = {weighted_f1:.4f}", flush=True)
print(f"  Rank Average:      OOF F1 = {rank_f1:.4f}", flush=True)
print(f"  Stacking:          OOF F1 = {stack_f1:.4f}", flush=True)
print(f"  Optimal Weights:   OOF F1 = {optimal_f1:.4f}", flush=True)

print(f"\n  BEST: {best_name} = {best_f1_final:.4f}", flush=True)

print(f"\nComparison to previous best:", flush=True)
print(f"  v21 (XGB only):    OOF F1 = 0.6708, LB = 0.6649", flush=True)
print(f"  v25 ({best_name}): OOF F1 = {best_f1_final:.4f}", flush=True)

delta = best_f1_final - 0.6708
if delta > 0:
    print(f"\n  +{delta*100:.2f}% improvement!", flush=True)
else:
    print(f"\n  {delta*100:.2f}% vs v21", flush=True)

print("=" * 60, flush=True)

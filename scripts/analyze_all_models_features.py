"""
MALLORN: Feature Analysis for All Models
========================================

After discovering that CatBoost improved from 0.6289 to 0.6882 with
feature reduction (230 -> 75), let's check if XGBoost and LightGBM
can also benefit from similar optimization.

Models to analyze:
1. v92d XGBoost (best LB: 0.6986)
2. v34a XGBoost (LB: 0.6907)
3. v114d LightGBM (LB: 0.6797)
"""

import sys
import pickle
import gzip
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import xgboost as xgb
import lightgbm as lgb
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

base_path = Path(__file__).parent.parent

print("=" * 70)
print("MALLORN: Feature Analysis for All Models")
print("=" * 70)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/4] Loading data...")

with gzip.open(base_path / 'data/kaggle_ensemble_package.pkl.gz', 'rb') as f:
    package = pickle.load(f)

train_features = package['train_features']
y = package['y']
sample_weights = package['sample_weights']

# Load model artifacts
with open(base_path / 'data/processed/v92_focal_adv_artifacts.pkl', 'rb') as f:
    v92_arts = pickle.load(f)
with open(base_path / 'data/processed/v34a_artifacts.pkl', 'rb') as f:
    v34a_arts = pickle.load(f)
with open(base_path / 'data/processed/v114_optimized_artifacts.pkl', 'rb') as f:
    v114_arts = pickle.load(f)

scale_pos_weight = len(y[y == 0]) / len(y[y == 1])

CV_SEED = 42
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
folds = list(skf.split(train_features, y))

def find_best_f1(pred, y_true):
    best_f1, best_t = 0, 0.1
    for t in np.linspace(0.03, 0.6, 100):
        f1 = f1_score(y_true, (pred > t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_f1, best_t

# ============================================================================
# 2. ANALYZE v92d XGBoost
# ============================================================================
print("\n" + "=" * 70)
print("[2/4] Analyzing v92d XGBoost...")
print("=" * 70)

# v92d uses v34a features - filter to only those available in train_features
v92d_features_raw = v34a_arts['feature_names']
v92d_features = [f for f in v92d_features_raw if f in train_features.columns]
print(f"   Current features: {len(v92d_features)} (filtered from {len(v92d_features_raw)})")

# Train to get feature importance
X = train_features[v92d_features].values
X = np.nan_to_num(X, nan=0, posinf=1e10, neginf=-1e10)

xgb_params = {
    'objective': 'binary:logistic',
    'max_depth': 5,
    'learning_rate': 0.025,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'reg_alpha': 0.2,
    'reg_lambda': 1.5,
    'scale_pos_weight': scale_pos_weight,
    'tree_method': 'hist',
    'random_state': 42,
}

print("   Training XGBoost to extract feature importance...")
dtrain = xgb.DMatrix(X, label=y, weight=sample_weights, feature_names=v92d_features)
model = xgb.train(xgb_params, dtrain, num_boost_round=500)

importance = model.get_score(importance_type='gain')
importance_df_v92d = pd.DataFrame([
    {'feature': k, 'importance': v} for k, v in importance.items()
]).sort_values('importance', ascending=False).reset_index(drop=True)

# Add features with zero importance
used_features = set(importance_df_v92d['feature'])
zero_features = [f for f in v92d_features if f not in used_features]
for f in zero_features:
    importance_df_v92d = pd.concat([importance_df_v92d, pd.DataFrame([{'feature': f, 'importance': 0}])], ignore_index=True)

print(f"\n   Top 20 features:")
for i, row in importance_df_v92d.head(20).iterrows():
    print(f"      {i+1}. {row['feature']}: {row['importance']:.1f}")

print(f"\n   Zero importance features: {len(zero_features)}")

# Test feature reduction
def evaluate_xgb(feature_list, use_adv_weights=True):
    X_sub = train_features[feature_list].values
    X_sub = np.nan_to_num(X_sub, nan=0, posinf=1e10, neginf=-1e10)

    oof_preds = np.zeros(len(y))
    for train_idx, val_idx in folds:
        X_tr, X_val = X_sub[train_idx], X_sub[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        w_tr = sample_weights[train_idx] if use_adv_weights else None

        dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr, feature_names=feature_list)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_list)

        model = xgb.train(xgb_params, dtrain, num_boost_round=500,
                          evals=[(dval, 'val')], early_stopping_rounds=50,
                          verbose_eval=False)
        oof_preds[val_idx] = model.predict(dval, iteration_range=(0, model.best_iteration + 1))

    return find_best_f1(oof_preds, y)[0]

print("\n   Testing feature reduction for v92d XGBoost...")
v92d_results = {}

# Baseline
print("   [A] All features...", end=" ", flush=True)
f1_all = evaluate_xgb(v92d_features)
print(f"F1={f1_all:.4f}")
v92d_results['all'] = {'count': len(v92d_features), 'f1': f1_all}

# Top N features
for n in [150, 120, 100, 80, 60]:
    top_n = importance_df_v92d.head(n)['feature'].tolist()
    print(f"   [*] Top {n} features...", end=" ", flush=True)
    f1_n = evaluate_xgb(top_n)
    print(f"F1={f1_n:.4f} ({f1_n - f1_all:+.4f})")
    v92d_results[f'top_{n}'] = {'count': n, 'f1': f1_n}

# Find best
best_v92d = max(v92d_results.items(), key=lambda x: x[1]['f1'])
print(f"\n   Best for v92d: {best_v92d[0]} (F1={best_v92d[1]['f1']:.4f})")

# ============================================================================
# 3. ANALYZE v34a XGBoost (without adversarial weights)
# ============================================================================
print("\n" + "=" * 70)
print("[3/4] Analyzing v34a XGBoost (no adversarial weights)...")
print("=" * 70)

print("\n   Testing feature reduction for v34a XGBoost...")
v34a_results = {}

# Baseline
print("   [A] All features...", end=" ", flush=True)
f1_all = evaluate_xgb(v92d_features, use_adv_weights=False)
print(f"F1={f1_all:.4f}")
v34a_results['all'] = {'count': len(v92d_features), 'f1': f1_all}

# Top N features
for n in [150, 120, 100, 80, 60]:
    top_n = importance_df_v92d.head(n)['feature'].tolist()
    print(f"   [*] Top {n} features...", end=" ", flush=True)
    f1_n = evaluate_xgb(top_n, use_adv_weights=False)
    print(f"F1={f1_n:.4f} ({f1_n - f1_all:+.4f})")
    v34a_results[f'top_{n}'] = {'count': n, 'f1': f1_n}

best_v34a = max(v34a_results.items(), key=lambda x: x[1]['f1'])
print(f"\n   Best for v34a: {best_v34a[0]} (F1={best_v34a[1]['f1']:.4f})")

# ============================================================================
# 4. ANALYZE v114d LightGBM
# ============================================================================
print("\n" + "=" * 70)
print("[4/4] Analyzing v114d LightGBM...")
print("=" * 70)

# v114d uses base + minimal research features - filter to available
v114d_config = v114_arts['results']['v114d_minimal_research']['config']
v114d_features_raw = list(v114_arts['results']['v114d_minimal_research']['feature_names'])
v114d_features = [f for f in v114d_features_raw if f in train_features.columns]
print(f"   Current features: {len(v114d_features)} (filtered from {len(v114d_features_raw)})")

# Get LightGBM feature importance
X = train_features[v114d_features].values
X = np.nan_to_num(X, nan=0, posinf=1e10, neginf=-1e10)

lgb_params = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'num_leaves': 15,
    'max_depth': 5,
    'learning_rate': 0.025,
    'feature_fraction': 0.4,
    'bagging_fraction': 0.65,
    'bagging_freq': 5,
    'reg_alpha': 3.0,
    'reg_lambda': 5.0,
    'min_child_samples': 40,
    'scale_pos_weight': scale_pos_weight,
    'verbose': -1,
    'random_state': 42,
}

print("   Training LightGBM to extract feature importance...")
train_data = lgb.Dataset(X, label=y, weight=sample_weights, feature_name=v114d_features)
model = lgb.train(lgb_params, train_data, num_boost_round=600)

importance = model.feature_importance(importance_type='gain')
importance_df_v114d = pd.DataFrame({
    'feature': v114d_features,
    'importance': importance
}).sort_values('importance', ascending=False).reset_index(drop=True)

print(f"\n   Top 20 features:")
for i, row in importance_df_v114d.head(20).iterrows():
    print(f"      {i+1}. {row['feature']}: {row['importance']:.1f}")

zero_imp = importance_df_v114d[importance_df_v114d['importance'] < 1]['feature'].tolist()
print(f"\n   Near-zero importance features: {len(zero_imp)}")

# Test feature reduction
def evaluate_lgb(feature_list):
    X_sub = train_features[feature_list].values
    X_sub = np.nan_to_num(X_sub, nan=0, posinf=1e10, neginf=-1e10)

    oof_preds = np.zeros(len(y))
    for train_idx, val_idx in folds:
        X_tr, X_val = X_sub[train_idx], X_sub[val_idx]
        y_tr = y[train_idx]
        w_tr = sample_weights[train_idx]

        train_data = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, feature_name=feature_list)
        val_data = lgb.Dataset(X_val, label=y[val_idx], reference=train_data, feature_name=feature_list)

        model = lgb.train(lgb_params, train_data, num_boost_round=600,
                          valid_sets=[val_data],
                          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)

    return find_best_f1(oof_preds, y)[0]

print("\n   Testing feature reduction for v114d LightGBM...")
v114d_results = {}

# Baseline
print("   [A] All features...", end=" ", flush=True)
f1_all = evaluate_lgb(v114d_features)
print(f"F1={f1_all:.4f}")
v114d_results['all'] = {'count': len(v114d_features), 'f1': f1_all}

# Top N features
for n in [150, 120, 100, 80, 60]:
    if n <= len(v114d_features):
        top_n = importance_df_v114d.head(n)['feature'].tolist()
        print(f"   [*] Top {n} features...", end=" ", flush=True)
        f1_n = evaluate_lgb(top_n)
        print(f"F1={f1_n:.4f} ({f1_n - f1_all:+.4f})")
        v114d_results[f'top_{n}'] = {'count': n, 'f1': f1_n}

best_v114d = max(v114d_results.items(), key=lambda x: x[1]['f1'])
print(f"\n   Best for v114d: {best_v114d[0]} (F1={best_v114d[1]['f1']:.4f})")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("FEATURE ANALYSIS SUMMARY")
print("=" * 70)

print(f"""
Model Feature Optimization Results:

v92d XGBoost (with adversarial weights):
   Original: {v92d_results['all']['count']} features -> F1={v92d_results['all']['f1']:.4f}
   Best: {best_v92d[0]} ({best_v92d[1]['count']} features) -> F1={best_v92d[1]['f1']:.4f}
   Change: {best_v92d[1]['f1'] - v92d_results['all']['f1']:+.4f}

v34a XGBoost (no adversarial weights):
   Original: {v34a_results['all']['count']} features -> F1={v34a_results['all']['f1']:.4f}
   Best: {best_v34a[0]} ({best_v34a[1]['count']} features) -> F1={best_v34a[1]['f1']:.4f}
   Change: {best_v34a[1]['f1'] - v34a_results['all']['f1']:+.4f}

v114d LightGBM:
   Original: {v114d_results['all']['count']} features -> F1={v114d_results['all']['f1']:.4f}
   Best: {best_v114d[0]} ({best_v114d[1]['count']} features) -> F1={best_v114d[1]['f1']:.4f}
   Change: {best_v114d[1]['f1'] - v114d_results['all']['f1']:+.4f}

CatBoost (for comparison):
   Original: 230 features -> F1=0.6571
   Best: 75 features -> F1=0.6971
   Change: +0.0400
""")

# Save results
results = {
    'v92d': {'importance_df': importance_df_v92d, 'results': v92d_results, 'best': best_v92d},
    'v34a': {'results': v34a_results, 'best': best_v34a},
    'v114d': {'importance_df': importance_df_v114d, 'results': v114d_results, 'best': best_v114d},
}

with open(base_path / 'data/processed/all_models_feature_analysis.pkl', 'wb') as f:
    pickle.dump(results, f)

print("Saved analysis to data/processed/all_models_feature_analysis.pkl")

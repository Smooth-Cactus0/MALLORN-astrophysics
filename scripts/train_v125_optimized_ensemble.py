"""
MALLORN v125: Fully Optimized Ensemble
======================================

After feature analysis revealed ALL models benefit from feature reduction:
- v92d XGBoost: 223 -> 120 features (+0.0086 OOF)
- v34a XGBoost: 223 -> 80 features (+0.0107 OOF)
- v114d LightGBM: 227 -> 120 features (+0.0419 OOF)
- CatBoost: 230 -> 75 features (+0.0400 OOF)

This script trains all 4 models with optimized feature sets and creates
the best possible ensemble.
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
from catboost import CatBoostClassifier, Pool
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

base_path = Path(__file__).parent.parent

print("=" * 70)
print("MALLORN v125: Fully Optimized Ensemble")
print("=" * 70)

# ============================================================================
# 1. LOAD DATA AND FEATURE ANALYSIS
# ============================================================================
print("\n[1/6] Loading data and feature analysis results...")

with gzip.open(base_path / 'data/kaggle_ensemble_package.pkl.gz', 'rb') as f:
    package = pickle.load(f)

train_features = package['train_features']
test_features = package['test_features']
y = package['y']
test_ids = package['test_ids']
sample_weights = package['sample_weights']

# Load feature analysis results
with open(base_path / 'data/processed/all_models_feature_analysis.pkl', 'rb') as f:
    feature_analysis = pickle.load(f)

with open(base_path / 'data/processed/catboost_feature_analysis.pkl', 'rb') as f:
    cb_feature_analysis = pickle.load(f)

# Get optimal feature sets
xgb_importance_df = feature_analysis['v92d']['importance_df']
lgb_importance_df = feature_analysis['v114d']['importance_df']
cb_importance_df = cb_feature_analysis['importance_df']

# Top features for each model
v92d_features = xgb_importance_df.head(120)['feature'].tolist()
v34a_features = xgb_importance_df.head(80)['feature'].tolist()
v114d_features = lgb_importance_df.head(120)['feature'].tolist()
cb_features = cb_importance_df.head(75)['feature'].tolist()

print(f"   v92d XGBoost: {len(v92d_features)} features")
print(f"   v34a XGBoost: {len(v34a_features)} features")
print(f"   v114d LightGBM: {len(v114d_features)} features")
print(f"   CatBoost: {len(cb_features)} features")

scale_pos_weight = len(y[y == 0]) / len(y[y == 1])

# CV Setup
MODEL_SEEDS = [42, 123, 456, 789, 2024]
CV_SEED = 42
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
folds = list(skf.split(train_features, y))

def find_best_threshold(y_true, y_pred):
    best_f1, best_t = 0, 0.1
    for t in np.linspace(0.03, 0.7, 100):
        f1 = f1_score(y_true, (y_pred > t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1

# ============================================================================
# 2. TRAIN v92d XGBoost (with adversarial weights) - 120 features
# ============================================================================
print("\n" + "=" * 70)
print("[2/6] Training v92d XGBoost (120 features, 5 seeds)...")
print("=" * 70)

X_v92d = train_features[v92d_features].values
X_v92d_test = test_features[v92d_features].values
X_v92d = np.nan_to_num(X_v92d, nan=0, posinf=1e10, neginf=-1e10)
X_v92d_test = np.nan_to_num(X_v92d_test, nan=0, posinf=1e10, neginf=-1e10)

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
}

v92d_all_oof = []
v92d_all_test = []

for seed in MODEL_SEEDS:
    print(f"   Seed {seed}...", end=" ", flush=True)
    xgb_params['random_state'] = seed

    oof_preds = np.zeros(len(y))
    test_preds = np.zeros(len(X_v92d_test))

    for train_idx, val_idx in folds:
        dtrain = xgb.DMatrix(X_v92d[train_idx], label=y[train_idx],
                            weight=sample_weights[train_idx], feature_names=v92d_features)
        dval = xgb.DMatrix(X_v92d[val_idx], label=y[val_idx], feature_names=v92d_features)
        dtest = xgb.DMatrix(X_v92d_test, feature_names=v92d_features)

        model = xgb.train(xgb_params, dtrain, num_boost_round=500,
                         evals=[(dval, 'val')], early_stopping_rounds=50, verbose_eval=False)
        oof_preds[val_idx] = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
        test_preds += model.predict(dtest, iteration_range=(0, model.best_iteration + 1)) / N_FOLDS

    _, f1 = find_best_threshold(y, oof_preds)
    print(f"OOF F1={f1:.4f}")
    v92d_all_oof.append(oof_preds)
    v92d_all_test.append(test_preds)

v92d_oof = np.mean(v92d_all_oof, axis=0)
v92d_test = np.mean(v92d_all_test, axis=0)
_, v92d_f1 = find_best_threshold(y, v92d_oof)
print(f"   v92d 5-seed avg OOF F1: {v92d_f1:.4f}")

# ============================================================================
# 3. TRAIN v34a XGBoost (no adversarial weights) - 80 features
# ============================================================================
print("\n" + "=" * 70)
print("[3/6] Training v34a XGBoost (80 features, 5 seeds)...")
print("=" * 70)

X_v34a = train_features[v34a_features].values
X_v34a_test = test_features[v34a_features].values
X_v34a = np.nan_to_num(X_v34a, nan=0, posinf=1e10, neginf=-1e10)
X_v34a_test = np.nan_to_num(X_v34a_test, nan=0, posinf=1e10, neginf=-1e10)

v34a_all_oof = []
v34a_all_test = []

for seed in MODEL_SEEDS:
    print(f"   Seed {seed}...", end=" ", flush=True)
    xgb_params['random_state'] = seed

    oof_preds = np.zeros(len(y))
    test_preds = np.zeros(len(X_v34a_test))

    for train_idx, val_idx in folds:
        # No adversarial weights for v34a
        dtrain = xgb.DMatrix(X_v34a[train_idx], label=y[train_idx], feature_names=v34a_features)
        dval = xgb.DMatrix(X_v34a[val_idx], label=y[val_idx], feature_names=v34a_features)
        dtest = xgb.DMatrix(X_v34a_test, feature_names=v34a_features)

        model = xgb.train(xgb_params, dtrain, num_boost_round=500,
                         evals=[(dval, 'val')], early_stopping_rounds=50, verbose_eval=False)
        oof_preds[val_idx] = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
        test_preds += model.predict(dtest, iteration_range=(0, model.best_iteration + 1)) / N_FOLDS

    _, f1 = find_best_threshold(y, oof_preds)
    print(f"OOF F1={f1:.4f}")
    v34a_all_oof.append(oof_preds)
    v34a_all_test.append(test_preds)

v34a_oof = np.mean(v34a_all_oof, axis=0)
v34a_test = np.mean(v34a_all_test, axis=0)
_, v34a_f1 = find_best_threshold(y, v34a_oof)
print(f"   v34a 5-seed avg OOF F1: {v34a_f1:.4f}")

# ============================================================================
# 4. TRAIN v114d LightGBM - 120 features
# ============================================================================
print("\n" + "=" * 70)
print("[4/6] Training v114d LightGBM (120 features, 5 seeds)...")
print("=" * 70)

X_lgb = train_features[v114d_features].values
X_lgb_test = test_features[v114d_features].values
X_lgb = np.nan_to_num(X_lgb, nan=0, posinf=1e10, neginf=-1e10)
X_lgb_test = np.nan_to_num(X_lgb_test, nan=0, posinf=1e10, neginf=-1e10)

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
}

lgb_all_oof = []
lgb_all_test = []

for seed in MODEL_SEEDS:
    print(f"   Seed {seed}...", end=" ", flush=True)
    lgb_params['random_state'] = seed

    oof_preds = np.zeros(len(y))
    test_preds = np.zeros(len(X_lgb_test))

    for train_idx, val_idx in folds:
        train_data = lgb.Dataset(X_lgb[train_idx], label=y[train_idx],
                                weight=sample_weights[train_idx], feature_name=v114d_features)
        val_data = lgb.Dataset(X_lgb[val_idx], label=y[val_idx], reference=train_data, feature_name=v114d_features)

        model = lgb.train(lgb_params, train_data, num_boost_round=600,
                         valid_sets=[val_data],
                         callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        oof_preds[val_idx] = model.predict(X_lgb[val_idx], num_iteration=model.best_iteration)
        test_preds += model.predict(X_lgb_test, num_iteration=model.best_iteration) / N_FOLDS

    _, f1 = find_best_threshold(y, oof_preds)
    print(f"OOF F1={f1:.4f}")
    lgb_all_oof.append(oof_preds)
    lgb_all_test.append(test_preds)

lgb_oof = np.mean(lgb_all_oof, axis=0)
lgb_test = np.mean(lgb_all_test, axis=0)
_, lgb_f1 = find_best_threshold(y, lgb_oof)
print(f"   LightGBM 5-seed avg OOF F1: {lgb_f1:.4f}")

# ============================================================================
# 5. TRAIN CatBoost - 75 features
# ============================================================================
print("\n" + "=" * 70)
print("[5/6] Training CatBoost (75 features, 5 seeds)...")
print("=" * 70)

X_cb = train_features[cb_features].values
X_cb_test = test_features[cb_features].values
X_cb = np.nan_to_num(X_cb, nan=0, posinf=1e10, neginf=-1e10)
X_cb_test = np.nan_to_num(X_cb_test, nan=0, posinf=1e10, neginf=-1e10)

# Load best CatBoost params from v118
with open(base_path / 'data/processed/v118_catboost_artifacts.pkl', 'rb') as f:
    cb_arts = pickle.load(f)
cb_best_params = cb_arts['best_params']

cb_all_oof = []
cb_all_test = []

for seed in MODEL_SEEDS:
    print(f"   Seed {seed}...", end=" ", flush=True)

    cb_params = {
        'loss_function': 'Logloss',
        'eval_metric': 'Logloss',
        'random_seed': seed,
        'verbose': False,
        'allow_writing_files': False,
        'scale_pos_weight': scale_pos_weight,
        **cb_best_params
    }

    oof_preds = np.zeros(len(y))
    test_preds = np.zeros(len(X_cb_test))

    for train_idx, val_idx in folds:
        train_pool = Pool(X_cb[train_idx], y[train_idx], weight=sample_weights[train_idx], feature_names=cb_features)
        val_pool = Pool(X_cb[val_idx], y[val_idx], feature_names=cb_features)
        test_pool = Pool(X_cb_test, feature_names=cb_features)

        model = CatBoostClassifier(**cb_params)
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, verbose=False)

        oof_preds[val_idx] = model.predict_proba(X_cb[val_idx])[:, 1]
        test_preds += model.predict_proba(X_cb_test)[:, 1] / N_FOLDS

    _, f1 = find_best_threshold(y, oof_preds)
    print(f"OOF F1={f1:.4f}")
    cb_all_oof.append(oof_preds)
    cb_all_test.append(test_preds)

cb_oof = np.mean(cb_all_oof, axis=0)
cb_test = np.mean(cb_all_test, axis=0)
_, cb_f1 = find_best_threshold(y, cb_oof)
print(f"   CatBoost 5-seed avg OOF F1: {cb_f1:.4f}")

# ============================================================================
# 6. COMPUTE OPTIMAL ENSEMBLE
# ============================================================================
print("\n" + "=" * 70)
print("[6/6] Computing optimal ensemble weights...")
print("=" * 70)

# Grid search for optimal weights
best_f1 = 0
best_weights = None

for w_v92d in np.linspace(0, 0.6, 13):
    for w_v34a in np.linspace(0, 0.6, 13):
        for w_lgb in np.linspace(0, 0.5, 11):
            w_cb = 1.0 - w_v92d - w_v34a - w_lgb
            if w_cb < 0 or w_cb > 0.6:
                continue

            ensemble_oof = (w_v92d * v92d_oof +
                          w_v34a * v34a_oof +
                          w_lgb * lgb_oof +
                          w_cb * cb_oof)

            _, f1 = find_best_threshold(y, ensemble_oof)
            if f1 > best_f1:
                best_f1 = f1
                best_weights = (w_v92d, w_v34a, w_lgb, w_cb)

print(f"\n   Optimal weights found:")
print(f"   v92d XGBoost:  {best_weights[0]:.2f}")
print(f"   v34a XGBoost:  {best_weights[1]:.2f}")
print(f"   LightGBM:      {best_weights[2]:.2f}")
print(f"   CatBoost:      {best_weights[3]:.2f}")
print(f"   Ensemble OOF F1: {best_f1:.4f}")

# Create final predictions
final_oof = (best_weights[0] * v92d_oof +
            best_weights[1] * v34a_oof +
            best_weights[2] * lgb_oof +
            best_weights[3] * cb_oof)

final_test = (best_weights[0] * v92d_test +
             best_weights[1] * v34a_test +
             best_weights[2] * lgb_test +
             best_weights[3] * cb_test)

final_threshold, final_f1 = find_best_threshold(y, final_oof)

# Save artifacts
artifacts = {
    'model_name': 'v125_optimized_ensemble',
    'feature_sets': {
        'v92d': v92d_features,
        'v34a': v34a_features,
        'lgb': v114d_features,
        'catboost': cb_features,
    },
    'individual_results': {
        'v92d': {'oof': v92d_oof, 'test': v92d_test, 'f1': v92d_f1},
        'v34a': {'oof': v34a_oof, 'test': v34a_test, 'f1': v34a_f1},
        'lgb': {'oof': lgb_oof, 'test': lgb_test, 'f1': lgb_f1},
        'catboost': {'oof': cb_oof, 'test': cb_test, 'f1': cb_f1},
    },
    'ensemble': {
        'weights': best_weights,
        'oof': final_oof,
        'test': final_test,
        'threshold': final_threshold,
        'f1': final_f1,
    }
}

with open(base_path / 'data/processed/v125_optimized_ensemble_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

# Create submissions
print("\n   Creating submissions...")

# Main optimized ensemble
binary_preds = (final_test > final_threshold).astype(int)
submission = pd.DataFrame({
    'object_id': test_ids,
    'target': binary_preds
})
submission.to_csv(base_path / 'submissions/submission_v125_optimized_ensemble.csv', index=False)
print(f"   submission_v125_optimized_ensemble.csv: {binary_preds.sum()} TDEs")

# Also create individual model submissions
for name, data in [('v92d', (v92d_oof, v92d_test)),
                   ('v34a', (v34a_oof, v34a_test)),
                   ('lgb', (lgb_oof, lgb_test)),
                   ('catboost', (cb_oof, cb_test))]:
    threshold, _ = find_best_threshold(y, data[0])
    binary = (data[1] > threshold).astype(int)
    sub = pd.DataFrame({'object_id': test_ids, 'target': binary})
    sub.to_csv(base_path / f'submissions/submission_v125_{name}_optimized.csv', index=False)
    print(f"   submission_v125_{name}_optimized.csv: {binary.sum()} TDEs")

# Create equal-weight ensemble
equal_oof = (v92d_oof + v34a_oof + lgb_oof + cb_oof) / 4
equal_test = (v92d_test + v34a_test + lgb_test + cb_test) / 4
equal_threshold, equal_f1 = find_best_threshold(y, equal_oof)
equal_binary = (equal_test > equal_threshold).astype(int)
sub_equal = pd.DataFrame({'object_id': test_ids, 'target': equal_binary})
sub_equal.to_csv(base_path / 'submissions/submission_v125_equal_ensemble.csv', index=False)
print(f"   submission_v125_equal_ensemble.csv: {equal_binary.sum()} TDEs (Equal weights F1={equal_f1:.4f})")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("FULLY OPTIMIZED ENSEMBLE COMPLETE")
print("=" * 70)

print(f"""
Individual Model Performance (after feature optimization):
   v92d XGBoost (120 feat): OOF F1 = {v92d_f1:.4f}
   v34a XGBoost (80 feat):  OOF F1 = {v34a_f1:.4f}
   LightGBM (120 feat):     OOF F1 = {lgb_f1:.4f}
   CatBoost (75 feat):      OOF F1 = {cb_f1:.4f}

Optimal Ensemble:
   Weights: v92d={best_weights[0]:.2f}, v34a={best_weights[1]:.2f}, lgb={best_weights[2]:.2f}, cb={best_weights[3]:.2f}
   OOF F1: {final_f1:.4f}
   Threshold: {final_threshold:.3f}

Comparison to Previous Best:
   Previous v92d (223 feat): OOF 0.6707
   Optimized v92d (120 feat): OOF {v92d_f1:.4f} (+{v92d_f1-0.6707:.4f})

   Previous LightGBM (227 feat): OOF 0.6477
   Optimized LightGBM (120 feat): OOF {lgb_f1:.4f} (+{lgb_f1-0.6477:.4f})

   Previous CatBoost (230 feat): OOF 0.6289
   Optimized CatBoost (75 feat): OOF {cb_f1:.4f} (+{cb_f1-0.6289:.4f})

Files saved:
   - data/processed/v125_optimized_ensemble_artifacts.pkl
   - submissions/submission_v125_optimized_ensemble.csv
   - submissions/submission_v125_v92d_optimized.csv
   - submissions/submission_v125_v34a_optimized.csv
   - submissions/submission_v125_lgb_optimized.csv
   - submissions/submission_v125_catboost_optimized.csv
   - submissions/submission_v125_equal_ensemble.csv
""")

"""
MALLORN v79: Curated Physics-Based Features

Tests three feature sets focused on high-SNR physics features:
- Set A (Ultra-lean): 20 features - top 20 physics only
- Set B (Balanced): 40 features - 30 physics + 10 stats
- Set C (Extended): 70 features - 50 physics + 20 stats

Trains both XGBoost (v34a params) and LightGBM (v77 params) on each set.
"""

import sys
import pickle
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

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
base_path = Path(__file__).parent.parent

print("=" * 80)
print("MALLORN v79: Curated Physics-Based Features")
print("=" * 80)

# ====================
# 1. LOAD DATA
# ====================
print("\n1. Loading data...")

from utils.data_loader import load_all_data
data = load_all_data()

train_meta = data['train_meta']
test_meta = data['test_meta']
train_ids = train_meta['object_id'].tolist()
test_ids = test_meta['object_id'].tolist()
y = train_meta['target'].values

print(f"   Training: {len(train_ids)} objects ({np.sum(y)} TDE)")

# ====================
# 2. LOAD ALL FEATURES
# ====================
print("\n2. Loading all features...")

cached = pd.read_pickle(base_path / 'data/processed/features_v4_cache.pkl')
train_base = cached['train_features']
test_base = cached['test_features']

tde_cached = pd.read_pickle(base_path / 'data/processed/tde_physics_cache.pkl')
train_tde = tde_cached['train']
test_tde = tde_cached['test']

with open(base_path / 'data/processed/multiband_gp_cache.pkl', 'rb') as f:
    gp2d_data = pickle.load(f)
train_gp2d = gp2d_data['train']
test_gp2d = gp2d_data['test']

with open(base_path / 'data/processed/bazin_features_cache.pkl', 'rb') as f:
    bazin_cache = pickle.load(f)
train_bazin = bazin_cache['train']
test_bazin = bazin_cache['test']

# Merge all
train_all = train_base.merge(train_tde, on='object_id', how='left')
train_all = train_all.merge(train_gp2d, on='object_id', how='left')
train_all = train_all.merge(train_bazin, on='object_id', how='left')

test_all = test_base.merge(test_tde, on='object_id', how='left')
test_all = test_all.merge(test_gp2d, on='object_id', how='left')
test_all = test_all.merge(test_bazin, on='object_id', how='left')

print(f"   Total available features: {len(train_all.columns) - 1}")

# ====================
# 3. LOAD FEATURE SETS
# ====================
print("\n3. Loading curated feature sets...")

with open(base_path / 'data/processed/feature_sets.pkl', 'rb') as f:
    feature_sets = pickle.load(f)

# Filter to available features
for name, features in feature_sets.items():
    available = [f for f in features if f in train_all.columns]
    feature_sets[name] = available
    print(f"   {name}: {len(available)} features")

# ====================
# 4. LOAD MODEL PARAMETERS
# ====================
print("\n4. Loading optimized model parameters...")

with open(base_path / 'data/processed/v34a_artifacts.pkl', 'rb') as f:
    v34a = pickle.load(f)

with open(base_path / 'data/processed/v77_artifacts.pkl', 'rb') as f:
    v77 = pickle.load(f)

# XGBoost params from v34a
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

# LightGBM params from v77
lgb_params = v77['best_params'].copy()

print(f"   XGBoost: v34a parameters")
print(f"   LightGBM: v77 Optuna-tuned parameters")

# ====================
# 5. TRAINING FUNCTION
# ====================
def train_and_evaluate(X_train, X_test, y, feature_names, model_type='xgb'):
    """Train model and return OOF F1 and test predictions."""
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(y))
    test_preds = np.zeros((len(X_test), n_folds))
    fold_f1s = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y), 1):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        if model_type == 'xgb':
            dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_names)
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
            dtest = xgb.DMatrix(X_test, feature_names=feature_names)

            model = xgb.train(
                xgb_params,
                dtrain,
                num_boost_round=500,
                evals=[(dval, 'val')],
                early_stopping_rounds=50,
                verbose_eval=False
            )

            oof_preds[val_idx] = model.predict(dval)
            test_preds[:, fold-1] = model.predict(dtest)

        else:  # LightGBM
            train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_names)
            val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_data)

            model = lgb.train(
                lgb_params,
                train_data,
                valid_sets=[val_data],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(period=0)
                ]
            )

            oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
            test_preds[:, fold-1] = model.predict(X_test, num_iteration=model.best_iteration)

        # Fold F1
        best_f1 = 0
        for t in np.linspace(0.03, 0.5, 50):
            f1 = f1_score(y_val, (oof_preds[val_idx] > t).astype(int))
            if f1 > best_f1:
                best_f1 = f1
        fold_f1s.append(best_f1)

    # Overall OOF F1
    best_f1 = 0
    best_thresh = 0.1
    for t in np.linspace(0.03, 0.5, 200):
        f1 = f1_score(y, (oof_preds > t).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    return {
        'oof_f1': best_f1,
        'threshold': best_thresh,
        'fold_f1s': fold_f1s,
        'fold_std': np.std(fold_f1s),
        'oof_preds': oof_preds,
        'test_preds': test_preds.mean(axis=1)
    }

# ====================
# 6. TRAIN ALL COMBINATIONS
# ====================
print("\n" + "=" * 80)
print("TRAINING ALL COMBINATIONS")
print("=" * 80)

results = {}

for set_name, features in feature_sets.items():
    print(f"\n{'-' * 40}")
    print(f"Feature Set: {set_name} ({len(features)} features)")
    print(f"{'-' * 40}")

    # Prepare data
    X_train = train_all[features].values
    X_test = test_all[features].values

    X_train = np.nan_to_num(X_train, nan=np.nan, posinf=1e10, neginf=-1e10)
    X_test = np.nan_to_num(X_test, nan=np.nan, posinf=1e10, neginf=-1e10)

    # Train XGBoost
    print(f"\n   Training XGBoost...")
    xgb_result = train_and_evaluate(X_train, X_test, y, features, 'xgb')
    results[f'{set_name}_xgb'] = xgb_result
    print(f"   XGBoost OOF F1: {xgb_result['oof_f1']:.4f} (std={xgb_result['fold_std']:.4f})")

    # Train LightGBM
    print(f"\n   Training LightGBM...")
    lgb_result = train_and_evaluate(X_train, X_test, y, features, 'lgb')
    results[f'{set_name}_lgb'] = lgb_result
    print(f"   LightGBM OOF F1: {lgb_result['oof_f1']:.4f} (std={lgb_result['fold_std']:.4f})")

# ====================
# 7. RESULTS SUMMARY
# ====================
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

print(f"\n{'Model':<30} {'Features':<10} {'OOF F1':<10} {'Fold Std':<10} {'Threshold':<10}")
print("-" * 75)

# Add baseline reference
print(f"{'v34a XGBoost (baseline)':<30} {'224':<10} {'0.6667':<10} {'-':<10} {'0.386':<10}")
print(f"{'v77 LightGBM (baseline)':<30} {'223':<10} {'0.6886':<10} {'-':<10} {'0.290':<10}")
print("-" * 75)

sorted_results = sorted(results.items(), key=lambda x: x[1]['oof_f1'], reverse=True)

for name, res in sorted_results:
    set_name = name.rsplit('_', 1)[0]
    n_features = len(feature_sets[set_name])
    print(f"{name:<30} {n_features:<10} {res['oof_f1']:<10.4f} {res['fold_std']:<10.4f} {res['threshold']:<10.3f}")

# ====================
# 8. BEST MODEL SUBMISSION
# ====================
print("\n" + "=" * 80)
print("SUBMISSIONS")
print("=" * 80)

# Save top 3 submissions
for i, (name, res) in enumerate(sorted_results[:3], 1):
    test_binary = (res['test_preds'] > res['threshold']).astype(int)

    submission = pd.DataFrame({
        'object_id': test_ids,
        'target': test_binary
    })

    filename = f"submission_v79_{name}.csv"
    submission.to_csv(base_path / f'submissions/{filename}', index=False)
    print(f"   {i}. {filename}: OOF={res['oof_f1']:.4f}, TDEs={test_binary.sum()}")

# ====================
# 9. SAVE ARTIFACTS
# ====================
artifacts = {
    'results': results,
    'feature_sets': feature_sets,
    'sorted_results': [(name, res['oof_f1']) for name, res in sorted_results]
}

with open(base_path / 'data/processed/v79_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

# ====================
# 10. ANALYSIS
# ====================
print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

# Best per algorithm
best_xgb = max([(k, v) for k, v in results.items() if 'xgb' in k], key=lambda x: x[1]['oof_f1'])
best_lgb = max([(k, v) for k, v in results.items() if 'lgb' in k], key=lambda x: x[1]['oof_f1'])

print(f"\n   Best XGBoost: {best_xgb[0]} (OOF F1={best_xgb[1]['oof_f1']:.4f})")
print(f"   Best LightGBM: {best_lgb[0]} (OOF F1={best_lgb[1]['oof_f1']:.4f})")

# Compare to baselines
print(f"\n   vs v34a baseline (0.6667):")
for name, res in sorted_results:
    diff = res['oof_f1'] - 0.6667
    print(f"      {name}: {diff:+.4f}")

print("\n" + "=" * 80)
print("v79 Complete")
print("=" * 80)

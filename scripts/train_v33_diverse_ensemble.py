"""
MALLORN v33: Extremely Diverse Ensemble (Experiment 3)

Train 3 models with VERY different configurations:
1. XGBoost: Shallow trees (depth=3), high learning rate
2. LightGBM: Deep trees (depth=8), low learning rate
3. CatBoost: Ordered boosting, moderate depth

Diversity reduces correlated errors and improves generalization.

Expected improvement: +1-3% over v21
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

try:
    from catboost import CatBoostClassifier, Pool
except ImportError:
    print("Installing CatBoost...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost"])
    from catboost import CatBoostClassifier, Pool

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
base_path = Path(__file__).parent.parent

print("=" * 80, flush=True)
print("MALLORN v33: Extremely Diverse Ensemble", flush=True)
print("=" * 80, flush=True)

# ====================
# 1. LOAD v21 FEATURES
# ====================
print("\n1. Loading v21 feature set...", flush=True)

from utils.data_loader import load_all_data
data = load_all_data()

train_meta = data['train_meta']
test_meta = data['test_meta']

train_ids = train_meta['object_id'].tolist()
test_ids = test_meta['object_id'].tolist()
y = train_meta['target'].values

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
tde_cols = [c for c in train_tde.columns if c != 'object_id']

train_base = train_base.merge(train_tde, on='object_id', how='left')
test_base = test_base.merge(test_tde, on='object_id', how='left')

with open(base_path / 'data/processed/multiband_gp_cache.pkl', 'rb') as f:
    gp2d_data = pickle.load(f)
train_gp2d = gp2d_data['train']
test_gp2d = gp2d_data['test']
gp2d_cols = [c for c in train_gp2d.columns if c != 'object_id']

train_combined = train_base[['object_id'] + selected_120].copy()
train_combined = train_combined.merge(train_tde, on='object_id', how='left')
train_combined = train_combined.merge(train_gp2d[['object_id'] + gp2d_cols], on='object_id', how='left')

test_combined = test_base[['object_id'] + selected_120].copy()
test_combined = test_combined.merge(test_tde, on='object_id', how='left')
test_combined = test_combined.merge(test_gp2d[['object_id'] + gp2d_cols], on='object_id', how='left')

X_train = train_combined.drop(columns=['object_id']).values
X_test = test_combined.drop(columns=['object_id']).values
feature_names = [c for c in train_combined.columns if c != 'object_id']

print(f"   Features: {len(feature_names)}", flush=True)

# ====================
# 2. MODEL CONFIGURATIONS
# ====================
print("\n2. Configuring 3 diverse models...", flush=True)

# Model 1: XGBoost - Shallow & Fast
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 3,              # SHALLOW
    'learning_rate': 0.05,       # HIGH
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 5,       # Conservative
    'reg_alpha': 0.5,
    'reg_lambda': 2.0,
    'scale_pos_weight': len(y[y==0]) / len(y[y==1]),
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1
}

# Model 2: LightGBM - Deep & Slow
lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'max_depth': 8,              # DEEP
    'learning_rate': 0.01,       # LOW
    'num_leaves': 127,           # Many leaves
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'min_child_samples': 10,
    'reg_alpha': 0.1,
    'reg_lambda': 0.5,
    'scale_pos_weight': len(y[y==0]) / len(y[y==1]),
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

# Model 3: CatBoost - Ordered boosting
cat_params = {
    'iterations': 800,
    'learning_rate': 0.03,
    'depth': 6,                   # MODERATE
    'boosting_type': 'Ordered',   # UNIQUE to CatBoost
    'l2_leaf_reg': 3.0,
    'random_strength': 1.0,
    'subsample': 0.8,
    'rsm': 0.8,
    'auto_class_weights': 'Balanced',
    'task_type': 'CPU',
    'random_seed': 42,
    'verbose': False,
    'early_stopping_rounds': 50
}

print("   Model 1 (XGBoost): depth=3, lr=0.05 (shallow & fast)", flush=True)
print("   Model 2 (LightGBM): depth=8, lr=0.01 (deep & slow)", flush=True)
print("   Model 3 (CatBoost): depth=6, ordered boosting (diverse)", flush=True)

# ====================
# 3. TRAIN WITH 5-FOLD CV
# ====================
print("\n3. Training ensemble with 5-fold CV...", flush=True)

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

oof_xgb = np.zeros(len(X_train))
oof_lgb = np.zeros(len(X_train))
oof_cat = np.zeros(len(X_train))

test_xgb = np.zeros((len(X_test), n_folds))
test_lgb = np.zeros((len(X_test), n_folds))
test_cat = np.zeros((len(X_test), n_folds))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y), 1):
    print(f"\n   Fold {fold}/{n_folds}:", flush=True)

    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    # ===== XGBoost =====
    print("      Training XGBoost (shallow)...", end=" ", flush=True)
    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test, feature_names=feature_names)

    model_xgb = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=500,
        evals=[(dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=False
    )

    oof_xgb[val_idx] = model_xgb.predict(dval)
    test_xgb[:, fold-1] = model_xgb.predict(dtest)
    print(f"Done (iters={model_xgb.best_iteration})", flush=True)

    # ===== LightGBM =====
    print("      Training LightGBM (deep)...", end=" ", flush=True)
    train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_names)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=feature_names)

    model_lgb = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )

    oof_lgb[val_idx] = model_lgb.predict(X_val)
    test_lgb[:, fold-1] = model_lgb.predict(X_test)
    print(f"Done (iters={model_lgb.best_iteration})", flush=True)

    # ===== CatBoost =====
    print("      Training CatBoost (ordered)...", end=" ", flush=True)
    train_pool = Pool(X_tr, y_tr, feature_names=feature_names)
    val_pool = Pool(X_val, y_val, feature_names=feature_names)
    test_pool = Pool(X_test, feature_names=feature_names)

    model_cat = CatBoostClassifier(**cat_params)
    model_cat.fit(train_pool, eval_set=val_pool, use_best_model=True, plot=False)

    oof_cat[val_idx] = model_cat.predict_proba(val_pool)[:, 1]
    test_cat[:, fold-1] = model_cat.predict_proba(test_pool)[:, 1]
    print(f"Done (iters={model_cat.best_iteration_})", flush=True)

    # Fold evaluation
    oof_ensemble = (oof_xgb[val_idx] + oof_lgb[val_idx] + oof_cat[val_idx]) / 3
    best_f1_fold = 0
    for t in np.linspace(0.05, 0.5, 50):
        f1 = f1_score(y_val, (oof_ensemble > t).astype(int))
        if f1 > best_f1_fold:
            best_f1_fold = f1

    print(f"      Ensemble F1: {best_f1_fold:.4f}", flush=True)

print("\n" + "=" * 80, flush=True)
print("CROSS-VALIDATION RESULTS", flush=True)
print("=" * 80, flush=True)

# Individual model scores
for name, oof_preds in [('XGBoost', oof_xgb), ('LightGBM', oof_lgb), ('CatBoost', oof_cat)]:
    best_f1 = 0
    best_thresh = 0.5
    for t in np.linspace(0.05, 0.5, 100):
        f1 = f1_score(y, (oof_preds > t).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    print(f"   {name:12s}: F1={best_f1:.4f} @ threshold={best_thresh:.2f}", flush=True)

# Ensemble scores
print(f"\n   Ensemble Strategies:", flush=True)
oof_ensemble_avg = (oof_xgb + oof_lgb + oof_cat) / 3
best_f1_ensemble = 0
best_thresh_ensemble = 0.5

for t in np.linspace(0.05, 0.5, 100):
    f1 = f1_score(y, (oof_ensemble_avg > t).astype(int))
    if f1 > best_f1_ensemble:
        best_f1_ensemble = f1
        best_thresh_ensemble = t

print(f"   Simple Average: F1={best_f1_ensemble:.4f} @ threshold={best_thresh_ensemble:.2f}", flush=True)

final_preds = (oof_ensemble_avg > best_thresh_ensemble).astype(int)
tp = np.sum((final_preds == 1) & (y == 1))
fp = np.sum((final_preds == 1) & (y == 0))
fn = np.sum((final_preds == 0) & (y == 1))
tn = np.sum((final_preds == 0) & (y == 0))

print(f"   Confusion: TP={tp}, FP={fp}, FN={fn}, TN={tn}", flush=True)
print(f"   Precision: {tp/(tp+fp):.4f}", flush=True)
print(f"   Recall: {tp/(tp+fn):.4f}", flush=True)

# ====================
# 4. CREATE SUBMISSION
# ====================
print("\n4. Creating submission...", flush=True)

test_ensemble = (test_xgb.mean(axis=1) + test_lgb.mean(axis=1) + test_cat.mean(axis=1)) / 3
test_final = (test_ensemble > best_thresh_ensemble).astype(int)

submission = pd.DataFrame({
    'object_id': test_ids,
    'target': test_final
})

submission_path = base_path / 'submissions/submission_v33_diverse_ensemble.csv'
submission.to_csv(submission_path, index=False)

print(f"   Submission saved: {submission_path.name}", flush=True)
print(f"   Predicted TDEs: {test_final.sum()} / {len(test_final)}", flush=True)

# Save artifacts
artifacts = {
    'oof_xgb': oof_xgb,
    'oof_lgb': oof_lgb,
    'oof_cat': oof_cat,
    'test_xgb': test_xgb.mean(axis=1),
    'test_lgb': test_lgb.mean(axis=1),
    'test_cat': test_cat.mean(axis=1),
    'test_ensemble': test_ensemble,
    'best_threshold': best_thresh_ensemble,
    'oof_f1': best_f1_ensemble
}

with open(base_path / 'data/processed/v33_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print("\n" + "=" * 80, flush=True)
print(f"MALLORN v33 (Diverse Ensemble) Complete: OOF F1 = {best_f1_ensemble:.4f}", flush=True)
print(f"Baseline v21 (XGBoost only): OOF F1 = 0.6708", flush=True)
change_pct = (best_f1_ensemble - 0.6708) * 100
change_abs = best_f1_ensemble - 0.6708
print(f"Change: {change_pct:+.2f}% ({change_abs:+.4f})", flush=True)

if best_f1_ensemble > 0.6708:
    print("SUCCESS: Diverse ensemble improved performance!", flush=True)
else:
    print("Ensemble did not improve beyond baseline", flush=True)
print("=" * 80, flush=True)

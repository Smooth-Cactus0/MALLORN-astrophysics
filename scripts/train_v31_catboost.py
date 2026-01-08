"""
MALLORN v31: CatBoost with Ordered Boosting (Experiment 1)

CatBoost advantages for MALLORN:
1. Ordered boosting reduces overfitting on small datasets (~3000 samples)
2. Native handling of missing values (common in sparse lightcurves)
3. Symmetric tree structure prevents overfitting to noise
4. Can use deeper trees (6-8) than XGBoost without overfitting

Expected improvement: +3-5% over v21's XGBoost
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

# Install catboost if needed
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
print("MALLORN v31: CatBoost with Ordered Boosting", flush=True)
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

# Load v21 features
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

# GP2D
with open(base_path / 'data/processed/multiband_gp_cache.pkl', 'rb') as f:
    gp2d_data = pickle.load(f)
train_gp2d = gp2d_data['train']
test_gp2d = gp2d_data['test']
gp2d_cols = [c for c in train_gp2d.columns if c != 'object_id']

# Combine
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
print(f"   Training shape: {X_train.shape}", flush=True)

# ====================
# 2. CATBOOST PARAMETERS
# ====================
print("\n2. CatBoost Configuration...", flush=True)

# CatBoost optimal parameters for small datasets
catboost_params = {
    # Core settings
    'iterations': 1000,
    'learning_rate': 0.03,
    'depth': 7,                    # CatBoost can go deeper

    # Ordered boosting (KEY for small datasets)
    'boosting_type': 'Ordered',    # Prevents overfitting

    # Regularization
    'l2_leaf_reg': 3.0,            # Leaf regularization
    'random_strength': 1.0,        # Randomness for robustness
    'bagging_temperature': 0.8,    # Bayesian bootstrap

    # Sampling
    'subsample': 0.8,
    'rsm': 0.8,                    # Random subspace method (like colsample)

    # Class imbalance
    'auto_class_weights': 'Balanced',

    # Performance
    'task_type': 'CPU',
    'thread_count': -1,
    'random_seed': 42,
    'verbose': False,

    # Early stopping
    'early_stopping_rounds': 50,
    'eval_metric': 'Logloss'
}

print("   Boosting type: Ordered (optimal for n=3043)", flush=True)
print("   Depth: 7 (deeper than XGBoost)", flush=True)
print("   Auto class weights: Balanced", flush=True)

# ====================
# 3. TRAIN WITH 5-FOLD CV
# ====================
print("\n3. Training CatBoost with 5-fold CV...", flush=True)

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X_train))
test_preds = np.zeros((len(X_test), n_folds))
feature_importance = np.zeros(len(feature_names))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y), 1):
    print(f"\n   Fold {fold}/{n_folds}:", flush=True)

    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    # Create CatBoost Pools
    train_pool = Pool(X_tr, y_tr, feature_names=feature_names)
    val_pool = Pool(X_val, y_val, feature_names=feature_names)
    test_pool = Pool(X_test, feature_names=feature_names)

    # Train
    model = CatBoostClassifier(**catboost_params)
    model.fit(
        train_pool,
        eval_set=val_pool,
        use_best_model=True,
        plot=False
    )

    # Predictions
    oof_preds[val_idx] = model.predict_proba(val_pool)[:, 1]
    test_preds[:, fold-1] = model.predict_proba(test_pool)[:, 1]

    # Feature importance
    importance = model.get_feature_importance()
    feature_importance += importance

    # Fold F1
    best_f1 = 0
    best_thresh = 0.5
    for t in np.linspace(0.05, 0.5, 50):
        preds_binary = (oof_preds[val_idx] > t).astype(int)
        f1 = f1_score(y_val, preds_binary)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    print(f"      Best iterations: {model.best_iteration_}", flush=True)
    print(f"      Best threshold: {best_thresh:.3f}, F1: {best_f1:.4f}", flush=True)

print("\n" + "=" * 80, flush=True)
print("CROSS-VALIDATION RESULTS", flush=True)
print("=" * 80, flush=True)

# Find optimal threshold
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
# 4. FEATURE IMPORTANCE
# ====================
print("\n4. Top 30 Features by Importance:", flush=True)

feature_importance = feature_importance / n_folds
importance_df_result = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(importance_df_result.head(30).to_string(index=False), flush=True)

# ====================
# 5. CREATE SUBMISSION
# ====================
print("\n5. Creating submission...", flush=True)

test_avg = test_preds.mean(axis=1)
test_final = (test_avg > best_thresh).astype(int)

submission = pd.DataFrame({
    'object_id': test_ids,
    'target': test_final
})

submission_path = base_path / 'submissions/submission_v31_catboost.csv'
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
    'catboost_params': catboost_params,
    'feature_names': feature_names
}

with open(base_path / 'data/processed/v31_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print("\n" + "=" * 80, flush=True)
print(f"MALLORN v31 (CatBoost) Complete: OOF F1 = {best_f1:.4f}", flush=True)
print(f"Baseline v21 (XGBoost): OOF F1 = 0.6708", flush=True)
change_pct = (best_f1 - 0.6708) * 100
change_abs = best_f1 - 0.6708
print(f"Change: {change_pct:+.2f}% ({change_abs:+.4f})", flush=True)

if best_f1 > 0.6708:
    print("SUCCESS: CatBoost improved over XGBoost!", flush=True)
else:
    print("CatBoost did not outperform XGBoost baseline", flush=True)
print("=" * 80, flush=True)

"""
MALLORN v75: CatBoost on v34a Features

Testing CatBoost as alternative to XGBoost.

CatBoost advantages:
1. Ordered boosting: reduces prediction shift/overfitting
2. Symmetric trees: faster inference
3. Better handling of high-cardinality features
4. Built-in regularization via random permutations

Using v34a feature set (224 features) since that achieved best LB (0.6907).
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier, Pool
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
base_path = Path(__file__).parent.parent

print("=" * 80, flush=True)
print("MALLORN v75: CatBoost on v34a Features", flush=True)
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

print(f"   Training: {len(train_ids)} objects ({np.sum(y)} TDE)", flush=True)

# ====================
# 2. LOAD v34a FEATURES
# ====================
print("\n2. Loading v34a features...", flush=True)

with open(base_path / 'data/processed/v34a_artifacts.pkl', 'rb') as f:
    v34a_artifacts = pickle.load(f)

feature_names = v34a_artifacts['feature_names']
print(f"   v34a features: {len(feature_names)}", flush=True)

# Load all feature data
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

# Select v34a features
available_features = [f for f in feature_names if f in train_all.columns]
print(f"   Available features: {len(available_features)}", flush=True)

X_train = train_all[available_features].values
X_test = test_all[available_features].values

# Handle infinities
X_train = np.nan_to_num(X_train, nan=np.nan, posinf=1e10, neginf=-1e10)
X_test = np.nan_to_num(X_test, nan=np.nan, posinf=1e10, neginf=-1e10)

# ====================
# 3. TRAIN CATBOOST
# ====================
print("\n3. Training CatBoost...", flush=True)

# CatBoost parameters (tuned for binary classification with imbalance)
catboost_params = {
    'iterations': 1000,
    'learning_rate': 0.03,
    'depth': 6,
    'l2_leaf_reg': 3.0,
    'border_count': 128,
    'bagging_temperature': 0.5,
    'random_strength': 1.0,
    'auto_class_weights': 'Balanced',  # Handle class imbalance
    'loss_function': 'Logloss',
    'eval_metric': 'F1',
    'random_seed': 42,
    'verbose': False,
    'early_stopping_rounds': 50,
    'task_type': 'CPU'
}

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

oof_preds = np.zeros(len(y))
test_preds = np.zeros((len(X_test), n_folds))
feature_importance = np.zeros(len(available_features))
fold_f1s = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y), 1):
    print(f"\n   Fold {fold}/{n_folds}:", flush=True)

    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    train_pool = Pool(X_tr, y_tr, feature_names=available_features)
    val_pool = Pool(X_val, y_val, feature_names=available_features)
    test_pool = Pool(X_test, feature_names=available_features)

    model = CatBoostClassifier(**catboost_params)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    oof_preds[val_idx] = model.predict_proba(val_pool)[:, 1]
    test_preds[:, fold-1] = model.predict_proba(test_pool)[:, 1]

    # Feature importance
    importance = model.get_feature_importance()
    feature_importance += importance

    # Fold F1
    best_f1 = 0
    for t in np.linspace(0.03, 0.5, 50):
        f1 = f1_score(y_val, (oof_preds[val_idx] > t).astype(int))
        if f1 > best_f1:
            best_f1 = f1
    fold_f1s.append(best_f1)
    print(f"      Fold F1: {best_f1:.4f}", flush=True)

# ====================
# 4. RESULTS
# ====================
print("\n" + "=" * 80, flush=True)
print("RESULTS", flush=True)
print("=" * 80, flush=True)

best_f1 = 0
best_thresh = 0.1
for t in np.linspace(0.03, 0.5, 200):
    f1 = f1_score(y, (oof_preds > t).astype(int))
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"\n   OOF F1: {best_f1:.4f} @ threshold={best_thresh:.3f}", flush=True)
print(f"   Fold F1s: {[f'{f:.4f}' for f in fold_f1s]}", flush=True)
print(f"   Fold Std: {np.std(fold_f1s):.4f}", flush=True)

final_preds = (oof_preds > best_thresh).astype(int)
tp = np.sum((final_preds == 1) & (y == 1))
fp = np.sum((final_preds == 1) & (y == 0))
fn = np.sum((final_preds == 0) & (y == 1))
print(f"\n   TP={tp}, FP={fp}, FN={fn}", flush=True)
print(f"   Precision: {tp/(tp+fp):.4f}, Recall: {tp/(tp+fn):.4f}", flush=True)

# Feature importance analysis
feature_importance = feature_importance / n_folds
importance_df = pd.DataFrame({
    'feature': available_features,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("\n   Top 20 Features:", flush=True)
print(importance_df.head(20).to_string(index=False), flush=True)

# ====================
# 5. SUBMISSION
# ====================
print("\n" + "=" * 80, flush=True)
print("SUBMISSION", flush=True)
print("=" * 80, flush=True)

test_avg = test_preds.mean(axis=1)
test_binary = (test_avg > best_thresh).astype(int)

submission = pd.DataFrame({
    'object_id': test_ids,
    'target': test_binary
})

submission_path = base_path / 'submissions/submission_v75_catboost.csv'
submission.to_csv(submission_path, index=False)

print(f"   Saved: {submission_path.name}", flush=True)
print(f"   Predicted TDEs: {test_binary.sum()}", flush=True)

# Save artifacts
artifacts = {
    'oof_preds': oof_preds,
    'test_preds': test_avg,
    'best_threshold': best_thresh,
    'oof_f1': best_f1,
    'feature_importance': importance_df,
    'feature_names': available_features,
    'fold_f1s': fold_f1s,
    'params': catboost_params
}

with open(base_path / 'data/processed/v75_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

# ====================
# 6. COMPARISON
# ====================
print("\n" + "=" * 80, flush=True)
print("COMPARISON", flush=True)
print("=" * 80, flush=True)

print(f"""
   Model                    Features   OOF F1   LB F1
   -----                    --------   ------   -----
   v34a XGBoost             224        0.6667   0.6907  <-- Best LB
   v75 CatBoost             {len(available_features)}        {best_f1:.4f}   ???

   CatBoost uses ordered boosting which may reduce overfitting.
   auto_class_weights='Balanced' handles TDE imbalance.
""", flush=True)

print("=" * 80, flush=True)
print("v75 Complete", flush=True)
print("=" * 80, flush=True)

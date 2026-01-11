"""
MALLORN v77: Optuna-Tuned LightGBM

Hyperparameter optimization for LightGBM using Optuna.
Similar approach to v34a's XGBoost tuning that achieved LB 0.6907.

Key parameters to tune:
- num_leaves, max_depth (tree structure)
- learning_rate, n_estimators (boosting)
- feature_fraction, bagging_fraction (regularization)
- reg_alpha, reg_lambda (L1/L2 regularization)
- min_child_samples (leaf constraints)
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
base_path = Path(__file__).parent.parent

print("=" * 80, flush=True)
print("MALLORN v77: Optuna-Tuned LightGBM", flush=True)
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

# Calculate class weight
n_pos = np.sum(y)
n_neg = len(y) - n_pos
scale_pos_weight = n_neg / n_pos

# ====================
# 3. OPTUNA HYPERPARAMETER SEARCH
# ====================
print("\n3. Running Optuna hyperparameter search...", flush=True)

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
folds = list(skf.split(X_train, y))

def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'n_jobs': -1,
        'random_state': 42,

        # Tree structure
        'num_leaves': trial.suggest_int('num_leaves', 15, 63),
        'max_depth': trial.suggest_int('max_depth', 4, 8),

        # Learning
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 200, 800),

        # Regularization
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.95),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.95),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 3.0),

        # Leaf constraints
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),

        # Class imbalance
        'scale_pos_weight': scale_pos_weight,
    }

    oof_preds = np.zeros(len(y))

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=0)
            ]
        )

        oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)

    # Find best threshold for F1
    best_f1 = 0
    for t in np.linspace(0.05, 0.5, 50):
        f1 = f1_score(y, (oof_preds > t).astype(int))
        if f1 > best_f1:
            best_f1 = f1

    return best_f1

# Run optimization
sampler = TPESampler(seed=42)
study = optuna.create_study(direction='maximize', sampler=sampler)

print("   Running 100 trials...", flush=True)
study.optimize(objective, n_trials=100, show_progress_bar=False,
               callbacks=[lambda study, trial: print(f"      Trial {trial.number}: F1={trial.value:.4f}", flush=True) if trial.number % 10 == 0 else None])

print(f"\n   Best trial: {study.best_trial.number}", flush=True)
print(f"   Best OOF F1: {study.best_value:.4f}", flush=True)
print(f"\n   Best parameters:", flush=True)
for k, v in study.best_params.items():
    print(f"      {k}: {v}", flush=True)

# ====================
# 4. TRAIN FINAL MODEL WITH BEST PARAMS
# ====================
print("\n4. Training final model with best parameters...", flush=True)

best_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'verbose': -1,
    'n_jobs': -1,
    'random_state': 42,
    'scale_pos_weight': scale_pos_weight,
    **study.best_params
}

oof_preds = np.zeros(len(y))
test_preds = np.zeros((len(X_test), n_folds))
feature_importance = np.zeros(len(available_features))
fold_f1s = []

for fold, (train_idx, val_idx) in enumerate(folds, 1):
    print(f"\n   Fold {fold}/{n_folds}:", flush=True)

    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=available_features)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=available_features, reference=train_data)

    model = lgb.train(
        best_params,
        train_data,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=0)
        ]
    )

    oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    test_preds[:, fold-1] = model.predict(X_test, num_iteration=model.best_iteration)

    # Feature importance
    importance = model.feature_importance(importance_type='gain')
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
# 5. RESULTS
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
# 6. SUBMISSION
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

submission_path = base_path / 'submissions/submission_v77_lgbm_optuna.csv'
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
    'best_params': best_params,
    'study_best_value': study.best_value
}

with open(base_path / 'data/processed/v77_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

# ====================
# 7. COMPARISON
# ====================
print("\n" + "=" * 80, flush=True)
print("COMPARISON", flush=True)
print("=" * 80, flush=True)

print(f"""
   Model                       Features   OOF F1   LB F1
   -----                       --------   ------   -----
   v34a XGBoost (Optuna)       224        0.6667   0.6907  <-- Best LB
   v76 LightGBM (default)      223        0.6583   ???
   v77 LightGBM (Optuna)       {len(available_features)}        {best_f1:.4f}   ???

   Optuna improvement: {best_f1 - 0.6583:+.4f} over default LightGBM
""", flush=True)

print("=" * 80, flush=True)
print("v77 Complete", flush=True)
print("=" * 80, flush=True)

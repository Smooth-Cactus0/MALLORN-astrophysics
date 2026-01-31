"""
MALLORN v88: Optuna Tuning with Adversarial Weights

Goal: Push OOF to 0.68+ using optimized hyperparameters with adversarial weights.
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
base_path = Path(__file__).parent.parent

print("=" * 80)
print("MALLORN v88: Optuna Tuning with Adversarial Weights")
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
# 2. LOAD FEATURES
# ====================
print("\n2. Loading features...")

with open(base_path / 'data/processed/v34a_artifacts.pkl', 'rb') as f:
    v34a = pickle.load(f)
v34a_features = v34a['feature_names']

with open(base_path / 'data/processed/adversarial_validation.pkl', 'rb') as f:
    adv_results = pickle.load(f)
sample_weights = adv_results['sample_weights']

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

# Remove shift features
shift_features = ['all_rise_time', 'all_asymmetry']
available_features = [f for f in v34a_features if f in train_all.columns and f not in shift_features]
print(f"   Features: {len(available_features)}")

X_train = train_all[available_features].values
X_test = test_all[available_features].values

X_train = np.nan_to_num(X_train, nan=np.nan, posinf=1e10, neginf=-1e10)
X_test = np.nan_to_num(X_test, nan=np.nan, posinf=1e10, neginf=-1e10)

# ====================
# 3. OPTUNA OBJECTIVE
# ====================
print("\n3. Running Optuna optimization (100 trials)...")

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

def objective(trial):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': trial.suggest_int('max_depth', 4, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 5.0, log=True),
        'scale_pos_weight': len(y[y==0]) / len(y[y==1]),
        'tree_method': 'hist',
        'random_state': 42,
        'n_jobs': -1
    }

    n_rounds = trial.suggest_int('n_rounds', 300, 1000)

    oof_preds = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y), 1):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        fold_weights = sample_weights[train_idx]

        dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=fold_weights)
        dval = xgb.DMatrix(X_val, label=y_val)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=n_rounds,
            evals=[(dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=False
        )

        oof_preds[val_idx] = model.predict(dval)

    # Find best threshold and F1
    best_f1 = 0
    for t in np.linspace(0.05, 0.5, 100):
        f1 = f1_score(y, (oof_preds > t).astype(int))
        if f1 > best_f1:
            best_f1 = f1

    return best_f1

# Run optimization
sampler = TPESampler(seed=42)
study = optuna.create_study(direction='maximize', sampler=sampler)
study.optimize(objective, n_trials=100, show_progress_bar=True)

print(f"\n   Best OOF F1: {study.best_value:.4f}")
print(f"   Best params: {study.best_params}")

# ====================
# 4. TRAIN WITH BEST PARAMS
# ====================
print("\n4. Training final model with best params...")

best_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': study.best_params['max_depth'],
    'learning_rate': study.best_params['learning_rate'],
    'subsample': study.best_params['subsample'],
    'colsample_bytree': study.best_params['colsample_bytree'],
    'min_child_weight': study.best_params['min_child_weight'],
    'reg_alpha': study.best_params['reg_alpha'],
    'reg_lambda': study.best_params['reg_lambda'],
    'scale_pos_weight': len(y[y==0]) / len(y[y==1]),
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1
}

n_rounds = study.best_params['n_rounds']

oof_preds = np.zeros(len(y))
test_preds = np.zeros((len(X_test), n_folds))
fold_f1s = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y), 1):
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    fold_weights = sample_weights[train_idx]

    dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=fold_weights, feature_names=available_features)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=available_features)
    dtest = xgb.DMatrix(X_test, feature_names=available_features)

    model = xgb.train(
        best_params,
        dtrain,
        num_boost_round=n_rounds,
        evals=[(dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=False
    )

    oof_preds[val_idx] = model.predict(dval)
    test_preds[:, fold-1] = model.predict(dtest)

    best_f1 = 0
    for t in np.linspace(0.05, 0.5, 50):
        f1 = f1_score(y_val, (oof_preds[val_idx] > t).astype(int))
        if f1 > best_f1:
            best_f1 = f1
    fold_f1s.append(best_f1)

# Find best threshold
best_f1 = 0
best_thresh = 0.3
for t in np.linspace(0.05, 0.5, 200):
    f1 = f1_score(y, (oof_preds > t).astype(int))
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"   Final OOF F1: {best_f1:.4f} @ threshold={best_thresh:.3f}")
print(f"   Fold Std: {np.std(fold_f1s):.4f}")

# ====================
# 5. SUBMISSION
# ====================
print("\n5. Creating submission...")

test_binary = (test_preds.mean(axis=1) > best_thresh).astype(int)

submission = pd.DataFrame({
    'object_id': test_ids,
    'target': test_binary
})

submission.to_csv(base_path / 'submissions/submission_v88_optuna_adv.csv', index=False)

expected_lb = best_f1 + 0.023
print(f"   submission_v88_optuna_adv.csv: OOF={best_f1:.4f}, Expected LB~{expected_lb:.4f}, TDEs={test_binary.sum()}")

# Save artifacts
with open(base_path / 'data/processed/v88_artifacts.pkl', 'wb') as f:
    pickle.dump({
        'best_params': study.best_params,
        'best_f1': best_f1,
        'threshold': best_thresh,
        'fold_f1s': fold_f1s,
        'oof_preds': oof_preds,
        'test_preds': test_preds.mean(axis=1),
        'study': study
    }, f)

print("\n" + "=" * 80)
print("v88 Complete")
print("=" * 80)

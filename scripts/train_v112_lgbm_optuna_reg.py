"""
MALLORN v112: Optuna LightGBM with Constrained Regularization Search

Unlike v77's Optuna search which allowed overfitting (OOF 0.6886 -> LB 0.6714),
this version uses CONSTRAINED search bounds that FAVOR REGULARIZATION:

Key differences from v77:
- Lower tree complexity: num_leaves 7-23 (v77: 15-63), max_depth 3-5 (v77: 4-8)
- More aggressive subsampling: feature_fraction 0.3-0.6 (v77: 0.6-0.95)
- Much higher regularization: reg_alpha 2-15 (v77: 0.01-1), reg_lambda 5-20 (v77: 0.1-3)
- Includes DART boosting option (v77: GBDT only)

Goal: Find optimal parameters within a regularization-favoring space to maximize
GENERALIZATION (LB), not just OOF F1.
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
print("MALLORN v112: Optuna LightGBM with Constrained Regularization Search", flush=True)
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
print(f"   Class imbalance ratio: {scale_pos_weight:.2f}", flush=True)

# ====================
# 3. OPTUNA HYPERPARAMETER SEARCH WITH CONSTRAINED BOUNDS
# ====================
print("\n3. Optuna hyperparameter search with CONSTRAINED regularization bounds...", flush=True)
print("\n   CONSTRAINED search space vs v77:", flush=True)
print("   Parameter            v112 (Constrained)      v77 (Overfitting)", flush=True)
print("   ---------            ------------------      -----------------", flush=True)
print("   num_leaves           7-23                    15-63", flush=True)
print("   max_depth            3-5                     4-8", flush=True)
print("   feature_fraction     0.3-0.6                 0.6-0.95", flush=True)
print("   bagging_fraction     0.4-0.7                 0.6-0.95", flush=True)
print("   reg_alpha            2.0-15.0                0.01-1.0", flush=True)
print("   reg_lambda           5.0-20.0                0.1-3.0", flush=True)
print("   boosting_type        gbdt/dart               gbdt only", flush=True)

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
folds = list(skf.split(X_train, y))

def objective(trial):
    boosting_type = trial.suggest_categorical('boosting_type', ['gbdt', 'dart'])

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': boosting_type,
        'verbose': -1,
        'n_jobs': -1,
        'random_state': 42,

        # CONSTRAINED: Lower tree complexity
        'num_leaves': trial.suggest_int('num_leaves', 7, 23),
        'max_depth': trial.suggest_int('max_depth', 3, 5),

        # Learning
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
        'n_estimators': trial.suggest_int('n_estimators', 400, 800),

        # CONSTRAINED: More aggressive subsampling
        'feature_fraction': trial.suggest_float('feature_fraction', 0.3, 0.6),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 0.7),
        'bagging_freq': trial.suggest_int('bagging_freq', 3, 7),

        # CONSTRAINED: Much higher regularization
        'reg_alpha': trial.suggest_float('reg_alpha', 2.0, 15.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 5.0, 20.0),

        # Leaf constraints
        'min_child_samples': trial.suggest_int('min_child_samples', 25, 60),

        'scale_pos_weight': scale_pos_weight,
    }

    # DART-specific params if selected
    if boosting_type == 'dart':
        params['drop_rate'] = trial.suggest_float('drop_rate', 0.1, 0.3)
        params['skip_drop'] = trial.suggest_float('skip_drop', 0.4, 0.6)

    oof_preds = np.zeros(len(y))

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        if boosting_type == 'dart':
            # DART: use fixed num_boost_round, no early stopping
            model = lgb.train(
                params,
                train_data,
                num_boost_round=params['n_estimators'],
                valid_sets=[val_data],
                callbacks=[lgb.log_evaluation(period=0)]
            )
            oof_preds[val_idx] = model.predict(X_val)
        else:
            # GBDT: use early stopping
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
    for t in np.linspace(0.03, 0.5, 50):
        f1 = f1_score(y, (oof_preds > t).astype(int))
        if f1 > best_f1:
            best_f1 = f1

    return best_f1

# Callback to print progress every 10 trials
def progress_callback(study, trial):
    if trial.number % 10 == 0:
        print(f"      Trial {trial.number}: F1={trial.value:.4f} (best so far: {study.best_value:.4f})", flush=True)

# Run optimization
sampler = TPESampler(seed=42)
study = optuna.create_study(direction='maximize', sampler=sampler)

print("\n   Running 80 trials...", flush=True)
study.optimize(objective, n_trials=80, show_progress_bar=False, callbacks=[progress_callback])

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
    'verbose': -1,
    'n_jobs': -1,
    'random_state': 42,
    'scale_pos_weight': scale_pos_weight,
    **study.best_params
}

# Check if DART was selected
is_dart = best_params.get('boosting_type', 'gbdt') == 'dart'
print(f"   Boosting type: {best_params.get('boosting_type', 'gbdt')}", flush=True)

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

    if is_dart:
        # DART: fixed iterations, no early stopping
        model = lgb.train(
            best_params,
            train_data,
            num_boost_round=best_params['n_estimators'],
            valid_sets=[val_data],
            callbacks=[lgb.log_evaluation(period=0)]
        )
        oof_preds[val_idx] = model.predict(X_val)
        test_preds[:, fold-1] = model.predict(X_test)
    else:
        # GBDT: early stopping
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
    best_fold_f1 = 0
    for t in np.linspace(0.03, 0.5, 50):
        f1 = f1_score(y_val, (oof_preds[val_idx] > t).astype(int))
        if f1 > best_fold_f1:
            best_fold_f1 = f1
    fold_f1s.append(best_fold_f1)
    print(f"      Fold F1: {best_fold_f1:.4f}", flush=True)

# ====================
# 5. FIND OPTIMAL THRESHOLD
# ====================
print("\n5. Finding optimal threshold...", flush=True)

best_f1 = 0
best_thresh = 0.1
for t in np.linspace(0.03, 0.5, 200):
    f1 = f1_score(y, (oof_preds > t).astype(int))
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"   Best threshold: {best_thresh:.4f}", flush=True)

# ====================
# 6. RESULTS
# ====================
print("\n" + "=" * 80, flush=True)
print("RESULTS", flush=True)
print("=" * 80, flush=True)

print(f"\n   OOF F1: {best_f1:.4f} @ threshold={best_thresh:.3f}", flush=True)
print(f"   Fold F1s: {[f'{f:.4f}' for f in fold_f1s]}", flush=True)
print(f"   Fold Std: {np.std(fold_f1s):.4f}", flush=True)

final_preds = (oof_preds > best_thresh).astype(int)
tp = np.sum((final_preds == 1) & (y == 1))
fp = np.sum((final_preds == 1) & (y == 0))
fn = np.sum((final_preds == 0) & (y == 1))
print(f"\n   TP={tp}, FP={fp}, FN={fn}", flush=True)
if tp + fp > 0:
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
# 7. SUBMISSION
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

submission_path = base_path / 'submissions/submission_v112_lgbm_optuna_reg.csv'
submission.to_csv(submission_path, index=False)

print(f"   Saved: {submission_path.name}", flush=True)
print(f"   Predicted TDEs: {test_binary.sum()}", flush=True)

# ====================
# 8. SAVE ARTIFACTS
# ====================
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

with open(base_path / 'data/processed/v112_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print(f"   Artifacts saved to: data/processed/v112_artifacts.pkl", flush=True)

# ====================
# 9. COMPARISON
# ====================
print("\n" + "=" * 80, flush=True)
print("COMPARISON", flush=True)
print("=" * 80, flush=True)

print(f"""
   Model                               Features   OOF F1    LB F1
   -----                               --------   ------    -----
   v34a XGBoost (Optuna)               224        0.6667    0.6907  <- Best LB
   v77 LightGBM (Optuna, overfitting)  223        0.6886    0.6714  <- Overfit
   v110 LightGBM (Heavy Reg GBDT)      224        0.6609    ???
   v111 LightGBM (DART)                224        0.6608    ???
   v112 LightGBM (Constrained Optuna)  {len(available_features)}        {best_f1:.4f}    ???

   v112 vs v77 key differences:
   - Tree complexity: {best_params.get('num_leaves', 'N/A')} leaves (v77: 15-63)
   - Depth: {best_params.get('max_depth', 'N/A')} (v77: 4-8)
   - Feature fraction: {best_params.get('feature_fraction', 'N/A'):.3f} (v77: 0.6-0.95)
   - Bagging fraction: {best_params.get('bagging_fraction', 'N/A'):.3f} (v77: 0.6-0.95)
   - reg_alpha: {best_params.get('reg_alpha', 'N/A'):.2f} (v77: 0.01-1.0)
   - reg_lambda: {best_params.get('reg_lambda', 'N/A'):.2f} (v77: 0.1-3.0)
   - Boosting: {best_params.get('boosting_type', 'gbdt')} (v77: gbdt only)

   Optuna found best params within CONSTRAINED regularization-favoring space.
   Goal: OOF similar to v34a (0.6667) for better generalization to LB.
   If LB > 0.6714 (v77), constrained search prevents overfitting.
   If LB > 0.6907 (v34a), this beats our best model!
""", flush=True)

print("=" * 80, flush=True)
print("v112 Complete", flush=True)
print("=" * 80, flush=True)

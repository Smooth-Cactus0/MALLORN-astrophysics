"""
MALLORN v100: LightGBM with Optuna Tuning

Previous LightGBM results:
- v77: OOF 0.6886, LB 0.6714 (best OOF LGB)

LightGBM advantages:
1. Leaf-wise growth (vs level-wise) - often better for imbalanced data
2. Faster training than XGBoost
3. Better handling of categorical features (though we don't have explicit cats)
4. Different regularization approach may complement XGBoost

Goal: Create a strong LightGBM model for potential ensembling with XGBoost.
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix
import warnings
import optuna
import lightgbm as lgb

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
base_path = Path(__file__).parent.parent

print("=" * 80)
print("MALLORN v100: LightGBM with Optuna Tuning")
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
y_train = train_meta['target'].values

print(f"   Training: {len(train_ids)} ({np.sum(y_train)} TDE)")
class_ratio = (len(y_train) - np.sum(y_train)) / np.sum(y_train)
print(f"   Class ratio: {class_ratio:.1f}:1")

# Load adversarial weights
with open(base_path / 'data/processed/adversarial_validation.pkl', 'rb') as f:
    adv_results = pickle.load(f)
sample_weights = adv_results['sample_weights']

# ====================
# 2. LOAD FEATURES
# ====================
print("\n2. Loading features...")

with open(base_path / 'data/processed/v34a_artifacts.pkl', 'rb') as f:
    v34a = pickle.load(f)
v34a_features = v34a['feature_names']

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

train_all = train_base.merge(train_tde, on='object_id', how='left')
train_all = train_all.merge(train_gp2d, on='object_id', how='left')
train_all = train_all.merge(train_bazin, on='object_id', how='left')

test_all = test_base.merge(test_tde, on='object_id', how='left')
test_all = test_all.merge(test_gp2d, on='object_id', how='left')
test_all = test_all.merge(test_bazin, on='object_id', how='left')

# Remove features that caused overfit
shift_features = ['all_rise_time', 'all_asymmetry']
available_features = [f for f in v34a_features if f in train_all.columns and f not in shift_features]

X_train = train_all[available_features].values
X_test = test_all[available_features].values

# Replace inf/nan
X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

print(f"   Features: {len(available_features)}")

# ====================
# 3. OPTUNA-TUNED LIGHTGBM
# ====================
print("\n3. Optuna-tuned LightGBM (50 trials)...")

def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 16, 128),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'scale_pos_weight': class_ratio,
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1,
    }

    # 3-fold for speed
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    fold_f1s = []

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        w_tr = sample_weights[train_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            sample_weight=w_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False)]
        )

        val_probs = model.predict_proba(X_val)[:, 1]

        # Find best threshold
        best_f1 = 0
        for t in np.linspace(0.05, 0.5, 50):
            f1 = f1_score(y_val, (val_probs > t).astype(int))
            if f1 > best_f1:
                best_f1 = f1
        fold_f1s.append(best_f1)

    return np.mean(fold_f1s)

print("   Running Optuna optimization...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"   Best trial F1: {study.best_value:.4f}")
print(f"   Best params: {study.best_params}")

# ====================
# 4. TRAIN WITH BEST PARAMS
# ====================
print("\n4. Training v100a with best params (5-fold CV)...")

best_params = study.best_params.copy()
best_params['objective'] = 'binary'
best_params['metric'] = 'binary_logloss'
best_params['boosting_type'] = 'gbdt'
best_params['scale_pos_weight'] = class_ratio
best_params['random_state'] = 42
best_params['verbose'] = -1
best_params['n_jobs'] = -1

results = {}

# v100a: Optuna-tuned with adversarial weights
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

oof_preds = np.zeros(len(y_train))
test_preds_folds = []
fold_f1s = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    w_tr = sample_weights[train_idx]

    model = lgb.LGBMClassifier(**best_params)
    model.fit(
        X_tr, y_tr,
        sample_weight=w_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds_folds.append(model.predict_proba(X_test)[:, 1])

    best_f1 = 0
    for t in np.linspace(0.05, 0.5, 50):
        f1 = f1_score(y_val, (oof_preds[val_idx] > t).astype(int))
        if f1 > best_f1:
            best_f1 = f1
    fold_f1s.append(best_f1)
    print(f"      Fold {fold} F1: {best_f1:.4f}")

test_preds = np.mean(test_preds_folds, axis=0)

best_f1 = 0
best_thresh = 0.3
for t in np.linspace(0.05, 0.5, 200):
    f1 = f1_score(y_train, (oof_preds > t).astype(int))
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

preds_binary = (oof_preds > best_thresh).astype(int)
cm = confusion_matrix(y_train, preds_binary)
tn, fp, fn, tp = cm.ravel()

results['v100a_optuna'] = {
    'oof_f1': best_f1,
    'threshold': best_thresh,
    'fold_f1s': fold_f1s,
    'fold_std': np.std(fold_f1s),
    'oof_preds': oof_preds,
    'test_preds': test_preds,
    'confusion': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
    'recall': tp / (tp + fn),
    'precision': tp / (tp + fp),
    'best_params': study.best_params,
}

print(f"      OOF F1: {best_f1:.4f} @ threshold={best_thresh:.3f}")
print(f"      Recall: {tp/(tp+fn):.1%} | Precision: {tp/(tp+fp):.1%}")
print(f"      FN: {fn} | FP: {fp}")

# ====================
# 5. TRY DIFFERENT CONFIGURATIONS
# ====================
print("\n5. Additional variants...")

# v100b: More regularized (to reduce overfit)
print("\n   v100b_reg: More regularized...")
reg_params = best_params.copy()
reg_params['reg_alpha'] = max(reg_params.get('reg_alpha', 0.1), 1.0)
reg_params['reg_lambda'] = max(reg_params.get('reg_lambda', 0.1), 1.0)
reg_params['num_leaves'] = min(reg_params.get('num_leaves', 31), 31)
reg_params['min_child_samples'] = max(reg_params.get('min_child_samples', 20), 30)

oof_preds = np.zeros(len(y_train))
test_preds_folds = []
fold_f1s = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    w_tr = sample_weights[train_idx]

    model = lgb.LGBMClassifier(**reg_params)
    model.fit(
        X_tr, y_tr,
        sample_weight=w_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds_folds.append(model.predict_proba(X_test)[:, 1])

    best_f1 = 0
    for t in np.linspace(0.05, 0.5, 50):
        f1 = f1_score(y_val, (oof_preds[val_idx] > t).astype(int))
        if f1 > best_f1:
            best_f1 = f1
    fold_f1s.append(best_f1)
    print(f"      Fold {fold} F1: {best_f1:.4f}")

test_preds = np.mean(test_preds_folds, axis=0)

best_f1 = 0
best_thresh = 0.3
for t in np.linspace(0.05, 0.5, 200):
    f1 = f1_score(y_train, (oof_preds > t).astype(int))
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

preds_binary = (oof_preds > best_thresh).astype(int)
cm = confusion_matrix(y_train, preds_binary)
tn, fp, fn, tp = cm.ravel()

results['v100b_reg'] = {
    'oof_f1': best_f1,
    'threshold': best_thresh,
    'fold_f1s': fold_f1s,
    'fold_std': np.std(fold_f1s),
    'oof_preds': oof_preds,
    'test_preds': test_preds,
    'confusion': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
    'recall': tp / (tp + fn),
    'precision': tp / (tp + fp),
}

print(f"      OOF F1: {best_f1:.4f} @ threshold={best_thresh:.3f}")
print(f"      Recall: {tp/(tp+fn):.1%} | Precision: {tp/(tp+fp):.1%}")

# ====================
# 6. RESULTS
# ====================
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

print(f"\n{'Variant':<20} {'OOF F1':<10} {'Recall':<10} {'Prec':<10} {'FN':<6} {'FP':<6}")
print("-" * 65)
print(f"{'v92d (LB=0.6986)':<20} {'0.6688':<10} {'69.6%':<10} {'64.4%':<10} {'45':<6} {'57':<6}")
print(f"{'v77 (LB=0.6714)':<20} {'0.6886':<10} {'--':<10} {'--':<10} {'--':<6} {'--':<6}")
print("-" * 65)

sorted_results = sorted(results.items(), key=lambda x: -x[1]['oof_f1'])

for name, res in sorted_results:
    recall_str = f"{100*res['recall']:.1f}%"
    prec_str = f"{100*res['precision']:.1f}%"
    print(f"{name:<20} {res['oof_f1']:<10.4f} {recall_str:<10} {prec_str:<10} {res['confusion']['fn']:<6} {res['confusion']['fp']:<6}")

# ====================
# 7. SUBMISSIONS
# ====================
print("\n" + "=" * 80)
print("SUBMISSIONS")
print("=" * 80)

for name, res in sorted_results:
    test_binary = (res['test_preds'] > res['threshold']).astype(int)

    submission = pd.DataFrame({
        'object_id': test_ids,
        'target': test_binary
    })

    filename = f"submission_{name}.csv"
    submission.to_csv(base_path / f'submissions/{filename}', index=False)

    print(f"   {filename}: OOF={res['oof_f1']:.4f}, TDEs={test_binary.sum()}")

with open(base_path / 'data/processed/v100_lightgbm_artifacts.pkl', 'wb') as f:
    pickle.dump(results, f)

print("\n" + "=" * 80)
print("v100 LightGBM Complete")
print("=" * 80)

"""
MALLORN v99: CatBoost with Adversarial Weights

CatBoost advantages over XGBoost/LightGBM for this problem:
1. Ordered boosting - reduces overfitting on small datasets
2. Symmetric trees - faster inference, more regularization
3. Native handling of class imbalance via class_weights
4. Different gradient estimation reduces prediction shift

Competitors are using CatBoost with success - let's see if it beats v92d (LB=0.6986).
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

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
base_path = Path(__file__).parent.parent

print("=" * 80)
print("MALLORN v99: CatBoost with Adversarial Weights")
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
# 3. CATBOOST TRAINING
# ====================
print("\n3. Training CatBoost variants...")

from catboost import CatBoostClassifier, Pool

# Configurations to test
configs = {
    # v99a: Basic CatBoost with class weights
    'v99a_basic': {
        'iterations': 1000,
        'learning_rate': 0.03,
        'depth': 6,
        'l2_leaf_reg': 3,
        'class_weights': {0: 1, 1: class_ratio},
        'random_seed': 42,
        'verbose': False,
        'use_adv_weights': False,
    },

    # v99b: CatBoost with adversarial weights
    'v99b_adv': {
        'iterations': 1000,
        'learning_rate': 0.03,
        'depth': 6,
        'l2_leaf_reg': 3,
        'class_weights': {0: 1, 1: class_ratio},
        'random_seed': 42,
        'verbose': False,
        'use_adv_weights': True,
    },

    # v99c: More regularized
    'v99c_reg': {
        'iterations': 1000,
        'learning_rate': 0.02,
        'depth': 4,
        'l2_leaf_reg': 10,
        'class_weights': {0: 1, 1: class_ratio},
        'random_seed': 42,
        'verbose': False,
        'use_adv_weights': True,
    },

    # v99d: Deeper trees with subsample
    'v99d_deep': {
        'iterations': 1500,
        'learning_rate': 0.02,
        'depth': 8,
        'l2_leaf_reg': 5,
        'bagging_temperature': 1.0,
        'class_weights': {0: 1, 1: class_ratio},
        'random_seed': 42,
        'verbose': False,
        'use_adv_weights': True,
    },
}

results = {}

for variant_name, config in configs.items():
    print(f"\n   {variant_name}:")

    use_adv_weights = config.pop('use_adv_weights')

    # 5-fold CV
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(y_train))
    test_preds_folds = []
    fold_f1s = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        # Sample weights for this fold
        if use_adv_weights:
            w_tr = sample_weights[train_idx]
        else:
            w_tr = None

        # Create CatBoost model
        model = CatBoostClassifier(**config)

        # Train
        train_pool = Pool(X_tr, y_tr, weight=w_tr)
        val_pool = Pool(X_val, y_val)

        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, verbose=False)

        # Predict
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        test_preds_folds.append(model.predict_proba(X_test)[:, 1])

        # Fold F1
        best_f1 = 0
        for t in np.linspace(0.1, 0.5, 50):
            f1 = f1_score(y_val, (oof_preds[val_idx] > t).astype(int))
            if f1 > best_f1:
                best_f1 = f1
        fold_f1s.append(best_f1)
        print(f"      Fold {fold} F1: {best_f1:.4f}")

    # Average test predictions
    test_preds = np.mean(test_preds_folds, axis=0)

    # OOF F1
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

    results[variant_name] = {
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
    print(f"      FN: {fn} | FP: {fp}")

    # Restore config for next iteration
    config['use_adv_weights'] = use_adv_weights

# ====================
# 4. OPTUNA-TUNED CATBOOST
# ====================
print("\n4. Optuna-tuned CatBoost...")

def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 500, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 30, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 2),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'class_weights': {0: 1, 1: class_ratio},
        'random_seed': 42,
        'verbose': False,
    }

    # 3-fold for speed
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    fold_f1s = []

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        w_tr = sample_weights[train_idx]

        model = CatBoostClassifier(**params)
        train_pool = Pool(X_tr, y_tr, weight=w_tr)
        val_pool = Pool(X_val, y_val)

        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=30, verbose=False)

        val_probs = model.predict_proba(X_val)[:, 1]

        # Find best threshold
        best_f1 = 0
        for t in np.linspace(0.05, 0.5, 50):
            f1 = f1_score(y_val, (val_probs > t).astype(int))
            if f1 > best_f1:
                best_f1 = f1
        fold_f1s.append(best_f1)

    return np.mean(fold_f1s)

print("   Running 50 Optuna trials...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"   Best trial F1: {study.best_value:.4f}")
print(f"   Best params: {study.best_params}")

# Train with best params on full 5-fold CV
print("\n   Training v99e with best params...")
best_params = study.best_params
best_params['class_weights'] = {0: 1, 1: class_ratio}
best_params['random_seed'] = 42
best_params['verbose'] = False

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

oof_preds = np.zeros(len(y_train))
test_preds_folds = []
fold_f1s = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    w_tr = sample_weights[train_idx]

    model = CatBoostClassifier(**best_params)
    train_pool = Pool(X_tr, y_tr, weight=w_tr)
    val_pool = Pool(X_val, y_val)

    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, verbose=False)

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

results['v99e_optuna'] = {
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
# 5. RESULTS
# ====================
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

print(f"\n{'Variant':<20} {'OOF F1':<10} {'Recall':<10} {'Prec':<10} {'FN':<6} {'FP':<6}")
print("-" * 65)
print(f"{'v92d (LB=0.6986)':<20} {'0.6688':<10} {'69.6%':<10} {'64.4%':<10} {'45':<6} {'57':<6}")
print("-" * 65)

sorted_results = sorted(results.items(), key=lambda x: -x[1]['oof_f1'])

for name, res in sorted_results:
    recall_str = f"{100*res['recall']:.1f}%"
    prec_str = f"{100*res['precision']:.1f}%"
    print(f"{name:<20} {res['oof_f1']:<10.4f} {recall_str:<10} {prec_str:<10} {res['confusion']['fn']:<6} {res['confusion']['fp']:<6}")

# ====================
# 6. SUBMISSIONS
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

with open(base_path / 'data/processed/v99_catboost_artifacts.pkl', 'wb') as f:
    pickle.dump(results, f)

print("\n" + "=" * 80)
print("v99 CatBoost Complete")
print("=" * 80)

"""
MALLORN v118: CatBoost for Ensemble Diversity
=============================================

CatBoost offers different tree-building (symmetric trees) and often
produces predictions with low correlation to XGBoost/LightGBM.

Strategy:
1. Optuna tuning to find best CatBoost params
2. Train with multiple seeds
3. Save predictions for ensemble with XGB + LGB
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier, Pool
import optuna
from optuna.samplers import TPESampler
import warnings

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
base_path = Path(__file__).parent.parent

print("=" * 70)
print("MALLORN v118: CatBoost for Ensemble Diversity")
print("=" * 70)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/5] Loading data...")

# Load feature package
import gzip
with gzip.open(base_path / 'data/kaggle_ensemble_package.pkl.gz', 'rb') as f:
    package = pickle.load(f)

train_features = package['train_features']
test_features = package['test_features']
y = package['y']
train_ids = package['train_ids']
test_ids = package['test_ids']
sample_weights = package['sample_weights']

print(f"   Train: {train_features.shape}")
print(f"   Test: {test_features.shape}")
print(f"   TDEs: {np.sum(y)} / {len(y)}")

# ============================================================================
# 2. FEATURE ENGINEERING FOR CATBOOST
# ============================================================================
print("\n[2/5] Feature engineering for CatBoost...")

# Load v34a feature list as base
with open(base_path / 'data/processed/v34a_artifacts.pkl', 'rb') as f:
    v34a = pickle.load(f)
v34a_features = v34a['feature_names']

# Remove adversarial-discriminative features
adv_discriminative = ['all_rise_time', 'all_asymmetry']
base_features = [f for f in v34a_features if f not in adv_discriminative]

# Add research features that helped LightGBM
# These showed good LB improvement in v113/v114
research_features = [
    'nuclear_concentration', 'nuclear_smoothness',
    'g_r_color_at_peak', 'r_i_color_at_peak',
    'mhps_10_100_ratio', 'mhps_30_100_ratio',
    # Extended set that helped v115c (highest delta)
    'nuclear_position_score', 'mhps_10d', 'mhps_30d',
]

catboost_features = base_features + [f for f in research_features if f in train_features.columns]
catboost_features = [f for f in catboost_features if f in train_features.columns]
catboost_features = list(dict.fromkeys(catboost_features))  # Remove duplicates

print(f"   CatBoost features: {len(catboost_features)}")

# Prepare data
X_train = train_features[catboost_features].values
X_test = test_features[catboost_features].values

# Handle NaN/Inf - CatBoost handles NaN natively but not Inf
X_train = np.nan_to_num(X_train, nan=np.nan, posinf=1e10, neginf=-1e10)
X_test = np.nan_to_num(X_test, nan=np.nan, posinf=1e10, neginf=-1e10)

# Class imbalance
scale_pos_weight = len(y[y == 0]) / len(y[y == 1])
print(f"   Scale pos weight: {scale_pos_weight:.2f}")

# CV setup
CV_SEED = 42
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
folds = list(skf.split(X_train, y))

# ============================================================================
# 3. OPTUNA TUNING FOR CATBOOST
# ============================================================================
print("\n[3/5] Optuna tuning for CatBoost (50 trials)...")

def objective(trial):
    params = {
        'loss_function': 'Logloss',
        'eval_metric': 'Logloss',
        'random_seed': 42,
        'verbose': False,
        'allow_writing_files': False,
        'scale_pos_weight': scale_pos_weight,

        # Constrained search space (prevent overfitting)
        'depth': trial.suggest_int('depth', 3, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.08),
        'iterations': trial.suggest_int('iterations', 400, 800),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 2.0, 10.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.5, 2.0),
        'random_strength': trial.suggest_float('random_strength', 0.5, 2.0),
        'border_count': trial.suggest_int('border_count', 32, 128),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 50),
    }

    oof_preds = np.zeros(len(y))

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        fold_weights = sample_weights[train_idx]

        train_pool = Pool(X_tr, y_tr, weight=fold_weights, feature_names=catboost_features)
        val_pool = Pool(X_val, y_val, feature_names=catboost_features)

        model = CatBoostClassifier(**params)
        model.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=50,
            verbose=False
        )

        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

    # Find best threshold
    best_f1 = 0
    for t in np.linspace(0.05, 0.5, 50):
        f1 = f1_score(y, (oof_preds > t).astype(int))
        if f1 > best_f1:
            best_f1 = f1

    return best_f1

# Progress callback
def callback(study, trial):
    if trial.number % 10 == 0:
        print(f"   Trial {trial.number}: F1={trial.value:.4f} (best: {study.best_value:.4f})")

# Run optimization
sampler = TPESampler(seed=42)
study = optuna.create_study(direction='maximize', sampler=sampler)
study.optimize(objective, n_trials=50, callbacks=[callback])

print(f"\n   Best trial: {study.best_trial.number}")
print(f"   Best OOF F1: {study.best_value:.4f}")
print(f"\n   Best parameters:")
for k, v in study.best_params.items():
    print(f"      {k}: {v}")

best_params = study.best_params

# ============================================================================
# 4. TRAIN WITH MULTIPLE SEEDS
# ============================================================================
print("\n[4/5] Training CatBoost with 5 seeds...")

MODEL_SEEDS = [42, 123, 456, 789, 2024]

def find_best_threshold(y_true, y_pred):
    best_f1, best_t = 0, 0.1
    for t in np.linspace(0.03, 0.5, 100):
        f1 = f1_score(y_true, (y_pred > t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1

all_oof_preds = []
all_test_preds = []
all_fold_scores = []

for seed in MODEL_SEEDS:
    print(f"\n   Seed {seed}...", end=" ", flush=True)

    params = {
        'loss_function': 'Logloss',
        'eval_metric': 'Logloss',
        'random_seed': seed,
        'verbose': False,
        'allow_writing_files': False,
        'scale_pos_weight': scale_pos_weight,
        **best_params
    }

    oof_preds = np.zeros(len(y))
    test_preds = np.zeros(len(X_test))
    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        fold_weights = sample_weights[train_idx]

        train_pool = Pool(X_tr, y_tr, weight=fold_weights, feature_names=catboost_features)
        val_pool = Pool(X_val, y_val, feature_names=catboost_features)
        test_pool = Pool(X_test, feature_names=catboost_features)

        model = CatBoostClassifier(**params)
        model.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=50,
            verbose=False
        )

        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / N_FOLDS

        # Fold F1
        t, f1 = find_best_threshold(y_val, oof_preds[val_idx])
        fold_scores.append(f1)

    threshold, oof_f1 = find_best_threshold(y, oof_preds)
    print(f"OOF F1: {oof_f1:.4f}, Fold std: {np.std(fold_scores):.4f}")

    all_oof_preds.append(oof_preds)
    all_test_preds.append(test_preds)
    all_fold_scores.append(fold_scores)

# Average across seeds
avg_oof = np.mean(all_oof_preds, axis=0)
avg_test = np.mean(all_test_preds, axis=0)
avg_threshold, avg_f1 = find_best_threshold(y, avg_oof)

print(f"\n   5-Seed Averaged OOF F1: {avg_f1:.4f} @ threshold {avg_threshold:.3f}")
print(f"   Fold std (averaged): {np.mean([np.std(fs) for fs in all_fold_scores]):.4f}")

# ============================================================================
# 5. SAVE RESULTS
# ============================================================================
print("\n[5/5] Saving results...")

# Save artifacts
artifacts = {
    'model_name': 'v118_catboost',
    'best_params': best_params,
    'optuna_best_f1': study.best_value,
    'avg_oof_f1': avg_f1,
    'avg_threshold': avg_threshold,
    'all_oof_preds': all_oof_preds,
    'all_test_preds': all_test_preds,
    'avg_oof_preds': avg_oof,
    'avg_test_preds': avg_test,
    'feature_names': catboost_features,
    'model_seeds': MODEL_SEEDS,
    'fold_scores': all_fold_scores,
}

with open(base_path / 'data/processed/v118_catboost_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)
print(f"   Saved artifacts to data/processed/v118_catboost_artifacts.pkl")

# Save submission
binary_preds = (avg_test > avg_threshold).astype(int)
submission = pd.DataFrame({
    'object_id': test_ids,
    'target': binary_preds
})
submission.to_csv(base_path / 'submissions/submission_v118_catboost.csv', index=False)
print(f"   Saved submission: {binary_preds.sum()} TDEs predicted")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("CATBOOST TRAINING COMPLETE")
print("=" * 70)

print(f"""
Results:
   Optuna Best OOF F1: {study.best_value:.4f}
   5-Seed Avg OOF F1:  {avg_f1:.4f}
   Threshold:          {avg_threshold:.3f}
   TDEs predicted:     {binary_preds.sum()}

Best Parameters:
   depth:              {best_params['depth']}
   learning_rate:      {best_params['learning_rate']:.4f}
   iterations:         {best_params['iterations']}
   l2_leaf_reg:        {best_params['l2_leaf_reg']:.2f}
   bagging_temperature:{best_params['bagging_temperature']:.2f}
   min_data_in_leaf:   {best_params['min_data_in_leaf']}

Comparison with other models:
   v92d XGBoost:  OOF ~0.6625 -> LB 0.6986
   v34a XGBoost:  OOF ~0.6667 -> LB 0.6907
   v114d LightGBM: OOF ~0.6761 -> LB 0.6797
   v118 CatBoost: OOF {avg_f1:.4f} -> LB ???

Next: Add CatBoost to ensemble!
""")

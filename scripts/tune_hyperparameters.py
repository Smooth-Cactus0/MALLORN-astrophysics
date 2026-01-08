"""
MALLORN Hyperparameter Tuning with Optuna

This script uses Optuna to find optimal hyperparameters for our 3-model ensemble.
We tune XGBoost, LightGBM, and CatBoost separately, then combine.

Key hyperparameters to tune:
- Tree depth (model complexity)
- Learning rate (step size)
- Regularization (prevent overfitting)
- Subsampling (reduce variance)
- Number of estimators (with early stopping)
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from utils.data_loader import load_all_data

# Suppress Optuna logs for cleaner output
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_data():
    """Load features and prepare training data."""
    base_path = Path(__file__).parent.parent

    # Load base features
    cached = pd.read_pickle(base_path / 'data/processed/features_v4_cache.pkl')
    train_features = cached['train_features']

    # Load feature selection
    selection = pd.read_pickle(base_path / 'data/processed/selected_features.pkl')
    importance_df = selection['importance_df']
    high_corr_df = selection['high_corr_df']

    # Get top 120 non-correlated features
    corr_to_drop = set()
    for _, row in high_corr_df.iterrows():
        if row['feature_1'] not in corr_to_drop:
            corr_to_drop.add(row['feature_2'])
    clean_features = importance_df[~importance_df['feature'].isin(corr_to_drop)]
    selected_120 = clean_features.head(120)['feature'].tolist()

    # Load TDE physics features
    tde_cached = pd.read_pickle(base_path / 'data/processed/tde_physics_cache.pkl')
    train_tde = tde_cached['train']
    tde_cols = [c for c in train_tde.columns if c != 'object_id']

    # Combine
    train_combined = train_features[['object_id'] + selected_120].merge(
        train_tde, on='object_id', how='left'
    )

    data = load_all_data()
    train_combined = train_combined.merge(
        data['train_meta'][['object_id', 'target']], on='object_id'
    )

    feature_cols = selected_120 + tde_cols

    X = train_combined[feature_cols].values
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    y = train_combined['target'].values

    return X, y, feature_cols


def find_optimal_threshold(y_true, y_prob):
    """Find threshold that maximizes F1."""
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.1, 0.7, 0.02):
        f1 = f1_score(y_true, (y_prob >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


class XGBObjective:
    """Optuna objective for XGBoost."""

    def __init__(self, X, y, n_splits=5):
        self.X = X
        self.y = y
        self.n_splits = n_splits
        self.scale_pos_weight = np.bincount(y)[0] / np.bincount(y)[1]

    def __call__(self, trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.8),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10, log=True),
            'scale_pos_weight': self.scale_pos_weight,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        oof = np.zeros(len(self.y))

        for train_idx, val_idx in skf.split(self.X, self.y):
            model = xgb.XGBClassifier(**params)
            model.fit(
                self.X[train_idx], self.y[train_idx],
                eval_set=[(self.X[val_idx], self.y[val_idx])],
                verbose=False
            )
            oof[val_idx] = model.predict_proba(self.X[val_idx])[:, 1]

        _, f1 = find_optimal_threshold(self.y, oof)
        return f1


class LGBObjective:
    """Optuna objective for LightGBM."""

    def __init__(self, X, y, n_splits=5):
        self.X = X
        self.y = y
        self.n_splits = n_splits
        self.scale_pos_weight = np.bincount(y)[0] / np.bincount(y)[1]

    def __call__(self, trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.8),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10, log=True),
            'scale_pos_weight': self.scale_pos_weight,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        oof = np.zeros(len(self.y))

        for train_idx, val_idx in skf.split(self.X, self.y):
            model = lgb.LGBMClassifier(**params)
            model.fit(
                self.X[train_idx], self.y[train_idx],
                eval_set=[(self.X[val_idx], self.y[val_idx])]
            )
            oof[val_idx] = model.predict_proba(self.X[val_idx])[:, 1]

        _, f1 = find_optimal_threshold(self.y, oof)
        return f1


class CatBoostObjective:
    """Optuna objective for CatBoost."""

    def __init__(self, X, y, n_splits=5):
        self.X = X
        self.y = y
        self.n_splits = n_splits
        self.scale_pos_weight = np.bincount(y)[0] / np.bincount(y)[1]

    def __call__(self, trial):
        params = {
            'depth': trial.suggest_int('depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'iterations': trial.suggest_int('iterations', 500, 1500),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10, log=True),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'scale_pos_weight': self.scale_pos_weight,
            'random_seed': 42,
            'verbose': False,
            'allow_writing_files': False
        }

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        oof = np.zeros(len(self.y))

        for train_idx, val_idx in skf.split(self.X, self.y):
            model = CatBoostClassifier(**params)
            model.fit(
                self.X[train_idx], self.y[train_idx],
                eval_set=(self.X[val_idx], self.y[val_idx]),
                early_stopping_rounds=100
            )
            oof[val_idx] = model.predict_proba(self.X[val_idx])[:, 1]

        _, f1 = find_optimal_threshold(self.y, oof)
        return f1


def main():
    print("=" * 60)
    print("MALLORN Hyperparameter Tuning with Optuna")
    print("=" * 60)

    # Load data
    print("\n1. Loading data...")
    X, y, feature_cols = load_data()
    print(f"   Features: {X.shape[1]}, Samples: {X.shape[0]}")

    n_trials = 30  # Trials per model (adjust based on time)

    # Tune XGBoost
    print(f"\n2. Tuning XGBoost ({n_trials} trials)...")
    xgb_study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42)
    )
    xgb_study.optimize(XGBObjective(X, y), n_trials=n_trials, show_progress_bar=True)
    print(f"   Best XGBoost F1: {xgb_study.best_value:.4f}")
    print(f"   Best params: {xgb_study.best_params}")

    # Tune LightGBM
    print(f"\n3. Tuning LightGBM ({n_trials} trials)...")
    lgb_study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42)
    )
    lgb_study.optimize(LGBObjective(X, y), n_trials=n_trials, show_progress_bar=True)
    print(f"   Best LightGBM F1: {lgb_study.best_value:.4f}")
    print(f"   Best params: {lgb_study.best_params}")

    # Tune CatBoost
    print(f"\n4. Tuning CatBoost ({n_trials} trials)...")
    cat_study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42)
    )
    cat_study.optimize(CatBoostObjective(X, y), n_trials=n_trials, show_progress_bar=True)
    print(f"   Best CatBoost F1: {cat_study.best_value:.4f}")
    print(f"   Best params: {cat_study.best_params}")

    # Save results
    base_path = Path(__file__).parent.parent
    results = {
        'xgb_study': xgb_study,
        'lgb_study': lgb_study,
        'cat_study': cat_study,
        'xgb_best_params': xgb_study.best_params,
        'lgb_best_params': lgb_study.best_params,
        'cat_best_params': cat_study.best_params
    }

    with open(base_path / 'data/processed/optuna_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    # Summary
    print("\n" + "=" * 60)
    print("TUNING COMPLETE")
    print("=" * 60)
    print(f"\nBest F1 scores:")
    print(f"  XGBoost:  {xgb_study.best_value:.4f}")
    print(f"  LightGBM: {lgb_study.best_value:.4f}")
    print(f"  CatBoost: {cat_study.best_value:.4f}")

    print("\nBest hyperparameters saved to data/processed/optuna_results.pkl")
    print("Run train_v8_tuned.py to train with optimized hyperparameters.")


if __name__ == "__main__":
    main()

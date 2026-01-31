"""
MALLORN v116: Optuna Hyperparameter Tuning

Run this AFTER getting LB results from v114/v115 submissions.
Tunes the best performing model(s) with constrained search space
to prevent overfitting.

Usage:
    python scripts/train_v116_optuna_tuning.py --model lgb  # Tune LightGBM
    python scripts/train_v116_optuna_tuning.py --model xgb  # Tune XGBoost
    python scripts/train_v116_optuna_tuning.py --model both # Tune both
"""

import sys
import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
base_path = Path(__file__).parent.parent


def load_data():
    """Load all required data."""
    print("Loading data...")

    from utils.data_loader import load_all_data
    data = load_all_data()

    train_meta = data['train_meta']
    test_meta = data['test_meta']
    train_ids = train_meta['object_id'].tolist()
    test_ids = test_meta['object_id'].tolist()
    y = train_meta['target'].values

    # Load adversarial weights
    with open(base_path / 'data/processed/adversarial_validation.pkl', 'rb') as f:
        adv_results = pickle.load(f)
    sample_weights = adv_results['sample_weights']

    # Load all features
    cached = pd.read_pickle(base_path / 'data/processed/features_v4_cache.pkl')
    train_base = cached['train_features']
    test_base = cached['test_features']

    tde_cached = pd.read_pickle(base_path / 'data/processed/tde_physics_cache.pkl')
    with open(base_path / 'data/processed/multiband_gp_cache.pkl', 'rb') as f:
        gp2d_data = pickle.load(f)
    with open(base_path / 'data/processed/bazin_features_cache.pkl', 'rb') as f:
        bazin_cache = pickle.load(f)
    with open(base_path / 'data/processed/research_features_cache.pkl', 'rb') as f:
        research_cache = pickle.load(f)

    # Merge all
    train_all = train_base.merge(tde_cached['train'], on='object_id', how='left')
    train_all = train_all.merge(gp2d_data['train'], on='object_id', how='left')
    train_all = train_all.merge(bazin_cache['train'], on='object_id', how='left')
    train_all = train_all.merge(research_cache['train'], on='object_id', how='left')

    test_all = test_base.merge(tde_cached['test'], on='object_id', how='left')
    test_all = test_all.merge(gp2d_data['test'], on='object_id', how='left')
    test_all = test_all.merge(bazin_cache['test'], on='object_id', how='left')
    test_all = test_all.merge(research_cache['test'], on='object_id', how='left')

    print(f"   Training: {len(train_ids)} objects ({np.sum(y)} TDE)")

    return {
        'train_all': train_all,
        'test_all': test_all,
        'train_ids': train_ids,
        'test_ids': test_ids,
        'y': y,
        'sample_weights': sample_weights
    }


def get_feature_set(model_type='lgb_minimal'):
    """Get feature set based on model type."""
    # Load v34a features
    with open(base_path / 'data/processed/v34a_artifacts.pkl', 'rb') as f:
        v34a = pickle.load(f)
    v34a_features = v34a['feature_names']

    # Remove adversarial-discriminative features
    adv_discriminative = ['all_rise_time', 'all_asymmetry']
    base_features = [f for f in v34a_features if f not in adv_discriminative]

    # Minimal research features (from v114d success)
    minimal_research = [
        'nuclear_concentration', 'nuclear_smoothness',
        'g_r_color_at_peak', 'r_i_color_at_peak',
        'mhps_10_100_ratio', 'mhps_30_100_ratio'
    ]

    if model_type == 'lgb_minimal':
        return base_features + minimal_research
    elif model_type == 'xgb_baseline':
        return base_features
    elif model_type == 'xgb_minimal':
        return base_features + minimal_research
    else:
        return base_features


def optimize_lightgbm(data, feature_names, n_trials=100):
    """Optuna optimization for LightGBM."""
    print("\n" + "=" * 60)
    print("OPTUNA: LightGBM Optimization")
    print("=" * 60)

    X_train = data['train_all'][feature_names].values
    y = data['y']
    sample_weights = data['sample_weights']

    X_train = np.nan_to_num(X_train, nan=np.nan, posinf=1e10, neginf=-1e10)

    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    folds = list(skf.split(X_train, y))

    scale_pos_weight = len(y[y==0]) / len(y[y==1])

    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'verbose': -1,
            'n_jobs': -1,
            'random_state': 42,
            'scale_pos_weight': scale_pos_weight,

            # Constrained search space (prevent overfitting)
            'num_leaves': trial.suggest_int('num_leaves', 8, 20),
            'max_depth': trial.suggest_int('max_depth', 3, 5),
            'learning_rate': trial.suggest_float('learning_rate', 0.015, 0.04),
            'n_estimators': trial.suggest_int('n_estimators', 400, 700),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.35, 0.6),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.75),
            'bagging_freq': trial.suggest_int('bagging_freq', 3, 7),
            'reg_alpha': trial.suggest_float('reg_alpha', 1.5, 6.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 2.0, 8.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 30, 60),
        }

        oof_preds = np.zeros(len(y))

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            fold_weights = sample_weights[train_idx]

            train_data = lgb.Dataset(X_tr, label=y_tr, weight=fold_weights)
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

        # Find best threshold
        best_f1 = 0
        for t in np.linspace(0.03, 0.5, 50):
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

    print(f"\n   Running {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials, callbacks=[callback])

    print(f"\n   Best trial: {study.best_trial.number}")
    print(f"   Best OOF F1: {study.best_value:.4f}")
    print(f"\n   Best parameters:")
    for k, v in study.best_params.items():
        print(f"      {k}: {v}")

    return study


def optimize_xgboost(data, feature_names, n_trials=100):
    """Optuna optimization for XGBoost."""
    print("\n" + "=" * 60)
    print("OPTUNA: XGBoost Optimization")
    print("=" * 60)

    X_train = data['train_all'][feature_names].values
    y = data['y']
    sample_weights = data['sample_weights']

    X_train = np.nan_to_num(X_train, nan=np.nan, posinf=1e10, neginf=-1e10)

    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    folds = list(skf.split(X_train, y))

    scale_pos_weight = len(y[y==0]) / len(y[y==1])

    def objective(trial):
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'hist',
            'random_state': 42,
            'n_jobs': -1,
            'scale_pos_weight': scale_pos_weight,

            # Constrained search space
            'max_depth': trial.suggest_int('max_depth', 3, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.015, 0.04),
            'subsample': trial.suggest_float('subsample', 0.65, 0.85),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.85),
            'min_child_weight': trial.suggest_int('min_child_weight', 2, 8),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 1.5),
            'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 4.0),
        }

        oof_preds = np.zeros(len(y))

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            fold_weights = sample_weights[train_idx]

            dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=fold_weights)
            dval = xgb.DMatrix(X_val, label=y_val)

            model = xgb.train(
                params,
                dtrain,
                num_boost_round=600,
                evals=[(dval, 'val')],
                early_stopping_rounds=50,
                verbose_eval=False
            )
            oof_preds[val_idx] = model.predict(dval)

        # Find best threshold
        best_f1 = 0
        for t in np.linspace(0.03, 0.5, 50):
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

    print(f"\n   Running {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials, callbacks=[callback])

    print(f"\n   Best trial: {study.best_trial.number}")
    print(f"   Best OOF F1: {study.best_value:.4f}")
    print(f"\n   Best parameters:")
    for k, v in study.best_params.items():
        print(f"      {k}: {v}")

    return study


def train_final_model(data, feature_names, params, model_type, version_name):
    """Train final model with best parameters and create submission."""
    print(f"\n   Training final {model_type} model...")

    X_train = data['train_all'][feature_names].values
    X_test = data['test_all'][feature_names].values
    y = data['y']
    sample_weights = data['sample_weights']
    test_ids = data['test_ids']

    X_train = np.nan_to_num(X_train, nan=np.nan, posinf=1e10, neginf=-1e10)
    X_test = np.nan_to_num(X_test, nan=np.nan, posinf=1e10, neginf=-1e10)

    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(y))
    test_preds = np.zeros((len(X_test), n_folds))
    fold_f1s = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y), 1):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        fold_weights = sample_weights[train_idx]

        if model_type == 'lgb':
            train_data = lgb.Dataset(X_tr, label=y_tr, weight=fold_weights, feature_name=feature_names)
            val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_data)

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
            test_preds[:, fold-1] = model.predict(X_test, num_iteration=model.best_iteration)
        else:  # xgb
            dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=fold_weights, feature_names=feature_names)
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
            dtest = xgb.DMatrix(X_test, feature_names=feature_names)

            model = xgb.train(
                params,
                dtrain,
                num_boost_round=600,
                evals=[(dval, 'val')],
                early_stopping_rounds=50,
                verbose_eval=False
            )
            oof_preds[val_idx] = model.predict(dval)
            test_preds[:, fold-1] = model.predict(dtest)

        # Fold F1
        best_fold_f1 = 0
        for t in np.linspace(0.03, 0.5, 50):
            f1 = f1_score(y_val, (oof_preds[val_idx] > t).astype(int))
            if f1 > best_fold_f1:
                best_fold_f1 = f1
        fold_f1s.append(best_fold_f1)

    # Find optimal threshold
    best_f1 = 0
    best_thresh = 0.1
    for t in np.linspace(0.03, 0.5, 200):
        f1 = f1_score(y, (oof_preds > t).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    print(f"   OOF F1: {best_f1:.4f} @ threshold={best_thresh:.3f}")
    print(f"   Fold F1s: {[f'{f:.4f}' for f in fold_f1s]}")
    print(f"   Fold std: {np.std(fold_f1s):.4f}")

    # Create submission
    test_avg = test_preds.mean(axis=1)
    test_binary = (test_avg > best_thresh).astype(int)

    submission = pd.DataFrame({
        'object_id': test_ids,
        'target': test_binary
    })

    filename = f"submission_{version_name}.csv"
    submission.to_csv(base_path / f'submissions/{filename}', index=False)
    print(f"   Saved: {filename} (TDEs: {test_binary.sum()})")

    return {
        'oof_f1': best_f1,
        'threshold': best_thresh,
        'fold_f1s': fold_f1s,
        'fold_std': np.std(fold_f1s),
        'oof_preds': oof_preds,
        'test_preds': test_avg,
        'params': params
    }


def main():
    parser = argparse.ArgumentParser(description='Optuna hyperparameter tuning')
    parser.add_argument('--model', type=str, default='both',
                       choices=['lgb', 'xgb', 'both'],
                       help='Model type to tune')
    parser.add_argument('--trials', type=int, default=80,
                       help='Number of Optuna trials')
    args = parser.parse_args()

    print("=" * 80)
    print("MALLORN v116: Optuna Hyperparameter Tuning")
    print("=" * 80)

    # Load data
    data = load_data()

    results = {}

    if args.model in ['lgb', 'both']:
        # LightGBM tuning
        feature_names = get_feature_set('lgb_minimal')
        feature_names = [f for f in feature_names if f in data['train_all'].columns]
        print(f"\n   LightGBM features: {len(feature_names)}")

        lgb_study = optimize_lightgbm(data, feature_names, n_trials=args.trials)

        # Train final model with best params
        best_lgb_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'verbose': -1,
            'n_jobs': -1,
            'random_state': 42,
            'scale_pos_weight': len(data['y'][data['y']==0]) / len(data['y'][data['y']==1]),
            **lgb_study.best_params
        }

        lgb_result = train_final_model(data, feature_names, best_lgb_params, 'lgb', 'v116_lgb_optuna')
        results['lgb'] = {
            'study': lgb_study,
            'result': lgb_result
        }

    if args.model in ['xgb', 'both']:
        # XGBoost tuning
        feature_names = get_feature_set('xgb_baseline')
        feature_names = [f for f in feature_names if f in data['train_all'].columns]
        print(f"\n   XGBoost features: {len(feature_names)}")

        xgb_study = optimize_xgboost(data, feature_names, n_trials=args.trials)

        # Train final model with best params
        best_xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'hist',
            'random_state': 42,
            'n_jobs': -1,
            'scale_pos_weight': len(data['y'][data['y']==0]) / len(data['y'][data['y']==1]),
            **xgb_study.best_params
        }

        xgb_result = train_final_model(data, feature_names, best_xgb_params, 'xgb', 'v116_xgb_optuna')
        results['xgb'] = {
            'study': xgb_study,
            'result': xgb_result
        }

    # Save results
    with open(base_path / 'data/processed/v116_optuna_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print("\n" + "=" * 80)
    print("OPTUNA TUNING COMPLETE")
    print("=" * 80)

    for model_name, model_results in results.items():
        print(f"\n   {model_name.upper()}:")
        print(f"      Best OOF F1: {model_results['result']['oof_f1']:.4f}")
        print(f"      Fold std: {model_results['result']['fold_std']:.4f}")


if __name__ == "__main__":
    main()

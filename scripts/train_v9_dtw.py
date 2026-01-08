"""
MALLORN v9: DTW Shape Features

Adds Dynamic Time Warping features to capture lightcurve shape similarity.
Key insight: TDEs have characteristic shapes that can be matched regardless
of time dilation (redshift) or observation cadence.

Combines:
- 120 selected features (from v6b)
- 25 TDE physics features (from v7)
- ~21 DTW shape features (NEW)
- Optuna-tuned hyperparameters (from v8)
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from utils.data_loader import load_all_data
from features.dtw_features import create_templates, extract_dtw_features


def find_optimal_threshold(y_true, y_prob):
    """Find threshold that maximizes F1 score."""
    best_f1, best_thresh = 0, 0.5
    for t in np.arange(0.05, 0.95, 0.01):
        f1 = f1_score(y_true, (y_prob >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    return best_thresh, best_f1


def main():
    print("=" * 60)
    print("MALLORN v9: DTW Shape Features")
    print("=" * 60)

    base_path = Path(__file__).parent.parent

    # 1. Load existing features and Optuna params
    print("\n1. Loading existing features and Optuna parameters...")

    # Load Optuna-tuned hyperparameters
    optuna_path = base_path / 'data/processed/optuna_results.pkl'
    with open(optuna_path, 'rb') as f:
        optuna_results = pickle.load(f)
    xgb_best = optuna_results['xgb_best_params']
    lgb_best = optuna_results['lgb_best_params']
    cat_best = optuna_results['cat_best_params']

    # Load data
    data = load_all_data()

    # Load v6 features (120 best)
    cached = pd.read_pickle(base_path / 'data/processed/features_v4_cache.pkl')
    train_features = cached['train_features']
    test_features = cached['test_features']

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
    test_tde = tde_cached['test']
    tde_cols = [c for c in train_tde.columns if c != 'object_id']

    print(f"   Base features: {len(selected_120)}")
    print(f"   TDE physics features: {len(tde_cols)}")

    # 2. Extract DTW features
    print("\n2. Extracting DTW shape features...")

    dtw_cache = base_path / 'data/processed/dtw_features_cache.pkl'

    if dtw_cache.exists():
        print("   Loading cached DTW features...")
        dtw_cached = pd.read_pickle(dtw_cache)
        train_dtw = dtw_cached['train']
        test_dtw = dtw_cached['test']
        templates = dtw_cached['templates']
    else:
        # Create templates from training data
        print("   Creating TDE/non-TDE templates...")
        templates = create_templates(data['train_lc'], data['train_meta'])

        # Extract DTW features for training
        print("   Processing training data...")
        train_dtw = extract_dtw_features(
            data['train_lc'],
            data['train_meta']['object_id'].tolist(),
            templates
        )

        # Extract DTW features for test
        print("   Processing test data...")
        test_dtw = extract_dtw_features(
            data['test_lc'],
            data['test_meta']['object_id'].tolist(),
            templates
        )

        # Cache results
        pd.to_pickle({
            'train': train_dtw,
            'test': test_dtw,
            'templates': templates
        }, dtw_cache)
        print(f"   Cached DTW features to {dtw_cache}")

    dtw_cols = [c for c in train_dtw.columns if c != 'object_id']
    print(f"   DTW features: {len(dtw_cols)}")

    # 3. Combine all features
    print("\n3. Combining all features...")

    # Merge all features
    train_combined = train_features[['object_id'] + selected_120].merge(
        train_tde, on='object_id', how='left'
    ).merge(
        train_dtw, on='object_id', how='left'
    )

    test_combined = test_features[['object_id'] + selected_120].merge(
        test_tde, on='object_id', how='left'
    ).merge(
        test_dtw, on='object_id', how='left'
    )

    # Add target
    train_combined = train_combined.merge(
        data['train_meta'][['object_id', 'target']], on='object_id'
    )

    all_feature_cols = selected_120 + tde_cols + dtw_cols
    print(f"   Total features: {len(all_feature_cols)}")
    print(f"     - Base: {len(selected_120)}")
    print(f"     - TDE physics: {len(tde_cols)}")
    print(f"     - DTW shape: {len(dtw_cols)}")

    # Prepare data
    X = train_combined[all_feature_cols].values
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    y = train_combined['target'].values

    n_neg, n_pos = np.bincount(y)
    scale_pos_weight = n_neg / n_pos
    print(f"\n   Samples: {len(y)} ({n_pos} TDE, {n_neg} non-TDE)")

    # 4. Prepare model parameters
    print("\n4. Preparing Optuna-tuned model parameters...")

    xgb_params = {
        'max_depth': xgb_best['max_depth'],
        'learning_rate': xgb_best['learning_rate'],
        'n_estimators': xgb_best['n_estimators'],
        'min_child_weight': xgb_best['min_child_weight'],
        'subsample': xgb_best['subsample'],
        'colsample_bytree': xgb_best['colsample_bytree'],
        'reg_alpha': xgb_best['reg_alpha'],
        'reg_lambda': xgb_best['reg_lambda'],
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    }

    lgb_params = {
        'max_depth': lgb_best['max_depth'],
        'learning_rate': lgb_best['learning_rate'],
        'n_estimators': lgb_best['n_estimators'],
        'num_leaves': lgb_best['num_leaves'],
        'min_child_samples': lgb_best['min_child_samples'],
        'subsample': lgb_best['subsample'],
        'colsample_bytree': lgb_best['colsample_bytree'],
        'reg_alpha': lgb_best['reg_alpha'],
        'reg_lambda': lgb_best['reg_lambda'],
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    cat_params = {
        'depth': cat_best['depth'],
        'learning_rate': cat_best['learning_rate'],
        'iterations': cat_best['iterations'],
        'l2_leaf_reg': cat_best['l2_leaf_reg'],
        'border_count': cat_best['border_count'],
        'scale_pos_weight': scale_pos_weight,
        'random_seed': 42,
        'verbose': False,
        'allow_writing_files': False
    }

    # 5. Train ensemble
    print("\n5. Training 3-model ensemble...")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    xgb_oof = np.zeros(len(y))
    lgb_oof = np.zeros(len(y))
    cat_oof = np.zeros(len(y))

    xgb_models, lgb_models, cat_models = [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # XGBoost
        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        xgb_oof[val_idx] = xgb_model.predict_proba(X_val)[:, 1]
        xgb_models.append(xgb_model)

        # LightGBM
        lgb_model = lgb.LGBMClassifier(**lgb_params)
        lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        lgb_oof[val_idx] = lgb_model.predict_proba(X_val)[:, 1]
        lgb_models.append(lgb_model)

        # CatBoost
        cat_model = CatBoostClassifier(**cat_params)
        cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=100)
        cat_oof[val_idx] = cat_model.predict_proba(X_val)[:, 1]
        cat_models.append(cat_model)

        # Fold results
        fold_ens = (xgb_oof[val_idx] + lgb_oof[val_idx] + cat_oof[val_idx]) / 3
        _, fold_f1 = find_optimal_threshold(y_val, fold_ens)
        print(f"   Fold {fold+1}: F1={fold_f1:.4f}")

    # 6. Optimize ensemble weights
    print("\n6. Optimizing ensemble weights...")

    best_f1, best_weights = 0, (1/3, 1/3, 1/3)
    for w1 in np.arange(0.15, 0.55, 0.05):
        for w2 in np.arange(0.15, 0.55, 0.05):
            w3 = 1 - w1 - w2
            if w3 < 0.1 or w3 > 0.55:
                continue
            weighted = w1 * xgb_oof + w2 * lgb_oof + w3 * cat_oof
            _, f1 = find_optimal_threshold(y, weighted)
            if f1 > best_f1:
                best_f1, best_weights = f1, (w1, w2, w3)

    ens_oof = best_weights[0] * xgb_oof + best_weights[1] * lgb_oof + best_weights[2] * cat_oof
    best_thresh, best_f1 = find_optimal_threshold(y, ens_oof)

    print(f"   Best weights: XGB={best_weights[0]:.2f}, LGB={best_weights[1]:.2f}, CAT={best_weights[2]:.2f}")
    print(f"   OOF F1: {best_f1:.4f} @ threshold={best_thresh:.2f}")

    # Individual model OOF scores
    _, xgb_f1 = find_optimal_threshold(y, xgb_oof)
    _, lgb_f1 = find_optimal_threshold(y, lgb_oof)
    _, cat_f1 = find_optimal_threshold(y, cat_oof)
    print(f"\n   Individual OOF F1: XGB={xgb_f1:.4f}, LGB={lgb_f1:.4f}, CAT={cat_f1:.4f}")

    # Confusion matrix
    final_preds = (ens_oof >= best_thresh).astype(int)
    cm = confusion_matrix(y, final_preds)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n   Confusion: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"   Precision: {precision_score(y, final_preds):.4f}")
    print(f"   Recall: {recall_score(y, final_preds):.4f}")

    # 7. Analyze DTW feature importance
    print("\n7. DTW Feature Importance Analysis:")

    importance = np.mean([m.feature_importances_ for m in xgb_models], axis=0)
    importance_df_new = pd.DataFrame({
        'feature': all_feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)

    # Show DTW features
    dtw_importance = importance_df_new[importance_df_new['feature'].isin(dtw_cols)]
    print("\n   DTW feature importance (top 10):")
    for _, row in dtw_importance.head(10).iterrows():
        print(f"   {row['importance']:.4f}: {row['feature']}")

    total_dtw_imp = dtw_importance['importance'].sum()
    total_imp = importance_df_new['importance'].sum()
    print(f"\n   DTW features contribution: {total_dtw_imp/total_imp*100:.1f}%")

    # 8. Create submission
    print("\n8. Creating submission...")

    X_test = test_combined[all_feature_cols].values
    X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

    xgb_test = np.mean([m.predict_proba(X_test)[:, 1] for m in xgb_models], axis=0)
    lgb_test = np.mean([m.predict_proba(X_test)[:, 1] for m in lgb_models], axis=0)
    cat_test = np.mean([m.predict_proba(X_test)[:, 1] for m in cat_models], axis=0)

    test_probs = best_weights[0] * xgb_test + best_weights[1] * lgb_test + best_weights[2] * cat_test
    test_preds = (test_probs >= best_thresh).astype(int)

    submission = pd.DataFrame({
        'object_id': test_combined['object_id'],
        'target': test_preds
    })

    submission_path = base_path / 'submissions' / 'submission_v9_dtw.csv'
    submission.to_csv(submission_path, index=False)
    print(f"   Saved to {submission_path}")
    print(f"   Predictions: {test_preds.sum()} TDEs / {len(test_preds)} total")

    # 9. Save models
    models_path = base_path / 'data/processed/models_v9.pkl'
    with open(models_path, 'wb') as f:
        pickle.dump({
            'xgb_models': xgb_models,
            'lgb_models': lgb_models,
            'cat_models': cat_models,
            'feature_cols': all_feature_cols,
            'best_weights': best_weights,
            'best_thresh': best_thresh,
            'dtw_templates': templates
        }, f)
    print(f"   Models saved to {models_path}")

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nVersion History:")
    print(f"  v8 (Optuna-tuned): OOF=0.6262, LB=0.6481 (145 features)")
    print(f"  v9 (DTW shapes):   OOF={best_f1:.4f} ({len(all_feature_cols)} features)")
    print(f"\nDTW impact: {(best_f1 - 0.6262)/0.6262*100:+.2f}% OOF change")
    print(f"DTW features added: {len(dtw_cols)}")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
MALLORN v18: GP Features + Enhanced Colors

Builds on v8 (best GBM, F1=0.6481 LB) by adding:
- GP length scale features (21 features)
- Enhanced color features with more time points and temperature tracking
- Color curvature and stability metrics

This follows the PLAsTiCC 1st place strategy: better features > complex models.
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
from features.gaussian_process import extract_gp_features
from features.colors import extract_color_features


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
    print("MALLORN v18: GP Features + Enhanced Colors")
    print("=" * 60)

    base_path = Path(__file__).parent.parent

    # 1. Load data
    print("\n1. Loading data...")
    data = load_all_data()
    train_lc = data['train_lc']
    test_lc = data['test_lc']
    train_meta = data['train_meta']
    test_meta = data['test_meta']

    train_ids = train_meta['object_id'].tolist()
    test_ids = test_meta['object_id'].tolist()

    print(f"   Train: {len(train_ids)} objects")
    print(f"   Test: {len(test_ids)} objects")

    # 2. Load v8 base features (already computed)
    print("\n2. Loading v8 base features...")
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

    print(f"   Base features: {len(selected_120)} + {len(tde_cols)} TDE physics")

    # 3. Extract new GP features
    print("\n3. Extracting GP length scale features...")
    gp_cache_path = base_path / 'data/processed/gp_features_cache.pkl'

    if gp_cache_path.exists():
        print("   Loading from cache...")
        with open(gp_cache_path, 'rb') as f:
            gp_cache = pickle.load(f)
        train_gp = gp_cache['train']
        test_gp = gp_cache['test']
    else:
        print("   Computing GP features (this may take a while)...")
        train_gp = extract_gp_features(train_lc, train_meta, train_ids, verbose=True)
        test_gp = extract_gp_features(test_lc, test_meta, test_ids, verbose=True)

        with open(gp_cache_path, 'wb') as f:
            pickle.dump({'train': train_gp, 'test': test_gp}, f)
        print(f"   Saved to {gp_cache_path}")

    gp_cols = [c for c in train_gp.columns if c != 'object_id']
    print(f"   GP features: {len(gp_cols)}")

    # 4. Extract enhanced color features
    print("\n4. Extracting enhanced color features...")
    color_cache_path = base_path / 'data/processed/enhanced_colors_cache.pkl'

    if color_cache_path.exists():
        print("   Loading from cache...")
        with open(color_cache_path, 'rb') as f:
            color_cache = pickle.load(f)
        train_colors = color_cache['train']
        test_colors = color_cache['test']
    else:
        print("   Computing enhanced color features...")
        train_colors = extract_color_features(train_lc, train_ids)
        test_colors = extract_color_features(test_lc, test_ids)

        with open(color_cache_path, 'wb') as f:
            pickle.dump({'train': train_colors, 'test': test_colors}, f)
        print(f"   Saved to {color_cache_path}")

    color_cols = [c for c in train_colors.columns if c != 'object_id']
    print(f"   Enhanced color features: {len(color_cols)}")

    # 5. Combine all features
    print("\n5. Combining features...")

    train_combined = train_features[['object_id'] + selected_120].copy()
    train_combined = train_combined.merge(train_tde, on='object_id', how='left')
    train_combined = train_combined.merge(train_gp, on='object_id', how='left')
    train_combined = train_combined.merge(train_colors, on='object_id', how='left')
    train_combined = train_combined.merge(
        train_meta[['object_id', 'target']], on='object_id'
    )

    test_combined = test_features[['object_id'] + selected_120].copy()
    test_combined = test_combined.merge(test_tde, on='object_id', how='left')
    test_combined = test_combined.merge(test_gp, on='object_id', how='left')
    test_combined = test_combined.merge(test_colors, on='object_id', how='left')

    # Identify which color columns are actually in the merged dataframe
    available_cols = set(train_combined.columns)

    # Filter to only columns that exist
    all_feature_cols = []
    for col in selected_120:
        if col in available_cols:
            all_feature_cols.append(col)
    for col in tde_cols:
        if col in available_cols and col not in all_feature_cols:
            all_feature_cols.append(col)
    for col in gp_cols:
        if col in available_cols and col not in all_feature_cols:
            all_feature_cols.append(col)
    for col in color_cols:
        if col in available_cols and col not in all_feature_cols:
            all_feature_cols.append(col)

    print(f"   Total features: {len(all_feature_cols)}")
    print(f"     - Base (v8): {len(selected_120)}")
    print(f"     - TDE physics: {len(tde_cols)}")
    print(f"     - GP features: {len(gp_cols)}")
    print(f"     - Enhanced colors: {len(color_cols)}")

    # Prepare data
    X = train_combined[all_feature_cols].values
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    y = train_combined['target'].values

    n_neg, n_pos = np.bincount(y)
    scale_pos_weight = n_neg / n_pos
    print(f"\n   Samples: {len(y)} ({n_pos} TDE, {n_neg} non-TDE)")

    # 6. Load tuned hyperparameters
    print("\n6. Loading Optuna-tuned hyperparameters...")
    optuna_path = base_path / 'data/processed/optuna_results.pkl'
    with open(optuna_path, 'rb') as f:
        optuna_results = pickle.load(f)

    xgb_best = optuna_results['xgb_best_params']
    lgb_best = optuna_results['lgb_best_params']
    cat_best = optuna_results['cat_best_params']

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

    # 7. Train ensemble
    print("\n7. Training 3-model ensemble...")

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

    # 8. Optimize ensemble weights
    print("\n8. Optimizing ensemble weights...")

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

    # 9. Feature importance analysis
    print("\n9. Analyzing feature importance...")

    # Get importance from LightGBM (it's usually most interpretable)
    importances = np.zeros(len(all_feature_cols))
    for model in lgb_models:
        importances += model.feature_importances_

    importance_df = pd.DataFrame({
        'feature': all_feature_cols,
        'importance': importances / 5
    }).sort_values('importance', ascending=False)

    print("\n   Top 20 features:")
    for i, row in importance_df.head(20).iterrows():
        feature_type = 'GP' if 'gp_' in row['feature'] else \
                       'COLOR' if any(x in row['feature'] for x in ['_r_', '_g_', 'temp_']) else \
                       'TDE' if row['feature'] in tde_cols else 'BASE'
        print(f"     {row['feature']}: {row['importance']:.0f} [{feature_type}]")

    # Count new features in top 50
    top_50 = importance_df.head(50)['feature'].tolist()
    gp_in_top50 = sum(1 for f in top_50 if 'gp_' in f)
    color_in_top50 = sum(1 for f in top_50 if any(x in f for x in ['_r_', '_g_', 'temp_', 'curvature', 'stability']))

    print(f"\n   New features in top 50: {gp_in_top50} GP, ~{color_in_top50} enhanced color")

    # 10. Create submission
    print("\n10. Creating submission...")

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

    submission_path = base_path / 'submissions' / 'submission_v18_gp_enhanced.csv'
    submission.to_csv(submission_path, index=False)
    print(f"   Saved to {submission_path}")
    print(f"   Predictions: {test_preds.sum()} TDEs / {len(test_preds)} total ({test_preds.sum()/len(test_preds)*100:.1f}%)")

    # 11. Save models
    models_path = base_path / 'data/processed/models_v18.pkl'
    with open(models_path, 'wb') as f:
        pickle.dump({
            'xgb_models': xgb_models,
            'lgb_models': lgb_models,
            'cat_models': cat_models,
            'feature_cols': all_feature_cols,
            'best_weights': best_weights,
            'best_thresh': best_thresh,
            'oof_probs': ens_oof,
            'oof_f1': best_f1,
            'importance_df': importance_df
        }, f)
    print(f"   Models saved to {models_path}")

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nVersion Comparison:")
    print(f"  v8 (Baseline):     OOF F1 = 0.6262, LB = 0.6481")
    print(f"  v18 (GP+Colors):   OOF F1 = {best_f1:.4f}")
    print(f"\n  Improvement: {(best_f1 - 0.6262) / 0.6262 * 100:+.2f}%")
    print(f"\nNew features added: {len(gp_cols)} GP + {len(color_cols)} enhanced colors")
    print("=" * 60)


if __name__ == "__main__":
    main()

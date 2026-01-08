"""
MALLORN v7: TDE Physics Features

Adding literature-informed TDE-specific features:
1. Color variance (TDEs have constant color, SNe redden)
2. Temperature stability (TDEs stay ~12,000K)
3. Late-time decay slope (t^-5/12 for TDEs)
4. Rise characteristics

Based on:
- Gezari 2021 (arxiv:2104.14580)
- van Velzen 2020 (arxiv:2008.05461)
- ALeRCE TDE classifier (arxiv:2503.19698)
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
from features.tde_physics import extract_tde_physics_features


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
    print("MALLORN v7: TDE Physics Features")
    print("=" * 60)

    base_path = Path(__file__).parent.parent

    # 1. Load data
    print("\n1. Loading data...")
    data = load_all_data()

    # Load v6 features (120 best)
    cached = pd.read_pickle(base_path / 'data/processed/features_v4_cache.pkl')
    train_features = cached['train_features']
    test_features = cached['test_features']

    # Load feature selection results
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

    print(f"  Base features (v6): {len(selected_120)}")

    # 2. Extract new TDE physics features
    print("\n2. Extracting TDE physics features...")

    tde_cache = base_path / 'data/processed/tde_physics_cache.pkl'

    if tde_cache.exists():
        print("  Loading cached TDE features...")
        tde_cached = pd.read_pickle(tde_cache)
        train_tde = tde_cached['train']
        test_tde = tde_cached['test']
    else:
        print("  Processing training data...")
        train_tde = extract_tde_physics_features(
            data['train_lc'],
            data['train_meta']['object_id'].tolist()
        )

        print("  Processing test data...")
        test_tde = extract_tde_physics_features(
            data['test_lc'],
            data['test_meta']['object_id'].tolist()
        )

        pd.to_pickle({'train': train_tde, 'test': test_tde}, tde_cache)
        print(f"  Cached to {tde_cache}")

    tde_feature_cols = [c for c in train_tde.columns if c != 'object_id']
    print(f"  New TDE features: {len(tde_feature_cols)}")

    # 3. Combine features
    print("\n3. Combining features...")

    # Merge TDE features with base features
    train_combined = train_features[['object_id'] + selected_120].merge(
        train_tde, on='object_id', how='left'
    )
    test_combined = test_features[['object_id'] + selected_120].merge(
        test_tde, on='object_id', how='left'
    )

    # Add target
    train_combined = train_combined.merge(
        data['train_meta'][['object_id', 'target']], on='object_id'
    )

    all_feature_cols = selected_120 + tde_feature_cols
    print(f"  Total features: {len(all_feature_cols)} (120 base + {len(tde_feature_cols)} TDE)")

    # 4. Prepare data
    X = train_combined[all_feature_cols].values
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    y = train_combined['target'].values

    n_neg, n_pos = np.bincount(y)
    scale_pos_weight = n_neg / n_pos

    print(f"\n4. Training data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"   Class balance: {n_pos} TDE / {n_neg} non-TDE")

    # 5. Train ensemble
    print("\n5. Training 3-model ensemble...")

    xgb_params = {
        'max_depth': 5, 'learning_rate': 0.015, 'n_estimators': 1000,
        'subsample': 0.7, 'colsample_bytree': 0.5, 'min_child_weight': 3,
        'scale_pos_weight': scale_pos_weight, 'random_state': 42,
        'n_jobs': -1, 'verbosity': 0
    }
    lgb_params = {
        'max_depth': 5, 'learning_rate': 0.015, 'n_estimators': 1000,
        'subsample': 0.7, 'colsample_bytree': 0.5,
        'scale_pos_weight': scale_pos_weight, 'random_state': 42,
        'n_jobs': -1, 'verbose': -1
    }
    cat_params = {
        'depth': 5, 'learning_rate': 0.03, 'iterations': 800,
        'l2_leaf_reg': 3.0, 'scale_pos_weight': scale_pos_weight,
        'random_seed': 42, 'verbose': False, 'allow_writing_files': False
    }

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

    # 6. Find best ensemble weights
    print("\n6. Optimizing ensemble weights...")

    best_f1, best_weights = 0, (1/3, 1/3, 1/3)
    for w1 in np.arange(0.2, 0.5, 0.05):
        for w2 in np.arange(0.2, 0.5, 0.05):
            w3 = 1 - w1 - w2
            if w3 < 0.1 or w3 > 0.5:
                continue
            weighted = w1 * xgb_oof + w2 * lgb_oof + w3 * cat_oof
            _, f1 = find_optimal_threshold(y, weighted)
            if f1 > best_f1:
                best_f1, best_weights = f1, (w1, w2, w3)

    ens_oof = best_weights[0] * xgb_oof + best_weights[1] * lgb_oof + best_weights[2] * cat_oof
    best_thresh, best_f1 = find_optimal_threshold(y, ens_oof)

    print(f"   Best weights: XGB={best_weights[0]:.2f}, LGB={best_weights[1]:.2f}, CAT={best_weights[2]:.2f}")
    print(f"   OOF F1: {best_f1:.4f} @ threshold={best_thresh:.2f}")

    # Confusion matrix
    final_preds = (ens_oof >= best_thresh).astype(int)
    cm = confusion_matrix(y, final_preds)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n   Confusion: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"   Precision: {precision_score(y, final_preds):.4f}")
    print(f"   Recall: {recall_score(y, final_preds):.4f}")

    # 7. Analyze new TDE feature importance
    print("\n7. TDE Physics Feature Importance:")

    # Get XGBoost importance
    importance = np.mean([m.feature_importances_ for m in xgb_models], axis=0)
    importance_df_new = pd.DataFrame({
        'feature': all_feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)

    # Show TDE features
    tde_importance = importance_df_new[importance_df_new['feature'].isin(tde_feature_cols)]
    print("\n   New TDE feature importance:")
    for _, row in tde_importance.head(10).iterrows():
        print(f"   {row['importance']:.4f}: {row['feature']}")

    total_tde_imp = tde_importance['importance'].sum()
    total_imp = importance_df_new['importance'].sum()
    print(f"\n   TDE features contribution: {total_tde_imp/total_imp*100:.1f}%")

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

    submission_path = base_path / 'submissions' / 'submission_v7_tde_physics.csv'
    submission.to_csv(submission_path, index=False)
    print(f"   Saved to {submission_path}")
    print(f"   Predictions: {test_preds.sum()} TDEs / {len(test_preds)} total")

    # 9. Save models
    models_path = base_path / 'data/processed/models_v7.pkl'
    with open(models_path, 'wb') as f:
        pickle.dump({
            'xgb_models': xgb_models,
            'lgb_models': lgb_models,
            'cat_models': cat_models,
            'feature_cols': all_feature_cols,
            'best_weights': best_weights,
            'best_thresh': best_thresh
        }, f)

    # Summary
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"v6b OOF F1: 0.6092 (120 features)")
    print(f"v7 OOF F1: {best_f1:.4f} ({len(all_feature_cols)} features)")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
MALLORN v5: Adding CatBoost for Ensemble Diversity

Building on v4's 271 features, we add CatBoost as a third model.
CatBoost uses ordered boosting which can reduce overfitting on small datasets.

Expected improvement: Ensemble diversity should boost F1 by ~1-3%
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import xgboost as xgb
import lightgbm as lgb

# Try to import CatBoost
try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    print("CatBoost not installed. Run: pip install catboost")
    HAS_CATBOOST = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from utils.data_loader import load_all_data
from features.statistical import extract_statistical_features, add_metadata_features
from features.colors import extract_color_features
from features.lightcurve_shape import extract_shape_features
from features.physics_based import extract_physics_features


def load_or_extract_features(data, cache_path):
    """Load cached features or extract new ones."""
    if cache_path.exists():
        print(f"Loading cached features from {cache_path}")
        cached = pd.read_pickle(cache_path)
        return cached['train_features'], cached['test_features']

    print("Extracting training features...")
    train_ids = data['train_meta']['object_id'].tolist()

    # Statistical features
    print("  Statistical features...")
    train_stat = extract_statistical_features(data['train_lc'], train_ids)
    train_stat = add_metadata_features(train_stat, data['train_meta'])

    # Color features
    print("  Color features...")
    train_colors = extract_color_features(data['train_lc'], train_ids)

    # Shape features
    print("  Shape features...")
    train_shapes = extract_shape_features(data['train_lc'], train_ids)

    # Physics features
    print("  Physics features...")
    train_physics = extract_physics_features(data['train_lc'], data['train_meta'], train_ids)

    # Merge all
    train_features = train_stat.merge(train_colors, on='object_id', how='left')
    train_features = train_features.merge(train_shapes, on='object_id', how='left')
    train_features = train_features.merge(train_physics, on='object_id', how='left')

    print("\nExtracting test features...")
    test_ids = data['test_meta']['object_id'].tolist()

    print("  Statistical features...")
    test_stat = extract_statistical_features(data['test_lc'], test_ids)
    test_stat = add_metadata_features(test_stat, data['test_meta'])

    print("  Color features...")
    test_colors = extract_color_features(data['test_lc'], test_ids)

    print("  Shape features...")
    test_shapes = extract_shape_features(data['test_lc'], test_ids)

    print("  Physics features...")
    test_physics = extract_physics_features(data['test_lc'], data['test_meta'], test_ids)

    test_features = test_stat.merge(test_colors, on='object_id', how='left')
    test_features = test_features.merge(test_shapes, on='object_id', how='left')
    test_features = test_features.merge(test_physics, on='object_id', how='left')

    # Cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle({'train_features': train_features, 'test_features': test_features}, cache_path)
    print(f"\nCached features to {cache_path}")

    return train_features, test_features


def find_optimal_threshold(y_true, y_prob, thresholds=np.arange(0.05, 0.95, 0.01)):
    """Find threshold that maximizes F1 score."""
    best_f1, best_thresh = 0, 0.5
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return best_thresh, best_f1


def main():
    if not HAS_CATBOOST:
        print("Please install CatBoost: pip install catboost")
        return

    print("=" * 60)
    print("MALLORN v5: Adding CatBoost for Ensemble Diversity")
    print("=" * 60)

    # 1. Load data
    print("\n1. Loading data...")
    data = load_all_data()

    # 2. Load/extract features (use v4 cache if available)
    print("\n2. Loading features...")
    cache_path = Path(__file__).parent.parent / 'data' / 'processed' / 'features_v4_cache.pkl'
    train_features, test_features = load_or_extract_features(data, cache_path)

    # 3. Prepare training data
    print("\n3. Preparing training data...")
    train_data = train_features.merge(
        data['train_meta'][['object_id', 'target']],
        on='object_id'
    )

    feature_cols = [c for c in train_features.columns
                   if c not in ['object_id', 'target', 'SpecType', 'English Translation', 'split']]

    X = train_data[feature_cols].values
    y = train_data['target'].values

    # Handle NaN/inf
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    print(f"Training data shape: X={X.shape}, y={y.shape}")
    print(f"Class distribution: {np.bincount(y)}")

    # Calculate class weight
    n_neg, n_pos = np.bincount(y)
    scale_pos_weight = n_neg / n_pos
    print(f"Scale pos weight: {scale_pos_weight:.2f}")

    # 4. Train models with 5-fold CV
    print("\n4. Training models...")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    xgb_models = []
    lgb_models = []
    cat_models = []

    xgb_oof = np.zeros(len(y))
    lgb_oof = np.zeros(len(y))
    cat_oof = np.zeros(len(y))

    # XGBoost params (same as v4)
    xgb_params = {
        'max_depth': 5,
        'learning_rate': 0.015,
        'n_estimators': 1200,
        'subsample': 0.7,
        'colsample_bytree': 0.5,
        'min_child_weight': 3,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    }

    # LightGBM params (same as v4)
    lgb_params = {
        'max_depth': 5,
        'learning_rate': 0.015,
        'n_estimators': 1200,
        'subsample': 0.7,
        'colsample_bytree': 0.5,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    # CatBoost params (tuned for this problem)
    cat_params = {
        'depth': 5,
        'learning_rate': 0.03,
        'iterations': 1000,
        'l2_leaf_reg': 3.0,
        'border_count': 128,
        'scale_pos_weight': scale_pos_weight,
        'random_seed': 42,
        'verbose': False,
        'allow_writing_files': False
    }

    print("\n" + "=" * 50)
    print("Training XGBoost, LightGBM, and CatBoost")
    print("=" * 50)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # XGBoost
        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        xgb_pred = xgb_model.predict_proba(X_val)[:, 1]
        xgb_oof[val_idx] = xgb_pred
        xgb_models.append(xgb_model)

        # LightGBM
        lgb_model = lgb.LGBMClassifier(**lgb_params)
        lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        lgb_pred = lgb_model.predict_proba(X_val)[:, 1]
        lgb_oof[val_idx] = lgb_pred
        lgb_models.append(lgb_model)

        # CatBoost
        cat_model = CatBoostClassifier(**cat_params)
        cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=100)
        cat_pred = cat_model.predict_proba(X_val)[:, 1]
        cat_oof[val_idx] = cat_pred
        cat_models.append(cat_model)

        # Report fold results
        xgb_thresh, xgb_f1 = find_optimal_threshold(y_val, xgb_pred)
        lgb_thresh, lgb_f1 = find_optimal_threshold(y_val, lgb_pred)
        cat_thresh, cat_f1 = find_optimal_threshold(y_val, cat_pred)

        print(f"Fold {fold+1}: XGB={xgb_f1:.4f} | LGB={lgb_f1:.4f} | CAT={cat_f1:.4f}")

    # 5. Evaluate individual models
    print("\n" + "=" * 50)
    print("Individual Model OOF Results")
    print("=" * 50)

    for name, oof_probs in [('XGBoost', xgb_oof), ('LightGBM', lgb_oof), ('CatBoost', cat_oof)]:
        thresh, f1_opt = find_optimal_threshold(y, oof_probs)
        f1_50 = f1_score(y, (oof_probs >= 0.5).astype(int))
        print(f"\n{name}:")
        print(f"  F1 @ 0.5: {f1_50:.4f}")
        print(f"  F1 @ optimal ({thresh:.2f}): {f1_opt:.4f}")

    # 6. Ensemble strategies
    print("\n" + "=" * 50)
    print("Ensemble Strategies")
    print("=" * 50)

    # Simple average (2 models - original)
    ens_2_probs = (xgb_oof + lgb_oof) / 2
    thresh_2, f1_2 = find_optimal_threshold(y, ens_2_probs)

    # Simple average (3 models)
    ens_3_probs = (xgb_oof + lgb_oof + cat_oof) / 3
    thresh_3, f1_3 = find_optimal_threshold(y, ens_3_probs)

    # Weighted average (optimized weights)
    best_f1_weighted = 0
    best_weights = (1/3, 1/3, 1/3)
    for w1 in np.arange(0.2, 0.5, 0.05):
        for w2 in np.arange(0.2, 0.5, 0.05):
            w3 = 1 - w1 - w2
            if w3 < 0.1 or w3 > 0.5:
                continue
            weighted = w1 * xgb_oof + w2 * lgb_oof + w3 * cat_oof
            _, f1 = find_optimal_threshold(y, weighted)
            if f1 > best_f1_weighted:
                best_f1_weighted = f1
                best_weights = (w1, w2, w3)

    ens_weighted_probs = best_weights[0] * xgb_oof + best_weights[1] * lgb_oof + best_weights[2] * cat_oof
    thresh_w, f1_w = find_optimal_threshold(y, ens_weighted_probs)

    print(f"\n2-Model Ensemble (XGB+LGB):")
    print(f"  F1 @ optimal ({thresh_2:.2f}): {f1_2:.4f}")

    print(f"\n3-Model Ensemble (equal weights):")
    print(f"  F1 @ optimal ({thresh_3:.2f}): {f1_3:.4f}")

    print(f"\n3-Model Ensemble (optimized weights):")
    print(f"  Weights: XGB={best_weights[0]:.2f}, LGB={best_weights[1]:.2f}, CAT={best_weights[2]:.2f}")
    print(f"  F1 @ optimal ({thresh_w:.2f}): {f1_w:.4f}")

    # Choose best ensemble
    if f1_w >= f1_3:
        best_ens_probs = ens_weighted_probs
        best_thresh = thresh_w
        best_f1 = f1_w
        ensemble_type = "weighted"
    else:
        best_ens_probs = ens_3_probs
        best_thresh = thresh_3
        best_f1 = f1_3
        ensemble_type = "equal"

    print(f"\nBest ensemble: {ensemble_type} (F1={best_f1:.4f})")

    # Confusion matrix
    final_preds = (best_ens_probs >= best_thresh).astype(int)
    cm = confusion_matrix(y, final_preds)
    tn, fp, fn, tp = cm.ravel()

    print(f"\nConfusion Matrix:")
    print(f"[[{tn} {fp}]")
    print(f" [{fn} {tp}]]")
    print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"  Precision: {precision_score(y, final_preds):.4f}")
    print(f"  Recall: {recall_score(y, final_preds):.4f}")

    # 7. Model agreement analysis
    print("\n" + "=" * 50)
    print("Model Agreement Analysis")
    print("=" * 50)

    xgb_preds = (xgb_oof >= 0.15).astype(int)
    lgb_preds = (lgb_oof >= 0.15).astype(int)
    cat_preds = (cat_oof >= 0.15).astype(int)

    all_agree = ((xgb_preds == lgb_preds) & (lgb_preds == cat_preds)).sum()
    two_agree = ((xgb_preds == lgb_preds) | (lgb_preds == cat_preds) | (xgb_preds == cat_preds)).sum()

    print(f"All 3 models agree: {all_agree}/{len(y)} ({all_agree/len(y)*100:.1f}%)")
    print(f"At least 2 models agree: {two_agree}/{len(y)} ({two_agree/len(y)*100:.1f}%)")

    # 8. Create submission
    print("\n6. Creating submission...")

    X_test = test_features[feature_cols].values
    X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

    # Get predictions from all models
    xgb_test_probs = np.zeros(len(X_test))
    lgb_test_probs = np.zeros(len(X_test))
    cat_test_probs = np.zeros(len(X_test))

    for model in xgb_models:
        xgb_test_probs += model.predict_proba(X_test)[:, 1] / len(xgb_models)
    for model in lgb_models:
        lgb_test_probs += model.predict_proba(X_test)[:, 1] / len(lgb_models)
    for model in cat_models:
        cat_test_probs += model.predict_proba(X_test)[:, 1] / len(cat_models)

    # Use best ensemble strategy
    if ensemble_type == "weighted":
        test_probs = best_weights[0] * xgb_test_probs + best_weights[1] * lgb_test_probs + best_weights[2] * cat_test_probs
    else:
        test_probs = (xgb_test_probs + lgb_test_probs + cat_test_probs) / 3

    test_preds = (test_probs >= best_thresh).astype(int)

    # Create submission
    submission = pd.DataFrame({
        'object_id': test_features['object_id'],
        'target': test_preds
    })

    submission_path = Path(__file__).parent.parent / 'submissions' / 'submission_v5_catboost.csv'
    submission.to_csv(submission_path, index=False)

    print(f"\nSubmission saved to {submission_path}")
    print(f"Using threshold: {best_thresh:.2f}")
    print(f"Predictions: {test_preds.sum()} TDEs / {len(test_preds)} total")

    # 9. Save models
    models_path = Path(__file__).parent.parent / 'data' / 'processed' / 'models_v5.pkl'
    with open(models_path, 'wb') as f:
        pickle.dump({
            'xgb_models': xgb_models,
            'lgb_models': lgb_models,
            'cat_models': cat_models,
            'feature_cols': feature_cols,
            'best_weights': best_weights,
            'best_thresh': best_thresh,
            'ensemble_type': ensemble_type
        }, f)
    print(f"Models saved to {models_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"v4 OOF F1: 0.585 -> v5 OOF F1: {best_f1:.4f}")
    print(f"Improvement: {(best_f1 - 0.585) / 0.585 * 100:+.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()

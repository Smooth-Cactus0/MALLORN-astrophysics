"""
Version 2: Training with color features + threshold optimization.

Improvements over baseline:
1. Color features (49 new features capturing color evolution)
2. Threshold optimization for F1 score
3. Feature importance analysis

Usage:
    python scripts/train_v2_colors.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import pickle
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from utils.data_loader import load_all_data
from features.statistical import extract_statistical_features, add_metadata_features
from features.colors import extract_color_features

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


def extract_all_features_v2(data: dict, cache_path: Path = None) -> tuple:
    """Extract statistical + color features for train and test sets."""
    if cache_path and cache_path.exists():
        print(f"Loading cached features from {cache_path}")
        cached = pd.read_pickle(cache_path)
        return cached['train_features'], cached['test_features']

    # Training features
    print("Extracting training features...")
    train_ids = data['train_meta']['object_id'].tolist()

    print("  Statistical features...")
    train_stat = extract_statistical_features(data['train_lc'], train_ids)
    train_stat = add_metadata_features(train_stat, data['train_meta'])

    print("  Color features...")
    train_color = extract_color_features(data['train_lc'], train_ids)

    # Merge
    train_features = train_stat.merge(train_color, on='object_id', how='left')

    # Test features
    print("\nExtracting test features...")
    test_ids = data['test_meta']['object_id'].tolist()

    print("  Statistical features...")
    test_stat = extract_statistical_features(data['test_lc'], test_ids)
    test_stat = add_metadata_features(test_stat, data['test_meta'])

    print("  Color features...")
    test_color = extract_color_features(data['test_lc'], test_ids)

    # Merge
    test_features = test_stat.merge(test_color, on='object_id', how='left')

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle({'train_features': train_features, 'test_features': test_features}, cache_path)
        print(f"\nCached features to {cache_path}")

    return train_features, test_features


def prepare_data(train_features: pd.DataFrame, train_meta: pd.DataFrame):
    """Prepare X, y arrays for training."""
    train_data = train_features.merge(
        train_meta[['object_id', 'target']],
        on='object_id'
    )

    # Feature columns (exclude identifiers and target)
    exclude_cols = ['object_id', 'target', 'peak_mjd']
    feature_cols = [c for c in train_data.columns if c not in exclude_cols]

    X = train_data[feature_cols].values
    y = train_data['target'].values
    object_ids = train_data['object_id'].values

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    print(f"Training data shape: X={X.shape}, y={y.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    print(f"Total features: {len(feature_cols)}")

    return X, y, object_ids, feature_cols


def find_optimal_threshold(y_true, y_probs):
    """Find threshold that maximizes F1 score."""
    thresholds = np.arange(0.05, 0.95, 0.01)
    f1_scores = []

    for t in thresholds:
        preds = (y_probs > t).astype(int)
        f1_scores.append(f1_score(y_true, preds))

    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]


def train_xgboost(X, y, n_splits=5):
    """Train XGBoost with stratified K-fold CV."""
    if not HAS_XGB:
        return None, None

    print("\n" + "="*50)
    print("Training XGBoost")
    print("="*50)

    scale_pos_weight = (y == 0).sum() / (y == 1).sum()

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'scale_pos_weight': scale_pos_weight,
        'max_depth': 6,
        'learning_rate': 0.03,  # Lower LR for more features
        'n_estimators': 800,    # More trees
        'subsample': 0.8,
        'colsample_bytree': 0.7,  # Slightly lower for more features
        'min_child_weight': 3,
        'gamma': 0.1,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    }

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))
    models = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        val_preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds

        # Find fold-specific optimal threshold
        opt_t, opt_f1 = find_optimal_threshold(y_val, val_preds)
        print(f"Fold {fold+1}: F1@0.5={f1_score(y_val, (val_preds>0.5).astype(int)):.4f}, "
              f"F1@opt({opt_t:.2f})={opt_f1:.4f}")
        models.append(model)

    # Overall results
    opt_threshold, opt_f1 = find_optimal_threshold(y, oof_preds)

    print(f"\nXGBoost OOF Results:")
    print(f"  F1 @ 0.5: {f1_score(y, (oof_preds>0.5).astype(int)):.4f}")
    print(f"  F1 @ optimal ({opt_threshold:.2f}): {opt_f1:.4f}")
    print(f"  Precision @ opt: {precision_score(y, (oof_preds>opt_threshold).astype(int)):.4f}")
    print(f"  Recall @ opt: {recall_score(y, (oof_preds>opt_threshold).astype(int)):.4f}")

    return models, oof_preds


def train_lightgbm(X, y, n_splits=5):
    """Train LightGBM with stratified K-fold CV."""
    if not HAS_LGB:
        return None, None

    print("\n" + "="*50)
    print("Training LightGBM")
    print("="*50)

    scale_pos_weight = (y == 0).sum() / (y == 1).sum()

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'scale_pos_weight': scale_pos_weight,
        'max_depth': 6,
        'learning_rate': 0.03,
        'n_estimators': 800,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))
    models = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

        val_preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds

        opt_t, opt_f1 = find_optimal_threshold(y_val, val_preds)
        print(f"Fold {fold+1}: F1@0.5={f1_score(y_val, (val_preds>0.5).astype(int)):.4f}, "
              f"F1@opt({opt_t:.2f})={opt_f1:.4f}")
        models.append(model)

    opt_threshold, opt_f1 = find_optimal_threshold(y, oof_preds)

    print(f"\nLightGBM OOF Results:")
    print(f"  F1 @ 0.5: {f1_score(y, (oof_preds>0.5).astype(int)):.4f}")
    print(f"  F1 @ optimal ({opt_threshold:.2f}): {opt_f1:.4f}")
    print(f"  Precision @ opt: {precision_score(y, (oof_preds>opt_threshold).astype(int)):.4f}")
    print(f"  Recall @ opt: {recall_score(y, (oof_preds>opt_threshold).astype(int)):.4f}")

    return models, oof_preds


def analyze_feature_importance(models, feature_cols, top_n=30):
    """Analyze and display top feature importances."""
    print("\n" + "="*50)
    print(f"Top {top_n} Feature Importances")
    print("="*50)

    # Average importance across folds
    importances = np.zeros(len(feature_cols))
    for model in models:
        importances += model.feature_importances_
    importances /= len(models)

    # Sort by importance
    indices = np.argsort(importances)[::-1]

    print("\nRank | Feature | Importance")
    print("-" * 50)
    for i in range(min(top_n, len(feature_cols))):
        idx = indices[i]
        print(f"{i+1:4d} | {feature_cols[idx]:30s} | {importances[idx]:.4f}")

    # Categorize features
    color_feats = [f for f in feature_cols if any(x in f for x in ['_g_', '_r_', '_i_', '_u_', '_z_', 'slope', 'peak_flux', 'lag'])]
    stat_feats = [f for f in feature_cols if f not in color_feats]

    color_imp = sum(importances[feature_cols.index(f)] for f in color_feats if f in feature_cols)
    stat_imp = sum(importances[feature_cols.index(f)] for f in stat_feats if f in feature_cols)

    print(f"\nFeature Category Importance:")
    print(f"  Color features: {color_imp:.4f} ({color_imp/(color_imp+stat_imp)*100:.1f}%)")
    print(f"  Statistical features: {stat_imp:.4f} ({stat_imp/(color_imp+stat_imp)*100:.1f}%)")


def main():
    print("="*60)
    print("MALLORN v2: Color Features + Threshold Optimization")
    print("="*60)

    project_root = Path(__file__).parent.parent
    cache_path = project_root / 'data' / 'processed' / 'features_v2_cache.pkl'
    submission_path = project_root / 'submissions' / 'submission_v2_colors.csv'

    # Load data
    print("\n1. Loading data...")
    data = load_all_data()

    # Extract features
    print("\n2. Extracting features (statistical + color)...")
    train_features, test_features = extract_all_features_v2(data, cache_path)

    # Prepare data
    print("\n3. Preparing training data...")
    X, y, train_ids, feature_cols = prepare_data(train_features, data['train_meta'])

    # Prepare test data
    test_feature_cols = [c for c in feature_cols if c in test_features.columns]
    X_test = test_features[test_feature_cols].values
    X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)
    test_ids = test_features['object_id'].values

    # Train models
    print("\n4. Training models...")
    xgb_models, xgb_oof = train_xgboost(X, y)
    lgb_models, lgb_oof = train_lightgbm(X, y)

    # Feature importance analysis
    if xgb_models:
        analyze_feature_importance(xgb_models, feature_cols)

    # Ensemble predictions
    print("\n5. Generating ensemble predictions...")
    ensemble_oof = (xgb_oof + lgb_oof) / 2
    opt_threshold, opt_f1 = find_optimal_threshold(y, ensemble_oof)

    print(f"\nEnsemble OOF Results:")
    print(f"  F1 @ 0.5: {f1_score(y, (ensemble_oof>0.5).astype(int)):.4f}")
    print(f"  F1 @ optimal ({opt_threshold:.2f}): {opt_f1:.4f}")
    print(f"\nConfusion Matrix @ optimal threshold:")
    print(confusion_matrix(y, (ensemble_oof > opt_threshold).astype(int)))

    # Generate test predictions
    xgb_test = np.zeros(len(X_test))
    lgb_test = np.zeros(len(X_test))

    for model in xgb_models:
        xgb_test += model.predict_proba(X_test)[:, 1]
    xgb_test /= len(xgb_models)

    for model in lgb_models:
        lgb_test += model.predict_proba(X_test)[:, 1]
    lgb_test /= len(lgb_models)

    test_preds = (xgb_test + lgb_test) / 2

    # Create submission with optimal threshold
    print("\n6. Creating submission...")
    submission_path.parent.mkdir(parents=True, exist_ok=True)

    binary_preds = (test_preds > opt_threshold).astype(int)
    submission = pd.DataFrame({
        'object_id': test_ids,
        'target': binary_preds
    })
    submission.to_csv(submission_path, index=False)

    print(f"\nSubmission saved to {submission_path}")
    print(f"Using threshold: {opt_threshold:.2f}")
    print(f"Predictions: {binary_preds.sum()} TDEs / {len(binary_preds)} total")

    # Save models and config
    models_path = project_root / 'data' / 'processed' / 'models_v2.pkl'
    with open(models_path, 'wb') as f:
        pickle.dump({
            'xgb_models': xgb_models,
            'lgb_models': lgb_models,
            'feature_cols': feature_cols,
            'optimal_threshold': opt_threshold,
            'oof_f1': opt_f1
        }, f)
    print(f"Models saved to {models_path}")

    print("\n" + "="*60)
    print("Training complete!")
    print(f"Expected improvement: OOF F1 {opt_f1:.4f} vs baseline 0.30")
    print("="*60)


if __name__ == "__main__":
    main()

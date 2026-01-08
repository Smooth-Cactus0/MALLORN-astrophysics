"""
Baseline model training for MALLORN Astronomical Classification.

This script:
1. Extracts statistical features from all training lightcurves
2. Trains XGBoost and LightGBM models with stratified K-fold CV
3. Generates predictions on test set
4. Creates submission file

Usage:
    python scripts/train_baseline.py
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

# Check for ML libraries
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Warning: xgboost not installed")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("Warning: lightgbm not installed")


def extract_all_features(data: dict, cache_path: Path = None) -> tuple:
    """
    Extract features for train and test sets.
    Uses cache if available to speed up repeated runs.
    """
    if cache_path and cache_path.exists():
        print(f"Loading cached features from {cache_path}")
        cached = pd.read_pickle(cache_path)
        return cached['train_features'], cached['test_features']

    print("Extracting training features...")
    train_ids = data['train_meta']['object_id'].tolist()
    train_features = extract_statistical_features(data['train_lc'], train_ids)
    train_features = add_metadata_features(train_features, data['train_meta'])

    print("Extracting test features...")
    test_ids = data['test_meta']['object_id'].tolist()
    test_features = extract_statistical_features(data['test_lc'], test_ids)
    test_features = add_metadata_features(test_features, data['test_meta'])

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle({'train_features': train_features, 'test_features': test_features}, cache_path)
        print(f"Cached features to {cache_path}")

    return train_features, test_features


def prepare_data(train_features: pd.DataFrame, train_meta: pd.DataFrame):
    """Prepare X, y arrays for training."""
    # Merge with target
    train_data = train_features.merge(
        train_meta[['object_id', 'target']],
        on='object_id'
    )

    # Feature columns (exclude object_id and target)
    feature_cols = [c for c in train_data.columns if c not in ['object_id', 'target']]

    X = train_data[feature_cols].values
    y = train_data['target'].values
    object_ids = train_data['object_id'].values

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    print(f"Training data shape: X={X.shape}, y={y.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    print(f"Features: {len(feature_cols)}")

    return X, y, object_ids, feature_cols


def train_xgboost(X, y, n_splits=5):
    """Train XGBoost with stratified K-fold CV."""
    if not HAS_XGB:
        print("XGBoost not available")
        return None, None

    print("\n" + "="*50)
    print("Training XGBoost")
    print("="*50)

    # Calculate scale_pos_weight for class imbalance
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    print(f"Scale pos weight: {scale_pos_weight:.2f}")

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'scale_pos_weight': scale_pos_weight,
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    }

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))
    models = []
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        val_preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds

        # Use threshold 0.5 for predictions
        val_pred_binary = (val_preds > 0.5).astype(int)
        fold_f1 = f1_score(y_val, val_pred_binary)
        fold_scores.append(fold_f1)

        print(f"Fold {fold+1}: F1={fold_f1:.4f}")
        models.append(model)

    # Overall OOF score
    oof_binary = (oof_preds > 0.5).astype(int)
    overall_f1 = f1_score(y, oof_binary)
    precision = precision_score(y, oof_binary)
    recall = recall_score(y, oof_binary)

    print(f"\nXGBoost OOF Results:")
    print(f"  F1 Score: {overall_f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  Mean CV F1: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y, oof_binary)}")

    return models, oof_preds


def train_lightgbm(X, y, n_splits=5):
    """Train LightGBM with stratified K-fold CV."""
    if not HAS_LGB:
        print("LightGBM not available")
        return None, None

    print("\n" + "="*50)
    print("Training LightGBM")
    print("="*50)

    # Calculate scale_pos_weight for class imbalance
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'scale_pos_weight': scale_pos_weight,
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))
    models = []
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
        )

        val_preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds

        val_pred_binary = (val_preds > 0.5).astype(int)
        fold_f1 = f1_score(y_val, val_pred_binary)
        fold_scores.append(fold_f1)

        print(f"Fold {fold+1}: F1={fold_f1:.4f}")
        models.append(model)

    # Overall OOF score
    oof_binary = (oof_preds > 0.5).astype(int)
    overall_f1 = f1_score(y, oof_binary)
    precision = precision_score(y, oof_binary)
    recall = recall_score(y, oof_binary)

    print(f"\nLightGBM OOF Results:")
    print(f"  F1 Score: {overall_f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  Mean CV F1: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y, oof_binary)}")

    return models, oof_preds


def generate_predictions(models, X_test, feature_cols):
    """Generate test predictions by averaging model predictions."""
    preds = np.zeros(len(X_test))

    for model in models:
        preds += model.predict_proba(X_test)[:, 1]

    preds /= len(models)
    return preds


def create_submission(test_ids, predictions, output_path, threshold=0.5):
    """Create submission file."""
    binary_preds = (predictions > threshold).astype(int)

    submission = pd.DataFrame({
        'object_id': test_ids,
        'target': binary_preds
    })

    submission.to_csv(output_path, index=False)
    print(f"\nSubmission saved to {output_path}")
    print(f"Predictions: {binary_preds.sum()} TDEs / {len(binary_preds)} total")

    return submission


def main():
    print("="*60)
    print("MALLORN Astronomical Classification - Baseline Training")
    print("="*60)

    # Paths
    project_root = Path(__file__).parent.parent
    cache_path = project_root / 'data' / 'processed' / 'features_cache.pkl'
    submission_path = project_root / 'submissions' / 'submission_baseline.csv'

    # Load data
    print("\n1. Loading data...")
    data = load_all_data()

    # Extract features
    print("\n2. Extracting features...")
    train_features, test_features = extract_all_features(data, cache_path)

    # Prepare training data
    print("\n3. Preparing training data...")
    X, y, train_ids, feature_cols = prepare_data(train_features, data['train_meta'])

    # Prepare test data
    X_test = test_features[[c for c in feature_cols if c in test_features.columns]].values
    X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)
    test_ids = test_features['object_id'].values

    # Train models
    print("\n4. Training models...")
    xgb_models, xgb_oof = train_xgboost(X, y)
    lgb_models, lgb_oof = train_lightgbm(X, y)

    # Generate test predictions
    print("\n5. Generating predictions...")
    if xgb_models and lgb_models:
        xgb_test_preds = generate_predictions(xgb_models, X_test, feature_cols)
        lgb_test_preds = generate_predictions(lgb_models, X_test, feature_cols)
        # Average ensemble
        test_preds = (xgb_test_preds + lgb_test_preds) / 2
        print("Using XGBoost + LightGBM ensemble")
    elif xgb_models:
        test_preds = generate_predictions(xgb_models, X_test, feature_cols)
        print("Using XGBoost only")
    elif lgb_models:
        test_preds = generate_predictions(lgb_models, X_test, feature_cols)
        print("Using LightGBM only")
    else:
        print("No models trained!")
        return

    # Create submission
    print("\n6. Creating submission...")
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    create_submission(test_ids, test_preds, submission_path)

    # Save models
    models_path = project_root / 'data' / 'processed' / 'models.pkl'
    with open(models_path, 'wb') as f:
        pickle.dump({
            'xgb_models': xgb_models,
            'lgb_models': lgb_models,
            'feature_cols': feature_cols
        }, f)
    print(f"Models saved to {models_path}")

    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)


if __name__ == "__main__":
    main()

"""
MALLORN v6: Feature Selection Optimization

Test different feature counts with the full 3-model ensemble to find the sweet spot.
Goal: Reduce noise without losing signal.

Strategy: Compare 100, 150, 200, and full (273) features
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


def train_ensemble(X, y, n_splits=5):
    """Train 3-model ensemble and return OOF predictions."""
    n_neg, n_pos = np.bincount(y)
    scale_pos_weight = n_neg / n_pos

    xgb_params = {
        'max_depth': 5, 'learning_rate': 0.015, 'n_estimators': 1000,
        'subsample': 0.7, 'colsample_bytree': 0.5, 'min_child_weight': 3,
        'scale_pos_weight': scale_pos_weight, 'random_state': 42, 'n_jobs': -1, 'verbosity': 0
    }
    lgb_params = {
        'max_depth': 5, 'learning_rate': 0.015, 'n_estimators': 1000,
        'subsample': 0.7, 'colsample_bytree': 0.5,
        'scale_pos_weight': scale_pos_weight, 'random_state': 42, 'n_jobs': -1, 'verbose': -1
    }
    cat_params = {
        'depth': 5, 'learning_rate': 0.03, 'iterations': 800,
        'l2_leaf_reg': 3.0, 'scale_pos_weight': scale_pos_weight,
        'random_seed': 42, 'verbose': False, 'allow_writing_files': False
    }

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

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

    # Find best ensemble weights
    best_f1 = 0
    best_weights = (1/3, 1/3, 1/3)
    for w1 in np.arange(0.2, 0.5, 0.05):
        for w2 in np.arange(0.2, 0.5, 0.05):
            w3 = 1 - w1 - w2
            if w3 < 0.1 or w3 > 0.5:
                continue
            weighted = w1 * xgb_oof + w2 * lgb_oof + w3 * cat_oof
            _, f1 = find_optimal_threshold(y, weighted)
            if f1 > best_f1:
                best_f1 = f1
                best_weights = (w1, w2, w3)

    ens_oof = best_weights[0] * xgb_oof + best_weights[1] * lgb_oof + best_weights[2] * cat_oof

    return ens_oof, best_weights, (xgb_models, lgb_models, cat_models)


def main():
    print("=" * 60)
    print("MALLORN v6: Feature Selection Optimization")
    print("=" * 60)

    # Load data
    print("\n1. Loading data...")
    base_path = Path(__file__).parent.parent

    cache_path = base_path / 'data' / 'processed' / 'features_v4_cache.pkl'
    cached = pd.read_pickle(cache_path)
    train_features = cached['train_features']
    test_features = cached['test_features']

    selection_path = base_path / 'data' / 'processed' / 'selected_features.pkl'
    selection = pd.read_pickle(selection_path)
    importance_df = selection['importance_df']
    high_corr_df = selection['high_corr_df']

    data = load_all_data()
    train_data = train_features.merge(data['train_meta'][['object_id', 'target']], on='object_id')

    models_path = base_path / 'data' / 'processed' / 'models_v5.pkl'
    with open(models_path, 'rb') as f:
        models = pickle.load(f)
    all_feature_cols = models['feature_cols']

    y = train_data['target'].values

    # Identify correlated features to drop
    corr_to_drop = set()
    for _, row in high_corr_df.iterrows():
        if row['feature_1'] not in corr_to_drop:
            corr_to_drop.add(row['feature_2'])

    # Get features sorted by importance, excluding correlated ones
    clean_features = importance_df[~importance_df['feature'].isin(corr_to_drop)]
    clean_features = clean_features.sort_values('combined', ascending=False)

    # Test different feature counts
    feature_counts = [80, 100, 120, 150, 180, 200, len(clean_features)]
    results = []

    print(f"\n2. Testing different feature counts...")
    print(f"   Available features after removing correlations: {len(clean_features)}")

    for n_features in feature_counts:
        n_features = min(n_features, len(clean_features))
        selected = clean_features.head(n_features)['feature'].tolist()

        # Prepare data
        X = train_data[selected].values
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

        print(f"\n   Testing {n_features} features...")
        ens_oof, weights, _ = train_ensemble(X, y)
        thresh, f1 = find_optimal_threshold(y, ens_oof)

        results.append({
            'n_features': n_features,
            'f1': f1,
            'threshold': thresh,
            'weights': weights
        })
        print(f"   -> F1={f1:.4f} @ thresh={thresh:.2f} | weights=({weights[0]:.2f},{weights[1]:.2f},{weights[2]:.2f})")

    # Find best
    results_df = pd.DataFrame(results)
    best_idx = results_df['f1'].idxmax()
    best_result = results_df.loc[best_idx]

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print("\n   Features  |   F1 Score  |  vs Full")
    print("   ----------|-------------|----------")
    full_f1 = results_df[results_df['n_features'] == results_df['n_features'].max()]['f1'].values[0]
    for _, row in results_df.iterrows():
        diff = (row['f1'] - full_f1) / full_f1 * 100
        marker = " <-- BEST" if row['n_features'] == best_result['n_features'] else ""
        print(f"   {row['n_features']:8d}  |   {row['f1']:.4f}    | {diff:+.2f}%{marker}")

    # Use best configuration
    best_n = int(best_result['n_features'])
    print(f"\n{'='*60}")
    print(f"BEST: {best_n} features with F1={best_result['f1']:.4f}")
    print(f"{'='*60}")

    # Train final model with best feature count
    print(f"\n3. Training final model with {best_n} features...")
    selected_features = clean_features.head(best_n)['feature'].tolist()

    X_train = train_data[selected_features].values
    X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)

    ens_oof, best_weights, (xgb_models, lgb_models, cat_models) = train_ensemble(X_train, y)
    best_thresh, best_f1 = find_optimal_threshold(y, ens_oof)

    # Confusion matrix
    final_preds = (ens_oof >= best_thresh).astype(int)
    cm = confusion_matrix(y, final_preds)
    tn, fp, fn, tp = cm.ravel()

    print(f"\nFinal Results:")
    print(f"  F1 @ optimal ({best_thresh:.2f}): {best_f1:.4f}")
    print(f"  Precision: {precision_score(y, final_preds):.4f}")
    print(f"  Recall: {recall_score(y, final_preds):.4f}")
    print(f"  Confusion: TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    # Create submission
    print("\n4. Creating submission...")
    X_test = test_features[selected_features].values
    X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

    xgb_test = np.mean([m.predict_proba(X_test)[:, 1] for m in xgb_models], axis=0)
    lgb_test = np.mean([m.predict_proba(X_test)[:, 1] for m in lgb_models], axis=0)
    cat_test = np.mean([m.predict_proba(X_test)[:, 1] for m in cat_models], axis=0)

    test_probs = best_weights[0] * xgb_test + best_weights[1] * lgb_test + best_weights[2] * cat_test
    test_preds = (test_probs >= best_thresh).astype(int)

    submission = pd.DataFrame({
        'object_id': test_features['object_id'],
        'target': test_preds
    })

    submission_path = base_path / 'submissions' / f'submission_v6_{best_n}features.csv'
    submission.to_csv(submission_path, index=False)

    print(f"\nSubmission saved to {submission_path}")
    print(f"Predictions: {test_preds.sum()} TDEs / {len(test_preds)} total")

    # Save models and config
    models_path = base_path / 'data' / 'processed' / 'models_v6.pkl'
    with open(models_path, 'wb') as f:
        pickle.dump({
            'xgb_models': xgb_models,
            'lgb_models': lgb_models,
            'cat_models': cat_models,
            'feature_cols': selected_features,
            'best_weights': best_weights,
            'best_thresh': best_thresh,
            'n_features': best_n
        }, f)
    print(f"Models saved to {models_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"v5 OOF F1: 0.6014 (273 features)")
    print(f"v6 OOF F1: {best_f1:.4f} ({best_n} features)")
    print(f"Feature reduction: {273 - best_n} fewer ({(273-best_n)/273*100:.0f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()

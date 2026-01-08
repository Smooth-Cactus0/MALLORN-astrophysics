"""
MALLORN Feature Selection Analysis

Goals:
1. Identify highly correlated features (redundant)
2. Rank features by importance across all 3 models
3. Find features that are consistently low-importance
4. Create a pruned feature set for v6

With 273 features, we likely have:
- Redundant features (high correlation)
- Noise features (near-zero importance)
- Unstable features (high variance across folds)
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from utils.data_loader import load_all_data


def load_features_and_models():
    """Load cached features and trained models."""
    base_path = Path(__file__).parent.parent

    # Load features
    cache_path = base_path / 'data' / 'processed' / 'features_v4_cache.pkl'
    cached = pd.read_pickle(cache_path)
    train_features = cached['train_features']
    test_features = cached['test_features']

    # Load models
    models_path = base_path / 'data' / 'processed' / 'models_v5.pkl'
    with open(models_path, 'rb') as f:
        models = pickle.load(f)

    return train_features, test_features, models


def analyze_correlations(X, feature_cols, threshold=0.95):
    """Find highly correlated feature pairs."""
    print(f"\n{'='*60}")
    print("CORRELATION ANALYSIS")
    print(f"{'='*60}")

    # Compute correlation matrix
    df = pd.DataFrame(X, columns=feature_cols)
    corr_matrix = df.corr().abs()

    # Find pairs above threshold
    high_corr_pairs = []
    for i in range(len(feature_cols)):
        for j in range(i+1, len(feature_cols)):
            if corr_matrix.iloc[i, j] >= threshold:
                high_corr_pairs.append({
                    'feature_1': feature_cols[i],
                    'feature_2': feature_cols[j],
                    'correlation': corr_matrix.iloc[i, j]
                })

    high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('correlation', ascending=False)

    print(f"\nFeatures with correlation >= {threshold}: {len(high_corr_df)} pairs")

    if len(high_corr_df) > 0:
        print("\nTop 20 correlated pairs:")
        for i, row in high_corr_df.head(20).iterrows():
            print(f"  {row['correlation']:.3f}: {row['feature_1']} <-> {row['feature_2']}")

    # Identify features to potentially drop (keep one from each correlated pair)
    features_to_drop = set()
    for _, row in high_corr_df.iterrows():
        # Keep the first feature, mark second for potential removal
        if row['feature_1'] not in features_to_drop:
            features_to_drop.add(row['feature_2'])

    print(f"\nFeatures that could be dropped (redundant): {len(features_to_drop)}")

    return high_corr_df, features_to_drop


def analyze_feature_importance(models, feature_cols):
    """Aggregate feature importance across all models and folds."""
    print(f"\n{'='*60}")
    print("FEATURE IMPORTANCE ANALYSIS")
    print(f"{'='*60}")

    importance_df = pd.DataFrame({'feature': feature_cols})

    # XGBoost importance
    xgb_importances = np.zeros((len(models['xgb_models']), len(feature_cols)))
    for i, model in enumerate(models['xgb_models']):
        xgb_importances[i] = model.feature_importances_
    importance_df['xgb_mean'] = xgb_importances.mean(axis=0)
    importance_df['xgb_std'] = xgb_importances.std(axis=0)

    # LightGBM importance
    lgb_importances = np.zeros((len(models['lgb_models']), len(feature_cols)))
    for i, model in enumerate(models['lgb_models']):
        lgb_importances[i] = model.feature_importances_
    importance_df['lgb_mean'] = lgb_importances.mean(axis=0)
    importance_df['lgb_std'] = lgb_importances.std(axis=0)

    # CatBoost importance
    cat_importances = np.zeros((len(models['cat_models']), len(feature_cols)))
    for i, model in enumerate(models['cat_models']):
        cat_importances[i] = model.feature_importances_
    importance_df['cat_mean'] = cat_importances.mean(axis=0)
    importance_df['cat_std'] = cat_importances.std(axis=0)

    # Normalize each model's importance to sum to 1
    importance_df['xgb_norm'] = importance_df['xgb_mean'] / importance_df['xgb_mean'].sum()
    importance_df['lgb_norm'] = importance_df['lgb_mean'] / importance_df['lgb_mean'].sum()
    importance_df['cat_norm'] = importance_df['cat_mean'] / importance_df['cat_mean'].sum()

    # Combined importance (average of normalized)
    importance_df['combined'] = (importance_df['xgb_norm'] + importance_df['lgb_norm'] + importance_df['cat_norm']) / 3

    # Stability score (lower std relative to mean = more stable)
    importance_df['stability'] = 1 / (1 + (importance_df['xgb_std'] / (importance_df['xgb_mean'] + 1e-10)))

    # Sort by combined importance
    importance_df = importance_df.sort_values('combined', ascending=False)

    print("\nTop 30 features by combined importance:")
    for i, row in importance_df.head(30).iterrows():
        print(f"  {row['combined']:.4f}: {row['feature']} (XGB:{row['xgb_norm']:.4f}, LGB:{row['lgb_norm']:.4f}, CAT:{row['cat_norm']:.4f})")

    # Find low-importance features
    cumsum = importance_df['combined'].cumsum()
    total = importance_df['combined'].sum()

    # Features that together contribute < 1% of total importance
    low_importance = importance_df[importance_df['combined'] < 0.001]['feature'].tolist()

    print(f"\nFeatures with < 0.1% importance: {len(low_importance)}")

    # Features contributing to top 80%, 90%, 95% of importance
    top_80 = importance_df[cumsum <= 0.80 * total].shape[0]
    top_90 = importance_df[cumsum <= 0.90 * total].shape[0]
    top_95 = importance_df[cumsum <= 0.95 * total].shape[0]

    print(f"\nCumulative importance coverage:")
    print(f"  Top {top_80} features cover 80% of importance")
    print(f"  Top {top_90} features cover 90% of importance")
    print(f"  Top {top_95} features cover 95% of importance")

    return importance_df, low_importance


def analyze_feature_categories(importance_df):
    """Analyze importance by feature category."""
    print(f"\n{'='*60}")
    print("FEATURE CATEGORY ANALYSIS")
    print(f"{'='*60}")

    # Categorize features
    categories = {
        'statistical': [],
        'color': [],
        'shape': [],
        'physics': [],
        'metadata': [],
        'other': []
    }

    for feat in importance_df['feature']:
        feat_lower = feat.lower()

        if any(x in feat_lower for x in ['stetson', 'sf_tau', 'sf_slope', 'bazin', 'temp_', 'rest_', 'excess_', 'snr']):
            categories['physics'].append(feat)
        elif any(x in feat_lower for x in ['_r_', 'g_r', 'r_i', 'i_z', 'u_g', 'color', 'slope_']):
            categories['color'].append(feat)
        elif any(x in feat_lower for x in ['rise', 'fade', 'asymm', 'duration', 'power_law', 'peak_time', 'lag']):
            categories['shape'].append(feat)
        elif feat in ['Z', 'EBV']:
            categories['metadata'].append(feat)
        elif any(x in feat_lower for x in ['mean', 'std', 'min', 'max', 'skew', 'kurt', 'amplitude', 'mad', 'beyond', 'n_obs']):
            categories['statistical'].append(feat)
        else:
            categories['other'].append(feat)

    print("\nFeatures by category:")
    for cat, feats in categories.items():
        total_imp = importance_df[importance_df['feature'].isin(feats)]['combined'].sum()
        print(f"  {cat}: {len(feats)} features, {total_imp:.1%} of total importance")

    # Top features per category
    print("\nTop 5 features per category:")
    for cat, feats in categories.items():
        if feats:
            cat_df = importance_df[importance_df['feature'].isin(feats)].head(5)
            print(f"\n  {cat.upper()}:")
            for _, row in cat_df.iterrows():
                print(f"    {row['combined']:.4f}: {row['feature']}")

    return categories


def select_features(importance_df, corr_features_to_drop, min_importance=0.0005, max_features=150):
    """Select optimal feature subset."""
    print(f"\n{'='*60}")
    print("FEATURE SELECTION")
    print(f"{'='*60}")

    # Start with all features sorted by importance
    candidates = importance_df.copy()

    # Remove low-importance features
    before = len(candidates)
    candidates = candidates[candidates['combined'] >= min_importance]
    print(f"After removing features with importance < {min_importance}: {before} -> {len(candidates)}")

    # Remove highly correlated features (keep more important one)
    before = len(candidates)
    candidates = candidates[~candidates['feature'].isin(corr_features_to_drop)]
    print(f"After removing correlated features: {before} -> {len(candidates)}")

    # Cap at max features if needed
    if len(candidates) > max_features:
        candidates = candidates.head(max_features)
        print(f"Capped at top {max_features} features")

    selected_features = candidates['feature'].tolist()

    print(f"\nFinal selected features: {len(selected_features)}")

    return selected_features


def evaluate_feature_subset(X, y, feature_cols, selected_features, n_splits=5):
    """Evaluate model performance with selected features."""
    print(f"\n{'='*60}")
    print("EVALUATING FEATURE SUBSET")
    print(f"{'='*60}")

    # Get indices of selected features
    selected_idx = [feature_cols.index(f) for f in selected_features if f in feature_cols]
    X_selected = X[:, selected_idx]

    print(f"Original features: {X.shape[1]}")
    print(f"Selected features: {X_selected.shape[1]}")

    # Class weight
    n_neg, n_pos = np.bincount(y)
    scale_pos_weight = n_neg / n_pos

    # Quick evaluation with XGBoost only
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_full = np.zeros(len(y))
    oof_selected = np.zeros(len(y))

    xgb_params = {
        'max_depth': 5,
        'learning_rate': 0.015,
        'n_estimators': 800,
        'subsample': 0.7,
        'colsample_bytree': 0.5,
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        # Full features
        model_full = xgb.XGBClassifier(**xgb_params)
        model_full.fit(X[train_idx], y[train_idx])
        oof_full[val_idx] = model_full.predict_proba(X[val_idx])[:, 1]

        # Selected features
        model_sel = xgb.XGBClassifier(**xgb_params)
        model_sel.fit(X_selected[train_idx], y[train_idx])
        oof_selected[val_idx] = model_sel.predict_proba(X_selected[val_idx])[:, 1]

    # Find optimal thresholds
    def find_best_f1(y_true, y_prob):
        best_f1, best_t = 0, 0.5
        for t in np.arange(0.05, 0.95, 0.01):
            f1 = f1_score(y_true, (y_prob >= t).astype(int))
            if f1 > best_f1:
                best_f1, best_t = f1, t
        return best_f1, best_t

    f1_full, t_full = find_best_f1(y, oof_full)
    f1_selected, t_selected = find_best_f1(y, oof_selected)

    print(f"\nXGBoost-only quick evaluation:")
    print(f"  Full features ({X.shape[1]}): F1={f1_full:.4f} @ threshold={t_full:.2f}")
    print(f"  Selected features ({X_selected.shape[1]}): F1={f1_selected:.4f} @ threshold={t_selected:.2f}")
    print(f"  Difference: {(f1_selected - f1_full) / f1_full * 100:+.2f}%")

    return f1_full, f1_selected, selected_idx


def main():
    print("=" * 60)
    print("MALLORN Feature Selection Analysis")
    print("=" * 60)

    # 1. Load data
    print("\n1. Loading data and models...")
    train_features, test_features, models = load_features_and_models()

    # Load metadata for target
    data = load_all_data()
    train_data = train_features.merge(data['train_meta'][['object_id', 'target']], on='object_id')

    feature_cols = models['feature_cols']
    X = train_data[feature_cols].values
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    y = train_data['target'].values

    print(f"Features: {len(feature_cols)}")
    print(f"Samples: {len(y)}")

    # 2. Correlation analysis
    high_corr_df, corr_features_to_drop = analyze_correlations(X, feature_cols, threshold=0.95)

    # 3. Feature importance analysis
    importance_df, low_importance = analyze_feature_importance(models, feature_cols)

    # 4. Category analysis
    categories = analyze_feature_categories(importance_df)

    # 5. Select features
    selected_features = select_features(
        importance_df,
        corr_features_to_drop,
        min_importance=0.0005,
        max_features=150
    )

    # 6. Evaluate subset
    f1_full, f1_selected, selected_idx = evaluate_feature_subset(X, y, feature_cols, selected_features)

    # 7. Save selected features
    output_path = Path(__file__).parent.parent / 'data' / 'processed' / 'selected_features.pkl'
    pd.to_pickle({
        'selected_features': selected_features,
        'importance_df': importance_df,
        'high_corr_df': high_corr_df,
        'categories': categories
    }, output_path)
    print(f"\nSaved feature selection results to {output_path}")

    # 8. Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Original features: {len(feature_cols)}")
    print(f"Highly correlated (redundant): {len(corr_features_to_drop)}")
    print(f"Low importance (< 0.1%): {len(low_importance)}")
    print(f"Selected features: {len(selected_features)}")
    print(f"\nExpected impact on F1: {(f1_selected - f1_full) / f1_full * 100:+.2f}%")

    # Print selected feature list for v6
    print("\n" + "=" * 60)
    print("SELECTED FEATURES FOR v6")
    print("=" * 60)
    for i, feat in enumerate(selected_features[:50]):
        imp = importance_df[importance_df['feature'] == feat]['combined'].values[0]
        print(f"  {i+1:3d}. {feat} ({imp:.4f})")
    if len(selected_features) > 50:
        print(f"  ... and {len(selected_features) - 50} more")


if __name__ == "__main__":
    main()
